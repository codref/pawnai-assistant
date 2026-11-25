"""Echo cancellation through PipeWire's ``module-echo-cancel`` (WebRTC AEC).

The assistant never cancels echo itself: it asks pipewire-pulse for a virtual
source/sink pair. Everything PAWN plays (beeps, Piper, the mpv radio) goes to
the echo-cancel *sink*; the wake loop records from the echo-cancel *source*,
which is the mic minus whatever went out through that sink. That is what makes
barge-in possible — saying the wake word over PAWN's own voice or the radio.

Requirements: PipeWire with pipewire-pulse and ``libspa-aec-webrtc`` (stock on
most distros), and a real mic *source* in PipeWire. A card left on an
output-only profile (the mic used through ``hw:X,Y`` instead) has none: pick
an "Analog Stereo Duplex" profile first (``pawn-assistant aec check``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from pawn_assistant.audio import routing

logger = logging.getLogger(__name__)


class AECUnavailable(RuntimeError):
    """Echo cancellation cannot be set up on this machine (reason in message)."""


@dataclass
class AECSettings:
    enabled: bool = False
    # webrtc | null (null = passthrough, handy to compare with/without).
    method: str = "webrtc"
    source_name: str = "pawn_ec_source"
    sink_name: str = "pawn_ec_sink"
    # Real devices behind the virtual pair. None = PipeWire defaults
    # (the default sink, and the default source unless that is a monitor).
    source_master: Optional[str] = None
    sink_master: Optional[str] = None
    # Extra WebRTC args, e.g. "webrtc.gain_control=false webrtc.noise_suppression=true".
    aec_args: str = ""
    # Leave the module loaded on exit (faster restarts, visible in pavucontrol).
    keep_loaded: bool = False

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "AECSettings":
        if not isinstance(raw, Mapping):
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class PulseState:
    sources: list[str] = field(default_factory=list)
    sinks: list[str] = field(default_factory=list)
    default_source: str = ""
    default_sink: str = ""
    server: str = ""


def _pactl(*args: str, timeout: float = 5.0) -> str:
    proc = subprocess.run(
        ["pactl", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if proc.returncode != 0:
        raise AECUnavailable(f"pactl {' '.join(args)}: {proc.stderr.strip() or proc.returncode}")
    return proc.stdout


def _short_names(listing: str) -> list[str]:
    names = []
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            names.append(parts[1])
    return names


def pulse_state() -> PulseState:
    if not shutil.which("pactl"):
        raise AECUnavailable("pactl not found (install pipewire-pulse / libpulse)")
    info = _pactl("info")
    state = PulseState(
        sources=_short_names(_pactl("list", "short", "sources")),
        sinks=_short_names(_pactl("list", "short", "sinks")),
    )
    for line in info.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if key == "Default Source":
            state.default_source = value
        elif key == "Default Sink":
            state.default_sink = value
        elif key == "Server Name":
            state.server = value
    return state


def mic_sources(state: PulseState, exclude: tuple[str, ...] = ()) -> list[str]:
    """Real capture sources: no sink monitors, none of our own virtual nodes."""
    return [s for s in state.sources if not s.endswith(".monitor") and s not in exclude]


def pick_source_master(state: PulseState, cfg: AECSettings) -> str:
    if cfg.source_master:
        if cfg.source_master not in state.sources:
            raise AECUnavailable(f"aec.source_master {cfg.source_master!r} is not a PipeWire source")
        return cfg.source_master
    mics = mic_sources(state, exclude=(cfg.source_name,))
    if not mics:
        raise AECUnavailable(
            "PipeWire has no microphone source (only sink monitors). Switch the "
            "sound card to a profile with an input, e.g. 'Analog Stereo Duplex' "
            "(pawn-assistant aec check --fix)."
        )
    if state.default_source in mics:
        return state.default_source
    return mics[0]


class EchoCancel:
    """Load (or reuse) a ``module-echo-cancel`` pair and route PAWN through it."""

    def __init__(self, cfg: AECSettings) -> None:
        self.cfg = cfg
        self.module_id: Optional[int] = None
        self.active = False
        self.reason = ""
        self.source_master = ""
        self.sink_master = ""
        self._prev_env: dict[str, Optional[str]] = {}

    def start(self) -> None:
        """Bring up AEC or raise :class:`AECUnavailable`."""
        cfg = self.cfg
        state = pulse_state()
        if "PipeWire" not in state.server:
            logger.warning("AEC: audio server %r is not PipeWire — trying anyway", state.server)
        have_source = cfg.source_name in state.sources
        have_sink = cfg.sink_name in state.sinks
        if have_source and have_sink:
            logger.info("AEC: reusing existing %s / %s", cfg.source_name, cfg.sink_name)
        else:
            self.source_master = pick_source_master(state, cfg)
            self.sink_master = cfg.sink_master or state.default_sink
            if cfg.sink_master and cfg.sink_master not in state.sinks:
                raise AECUnavailable(f"aec.sink_master {cfg.sink_master!r} is not a PipeWire sink")
            args = [
                "load-module",
                "module-echo-cancel",
                f"aec_method={cfg.method}",
                f"source_name={cfg.source_name}",
                f"sink_name={cfg.sink_name}",
                f"source_master={self.source_master}",
            ]
            if self.sink_master:
                args.append(f"sink_master={self.sink_master}")
            if cfg.aec_args:
                args.append(f"aec_args=\"{cfg.aec_args}\"")
            out = _pactl(*args).strip()
            try:
                self.module_id = int(out.splitlines()[-1])
            except (ValueError, IndexError):
                self.module_id = None
            logger.info(
                "AEC: loaded module-echo-cancel #%s (%s) mic=%s speaker=%s",
                self.module_id, cfg.method, self.source_master, self.sink_master or "default",
            )
            self._wait_for_nodes()
        self._route()
        self.active = True

    def _wait_for_nodes(self, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = pulse_state()
            if self.cfg.source_name in state.sources and self.cfg.sink_name in state.sinks:
                return
            time.sleep(0.1)
        self.stop()
        raise AECUnavailable(
            f"module-echo-cancel loaded but {self.cfg.source_name} did not appear "
            "(is libspa-aec-webrtc installed?)"
        )

    def _route(self) -> None:
        # libpulse clients we spawn or open (PortAudio "pulse", paplay, mpv
        # --ao=pulse) pick these up as their default devices.
        for key, value in (("PULSE_SOURCE", self.cfg.source_name), ("PULSE_SINK", self.cfg.sink_name)):
            self._prev_env.setdefault(key, os.environ.get(key))
            os.environ[key] = value
        routing.set_output_sink(self.cfg.sink_name)

    def stop(self) -> None:
        if self.active:
            routing.set_output_sink(None)
            for key, value in self._prev_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self._prev_env.clear()
        self.active = False
        if self.module_id is not None and not self.cfg.keep_loaded:
            try:
                _pactl("unload-module", str(self.module_id))
                logger.info("AEC: unloaded module #%s", self.module_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("AEC: unload failed: %s", exc)
        self.module_id = None


# --------------------------------------------------------------- card profiles


def duplex_profiles() -> list[tuple[str, str, str]]:
    """``(card, profile, description)`` for available output+input profiles."""
    out = _pactl("list", "cards")
    found: list[tuple[str, str, str]] = []
    card = ""
    in_profiles = False
    for raw in out.splitlines():
        line = raw.strip()
        if line.startswith("Name:"):
            card = line.split(":", 1)[1].strip()
            in_profiles = False
        elif line.startswith("Profiles:"):
            in_profiles = True
        elif in_profiles and not raw.startswith("\t\t"):
            in_profiles = False
        elif in_profiles:
            name, _, rest = line.partition(": ")
            if "input:" in name and "available: no" not in rest:
                found.append((card, name, rest.split(" (", 1)[0]))
    # Prefer analog duplex (speaker + mic on the same codec: best for AEC).
    found.sort(key=lambda t: (not t[1].startswith("output:analog"), t[1]))
    return found


def active_profile(card: str) -> str:
    out = _pactl("list", "cards")
    current = ""
    for raw in out.splitlines():
        line = raw.strip()
        if line.startswith("Name:"):
            current = line.split(":", 1)[1].strip()
        elif line.startswith("Active Profile:") and current == card:
            return line.split(":", 1)[1].strip()
    return ""


def set_profile(card: str, profile: str) -> None:
    _pactl("set-card-profile", card, profile)
