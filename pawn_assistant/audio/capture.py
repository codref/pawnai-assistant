"""16 kHz mono capture via sounddevice with a frame queue."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# ALSA "default" / bare pipewire often fail under PortAudio.
# Pulse can open but be silent when PipeWire has no mic Source (output-only profile).
_PREFERRED_NAMES = ("pulse", "sysdefault", "pipewire", "default")
_SKIP_NAME_PARTS = (
    "lavrate",
    "samplerate",
    "speexrate",
    "speex",
    "upmix",
    "vdownmix",
    "null",
    "hdmi",
    ".monitor",
)
# Prefer 16 kHz when the device accepts it; otherwise common natives.
# Device default is prepended at open time (see AudioCapture.start).
_OPEN_RATES = (16_000, 44_100, 48_000)


@dataclass
class DeviceInfo:
    index: Optional[int]
    name: str
    sample_rate: int
    channels: int
    stream_rate: int = 16_000


def list_input_devices() -> list[dict]:
    """Return sounddevice input devices as plain dicts."""
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    out: list[dict] = []
    for i, dev in enumerate(devices):
        if int(dev["max_input_channels"]) < 1:
            continue
        api = hostapis[dev["hostapi"]]["name"] if hostapis else ""
        out.append(
            {
                "index": i,
                "name": str(dev["name"]),
                "hostapi": api,
                "channels": int(dev["max_input_channels"]),
                "default_samplerate": float(dev["default_samplerate"]),
            }
        )
    return out


def _default_input_index() -> Optional[int]:
    default = sd.default.device
    try:
        raw = default[0]
    except (TypeError, IndexError):
        raw = default
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        return None
    return None if idx < 0 else idx


def _index_for_name(name: str) -> Optional[int]:
    try:
        info = sd.query_devices(name, "input")
    except Exception:
        return None
    idx = info.get("index")
    if idx is None:
        return None
    idx_i = int(idx)
    return idx_i if idx_i >= 0 else None


def _is_skippable(name: str) -> bool:
    lower = name.lower()
    return any(part in lower for part in _SKIP_NAME_PARTS)


def _candidate_indices(preferred: Optional[int | str] = None) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()

    def add(idx: Optional[int]) -> None:
        if idx is None or idx < 0 or idx in seen:
            return
        try:
            info = sd.query_devices(idx, "input")
        except Exception:
            return
        if int(info["max_input_channels"]) < 1:
            return
        if _is_skippable(str(info["name"])):
            return
        seen.add(idx)
        ordered.append(idx)

    if preferred is not None:
        if isinstance(preferred, int):
            add(preferred)
        else:
            add(_index_for_name(str(preferred)))

    for name in _PREFERRED_NAMES:
        add(_index_for_name(name))

    add(_default_input_index())
    for hostapi in sd.query_hostapis():
        add(int(hostapi.get("default_input_device", -1)))
    for i, dev in enumerate(sd.query_devices()):
        if int(dev["max_input_channels"]) > 0:
            add(i)
    return ordered


def resolve_device(device: Optional[int | str]) -> DeviceInfo:
    if device is not None:
        info = sd.query_devices(device, "input")
        if isinstance(device, int):
            idx: Optional[int] = device
        else:
            idx = _index_for_name(str(device))
            if idx is None:
                for i, dev in enumerate(sd.query_devices()):
                    if str(dev["name"]) == str(device) and int(dev["max_input_channels"]) > 0:
                        idx = i
                        break
        return DeviceInfo(
            index=idx,
            name=str(info["name"]),
            sample_rate=16_000,
            channels=1,
            stream_rate=int(float(info.get("default_samplerate", 16_000))),
        )

    candidates = _candidate_indices()
    if not candidates:
        raise RuntimeError("no input audio device found")
    idx = candidates[0]
    info = sd.query_devices(idx, "input")
    return DeviceInfo(
        index=idx,
        name=str(info["name"]),
        sample_rate=16_000,
        channels=1,
        stream_rate=int(float(info.get("default_samplerate", 16_000))),
    )


def resample_mono(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample whole buffer (prefer polyphase; fall back to linear)."""
    if src_rate == dst_rate or samples.size == 0:
        return np.asarray(samples, dtype=np.float32)
    try:
        from scipy.signal import resample_poly

        g = math.gcd(int(src_rate), int(dst_rate))
        up, down = int(dst_rate) // g, int(src_rate) // g
        n = (samples.size // down) * down
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        out = resample_poly(samples[:n].astype(np.float64), up, down)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        duration = samples.size / float(src_rate)
        n_out = max(1, int(round(duration * dst_rate)))
        x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.float32)


# Back-compat alias used by older call sites / tests.
_resample_mono = resample_mono


def _resample_with_remainder(
    samples: np.ndarray, src_rate: int, dst_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample keeping an unconsumed native-rate remainder for the next block."""
    if src_rate == dst_rate or samples.size == 0:
        return np.asarray(samples, dtype=np.float32), np.zeros(0, dtype=np.float32)
    g = math.gcd(int(src_rate), int(dst_rate))
    down = int(src_rate) // g
    n = (samples.size // down) * down
    if n <= 0:
        return np.zeros(0, dtype=np.float32), np.asarray(samples, dtype=np.float32)
    converted = _resample_mono(samples[:n], src_rate, dst_rate)
    remainder = np.asarray(samples[n:], dtype=np.float32)
    return converted, remainder


@contextlib.contextmanager
def quiet_portaudio():
    """Mute ALSA/JACK chatter PortAudio dumps to stderr while probing."""
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(stderr_fd)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)
        os.close(devnull)


def measure_device(
    device: int | str,
    seconds: float = 0.5,
    target_rate: int = 16_000,
) -> dict:
    """Open a device briefly and return level stats (for CLI probing)."""
    last_err: Optional[Exception] = None
    with quiet_portaudio():
        for rate in _OPEN_RATES:
            try:
                frames = max(1, int(seconds * rate))
                rec = sd.rec(
                    frames,
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    device=device,
                )
                sd.wait()
                mono = np.asarray(rec[:, 0], dtype=np.float32)
                if rate != target_rate:
                    mono = _resample_mono(mono, rate, target_rate)
                peak = float(np.max(np.abs(mono))) if mono.size else 0.0
                rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                peak_db = 20.0 * np.log10(peak) if peak > 1e-9 else -120.0
                rms_db = 20.0 * np.log10(rms) if rms > 1e-9 else -120.0
                info = sd.query_devices(device, "input")
                return {
                    "ok": True,
                    "index": device
                    if isinstance(device, int)
                    else _index_for_name(str(device)),
                    "name": str(info["name"]),
                    "stream_rate": rate,
                    "rms_db": rms_db,
                    "peak_db": peak_db,
                    "silent": peak < 1e-4,
                }
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
    name = str(device)
    try:
        name = str(sd.query_devices(device, "input")["name"])
    except Exception:
        pass
    err = str(last_err) if last_err else "open failed"
    # Keep the table readable — drop multi-line PortAudio host dumps.
    err = err.split("\n", 1)[0]
    return {
        "ok": False,
        "index": device if isinstance(device, int) else _index_for_name(str(device)),
        "name": name,
        "error": err,
    }


class AudioCapture:
    """Realtime capture with a *copy-only* PortAudio callback.

    Resampling used to run inside the callback (scipy polyphase). That made the
    callback miss deadlines → PortAudio input overflows → time-compressed,
    untranscribable utterances. Now:

    - callback: append native-rate PCM only
    - consumer ``read()``: resample to 16 kHz for wake/VAD
    - ``start_utt_record`` / ``stop_utt_record``: keep a native-rate copy for ASR
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_samples: int = 1280,
        device: Optional[int | str] = None,
        max_queue: int = 0,  # API compat; ignored
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._requested_device = device
        self._native: deque[np.ndarray] = deque()
        self._frames: deque[np.ndarray] = deque()
        self._cond = threading.Condition()
        self.overruns = 0
        self.dropped_frames = 0
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self.device_info = resolve_device(device)
        self.device: Optional[int | str] = self.device_info.index
        self._stream_rate = self.sample_rate
        self._native_carry = np.zeros(0, dtype=np.float32)
        self._pcm_carry = np.zeros(0, dtype=np.float32)
        self._utt_recording = False
        self._utt_native: list[np.ndarray] = []
        self.gain_db = 0.0
        self._gain_lin = 1.0

    def set_gain_db(self, db: float) -> None:
        """Digital gain applied to every callback block (cheap multiply)."""
        self.gain_db = float(db)
        self._gain_lin = 10.0 ** (self.gain_db / 20.0)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        # Keep this path tiny — any heavy work here causes PaInputOverflow.
        if status:
            self.overruns += 1
        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        if self._gain_lin != 1.0:
            np.clip(mono * self._gain_lin, -1.0, 1.0, out=mono)
        with self._cond:
            self._native.append(mono)
            if self._utt_recording:
                self._utt_native.append(mono.copy())
            self._cond.notify_all()

    def _open_stream(self, device: int, rate: int) -> sd.InputStream:
        # On this ALSA HDA path, ~100 ms blocks *cause* PaInputOverflow.
        # 256–1024 + high latency keeps realtime; blocksize=0 lets PortAudio pick.
        stream = sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocksize=512,
            device=device,
            callback=self._callback,
            latency="high",
        )
        stream.start()
        return stream

    def _take_native_batch(self) -> np.ndarray:
        """Pop pending native PCM + carry (caller holds ``_cond``)."""
        parts: list[np.ndarray] = []
        if self._native_carry.size:
            parts.append(self._native_carry)
            self._native_carry = np.zeros(0, dtype=np.float32)
        while self._native:
            parts.append(self._native.popleft())
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def _commit_converted(self, converted: np.ndarray, rem: np.ndarray) -> None:
        """Push 16 kHz frames; keep remainders (caller holds ``_cond``)."""
        self._native_carry = rem
        if self._pcm_carry.size:
            converted = np.concatenate([self._pcm_carry, converted])
        while converted.size >= self.frame_samples:
            self._frames.append(converted[: self.frame_samples].copy())
            converted = converted[self.frame_samples :]
        self._pcm_carry = converted

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            candidates = _candidate_indices(self._requested_device)
            if not candidates:
                raise RuntimeError("no input audio device found")

            errors: list[str] = []
            require_signal = self._requested_device is None
            silent_ok: list[tuple[int, str, int]] = []

            for idx in candidates:
                name = "?"
                try:
                    name = str(sd.query_devices(idx, "input")["name"])
                except Exception:
                    pass
                rates: list[int] = []
                try:
                    default_sr = int(
                        float(sd.query_devices(idx, "input")["default_samplerate"])
                    )
                    if default_sr > 0:
                        rates.append(default_sr)
                except Exception:  # noqa: BLE001
                    pass
                for r in _OPEN_RATES:
                    if r not in rates:
                        rates.append(r)
                for rate in rates:
                    try:
                        self._native_carry = np.zeros(0, dtype=np.float32)
                        self._pcm_carry = np.zeros(0, dtype=np.float32)
                        self.drain()
                        self._stream = self._open_stream(idx, rate)
                        self._stream_rate = rate
                        self.device = idx
                        self.device_info = DeviceInfo(
                            index=idx,
                            name=name,
                            sample_rate=self.sample_rate,
                            channels=1,
                            stream_rate=rate,
                        )
                        if require_signal and not self._hears_signal(timeout=0.45):
                            logger.warning(
                                "device %s (%s) @ %s Hz opens but is silent — trying next",
                                idx,
                                name,
                                rate,
                            )
                            silent_ok.append((idx, name, rate))
                            self._stream.stop()
                            self._stream.close()
                            self._stream = None
                            continue
                        logger.info(
                            "opened input device %s (%s) @ %s Hz → %s Hz "
                            "(callback=copy-only, resample on read)",
                            idx,
                            name,
                            rate,
                            self.sample_rate,
                        )
                        return
                    except Exception as exc:  # noqa: BLE001
                        if self._stream is not None:
                            try:
                                self._stream.stop()
                                self._stream.close()
                            except Exception:  # noqa: BLE001
                                pass
                            self._stream = None
                        errors.append(f"{idx}:{name}@{rate}: {exc}")

            if silent_ok:
                idx, name, rate = silent_ok[0]
                self._native_carry = np.zeros(0, dtype=np.float32)
                self._pcm_carry = np.zeros(0, dtype=np.float32)
                self.drain()
                self._stream = self._open_stream(idx, rate)
                self._stream_rate = rate
                self.device = idx
                self.device_info = DeviceInfo(
                    index=idx,
                    name=name,
                    sample_rate=self.sample_rate,
                    channels=1,
                    stream_rate=rate,
                )
                logger.warning(
                    "falling back to silent device %s (%s) — "
                    "enable a PipeWire mic Source or pass --device after "
                    "`pawn-assistant probe`",
                    idx,
                    name,
                )
                return

            raise RuntimeError(
                "could not open any input device; tried: " + "; ".join(errors[:8])
            )

    def _hears_signal(self, timeout: float = 0.45, peak_min: float = 1e-4) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            frame = self.read(timeout=remaining)
            if frame is None:
                continue
            if float(np.max(np.abs(frame))) >= peak_min:
                return True
        return False

    def stop(self) -> None:
        with self._lock:
            if self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        # Resample *outside* the condition lock so the PortAudio callback never
        # blocks on scipy — that was the overrun / time-compression root cause.
        deadline = time.monotonic() + timeout
        while True:
            with self._cond:
                if self._frames:
                    return self._frames.popleft()
                native = self._take_native_batch()
                stream_rate = self._stream_rate
                sample_rate = self.sample_rate
            if native.size:
                if stream_rate == sample_rate:
                    converted, rem = native, np.zeros(0, dtype=np.float32)
                else:
                    converted, rem = _resample_with_remainder(
                        native, stream_rate, sample_rate
                    )
                with self._cond:
                    self._commit_converted(converted, rem)
                    if self._frames:
                        return self._frames.popleft()
                    # More native may have arrived while we resampled.
                    continue
            with self._cond:
                if self._frames:
                    return self._frames.popleft()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)

    def drain(self) -> None:
        with self._cond:
            self._native.clear()
            self._frames.clear()
            self._native_carry = np.zeros(0, dtype=np.float32)
            self._pcm_carry = np.zeros(0, dtype=np.float32)

    def start_utt_record(self) -> None:
        """Begin capturing a native-rate utterance (for high-quality ASR)."""
        with self._cond:
            self._utt_native = []
            self._utt_recording = True

    def stop_utt_record(self) -> tuple[np.ndarray, int]:
        """Stop utterance capture; return ``(audio, native_sample_rate)``."""
        with self._cond:
            self._utt_recording = False
            if not self._utt_native:
                return np.zeros(0, dtype=np.float32), self._stream_rate
            audio = np.concatenate(self._utt_native)
            self._utt_native = []
            return audio, self._stream_rate

    @property
    def queue_depth(self) -> int:
        with self._cond:
            return len(self._frames) + len(self._native)
