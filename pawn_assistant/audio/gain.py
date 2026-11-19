"""Input gain control for the HUD knob.

Two backends, picked from the capture device:

- ``alsa``: the device is a raw ``hw:N,M`` ALSA node (PortAudio bypasses
  PulseAudio/PipeWire there), so the analog mixer control (``Capture``) is the
  only thing that can stop the mic clipping. Driven through ``amixer``.
- ``software``: anything else; a digital multiplier applied in the capture
  callback. It cannot undo clipping that already happened upstream, but it can
  boost a quiet source or trim overshoot.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_HW_RE = re.compile(r"\(hw:(\d+)(?:,\d+)?\)|^hw:(\d+)")
_PCT_RE = re.compile(r"\[(\d+)%\]")
_DB_RE = re.compile(r"\[(-?\d+(?:\.\d+)?)dB\]")

SOFT_MIN_DB = -24.0
SOFT_MAX_DB = 12.0


def alsa_card_from_name(name: str) -> Optional[int]:
    """``'HDA Intel PCH: ALC3204 Analog (hw:0,0)'`` → ``0``."""
    m = _HW_RE.search(name or "")
    if not m:
        return None
    return int(m.group(1) if m.group(1) is not None else m.group(2))


def parse_sget(output: str) -> Optional[tuple[float, Optional[float]]]:
    """Parse ``amixer sget`` → ``(percent, dB or None)`` from the first channel."""
    for line in output.splitlines():
        pct = _PCT_RE.search(line)
        if pct:
            db = _DB_RE.search(line)
            return float(pct.group(1)), (float(db.group(1)) if db else None)
    return None


class InputGain:
    """Reads/sets the input gain; ``state()`` is the JSON the HUD renders."""

    def __init__(self, capture: Any, control: str = "Capture") -> None:
        self.capture = capture
        self.control = control
        self._lock = threading.Lock()
        self._card = alsa_card_from_name(capture.device_info.name)
        self.backend = "software"
        if self._card is not None and shutil.which("amixer"):
            if self._amixer_get() is not None:
                self.backend = "alsa"

    # -- ALSA -------------------------------------------------------------

    def _amixer(self, *args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["amixer", "-c", str(self._card), *args],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("amixer failed: %s", exc)
            return None
        if proc.returncode != 0:
            logger.warning("amixer %s: %s", " ".join(args), proc.stderr.strip())
            return None
        return proc.stdout

    def _amixer_get(self) -> Optional[tuple[float, Optional[float]]]:
        out = self._amixer("sget", self.control)
        return parse_sget(out) if out else None

    # -- public -----------------------------------------------------------

    def set(self, value: float) -> dict[str, Any]:
        """Apply *value* (percent for ``alsa``, dB for ``software``)."""
        with self._lock:
            if self.backend == "alsa":
                pct = max(0.0, min(100.0, float(value)))
                self._amixer("-q", "sset", self.control, f"{pct:.0f}%")
            else:
                db = max(SOFT_MIN_DB, min(SOFT_MAX_DB, float(value)))
                self.capture.set_gain_db(db)
        return self.state()

    def state(self) -> dict[str, Any]:
        if self.backend == "alsa":
            got = self._amixer_get()
            pct, db = got if got else (0.0, None)
            return {
                "backend": "alsa",
                "label": f"mixer · {self.control}",
                "value": pct,
                "min": 0,
                "max": 100,
                "step": 1,
                "unit": "%",
                "db": db,
            }
        db = float(self.capture.gain_db)
        return {
            "backend": "software",
            "label": "software gain",
            "value": db,
            "min": SOFT_MIN_DB,
            "max": SOFT_MAX_DB,
            "step": 0.5,
            "unit": "dB",
            "db": db,
        }
