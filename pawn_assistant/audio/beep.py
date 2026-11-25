"""Short wake acknowledgement tone — never via sounddevice (avoids killing ALSA capture)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from pawn_assistant.audio import routing

logger = logging.getLogger(__name__)


def _wav_bytes(
    sample_rate: int,
    freq_hz: float,
    duration_ms: int,
    volume: float,
) -> bytes:
    n = max(1, int(sample_rate * duration_ms / 1000.0))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    wave_f = (volume * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    fade = min(64, n // 4)
    if fade > 0:
        wave_f[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wave_f[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    pcm = np.clip(wave_f * 32767.0, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


def _play_wav_file(
    path: Path, timeout: float, stop: Optional[threading.Event] = None
) -> bool:
    """Play through PipeWire/Pulse; returns early (killing the player) if *stop* is set."""
    for player in ("paplay", "pw-play", "aplay"):
        if not shutil.which(player):
            continue
        cmd = routing.player_command(str(path), player)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("playback via %s failed: %s", player, exc)
            continue
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    proc.wait(timeout=0.03)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if (stop is not None and stop.is_set()) or time.monotonic() > deadline:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
        finally:
            if proc.poll() is None:
                proc.kill()
        return True
    logger.warning("no paplay/pw-play/aplay — skipping audio playback")
    return False


def play_pcm(
    pcm: bytes,
    sample_rate: int,
    *,
    block: bool = True,
    timeout: float = 60.0,
    stop: Optional[threading.Event] = None,
) -> None:
    """Play mono int16 PCM through an external player (never sounddevice).

    Uses ``paplay`` / ``pw-play`` / ``aplay`` so exclusive ALSA ``hw:X,Y``
    input is not interrupted by a concurrent PortAudio output stream.
    Setting *stop* cuts playback short (barge-in).
    """

    def _run() -> None:
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = Path(tmp.name)
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm)
            if stop is not None and stop.is_set():
                return
            _play_wav_file(path, timeout, stop)
        except Exception:  # noqa: BLE001
            logger.exception("playback failed")
        finally:
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    if block:
        _run()
    else:
        threading.Thread(target=_run, name="pawn-play", daemon=True).start()


def tone_pcm(
    sample_rate: int = 16_000,
    freq_hz: float = 880.0,
    duration_ms: int = 120,
    volume: float = 0.25,
) -> bytes:
    return _wav_bytes(sample_rate, freq_hz, duration_ms, volume)


def play_beep(
    sample_rate: int = 16_000,
    freq_hz: float = 880.0,
    duration_ms: int = 120,
    volume: float = 0.25,
    *,
    block: bool = False,
) -> None:
    """Play a sine beep without touching the sounddevice capture stream."""
    play_pcm(
        _wav_bytes(sample_rate, freq_hz, duration_ms, volume),
        sample_rate,
        block=block,
        timeout=5.0,
    )
