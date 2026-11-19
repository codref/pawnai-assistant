"""Ring-buffer WAV dumps for wake / utterance debugging."""

from __future__ import annotations

import logging
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DebugAudioDump:
    """Keep a rolling mono float32 buffer and write WAVs on demand."""

    def __init__(
        self,
        directory: Path,
        sample_rate: int = 16_000,
        preroll_s: float = 8.0,
    ) -> None:
        self.directory = Path(directory)
        self.sample_rate = sample_rate
        self.preroll_s = preroll_s
        self._maxlen = max(1, int(preroll_s * sample_rate))
        self._buf: deque[np.ndarray] = deque()
        self._nsamples = 0
        self._lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        logger.info("debug audio dump → %s (preroll %.1fs)", self.directory, preroll_s)

    def push(self, frame: np.ndarray) -> None:
        chunk = np.asarray(frame, dtype=np.float32).reshape(-1).copy()
        if chunk.size == 0:
            return
        with self._lock:
            self._buf.append(chunk)
            self._nsamples += chunk.size
            while self._nsamples > self._maxlen and self._buf:
                dropped = self._buf.popleft()
                self._nsamples -= dropped.size

    def snapshot(self) -> np.ndarray:
        with self._lock:
            if not self._buf:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(list(self._buf))

    def save(self, label: str, audio: Optional[np.ndarray] = None, meta: str = "") -> Path:
        if audio is None:
            audio = self.snapshot()
        else:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:48]
        path = self.directory / f"{ts}_{safe}.wav"
        self._write_wav(path, audio)
        if meta:
            path.with_suffix(".txt").write_text(meta + "\n", encoding="utf-8")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        dur = audio.size / float(self.sample_rate)
        logger.info(
            "saved debug wav %s (%.2fs peak=%.3f)%s",
            path.name,
            dur,
            peak,
            f" — {meta}" if meta else "",
        )
        return path

    def _write_wav(self, path: Path, audio: np.ndarray) -> None:
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
