"""faster-whisper (int8, CPU) with lazy load / idle unload."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text: str
    duration_ms: float
    infer_ms: float

    @property
    def rtf(self) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return self.infer_ms / self.duration_ms


def _is_prompt_echo(text: str, prompt: str) -> bool:
    """True when *text* is (a long-enough slice of) the initial prompt."""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    if len(words) < 6:
        return False
    return " ".join(words) in " ".join(re.sub(r"[^\w\s]", " ", prompt.lower()).split())


class WhisperASR:
    """Lazy faster-whisper wrapper."""

    def __init__(
        self,
        model_size: str = "tiny",
        compute_type: str = "int8",
        idle_unload_s: float = 60.0,
        language: Optional[str] = None,
        initial_prompt: str = "",
        vad_filter: bool = True,
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.idle_unload_s = idle_unload_s
        # None → faster-whisper autodetection (Italian / English / …).
        self.language = language
        self.initial_prompt = initial_prompt.strip()
        self.vad_filter = vad_filter
        self._model = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._unload_timer: Optional[threading.Timer] = None

    @property
    def resident(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            logger.info(
                "loading faster-whisper %s (%s)",
                self.model_size,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
            )
            self._last_used = time.monotonic()

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            logger.info("unloading faster-whisper")
            self._model = None

    def _schedule_unload(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        if self.idle_unload_s <= 0:
            return

        def _fire() -> None:
            if time.monotonic() - self._last_used >= self.idle_unload_s:
                self.unload()

        self._unload_timer = threading.Timer(self.idle_unload_s, _fire)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> TranscriptResult:
        self.load()
        assert self._model is not None
        # Resampling can overshoot ±1.0; Whisper expects normalised samples.
        wav = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
        duration_ms = 1000.0 * len(wav) / float(sample_rate)
        t0 = time.perf_counter()
        with self._lock:
            segments, _info = self._model.transcribe(
                wav,
                language=self.language,
                beam_size=3,
                best_of=3,
                vad_filter=self.vad_filter,
                vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
                initial_prompt=self.initial_prompt or None,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            # On silence/noise Whisper may parrot the prompt back.
            if self.initial_prompt and _is_prompt_echo(text, self.initial_prompt):
                text = ""
            # Drop Whisper hallucinations that appear on short/noisy tails.
            if text.lower() in {
                "sottotitoli creati dalla comunità amara",
                "sottotitoli e review a cura di qtss",
                "thanks for watching",
                "thank you",
                "thank you for watching",
                "sottotitoli a cura di qtss",
            }:
                text = ""
            self._last_used = time.monotonic()
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self._schedule_unload()
        return TranscriptResult(text=text, duration_ms=duration_ms, infer_ms=infer_ms)
