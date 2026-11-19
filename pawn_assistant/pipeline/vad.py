"""Silero VAD via openWakeWord's bundled ONNX model (h/c state API)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SileroVAD:
    """Streaming speech probability.

    Uses the openWakeWord-packaged Silero checkpoint (``h``/``c`` inputs).
    The newer upstream ``state``-tensor ONNX build needs a different calling
    convention and was returning ~0 for all audio in our earlier wrapper.
    """

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold
        self._vad = None

    def load(self) -> None:
        if self._vad is not None:
            return
        from openwakeword.vad import VAD

        self._vad = VAD()
        logger.info("Silero VAD ready (openWakeWord bundle)")

    def reset(self) -> None:
        if self._vad is not None:
            self._vad.reset_states()

    def prob(self, frame: np.ndarray) -> float:
        if self._vad is None:
            self.load()
        assert self._vad is not None
        audio = np.asarray(frame, dtype=np.float32).reshape(-1)
        # openWakeWord VAD expects int16 PCM samples.
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        if pcm.size < 480:
            return 0.0
        # Frame size must divide the buffer; 640 = 40 ms @ 16 kHz.
        usable = (pcm.size // 640) * 640
        if usable <= 0:
            return 0.0
        return float(self._vad.predict(pcm[:usable], frame_size=640))

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.prob(frame) >= self.threshold
