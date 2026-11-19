"""openWakeWord wrapper (stock name or path to a custom .onnx)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class WakeDetector:
    """Score mono float32 frames against an openWakeWord model."""

    def __init__(self, model_name: str = "hey_jarvis") -> None:
        self.model_name = model_name
        self._model = None
        self._key: Optional[str] = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from openwakeword import utils as oww_utils

            oww_utils.download_models()
        except Exception as exc:  # noqa: BLE001
            logger.warning("openWakeWord download_models: %s", exc)

        from openwakeword.model import Model

        spec = self.model_name
        path = Path(spec).expanduser()
        if path.suffix.lower() == ".onnx" and path.exists():
            wakeword_models = [str(path.resolve())]
            logger.info("loading custom wake ONNX %s", path)
        else:
            wakeword_models = [spec]

        self._model = Model(
            wakeword_models=wakeword_models,
            inference_framework="onnx",
        )
        keys = list(self._model.models.keys())
        self._key = keys[0] if keys else path.stem if path.suffix else spec
        logger.info("wake model ready: %s (key=%s)", spec, self._key)

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def score(self, frame: np.ndarray) -> float:
        """Return wake score for one 1280-sample float32 frame."""
        if self._model is None:
            self.load()
        assert self._model is not None
        pcm = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
        preds = self._model.predict(pcm)
        if not preds:
            return 0.0
        if self._key and self._key in preds:
            return float(preds[self._key])
        return float(max(preds.values()))
