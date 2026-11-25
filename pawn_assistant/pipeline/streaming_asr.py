"""Streaming command ASR with a sherpa-onnx Zipformer transducer.

Unlike Whisper (decode after the turn ends), the stream is fed every 80 ms
frame while you talk: the HUD shows partial text live and the final result is
ready a few ms after the end-of-speech, so the silence budget can be shorter.

Any sherpa-onnx *online transducer* works; point ``asr.streaming.model_dir`` at
its folder (``encoder*.onnx``, ``decoder*.onnx``, ``joiner*.onnx``,
``tokens.txt``; int8 files are preferred). There is no official Italian
streaming model; community Zipformer2 builds (e.g. Kroko) exist — check their
licence. Install the runtime with ``uv sync --extra streaming``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from pawn_assistant.pipeline.asr import TranscriptResult

logger = logging.getLogger(__name__)


@dataclass
class StreamingSettings:
    model_dir: Optional[str] = None
    encoder: Optional[str] = None
    decoder: Optional[str] = None
    joiner: Optional[str] = None
    tokens: Optional[str] = None
    threads: int = 2
    # greedy_search | modified_beam_search (needed for hotwords)
    decoding_method: str = "greedy_search"
    # Optional hotwords file (one phrase per line) — biases toward commands.
    hotwords_file: str = ""
    hotwords_score: float = 1.5
    # Final result is instant, so the turn can end on less trailing silence.
    silence_end_ms: Optional[int] = 600
    # Empty streaming result → decode the same audio with Whisper.
    fallback_whisper: bool = True
    idle_unload_s: float = 300.0
    # Lowercase text (most transducer vocabularies are upper-case BPE).
    lowercase: bool = True

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "StreamingSettings":
        if not isinstance(raw, Mapping):
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


def _pick(folder: Path, stem: str) -> Optional[Path]:
    """``encoder*.onnx`` in *folder*, int8 first, then the shortest name."""
    cands = sorted(folder.glob(f"{stem}*.onnx"))
    if not cands:
        return None
    int8 = [c for c in cands if "int8" in c.name]
    pool = int8 or cands
    return min(pool, key=lambda p: (len(p.name), p.name))


def resolve_model_files(cfg: StreamingSettings) -> dict[str, str]:
    folder = Path(cfg.model_dir).expanduser() if cfg.model_dir else None
    files: dict[str, Optional[Path]] = {}
    for part in ("encoder", "decoder", "joiner"):
        explicit = getattr(cfg, part)
        files[part] = Path(explicit).expanduser() if explicit else (_pick(folder, part) if folder else None)
    tokens = cfg.tokens or (str(folder / "tokens.txt") if folder else None)
    files["tokens"] = Path(tokens).expanduser() if tokens else None
    missing = [k for k, v in files.items() if v is None or not v.is_file()]
    if missing:
        where = folder or "asr.streaming"
        raise FileNotFoundError(f"streaming ASR: missing {', '.join(missing)} in {where}")
    return {k: str(v) for k, v in files.items()}


class ZipformerASR:
    """Lazy sherpa-onnx OnlineRecognizer; one stream per utterance."""

    def __init__(self, cfg: StreamingSettings, sample_rate: int = 16_000) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self._recognizer = None
        self._stream = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._fed_samples = 0
        self._infer_s = 0.0
        self._partial = ""
        self._unload_timer: Optional[threading.Timer] = None

    @property
    def resident(self) -> bool:
        return self._recognizer is not None

    @property
    def active(self) -> bool:
        return self._stream is not None

    def load(self) -> None:
        with self._lock:
            if self._recognizer is not None:
                return
            try:
                import sherpa_onnx
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "asr.engine: zipformer needs sherpa-onnx (uv sync --extra streaming)"
                ) from exc
            files = resolve_model_files(self.cfg)
            t0 = time.monotonic()
            kwargs: dict[str, Any] = dict(
                tokens=files["tokens"],
                encoder=files["encoder"],
                decoder=files["decoder"],
                joiner=files["joiner"],
                num_threads=int(self.cfg.threads),
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method=self.cfg.decoding_method,
                enable_endpoint_detection=False,  # Silero + our silence rules own endpointing
            )
            if self.cfg.hotwords_file:
                kwargs.update(
                    hotwords_file=str(Path(self.cfg.hotwords_file).expanduser()),
                    hotwords_score=float(self.cfg.hotwords_score),
                    decoding_method="modified_beam_search",
                )
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
            logger.info(
                "zipformer loaded in %.0f ms (%s)",
                1000 * (time.monotonic() - t0), Path(files["encoder"]).parent,
            )

    # ------------------------------------------------------------ per utterance

    def begin(self) -> None:
        """New utterance (call from the audio thread on wake)."""
        self.load()
        self._cancel_unload()
        assert self._recognizer is not None
        self._stream = self._recognizer.create_stream()
        self._fed_samples = 0
        self._infer_s = 0.0
        self._partial = ""

    def feed(self, frame: np.ndarray) -> Optional[str]:
        """Push one frame; returns the partial text when it changed."""
        stream, rec = self._stream, self._recognizer
        if stream is None or rec is None:
            return None
        t0 = time.perf_counter()
        stream.accept_waveform(self.sample_rate, np.asarray(frame, dtype=np.float32))
        self._fed_samples += int(frame.size)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        self._infer_s += time.perf_counter() - t0
        text = self._text(rec.get_result(stream))
        if text != self._partial:
            self._partial = text
            return text
        return None

    def finish(self) -> TranscriptResult:
        """Flush the stream and return the final transcript."""
        stream, rec = self._stream, self._recognizer
        self._stream = None
        if stream is None or rec is None:
            return TranscriptResult("", 0.0, 0.0)
        t0 = time.perf_counter()
        # Tail padding lets the encoder's right context see the last word.
        stream.accept_waveform(self.sample_rate, np.zeros(int(0.3 * self.sample_rate), np.float32))
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        self._infer_s += time.perf_counter() - t0
        text = self._text(rec.get_result(stream))
        self._last_used = time.monotonic()
        self._schedule_unload()
        return TranscriptResult(
            text=text,
            duration_ms=1000.0 * self._fed_samples / self.sample_rate,
            infer_ms=1000.0 * self._infer_s,
        )

    def cancel(self) -> None:
        self._stream = None

    def _text(self, raw: Any) -> str:
        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        text = " ".join(str(text).split())
        return text.lower() if self.cfg.lowercase else text

    # ------------------------------------------------------------ idle unload

    def _cancel_unload(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _schedule_unload(self) -> None:
        self._cancel_unload()
        if self.cfg.idle_unload_s and self.cfg.idle_unload_s > 0:
            self._unload_timer = threading.Timer(self.cfg.idle_unload_s, self._unload_if_idle)
            self._unload_timer.daemon = True
            self._unload_timer.start()

    def _unload_if_idle(self) -> None:
        with self._lock:
            if self._stream is None and time.monotonic() - self._last_used >= self.cfg.idle_unload_s:
                self._recognizer = None
                logger.info("zipformer unloaded after idle")
