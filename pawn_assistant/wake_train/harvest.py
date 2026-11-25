"""Collect wake-word training clips from live use (``serve``).

Every wake keeps the ~2 s of audio that triggered it; what happens next labels
it, and the clip lands in ``<project>/inbox/<label>/``:

``confirmed``   wake → speech → a recognised command        (likely positive)
``barge``       wake word heard over PAWN's own reply (AEC)  (likely positive)
``false_wake``  wake → no speech / empty transcript          (likely negative)
``unsure``      wake → speech, but no command understood
``near_miss``   score in the stir band, no wake              (could be either)

Nothing in the inbox is used for training until it is accepted
(``wake-train review`` / ``wake-train accept``), which moves clips into
``positives/`` or ``negatives/live_<label>/``.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pawn_assistant.wake_train.io_wav import write_wav

logger = logging.getLogger(__name__)

LABELS = ("confirmed", "barge", "false_wake", "unsure", "near_miss")
# Where ``accept`` files each label (None = needs a human: review only).
ACCEPT_TO: dict[str, Optional[str]] = {
    "confirmed": "pos",
    "barge": "pos",
    "false_wake": "neg",
    "unsure": None,
    "near_miss": None,
}


def find_project_root(wake_model: str) -> Optional[Path]:
    """The wake-train project a model file belongs to (has ``manifest.json``)."""
    path = Path(wake_model).expanduser()
    if path.suffix.lower() != ".onnx" or not path.exists():
        return None
    for parent in (path.parent, path.parent.parent):
        if (parent / "manifest.json").is_file():
            return parent.resolve()
    return None


def resolve_harvest_root(setting: Any, wake_model: str) -> Optional[Path]:
    """``wake.harvest``: auto (model's project) | true | false | a project path."""
    if setting is None or setting is False:
        return None
    text = str(setting).strip().lower()
    if text in ("", "false", "off", "no", "none", "null"):
        return None
    if text in ("auto", "true", "on", "yes"):
        root = find_project_root(wake_model)
        if root is None and text != "auto":
            logger.warning("wake.harvest: %s is not inside a wake-train project", wake_model)
        return root
    root = Path(str(setting)).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        logger.warning("wake.harvest: %s has no manifest.json — harvesting off", root)
        return None
    return root


@dataclass
class _Pending:
    audio: np.ndarray
    score: float
    at: float


class WakeHarvester:
    """Ring buffer of recent 16 kHz frames + labelled clip writer."""

    def __init__(
        self,
        root: Path,
        *,
        sample_rate: int = 16_000,
        clip_s: float = 2.0,
        max_per_label: int = 500,
        near_miss_gap_s: float = 3.0,
    ) -> None:
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.sample_rate = sample_rate
        self.clip_samples = int(clip_s * sample_rate)
        self.max_per_label = max_per_label
        self.near_miss_gap_s = near_miss_gap_s
        self._frames: deque[np.ndarray] = deque()
        self._n = 0
        self._pending: Optional[_Pending] = None
        self._near: Optional[_Pending] = None
        self._last_near_saved = 0.0
        self._lock = threading.Lock()
        logger.info("wake harvest → %s", self.inbox)

    # ------------------------------------------------------------ audio thread

    def push(self, frame: np.ndarray) -> None:
        self._frames.append(np.asarray(frame, dtype=np.float32).reshape(-1))
        self._n += self._frames[-1].size
        while self._frames and self._n - self._frames[0].size >= self.clip_samples:
            self._n -= self._frames.popleft().size
        self._maybe_flush_near_miss()

    def _clip(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(list(self._frames))
        return audio[-self.clip_samples :]

    def on_wake(self, score: float) -> None:
        """The wake fired on this frame: hold its clip until the turn's outcome."""
        self._near = None
        with self._lock:
            self._pending = _Pending(self._clip(), score, time.time())

    def on_barge(self, score: float) -> None:
        self._save("barge", self._clip(), score, {})

    def on_score(self, score: float, stir: float, threshold: float) -> None:
        """Track the peak of a near-miss episode (stir ≤ score < threshold)."""
        if stir <= score < threshold:
            if self._near is None or score > self._near.score:
                self._near = _Pending(self._clip(), score, time.monotonic())

    def _maybe_flush_near_miss(self) -> None:
        near = self._near
        if near is None:
            return
        # Wait ~1 s past the peak so a wake that follows cancels it.
        if time.monotonic() - near.at < 1.0:
            return
        self._near = None
        if time.monotonic() - self._last_near_saved < self.near_miss_gap_s:
            return
        self._last_near_saved = time.monotonic()
        self._save("near_miss", near.audio, near.score, {})

    # ------------------------------------------------------------ any thread

    def resolve(self, label: str, **meta: Any) -> None:
        """Label the clip held since the last wake (no-op when none is pending)."""
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is None:
            return
        self._save(label, pending.audio, pending.score, meta)

    def _save(self, label: str, audio: np.ndarray, score: float, meta: dict[str, Any]) -> None:
        if audio.size < self.sample_rate // 2:
            return
        folder = self.inbox / label
        try:
            folder.mkdir(parents=True, exist_ok=True)
            existing = sorted(folder.glob("*.wav"))
            # Keep the newest clips: the inbox must not grow without bound.
            for old in existing[: max(0, len(existing) - self.max_per_label + 1)]:
                old.unlink(missing_ok=True)
                old.with_suffix(".json").unlink(missing_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
            path = folder / f"{stamp}_{score:.2f}.wav"
            write_wav(path, audio, self.sample_rate)
            info = {"label": label, "score": round(float(score), 4), "time": stamp, **meta}
            path.with_suffix(".json").write_text(
                json.dumps(info, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            logger.info("harvested %s clip %s", label, path.name)
        except OSError as exc:
            logger.warning("wake harvest write failed: %s", exc)


# ------------------------------------------------------------------ inbox ops


@dataclass
class InboxClip:
    path: Path
    label: str
    meta: dict[str, Any]

    @property
    def score(self) -> float:
        return float(self.meta.get("score", 0.0))


def inbox_clips(root: Path, labels: Optional[list[str]] = None) -> list[InboxClip]:
    inbox = Path(root) / "inbox"
    out: list[InboxClip] = []
    if not inbox.is_dir():
        return out
    for folder in sorted(p for p in inbox.iterdir() if p.is_dir()):
        if labels and folder.name not in labels:
            continue
        for wav in sorted(folder.glob("*.wav")):
            meta_path = wav.with_suffix(".json")
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            except (OSError, ValueError):
                meta = {}
            out.append(InboxClip(wav, folder.name, meta))
    return out


def inbox_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for clip in inbox_clips(root):
        counts[clip.label] = counts.get(clip.label, 0) + 1
    return counts


def file_clip(root: Path, clip: InboxClip, kind: str) -> Optional[Path]:
    """Move an inbox clip into training data: kind = pos | neg | drop."""
    root = Path(root)
    meta = clip.path.with_suffix(".json")
    if kind == "drop":
        clip.path.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        return None
    if kind == "pos":
        dest_dir = root / "positives"
        name = f"live_{clip.label}_{clip.path.name}"
    elif kind == "neg":
        dest_dir = root / "negatives" / f"live_{clip.label}"
        name = clip.path.name
    else:
        raise ValueError(f"unknown kind {kind!r}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    shutil.move(str(clip.path), dest)
    meta.unlink(missing_ok=True)
    return dest
