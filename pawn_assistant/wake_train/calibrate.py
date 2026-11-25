"""Pick a wake threshold from clips, scored the way ``serve`` scores them.

Each clip is streamed through openWakeWord in 80 ms frames between a second of
faint noise and half a second of tail (the live loop never sees a clip in
isolation), and its peak score is kept. Positives should peak high, negatives
low; the threshold is placed above the negatives' tail with a margin.

Training clips flatter the model (it has seen them). Inbox clips harvested from
live use but not yet accepted are the honest test set and are reported apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from pawn_assistant.wake_train.harvest import ACCEPT_TO, inbox_clips
from pawn_assistant.wake_train.io_wav import load_clip_16k
from pawn_assistant.wake_train.project import WakeProject

logger = logging.getLogger(__name__)

FRAME = 1280


@dataclass
class ScoreSet:
    pos: list[float] = field(default_factory=list)
    neg: list[float] = field(default_factory=list)
    worst_neg: list[tuple[float, str]] = field(default_factory=list)
    worst_pos: list[tuple[float, str]] = field(default_factory=list)


@dataclass
class Calibration:
    train: ScoreSet
    inbox: ScoreSet
    threshold: float
    recall: float
    false_positives: int
    negatives: int

    @property
    def barge_threshold(self) -> float:
        return round(min(0.95, self.threshold + 0.1), 2)


class ClipScorer:
    def __init__(self, model_path: Path, sample_rate: int = 16_000, seed: int = 0) -> None:
        from openwakeword.model import Model

        self.model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        self.key = next(iter(self.model.models))
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def peak(self, clip: np.ndarray) -> float:
        noise = lambda s: self.rng.normal(0.0, 3e-4, int(s * self.sample_rate)).astype(np.float32)  # noqa: E731
        audio = np.concatenate([noise(1.0), clip.astype(np.float32), noise(0.5)])
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        self.model.reset()
        best = 0.0
        for i in range(0, pcm.size - FRAME + 1, FRAME):
            preds = self.model.predict(pcm[i : i + FRAME])
            best = max(best, float(preds.get(self.key, 0.0)))
        return best


def _score_paths(scorer: ClipScorer, paths: list[Path], sr: int) -> list[tuple[float, str]]:
    out = []
    for i, path in enumerate(paths, 1):
        out.append((scorer.peak(load_clip_16k(path, sr)), str(path)))
        if i % 200 == 0:
            logger.info("scored %d/%d", i, len(paths))
    return out


def _fill(dest: ScoreSet, pos: list[tuple[float, str]], neg: list[tuple[float, str]]) -> None:
    dest.pos = [s for s, _ in pos]
    dest.neg = [s for s, _ in neg]
    dest.worst_neg = sorted(neg, reverse=True)[:5]
    dest.worst_pos = sorted(pos)[:5]


def choose_threshold(
    pos: list[float], neg: list[float], *, max_fpr: float = 0.0, margin: float = 0.05,
    floor: float = 0.2, ceiling: float = 0.9,
) -> float:
    """Just above the negatives' (1 - max_fpr) quantile, plus a margin."""
    if not neg:
        return 0.5
    neg_arr = np.sort(np.asarray(neg))
    allowed = int(np.floor(max_fpr * neg_arr.size))
    tail = float(neg_arr[-1 - allowed]) if allowed < neg_arr.size else 0.0
    thr = min(ceiling, max(floor, tail + margin))
    if pos:
        # Never above the median positive: a threshold nobody reaches is useless.
        thr = min(thr, max(floor, float(np.median(pos)) - 0.01))
    return round(thr, 2)


def rates(pos: list[float], neg: list[float], thr: float) -> tuple[float, int]:
    recall = float(np.mean(np.asarray(pos) >= thr)) if pos else 0.0
    fp = int(np.sum(np.asarray(neg) >= thr)) if neg else 0
    return recall, fp


def calibrate(
    project: WakeProject,
    *,
    model_path: Optional[Path] = None,
    max_fpr: float = 0.0,
    margin: float = 0.05,
    use_inbox: bool = True,
) -> Calibration:
    sr = project.sample_rate
    scorer = ClipScorer(model_path or project.model_path, sr)

    pos_paths = sorted(project.positives_dir.glob("*.wav"))
    neg_paths = sorted(project.negatives_dir.rglob("*.wav"))
    logger.info("scoring %d positives, %d negatives…", len(pos_paths), len(neg_paths))
    train = ScoreSet()
    _fill(train, _score_paths(scorer, pos_paths, sr), _score_paths(scorer, neg_paths, sr))

    inbox = ScoreSet()
    if use_inbox:
        clips = inbox_clips(project.root)
        ipos = [c.path for c in clips if ACCEPT_TO.get(c.label) == "pos"]
        ineg = [c.path for c in clips if ACCEPT_TO.get(c.label) == "neg"]
        if ipos or ineg:
            logger.info("scoring inbox: %d likely positives, %d likely negatives", len(ipos), len(ineg))
            _fill(inbox, _score_paths(scorer, ipos, sr), _score_paths(scorer, ineg, sr))

    all_pos = train.pos + inbox.pos
    all_neg = train.neg + inbox.neg
    thr = choose_threshold(all_pos, all_neg, max_fpr=max_fpr, margin=margin)
    recall, fp = rates(all_pos, all_neg, thr)
    return Calibration(train, inbox, thr, recall, fp, len(all_neg))
