"""Build openWakeWord embedding windows from recorded clips."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pawn_assistant.wake_train.io_wav import load_clip_16k
from pawn_assistant.wake_train.project import WakeProject

logger = logging.getLogger(__name__)

N_FRAMES = 16  # matches stock openWakeWord classifiers
N_FEATURES = 96


def _noise_floor(n: int, rng: np.random.Generator, level: float = 3e-4) -> np.ndarray:
    return (rng.normal(0.0, level, size=n)).astype(np.float32)


def place_in_window(
    clip: np.ndarray,
    window_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Put *clip* at a random offset; fill the rest with faint noise (never zeros)."""
    out = _noise_floor(window_samples, rng)
    if clip.size >= window_samples:
        start = int(rng.integers(0, max(1, clip.size - window_samples + 1)))
        return clip[start : start + window_samples].astype(np.float32)
    max_off = window_samples - clip.size
    offset = int(rng.integers(0, max_off + 1))
    out[offset : offset + clip.size] = clip.astype(np.float32)
    # Light gain jitter
    gain = float(rng.uniform(0.7, 1.15))
    out *= gain
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def negative_windows(
    clip: np.ndarray,
    window_samples: int,
    rng: np.random.Generator,
    max_crops: int = 4,
) -> list[np.ndarray]:
    """Short clips → one padded window; long clips → several overlapping crops.

    Long speech negatives carry many distinct "not the wake word" contexts; a
    single random crop wastes most of them.
    """
    if clip.size <= window_samples:
        return [place_in_window(clip, window_samples, rng)]
    hop = window_samples // 2
    starts = list(range(0, clip.size - window_samples + 1, hop))
    if starts[-1] != clip.size - window_samples:
        starts.append(clip.size - window_samples)  # speech ending at the window edge
    if len(starts) > max_crops:
        starts = sorted(rng.choice(starts, size=max_crops, replace=False).tolist())
    return [clip[s : s + window_samples].astype(np.float32) for s in starts]


def embed_window(audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Return shape (N_FRAMES, 96) openWakeWord embedding features."""
    from openwakeword.utils import AudioFeatures

    feats_obj = AudioFeatures(inference_framework="onnx")
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    # Feed the whole window; take the trailing feature frames.
    step = 1280
    for i in range(0, max(step, pcm.size), step):
        chunk = pcm[i : i + step]
        if chunk.size < step:
            pad = np.zeros(step - chunk.size, dtype=np.int16)
            chunk = np.concatenate([chunk, pad])
        feats_obj(chunk)
    feat = feats_obj.get_features(N_FRAMES)[0]
    if feat.shape != (N_FRAMES, N_FEATURES):
        # Pad/trim defensively.
        out = np.zeros((N_FRAMES, N_FEATURES), dtype=np.float32)
        n = min(N_FRAMES, feat.shape[0])
        m = min(N_FEATURES, feat.shape[1])
        out[:n, :m] = feat[:n, :m]
        return out
    return feat.astype(np.float32)


def build_feature_arrays(
    project: WakeProject,
    *,
    augment_positives: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns train_x, train_y, val_x, val_y.

    Positives are heavily augmented; negatives include recorded clips plus
    synthetic silence/noise so the model does not learn “digital quiet = wake”.
    """
    rng = np.random.default_rng(seed)
    window = int(project.clip_seconds * project.sample_rate)

    pos_paths = sorted(project.positives_dir.glob("*.wav"))
    neg_paths = sorted(project.negatives_dir.rglob("*.wav"))
    if len(pos_paths) < 8:
        raise RuntimeError(
            f"need at least 8 positive clips, found {len(pos_paths)} in {project.positives_dir}"
        )
    if len(neg_paths) < 8:
        raise RuntimeError(
            f"need at least 8 negative clips, found {len(neg_paths)}. "
            "Record categories: speech, keyboard, silence, ambient."
        )

    logger.info(
        "embedding %d positives (×%d aug) and %d negatives…",
        len(pos_paths),
        augment_positives,
        len(neg_paths),
    )

    pos_feats: list[np.ndarray] = []
    for path in pos_paths:
        clip = load_clip_16k(path, project.sample_rate)
        for _ in range(augment_positives):
            windowed = place_in_window(clip, window, rng)
            # Mild noise mix
            windowed = np.clip(
                windowed + _noise_floor(window, rng, level=float(rng.uniform(1e-4, 8e-4))),
                -1.0,
                1.0,
            )
            pos_feats.append(embed_window(windowed, project.sample_rate))

    neg_feats: list[np.ndarray] = []
    for i, path in enumerate(neg_paths, 1):
        clip = load_clip_16k(path, project.sample_rate)
        for windowed in negative_windows(clip, window, rng):
            neg_feats.append(embed_window(windowed, project.sample_rate))
        if i % 100 == 0:
            logger.info("negatives %d/%d (%d windows)", i, len(neg_paths), len(neg_feats))

    # Synthetic silence / faint-noise negatives (critical — see zero-pad pitfall).
    for _ in range(max(40, len(pos_feats) // 2)):
        level = float(rng.uniform(0.0, 5e-4))
        synth = _noise_floor(window, rng, level=level)
        neg_feats.append(embed_window(synth, project.sample_rate))

    X_pos = np.stack(pos_feats).astype(np.float32)
    X_neg = np.stack(neg_feats).astype(np.float32)

    # Hold out ~20% for validation
    def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        idx = rng.permutation(n)
        n_val = max(1, n // 5)
        return x[idx[n_val:]], x[idx[:n_val]]

    tr_pos, va_pos = split(X_pos)
    tr_neg, va_neg = split(X_neg)

    train_x = np.concatenate([tr_pos, tr_neg], axis=0)
    train_y = np.concatenate(
        [
            np.ones(tr_pos.shape[0], dtype=np.float32),
            np.zeros(tr_neg.shape[0], dtype=np.float32),
        ]
    )
    val_x = np.concatenate([va_pos, va_neg], axis=0)
    val_y = np.concatenate(
        [
            np.ones(va_pos.shape[0], dtype=np.float32),
            np.zeros(va_neg.shape[0], dtype=np.float32),
        ]
    )

    # Shuffle train
    perm = rng.permutation(train_x.shape[0])
    train_x, train_y = train_x[perm], train_y[perm]

    project.features_dir.mkdir(parents=True, exist_ok=True)
    np.save(project.features_dir / "train_x.npy", train_x)
    np.save(project.features_dir / "train_y.npy", train_y)
    np.save(project.features_dir / "val_x.npy", val_x)
    np.save(project.features_dir / "val_y.npy", val_y)
    logger.info(
        "features ready train=%s val=%s (pos_train=%d neg_train=%d)",
        train_x.shape,
        val_x.shape,
        int(tr_pos.shape[0]),
        int(tr_neg.shape[0]),
    )
    return train_x, train_y, val_x, val_y
