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


# ---------------------------------------------------------------- augmentation

# Negative categories that make useful backgrounds under a positive (people
# talking, the room, the radio) — not silence, not other wake-like clips.
_BACKGROUND_SKIP = ("silence",)
# Mistakes the live model already made: worth more than one look each.
_HARD_NEGATIVE_PREFIX = "live_"


def speed_perturb(clip: np.ndarray, factor: float) -> np.ndarray:
    """Resample so the phrase plays ``factor``× faster (pitch shifts with it)."""
    if abs(factor - 1.0) < 1e-3 or clip.size < 2:
        return clip.astype(np.float32)
    n_out = max(2, int(round(clip.size / factor)))
    x_old = np.linspace(0.0, 1.0, num=clip.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, clip.astype(np.float64)).astype(np.float32)


def synthetic_rir(rng: np.random.Generator, sample_rate: int = 16_000) -> np.ndarray:
    """Cheap room impulse response: direct path + exponentially decaying noise."""
    rt60 = float(rng.uniform(0.15, 0.6))
    n = int(rt60 * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    tail = rng.normal(0.0, 1.0, n).astype(np.float32) * np.exp(-6.9 * t / rt60).astype(np.float32)
    predelay = int(rng.uniform(0.002, 0.012) * sample_rate)
    tail[:predelay] = 0.0
    tail *= float(rng.uniform(0.15, 0.5)) / max(1e-6, float(np.sqrt(np.sum(tail**2))))
    tail[0] = 1.0
    return tail


def add_reverb(clip: np.ndarray, rng: np.random.Generator, sample_rate: int = 16_000) -> np.ndarray:
    from scipy.signal import fftconvolve

    wet = fftconvolve(clip.astype(np.float32), synthetic_rir(rng, sample_rate))[: clip.size]
    peak_in = float(np.max(np.abs(clip))) or 1.0
    peak_out = float(np.max(np.abs(wet))) or 1.0
    return (wet * (peak_in / peak_out)).astype(np.float32)


def mix_at_snr(
    clip: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Lay *noise* (looped/cropped to fit) under *clip* at ``snr_db``."""
    if noise.size == 0:
        return clip
    if noise.size < clip.size:
        noise = np.tile(noise, int(np.ceil(clip.size / noise.size)))
    start = int(rng.integers(0, noise.size - clip.size + 1))
    bed = noise[start : start + clip.size].astype(np.float32)
    p_clip = float(np.mean(clip**2)) or 1e-9
    p_bed = float(np.mean(bed**2)) or 1e-9
    scale = np.sqrt(p_clip / (p_bed * 10 ** (snr_db / 10.0)))
    return np.clip(clip + bed * scale, -1.0, 1.0).astype(np.float32)


def augment_positive(
    clip: np.ndarray,
    window_samples: int,
    rng: np.random.Generator,
    backgrounds: list[np.ndarray],
    sample_rate: int = 16_000,
) -> np.ndarray:
    """One training window: speed, room, background and level variation."""
    out = clip.astype(np.float32)
    if rng.random() < 0.7:
        out = speed_perturb(out, float(rng.uniform(0.9, 1.1)))
    if rng.random() < 0.5:
        out = add_reverb(out, rng, sample_rate)
    windowed = place_in_window(out, window_samples, rng)
    if backgrounds and rng.random() < 0.5:
        bed = backgrounds[int(rng.integers(0, len(backgrounds)))]
        windowed = mix_at_snr(windowed, bed, float(rng.uniform(5.0, 20.0)), rng)
    else:
        windowed = windowed + _noise_floor(window_samples, rng, level=float(rng.uniform(1e-4, 8e-4)))
    # Far-field / quiet speaker as well as close and loud.
    windowed = windowed * float(rng.uniform(0.35, 1.0))
    return np.clip(windowed, -1.0, 1.0).astype(np.float32)


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


def _load_backgrounds(
    neg_paths: list[Path], negatives_dir: Path, sample_rate: int, rng: np.random.Generator,
    limit: int = 200,
) -> list[np.ndarray]:
    cands = [
        p for p in neg_paths
        if not any(part in _BACKGROUND_SKIP for part in p.relative_to(negatives_dir).parts[:-1])
    ]
    if len(cands) > limit:
        cands = [cands[i] for i in sorted(rng.choice(len(cands), size=limit, replace=False))]
    out = []
    for path in cands:
        clip = load_clip_16k(path, sample_rate)
        if clip.size and float(np.max(np.abs(clip))) > 1e-3:
            out.append(clip)
    return out


def build_feature_arrays(
    project: WakeProject,
    *,
    augment_positives: int = 4,
    seed: int = 0,
    hard_negative_copies: int = 3,
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
        "embedding %d positives (×%d: speed/reverb/background) and %d negatives…",
        len(pos_paths),
        augment_positives,
        len(neg_paths),
    )

    backgrounds = _load_backgrounds(neg_paths, project.negatives_dir, project.sample_rate, rng)
    logger.info("%d background clips for positive mixing", len(backgrounds))

    pos_feats: list[np.ndarray] = []
    for path in pos_paths:
        clip = load_clip_16k(path, project.sample_rate)
        # First copy stays close to the recording; the rest are augmented.
        pos_feats.append(embed_window(place_in_window(clip, window, rng), project.sample_rate))
        for _ in range(max(0, augment_positives - 1)):
            windowed = augment_positive(clip, window, rng, backgrounds, project.sample_rate)
            pos_feats.append(embed_window(windowed, project.sample_rate))

    neg_feats: list[np.ndarray] = []
    for i, path in enumerate(neg_paths, 1):
        clip = load_clip_16k(path, project.sample_rate)
        for windowed in negative_windows(clip, window, rng):
            neg_feats.append(embed_window(windowed, project.sample_rate))
        if path.parent.name.startswith(_HARD_NEGATIVE_PREFIX):
            # Real false wakes: jittered extra copies so they weigh more.
            for _ in range(max(0, hard_negative_copies - 1)):
                jitter = place_in_window(clip, window, rng) * float(rng.uniform(0.6, 1.2))
                neg_feats.append(embed_window(np.clip(jitter, -1, 1), project.sample_rate))
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
