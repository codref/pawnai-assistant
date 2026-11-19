"""WAV helpers for wake-word clips (always 16 kHz mono int16)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(audio, dtype=np.float32) * 32767.0, -32768, 32767).astype(
        np.int16
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    else:
        raise ValueError(f"unsupported sample width {width} in {path}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, int(sr)


def resample_mono(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return np.asarray(audio, dtype=np.float32)
    duration = audio.size / float(src_rate)
    n_out = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)


def load_clip_16k(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    audio, sr = read_wav(path)
    return resample_mono(audio, sr, sample_rate)
