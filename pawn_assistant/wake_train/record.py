"""Guided mic recording for positives / negatives."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console

from pawn_assistant.audio.beep import play_beep
from pawn_assistant.audio.capture import _OPEN_RATES, quiet_portaudio
from pawn_assistant.wake_train.io_wav import resample_mono, write_wav
from pawn_assistant.wake_train.project import WakeProject

console = Console()

# Cached after first successful open in a session.
_STREAM_RATE: int | None = None


def _record_seconds(
    seconds: float,
    target_rate: int,
    device: int | None,
) -> np.ndarray:
    """Record *seconds* of mono audio, always returning *target_rate* samples.

    ALSA ``hw:0,0`` often rejects 16 kHz — try 48k / 44.1k and resample.
    """
    global _STREAM_RATE
    rates: list[int] = []
    if _STREAM_RATE is not None:
        rates.append(_STREAM_RATE)
    for rate in _OPEN_RATES:
        if rate not in rates:
            rates.append(rate)

    last_err: Exception | None = None
    with quiet_portaudio():
        for rate in rates:
            frames = max(1, int(seconds * rate))
            try:
                rec = sd.rec(
                    frames,
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    device=device,
                )
                sd.wait()
                audio = np.asarray(rec[:, 0], dtype=np.float32)
                _STREAM_RATE = rate
                if rate != target_rate:
                    audio = resample_mono(audio, rate, target_rate)
                return audio
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
    raise RuntimeError(
        f"could not open input device {device} at any of {_OPEN_RATES}: {last_err}"
    )


def record_session(
    project: WakeProject,
    *,
    kind: str,
    count: int,
    device: int | None,
    category: str = "misc",
    interval: float = 3.5,
) -> list[Path]:
    """Record *count* clips with an automatic cadence (no keyboard between takes)."""
    global _STREAM_RATE
    _STREAM_RATE = None

    if kind not in ("pos", "neg"):
        raise ValueError("kind must be pos or neg")

    out_dir = (
        project.positives_dir
        if kind == "pos"
        else project.negatives_dir / category
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    label = project.phrase if kind == "pos" else f"NOT wake ({category})"
    console.print(
        f"[bold cyan]Recording {count} × {kind}[/] — {label}\n"
        f"Device={device if device is not None else 'auto'}  "
        f"clip={project.record_seconds:.1f}s  cadence={interval:.1f}s\n"
        "[dim]Do not type between takes (keyboard noise poisons positives). Ctrl-C to stop.[/dim]"
    )

    # Probe once so we can show the host rate before the first countdown.
    try:
        probe = _record_seconds(0.05, project.sample_rate, device)
        console.print(
            f"[dim]Capture @ {_STREAM_RATE} Hz → {project.sample_rate} Hz "
            f"(probe peak={float(np.max(np.abs(probe))) if probe.size else 0:.3f})[/dim]"
        )
    except Exception as exc:
        console.print(f"[red]Mic open failed:[/] {exc}")
        raise

    saved: list[Path] = []
    try:
        for i in range(1, count + 1):
            console.print(f"\n[{i}/{count}] Get ready…")
            for n in (3, 2, 1):
                console.print(f"  {n}")
                time.sleep(0.55)
            play_beep(duration_ms=80, freq_hz=990.0, volume=0.2)
            time.sleep(0.15)
            console.print(f"  [bold green]SPEAK[/] — {label}")
            audio = _record_seconds(
                project.record_seconds, project.sample_rate, device
            )
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = out_dir / f"{kind}_{stamp}_{i:03d}.wav"
            write_wav(path, audio, project.sample_rate)
            saved.append(path)
            console.print(f"  saved {path.name}  peak={peak:.3f}")
            pause = max(0.4, interval - 0.2)
            if i < count:
                time.sleep(pause)
    except KeyboardInterrupt:
        console.print("\n[yellow]stopped early[/]")

    console.print(f"[bold]Done[/] — {len(saved)} clips → {out_dir}")
    return saved
