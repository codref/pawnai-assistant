"""CLI entry: pawn-assistant serve | list-devices | probe | monitor."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import uvicorn
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from pawn_assistant.audio.capture import (
    _is_skippable,
    list_input_devices,
    measure_device,
)
app = typer.Typer(
    name="pawn-assistant",
    help="Pawn always-listening wake loop and cyberpunk HUD.",
    no_args_is_help=True,
)
console = Console()


def _level_bar(db: float, width: int = 24) -> str:
    # Map -60..0 dBFS to bar.
    pct = max(0.0, min(1.0, (db + 60.0) / 60.0))
    filled = int(round(pct * width))
    return "█" * filled + "·" * (width - filled)


@app.command("list-devices")
def list_devices_cmd() -> None:
    """List sounddevice input devices."""
    devices = list_input_devices()
    table = Table(title="Input devices")
    table.add_column("index", justify="right")
    table.add_column("name")
    table.add_column("hostapi")
    table.add_column("ch", justify="right")
    table.add_column("rate", justify="right")
    table.add_column("note")
    for d in devices:
        note = "skip" if _is_skippable(d["name"]) else ""
        table.add_row(
            str(d["index"]),
            d["name"],
            d["hostapi"],
            str(d["channels"]),
            f"{d['default_samplerate']:.0f}",
            note,
        )
    console.print(table)
    console.print(
        "Tip: run [cyan]pawn-assistant probe[/] (speak during it), then "
        "[cyan]pawn-assistant monitor --device N[/]."
    )


@app.command("probe")
def probe(
    seconds: float = typer.Option(0.6, help="Capture length per device"),
) -> None:
    """Try each input briefly and report level — speak/tap the mic while it runs."""
    console.print(
        "[bold]Probing inputs[/] — make noise / speak for a few seconds…"
    )
    table = Table(title="Mic probe")
    table.add_column("index", justify="right")
    table.add_column("open")
    table.add_column("rate", justify="right")
    table.add_column("rms", justify="right")
    table.add_column("peak", justify="right")
    table.add_column("name")

    best: Optional[dict] = None
    for d in list_input_devices():
        if _is_skippable(d["name"]):
            table.add_row(str(d["index"]), "skip", "—", "—", "—", d["name"])
            continue
        result = measure_device(d["index"], seconds=seconds)
        if not result.get("ok"):
            err = result.get("error", "")
            table.add_row(
                str(d["index"]),
                "fail",
                "—",
                "—",
                "—",
                f"{d['name']}" + (f" — {err}" if err else ""),
            )
            continue
        silent = result["silent"]
        mark = "quiet" if silent else "LIVE"
        table.add_row(
            str(result["index"]),
            mark,
            str(result["stream_rate"]),
            f"{result['rms_db']:.1f}",
            f"{result['peak_db']:.1f}",
            result["name"],
        )
        if not silent and (best is None or result["peak_db"] > best["peak_db"]):
            best = result

    console.print(table)
    if best is None:
        console.print(
            "[yellow]No device showed signal.[/]\n"
            "On PipeWire this usually means the mic Source is disabled "
            "(profile is output-only). Check:\n"
            "  • System Settings → Audio → Input / enable Analog Stereo Duplex\n"
            "  • [cyan]wpctl status[/] — Audio → Sources should list a mic\n"
            "  • [cyan]pactl get-default-source[/] — should not end with [.monitor]\n"
            "Hardware ALSA ([cyan]hw:0,0[/]) may still work — try "
            "[cyan]pawn-assistant monitor --device 0[/]."
        )
        raise typer.Exit(code=1)

    console.print(
        f"Best signal: index [bold cyan]{best['index']}[/] "
        f"({best['name']}) @ {best['stream_rate']} Hz — "
        f"peak {best['peak_db']:.1f} dBFS\n"
        f"Serve with: [cyan]pawn-assistant serve --device {best['index']}[/]"
    )


@app.command("monitor")
def monitor(
    device: int = typer.Option(..., "--device", "-d", help="sounddevice input index"),
    seconds: float = typer.Option(0.0, help="Stop after N seconds (0 = until Ctrl-C)"),
) -> None:
    """Live level meter for one device — talk and watch the bar move."""
    from pawn_assistant.audio.capture import AudioCapture

    console.print(
        f"Monitoring device [cyan]{device}[/] — speak; Ctrl-C to stop."
    )
    cap = AudioCapture(device=device)
    try:
        cap.start()
    except Exception as exc:
        console.print(f"[red]Failed to open device {device}:[/] {exc}")
        raise typer.Exit(code=1) from exc

    info = cap.device_info
    console.print(
        f"Opened [cyan]{info.index}[/] {info.name} "
        f"@ {info.stream_rate} Hz → {info.sample_rate} Hz"
    )
    t0 = time.monotonic()
    try:
        with Live(console=console, refresh_per_second=12) as live:
            while True:
                if seconds > 0 and (time.monotonic() - t0) >= seconds:
                    break
                frame = cap.read(timeout=0.2)
                if frame is None:
                    continue
                peak = float(np.max(np.abs(frame)))
                rms = float(np.sqrt(np.mean(np.square(frame))))
                peak_db = 20.0 * np.log10(peak) if peak > 1e-9 else -120.0
                rms_db = 20.0 * np.log10(rms) if rms > 1e-9 else -120.0
                text = Text.assemble(
                    (f"rms {rms_db:7.1f} dBFS  ", "cyan"),
                    (f"peak {peak_db:7.1f}  ", "yellow"),
                    (_level_bar(peak_db), "bold green" if peak_db > -45 else "dim"),
                )
                live.update(text)
    except KeyboardInterrupt:
        console.print("\nstopped")
    finally:
        cap.stop()


@app.command("serve")
def serve(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML config (default: ./pawn-assistant.yaml or ~/.config/pawn-assistant/config.yaml)",
    ),
    host: Optional[str] = typer.Option(None, help="Bind address (overrides config)"),
    port: Optional[int] = typer.Option(None, help="Bind port (overrides config)"),
    device: Optional[int] = typer.Option(None, help="sounddevice input index"),
    wake_threshold: Optional[float] = typer.Option(None, help="Wake score threshold"),
    wake_model: Optional[str] = typer.Option(
        None, help="openWakeWord id or path to .onnx"
    ),
    whisper_model: Optional[str] = typer.Option(None, help="faster-whisper model size"),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        help="ASR language (it/en/…). Overrides config; use 'auto' for autodetection",
    ),
    debug_audio: Optional[str] = typer.Option(
        None,
        "--debug-audio",
        help="Directory for wake/utterance WAV dumps (e.g. ./debug-audio)",
    ),
    log_level: Optional[str] = typer.Option(None, help="Logging level"),
) -> None:
    """Start the wake loop and serve the HUD."""
    from pawn_assistant.config import load_settings

    settings = load_settings(
        config,
        overrides={
            "host": host,
            "port": port,
            "device": device,
            "wake_threshold": wake_threshold,
            "wake_model": wake_model,
            "whisper_model": whisper_model,
            "asr_language": language,
            "debug_audio_dir": debug_audio,
            "log_level": log_level,
        },
    )
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from pawn_assistant.server.app import create_app

    application = create_app(settings)
    lang = settings.asr_language or "auto"
    console.print(
        f"[bold cyan]PAWN[/] listening — HUD at http://{settings.host}:{settings.port}/  "
        f"(wake=[yellow]{settings.wake_model}[/], thr={settings.wake_threshold}, "
        f"asr_lang=[yellow]{lang}[/])"
    )
    if settings.config_path:
        console.print(f"[dim]config {settings.config_path}[/dim]")
    else:
        console.print(
            "[dim]No config file — copy [cyan]pawn-assistant.example.yaml[/] "
            "to [cyan]pawn-assistant.yaml[/].[/dim]"
        )
    if settings.device is None:
        console.print(
            "[dim]No audio.device set; auto-picks pulse/sysdefault/hw. "
            "If the HUD stays silent, run [cyan]pawn-assistant probe[/].[/dim]"
        )
    if settings.debug_audio_dir:
        console.print(
            f"[dim]Debug WAVs → [cyan]{settings.debug_audio_dir}[/] "
            f"(wake_*, near_miss_*, utt_*, empty_*)[/dim]"
        )
    uvicorn.run(
        application,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


wake_train_app = typer.Typer(
    name="wake-train",
    help="Train a custom openWakeWord ONNX model from your own recordings.",
    no_args_is_help=True,
)
app.add_typer(wake_train_app, name="wake-train")


@wake_train_app.command("init")
def wake_init(
    phrase: str = typer.Argument(..., help='Wake phrase, e.g. "hey pawn"'),
    out: str = typer.Option(
        "wakewords",
        "--out",
        help="Parent directory; creates <out>/<slug>/",
    ),
) -> None:
    """Create a training project folder and manifest."""
    from pawn_assistant.wake_train.project import WakeProject, slugify

    root = Path(out) / slugify(phrase)
    proj = WakeProject.create(root, phrase)
    console.print(f"[bold cyan]Project[/] {proj.root}")
    console.print(f"phrase=[yellow]{proj.phrase}[/]  model=[yellow]{proj.model_name}.onnx[/]")
    console.print(
        "Next:\n"
        f"  pawn-assistant wake-train record-pos --project {proj.root} --device 0 --count 40\n"
        f"  pawn-assistant wake-train record-neg --project {proj.root} --device 0 "
        "--category speech --count 60\n"
        f"  pawn-assistant wake-train record-neg --project {proj.root} --device 0 "
        "--category keyboard --count 40\n"
        f"  pawn-assistant wake-train record-neg --project {proj.root} --device 0 "
        "--category silence --count 30\n"
        f"  pawn-assistant wake-train train --project {proj.root}\n"
        f"  pawn-assistant serve --device 0 --wake-model {proj.model_path}"
    )


@wake_train_app.command("status")
def wake_status(
    project: str = typer.Option(..., "--project", "-p", help="Project directory"),
) -> None:
    """Show clip counts and whether a model exists."""
    from pawn_assistant.wake_train.project import WakeProject

    proj = WakeProject.load(Path(project))
    c = proj.counts()
    console.print(c)


@wake_train_app.command("record-pos")
def wake_record_pos(
    project: str = typer.Option(..., "--project", "-p"),
    device: Optional[int] = typer.Option(None, "--device", "-d"),
    count: int = typer.Option(40, help="Number of positive clips"),
    interval: float = typer.Option(3.5, help="Seconds between takes"),
) -> None:
    """Record positive clips of the wake phrase (hands-free cadence)."""
    from pawn_assistant.wake_train.project import WakeProject
    from pawn_assistant.wake_train.record import record_session

    proj = WakeProject.load(Path(project))
    record_session(
        proj, kind="pos", count=count, device=device, interval=interval
    )


@wake_train_app.command("record-neg")
def wake_record_neg(
    project: str = typer.Option(..., "--project", "-p"),
    device: Optional[int] = typer.Option(None, "--device", "-d"),
    count: int = typer.Option(40, help="Number of negative clips"),
    category: str = typer.Option(
        "speech",
        help="Category folder: speech | keyboard | silence | ambient | music | …",
    ),
    interval: float = typer.Option(2.5, help="Seconds between takes"),
) -> None:
    """Record negative clips (one category at a time)."""
    from pawn_assistant.wake_train.project import WakeProject
    from pawn_assistant.wake_train.record import record_session

    proj = WakeProject.load(Path(project))
    record_session(
        proj,
        kind="neg",
        count=count,
        device=device,
        category=category,
        interval=interval,
    )


@wake_train_app.command("train")
def wake_train_cmd(
    project: str = typer.Option(..., "--project", "-p"),
    steps: int = typer.Option(4000, help="Training steps"),
    augment: int = typer.Option(4, help="Augmentations per positive clip"),
) -> None:
    """Embed clips, train classifier, write <model_name>.onnx."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from pawn_assistant.wake_train.project import WakeProject
    from pawn_assistant.wake_train.train_model import train_and_export

    proj = WakeProject.load(Path(project))
    path = train_and_export(
        proj, steps=steps, augment_positives=augment
    )
    console.print(f"[bold green]Model ready[/] {path}")
    console.print(
        f"Test:  pawn-assistant wake-train test --project {proj.root} --device 0\n"
        f"Serve: pawn-assistant serve --device 0 --wake-model {path}"
    )


@wake_train_app.command("test")
def wake_test_cmd(
    project: str = typer.Option(..., "--project", "-p"),
    device: Optional[int] = typer.Option(None, "--device", "-d"),
    seconds: float = typer.Option(30.0, help="How long to monitor"),
    threshold: float = typer.Option(0.5, help="Score highlight threshold"),
    model: Optional[str] = typer.Option(
        None, help="Override ONNX path (default: project model)"
    ),
) -> None:
    """Live-score the trained model from the mic."""
    from pawn_assistant.wake_train.project import WakeProject
    from pawn_assistant.wake_train.train_model import live_test

    proj = WakeProject.load(Path(project))
    path = Path(model) if model else proj.model_path
    if not path.exists():
        console.print(f"[red]Missing model[/] {path}")
        raise typer.Exit(1)
    live_test(path, device=device, seconds=seconds, threshold=threshold)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
