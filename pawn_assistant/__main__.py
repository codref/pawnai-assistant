"""CLI entry: pawn-assistant serve | ask | say | models | list-devices | probe | monitor."""

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


def _command_settings(config: Optional[str]):
    from pawn_assistant.config import load_settings

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return load_settings(config)


def _build_router(settings, *, use_llm: bool = True):
    from pawn_assistant.commands import CommandContext, CommandRegistry, CommandRouter
    from pawn_assistant.commands.llm import IntentLLM

    # Same enabled set as `serve` (YAML + HUD toggles), but read-only state.
    state_path = settings.commands_state_path()
    registry = CommandRegistry(config=settings.commands, state_path=state_path)
    registry.state_path = None  # `ask` never writes plugin state
    llm = None
    if use_llm and settings.intent_enabled:
        llm = IntentLLM(
            model=settings.intent_model,
            n_threads=settings.intent_threads,
            idle_unload_s=0,
            intents=registry.intents(),
        )
    return CommandRouter(
        ctx=CommandContext(lang=settings.reply_language),
        llm=llm,
        fastpath_ratio=settings.intent_fastpath_ratio,
        registry=registry,
    )


def _active_commands(router) -> list[str]:
    return [c.name for c in router.registry.commands() if c.is_active()]


def _run_eval(router, path: Path) -> None:
    import yaml

    cases = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    table = Table(title=f"intent eval — {path.name}")
    for col in ("utterance", "expected", "got", "via", "ms", ""):
        table.add_column(col)
    ok = 0
    llm_ms: list[float] = []
    for case in cases:
        text, want = case["text"], case["intent"]
        want_slots = case.get("slots") or {}
        res = router.classify(text)
        good = res.intent == want and res.slots == want_slots
        ok += good
        if res.source == "llm":
            llm_ms.append(res.parse_ms)
        got = res.intent + (f" {res.slots}" if res.slots else "")
        table.add_row(
            text, want + (f" {want_slots}" if want_slots else ""), got, res.source,
            f"{res.parse_ms:.0f}", "[green]✓[/]" if good else "[red]✗[/]",
        )
    router.registry.shutdown()
    console.print(table)
    console.print(f"[bold]accuracy[/] {ok}/{len(cases)} = {100.0 * ok / max(1, len(cases)):.1f}%")
    if llm_ms:
        arr = np.array(llm_ms)
        console.print(
            f"[bold]llm[/] n={arr.size} p50={np.percentile(arr, 50):.0f} ms "
            f"p95={np.percentile(arr, 95):.0f} ms max={arr.max():.0f} ms"
        )


def _phrase_samples(router, settings, res, text: str, samples: int) -> str:
    """Print ``samples`` phrased variants with timings; return the first."""
    from pawn_assistant.commands.phrasing import PhraseSettings, ReplyPhraser

    llm = router.llm
    phraser = ReplyPhraser(llm, PhraseSettings.from_settings(settings), registry=router.registry)
    if not phraser.applies(res.intent):
        console.print(f"[bold green]PAWN:[/] {res.reply}  [dim](phrasing off for {res.intent})[/]")
        return res.reply
    with console.status("warming up LLM prompts…"):
        llm.load()
        phraser.warmup()
    console.print(f"[dim]canonical:[/] {res.reply}")
    first_ms: list[float] = []
    fallbacks = 0
    first_reply = ""
    for _ in range(samples):
        out, stats = phraser.phrase(res.intent, text, res.reply)
        first_reply = first_reply or out
        if stats.phrased:
            first_ms.append(stats.first_ms)
            note = f"first {stats.first_ms:.0f} ms · total {stats.gen_ms:.0f} ms"
            if stats.reason:
                note += f" · {stats.reason}"
            console.print(f"[bold green]PAWN:[/] {out}  [dim]{note}[/]")
        else:
            fallbacks += 1
            console.print(f"[bold yellow]PAWN:[/] {out}  [dim]fallback · {stats.reason}[/]")
    if samples > 1:
        line = f"[bold]phrased[/] {samples - fallbacks}/{samples}"
        if first_ms:
            arr = np.array(first_ms)
            line += (
                f" · first sentence p50={np.percentile(arr, 50):.0f} ms "
                f"p95={np.percentile(arr, 95):.0f} ms"
            )
        console.print(line)
    return first_reply


@app.command("ask")
def ask(
    text: Optional[str] = typer.Argument(None, help="Command text, e.g. \"che ora è\""),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config"),
    speak: bool = typer.Option(True, "--speak/--no-speak", help="Say the reply with Piper"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Fast-path only"),
    eval_file: Optional[Path] = typer.Option(
        None, "--eval", help="YAML list of {text, intent, slots?} — prints accuracy/latency"
    ),
    phrase: Optional[bool] = typer.Option(
        None, "--phrase/--no-phrase", help="Reword the reply with the LLM (default: config)"
    ),
    samples: int = typer.Option(1, "--samples", "-n", help="Phrased variants to generate"),
) -> None:
    """Route a typed command through fast-path → LLM → handler (no mic)."""
    settings = _command_settings(config)
    router = _build_router(settings, use_llm=not no_llm)
    if eval_file is not None:
        if router.llm is not None:
            with console.status("warming up intent LLM…"):
                router.llm.warmup()
        _run_eval(router, eval_file)
        return
    if not text:
        console.print("[red]give a command text or --eval FILE[/]")
        raise typer.Exit(2)
    res = router.handle(text)
    console.print(
        f"[magenta]{res.intent}[/] {res.slots or ''} "
        f"[dim]via {res.source} · {res.parse_ms:.0f} ms[/]"
    )
    reply = res.reply
    use_phrase = settings.phrase_enabled if phrase is None else phrase
    if use_phrase and router.llm is not None and res.reply:
        reply = _phrase_samples(router, settings, res, text, max(1, samples))
    else:
        console.print(f"[bold green]PAWN:[/] {res.reply}")
    if speak and settings.tts_enabled and reply:
        from pawn_assistant.audio.tts import PiperTTS

        tts = PiperTTS(
            voice=settings.tts_voice,
            length_scale=settings.tts_length_scale,
            volume=settings.tts_volume,
            idle_unload_s=0,
            speaker_id=settings.tts_speaker_id,
        )
        speech = tts.speak(reply)
        console.print(f"[dim]tts {speech.synth_ms:.0f} ms for {speech.duration_ms:.0f} ms audio[/]")
    active = _active_commands(router)
    if "radio" in active:
        console.print("[dim]radio playing — Ctrl+C to stop[/]")
        try:
            while "radio" in _active_commands(router):
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    elif active:
        console.print(f"[dim]{', '.join(active)} started — it only runs while `serve` is running[/]")
    router.registry.shutdown()


@app.command("say")
def say(
    text: str = typer.Argument(..., help="Text to speak"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config"),
    voice: Optional[str] = typer.Option(None, help="Piper voice id or .onnx path"),
) -> None:
    """Speak text with the configured Piper voice."""
    from pawn_assistant.audio.tts import PiperTTS

    settings = _command_settings(config)
    tts = PiperTTS(
        voice=voice or settings.tts_voice,
        length_scale=settings.tts_length_scale,
        volume=settings.tts_volume,
        idle_unload_s=0,
        speaker_id=settings.tts_speaker_id,
    )
    speech = tts.speak(text)
    console.print(
        f"[dim]{tts.voice}: synth {speech.synth_ms:.0f} ms for {speech.duration_ms:.0f} ms "
        f"audio (rtf {speech.synth_ms / max(1.0, speech.duration_ms):.2f})[/]"
    )


voice_app = typer.Typer(
    name="voice", help="Choose, audition and configure the Piper voice.", no_args_is_help=True
)
app.add_typer(voice_app, name="voice")

_CONFIG_OPT = typer.Option(None, "--config", "-c", help="YAML config")


def _make_tts(settings, voice: str, speed: Optional[float], volume: Optional[float],
              speaker: Optional[int]):
    from pawn_assistant.audio.tts import PiperTTS

    return PiperTTS(
        voice=voice,
        length_scale=speed if speed is not None else settings.tts_length_scale,
        volume=volume if volume is not None else settings.tts_volume,
        idle_unload_s=0,
        speaker_id=speaker if speaker is not None else settings.tts_speaker_id,
    )


def _audition(tts, text: str) -> bool:
    """Speak ``text``; print timings. Returns False when the voice failed to load."""
    try:
        with console.status(f"loading {tts.voice}…"):
            tts.load()
        speech = tts.speak(text)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]{tts.voice}: {exc}[/]")
        return False
    console.print(
        f"[dim]synth {speech.synth_ms:.0f} ms · audio {speech.duration_ms:.0f} ms · "
        f"rtf {speech.synth_ms / max(1.0, speech.duration_ms):.2f}[/]"
    )
    return True


def _default_lang(settings, lang: Optional[str]) -> str:
    return lang or settings.reply_language or "en"


@voice_app.command("current")
def voice_current(config: Optional[str] = _CONFIG_OPT) -> None:
    """Show the configured voice and its settings."""
    from pawn_assistant.audio.voices import is_installed

    s = _command_settings(config)
    console.print(f"voice      [bold cyan]{s.tts_voice}[/]")
    console.print(f"downloaded {'yes' if is_installed(s.tts_voice) or s.tts_voice.endswith('.onnx') else 'no'}")
    console.print(
        f"speed      {s.tts_length_scale} (length_scale; >1 slower)\n"
        f"volume     {s.tts_volume}\n"
        f"speaker_id {s.tts_speaker_id}\n"
        f"config     {s.config_path or '[yellow]none found[/]'}"
    )


@voice_app.command("list")
def voice_list(
    lang: Optional[str] = typer.Option(
        None, "--lang", "-l", help="Language/locale prefix (it, en_GB…). Default: reply_language"
    ),
    all_langs: bool = typer.Option(False, "--all", help="Do not filter by language"),
    quality: Optional[str] = typer.Option(None, "--quality", "-q", help="x_low|low|medium|high"),
    installed: bool = typer.Option(False, "--installed", help="Only downloaded voices"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """List Piper voices from the rhasspy/piper-voices catalog."""
    from pawn_assistant.audio.voices import filter_voices, load_catalog

    settings = _command_settings(config)
    with console.status("loading voice catalog…"):
        voices, online = load_catalog()
    if not online:
        console.print("[yellow]Catalog unreachable — showing locally cached voices only.[/]")
    shown = filter_voices(
        voices,
        lang=None if all_langs else _default_lang(settings, lang),
        quality=quality,
        installed=installed,
    )
    table = Table(title=f"Piper voices ({len(shown)})")
    for col, just in (("id", "left"), ("locale", "left"), ("quality", "left"),
                      ("spk", "right"), ("rate", "right"), ("size", "right"), ("", "left")):
        table.add_column(col, justify=just, no_wrap=True)
    for v in shown:
        mark = "[green]✓ installed[/]" if v.installed else ""
        if v.id == settings.tts_voice:
            mark += " [bold cyan]← current[/]"
        table.add_row(
            v.id, v.locale, v.quality, str(v.speakers), str(v.sample_rate),
            f"{v.size_mb:.0f} MB" if v.size_mb else "—", mark,
        )
    console.print(table)
    console.print(
        "Audition: [cyan]pawn-assistant voice test ID [ID…][/]   "
        "Interactive: [cyan]pawn-assistant voice pick[/]"
    )


@voice_app.command("test")
def voice_test(
    voices: Optional[list[str]] = typer.Argument(
        None, help="Voice ids / .onnx paths (default: the configured voice)"
    ),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Sentence to speak"),
    long: bool = typer.Option(False, "--long", help="Use the longer built-in sentence"),
    lang: Optional[str] = typer.Option(
        None, "--lang", "-l", help="With --all: audition every voice of this language"
    ),
    all_voices: bool = typer.Option(False, "--all", help="Audition every voice for --lang"),
    quality: Optional[str] = typer.Option(None, "--quality", "-q"),
    speed: Optional[float] = typer.Option(None, "--speed", help="length_scale (>1 slower)"),
    volume: Optional[float] = typer.Option(None, "--volume"),
    speaker: Optional[int] = typer.Option(None, "--speaker", help="Speaker id (multi-speaker)"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Speak the same sentence with one or more voices for A/B comparison."""
    from pawn_assistant.audio.voices import filter_voices, load_catalog, sample_text

    settings = _command_settings(config)
    ids = list(voices or [])
    if all_voices:
        cat, _ = load_catalog()
        ids += [v.id for v in filter_voices(cat, lang=_default_lang(settings, lang), quality=quality)]
    if not ids:
        ids = [settings.tts_voice]
    phrase = text or sample_text(_default_lang(settings, lang), long=long)
    console.print(f"[dim]“{phrase}”[/]")
    failed = 0
    for vid in ids:
        mark = " [bold cyan](current)[/]" if vid == settings.tts_voice else ""
        console.print(f"[bold]{vid}[/]{mark}")
        if not _audition(_make_tts(settings, vid, speed, volume, speaker), phrase):
            failed += 1
    if failed:
        raise typer.Exit(1)
    if len(ids) > 1:
        console.print("Keep one with: [cyan]pawn-assistant voice use ID[/]")


def _getkey() -> str:
    import sys

    if not sys.stdin.isatty():
        return (sys.stdin.readline() or "q")[:1].lower() or "q"
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


@voice_app.command("pick")
def voice_pick(
    lang: Optional[str] = typer.Option(None, "--lang", "-l"),
    quality: Optional[str] = typer.Option(None, "--quality", "-q"),
    installed: bool = typer.Option(False, "--installed", help="Only already-downloaded voices"),
    text: Optional[str] = typer.Option(None, "--text", "-t"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Browse voices interactively: n/p next/prev, r replay, +/- speed, enter=keep, q=quit."""
    from pawn_assistant.audio.voices import filter_voices, load_catalog, sample_text

    settings = _command_settings(config)
    use_lang = _default_lang(settings, lang)
    with console.status("loading voice catalog…"):
        cat, _ = load_catalog()
    pool = filter_voices(cat, lang=use_lang, quality=quality, installed=installed)
    if not pool:
        console.print(f"[red]No voices match lang={use_lang!r}.[/]")
        raise typer.Exit(1)
    phrase = text or sample_text(use_lang)
    idx = next((i for i, v in enumerate(pool) if v.id == settings.tts_voice), 0)
    speed = settings.tts_length_scale
    speaker: Optional[int] = settings.tts_speaker_id
    play = True
    while True:
        v = pool[idx]
        console.print(
            f"\n[bold cyan]{v.id}[/]  ({idx + 1}/{len(pool)})  speed {speed:.2f}"
            + (f"  speaker {speaker}" if v.speakers > 1 else "")
        )
        if play:
            _audition(_make_tts(settings, v.id, speed, None, speaker if v.speakers > 1 else None), phrase)
        play = True
        console.print("[dim]n/p next/prev · r replay · +/- speed · s speaker · enter keep · q quit[/]")
        key = _getkey()
        if key in ("n", " ", "\x1b"):
            idx = (idx + 1) % len(pool)
        elif key == "p":
            idx = (idx - 1) % len(pool)
        elif key == "r":
            pass
        elif key == "+":
            speed = max(0.5, round(speed - 0.05, 2))  # faster = smaller length_scale
        elif key == "-":
            speed = min(2.0, round(speed + 0.05, 2))
        elif key == "s" and v.speakers > 1:
            speaker = ((speaker or 0) + 1) % v.speakers
        elif key in ("\n", "\r"):
            _save_voice(settings, v.id, speed, speaker if v.speakers > 1 else None, yes=False)
            return
        elif key == "q":
            console.print("no change")
            return
        else:
            play = False


def _save_voice(settings, voice: str, speed: Optional[float], speaker: Optional[int],
                *, yes: bool, volume: Optional[float] = None, path: Optional[str] = None) -> None:
    from pawn_assistant.audio.voices import update_yaml_section

    target = Path(path) if path else (Path(settings.config_path) if settings.config_path else None)
    values: dict = {"voice": voice}
    if speed is not None:
        values["length_scale"] = speed
    if volume is not None:
        values["volume"] = volume
    if speaker is not None:
        values["speaker_id"] = speaker
    if target is None:
        console.print(
            "[yellow]No config file found.[/] Add to your YAML:\n"
            + "tts:\n" + "".join(f"  {k}: {v}\n" for k, v in values.items())
        )
        return
    if not yes and not typer.confirm(f"Write {values} to {target}?", default=True):
        console.print("no change")
        return
    update_yaml_section(target, "tts", values)
    console.print(f"[green]saved[/] tts.voice = [bold]{voice}[/] → {target}")


@voice_app.command("use")
def voice_use(
    voice: str = typer.Argument(..., help="Piper voice id or .onnx path"),
    speed: Optional[float] = typer.Option(None, "--speed", help="length_scale (>1 slower)"),
    volume: Optional[float] = typer.Option(None, "--volume"),
    speaker: Optional[int] = typer.Option(None, "--speaker"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before writing"),
    no_check: bool = typer.Option(False, "--no-check", help="Skip download/load validation"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Validate a voice (downloads it) and write it into the YAML config in place."""
    settings = _command_settings(config)
    if not no_check:
        tts = _make_tts(settings, voice, speed, volume, speaker)
        try:
            with console.status(f"downloading/loading {voice}…"):
                tts.load()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Cannot use {voice}:[/] {exc}")
            raise typer.Exit(1) from exc
    _save_voice(settings, voice, speed, speaker, yes=yes, volume=volume)
    console.print("Restart [cyan]serve[/] to pick it up. Check with: [cyan]pawn-assistant say \"ciao\"[/]")


models_app = typer.Typer(name="models", help="Manage local model downloads.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("pull")
def models_pull(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config"),
) -> None:
    """Pre-download Whisper, the intent GGUF and the Piper voice."""
    settings = _command_settings(config)
    with console.status(f"faster-whisper {settings.whisper_model}…"):
        from faster_whisper.utils import download_model

        console.print(f"whisper  → {download_model(settings.whisper_model)}")
    if settings.intent_enabled:
        from pawn_assistant.commands.llm import resolve_model_path

        with console.status(f"intent LLM {settings.intent_model}…"):
            console.print(f"intent   → {resolve_model_path(settings.intent_model)}")
    if settings.tts_enabled:
        from pawn_assistant.audio.tts import resolve_voice

        with console.status(f"Piper voice {settings.tts_voice}…"):
            console.print(f"voice    → {resolve_voice(settings.tts_voice)[0]}")


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


@wake_train_app.command("gen-neg")
def wake_gen_neg(
    project: str = typer.Option(..., "--project", "-p"),
    count: int = typer.Option(600, help="Number of synthetic speech clips"),
    category: str = typer.Option("speech_tts", help="Negatives sub-folder"),
    seed: int = typer.Option(0),
) -> None:
    """Synthesize speech negatives with Piper (it + en voices, near-miss words)."""
    from pawn_assistant.wake_train.project import WakeProject
    from pawn_assistant.wake_train.synth_neg import generate_speech_negatives

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    proj = WakeProject.load(Path(project))
    out = generate_speech_negatives(proj, count=count, category=category, seed=seed)
    console.print(f"[green]speech negatives →[/] {out}")


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
