"""Runtime configuration for the wake loop (YAML + CLI)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

_CONFIG_NAMES = (
    "pawn-assistant.yaml",
    "pawn-assistant.yml",
    ".pawn-assistant.yaml",
)


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8787
    log_level: str = "INFO"
    device: Optional[int | str] = None
    sample_rate: int = 16_000
    channels: int = 1
    # openWakeWord expects 80 ms frames (1280 samples @ 16 kHz).
    frame_samples: int = 1280
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.4
    stir_threshold: float = 0.2
    vad_threshold: float = 0.45
    # Peak dBFS above this counts as speech even if Silero is unsure.
    energy_speech_db: float = -28.0
    # Trailing silence after real speech before ending the turn.
    silence_end_ms: int = 900
    max_utterance_ms: int = 10_000
    # Abort LISTEN if no speech after the wake beep.
    speech_wait_ms: int = 8_000
    # Consecutive speech needed before latching "heard speech" (filters wake tail).
    speech_start_ms: int = 240
    # Do not end on silence until this much confirmed speech has been heard.
    min_speech_ms: int = 400
    # After beep, ignore energy fallback this long (wake-word echo / room).
    post_wake_arm_ms: int = 400
    # Drop this much audio from the head of each utterance: the wake beep leaks
    # from the speaker into the mic and Whisper would otherwise hear a tone.
    head_trim_ms: int = 400
    beep_mute_ms: int = 600
    # Refractory after a full turn so openWakeWord can refill features.
    wake_debounce_ms: int = 1_200
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_idle_unload_s: float = 60.0
    # Preferred ASR language (it/en/…). None → faster-whisper autodetection.
    asr_language: Optional[str] = None
    # Biases decoding toward the command vocabulary. Empty = no prompt.
    asr_initial_prompt: str = (
        "Comandi vocali: che ore sono, che giorno è oggi, imposta un timer "
        "di cinque minuti, quanto manca al timer, annulla il timer."
    )
    # Let faster-whisper's own Silero pass cut leading/trailing non-speech.
    asr_vad_filter: bool = True
    # Input gain. ALSA hw devices: percent of the mixer ``gain_control`` range
    # (applied at startup when set). Other devices: software gain in dB.
    input_gain_percent: Optional[float] = None
    software_gain_db: float = 0.0
    gain_control: str = "Capture"
    meter_hz: float = 20.0
    beep_freq_hz: float = 880.0
    beep_duration_ms: int = 120
    beep_volume: float = 0.25
    # Directory for wake/utterance WAV dumps; None disables.
    debug_audio_dir: Optional[str] = None
    debug_preroll_s: float = 8.0
    # Dump preroll when wake score crosses this (near-miss debugging).
    debug_wake_score_dump: float = 0.25

    # Point 2 — command intents (fast-path + local llama.cpp model).
    intent_enabled: bool = True
    # Alias (qwen2.5-1.5b / qwen2.5-0.5b) or a local .gguf path.
    intent_model: str = "qwen2.5-1.5b"
    intent_threads: int = 4
    intent_idle_unload_s: float = 300.0
    # difflib ratio needed for a fast-path (no LLM) match.
    intent_fastpath_ratio: float = 0.85
    # Reply language for handlers (it/en).
    reply_language: str = "it"

    # Point 2 — spoken replies (Piper).
    tts_enabled: bool = True
    # Piper voice id (rhasspy/piper-voices) or a local .onnx path.
    tts_voice: str = "it_IT-paola-medium"
    tts_length_scale: float = 1.0
    tts_volume: float = 1.0
    # Speaker index for multi-speaker voices (None = voice default).
    tts_speaker_id: Optional[int] = None
    tts_idle_unload_s: float = 300.0

    # Point 2 — LLM rephrasing of replies (same model as the intent parser).
    phrase_enabled: bool = True
    # System-prompt persona; empty = built-in default for reply_language.
    phrase_persona: str = ""
    phrase_temperature: float = 0.6
    phrase_top_p: float = 0.95
    phrase_max_tokens: int = 60
    phrase_max_sentences: int = 2
    # No valid first sentence within this budget → speak the canonical reply.
    phrase_first_sentence_timeout_s: float = 2.5
    # "all" or a list of intent names (timer.alert = spoken timer alerts).
    phrase_intents: Any = "all"

    # Command plugins: raw ``commands:`` YAML section — ``disabled: [...]``,
    # ``state_file``, plus one sub-mapping per plugin (e.g. ``radio: {...}``).
    commands: dict = field(default_factory=dict)

    # Path the settings were loaded from (informational).
    config_path: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def commands_state_path(self) -> Optional[Path]:
        """Runtime toggles + plugin state (JSON). ``state_file: null`` disables."""
        raw = self.commands.get("state_file", _MISSING) if isinstance(self.commands, dict) else _MISSING
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "none", "null", "off")):
            return None
        if raw is _MISSING:
            from pawn_assistant.commands.base import default_state_dir

            return default_state_dir() / "commands.json"
        return Path(str(raw)).expanduser()

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_samples / self.sample_rate


def discover_config_path(explicit: Optional[str | Path] = None) -> Optional[Path]:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"config not found: {path}")
        return path.resolve()
    cwd = Path.cwd()
    for name in _CONFIG_NAMES:
        cand = cwd / name
        if cand.is_file():
            return cand.resolve()
    home = Path.home() / ".config" / "pawn-assistant" / "config.yaml"
    if home.is_file():
        return home.resolve()
    return None


def _dig(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _flatten_yaml(data: Mapping[str, Any]) -> dict[str, Any]:
    """Map nested YAML sections onto Settings field names."""
    out: dict[str, Any] = {}

    mapping = {
        ("server", "host"): "host",
        ("server", "port"): "port",
        ("server", "log_level"): "log_level",
        ("audio", "device"): "device",
        ("audio", "sample_rate"): "sample_rate",
        ("audio", "channels"): "channels",
        ("audio", "frame_samples"): "frame_samples",
        ("audio", "gain_percent"): "input_gain_percent",
        ("audio", "software_gain_db"): "software_gain_db",
        ("audio", "gain_control"): "gain_control",
        ("wake", "model"): "wake_model",
        ("wake", "threshold"): "wake_threshold",
        ("wake", "stir_threshold"): "stir_threshold",
        ("wake", "beep_mute_ms"): "beep_mute_ms",
        ("wake", "debounce_ms"): "wake_debounce_ms",
        ("vad", "threshold"): "vad_threshold",
        ("vad", "energy_speech_db"): "energy_speech_db",
        ("vad", "silence_end_ms"): "silence_end_ms",
        ("vad", "max_utterance_ms"): "max_utterance_ms",
        ("vad", "speech_wait_ms"): "speech_wait_ms",
        ("vad", "speech_start_ms"): "speech_start_ms",
        ("vad", "min_speech_ms"): "min_speech_ms",
        ("vad", "post_wake_arm_ms"): "post_wake_arm_ms",
        ("vad", "head_trim_ms"): "head_trim_ms",
        ("asr", "model"): "whisper_model",
        ("asr", "compute_type"): "whisper_compute_type",
        ("asr", "language"): "asr_language",
        ("asr", "idle_unload_s"): "whisper_idle_unload_s",
        ("asr", "initial_prompt"): "asr_initial_prompt",
        ("asr", "vad_filter"): "asr_vad_filter",
        ("beep", "freq_hz"): "beep_freq_hz",
        ("beep", "duration_ms"): "beep_duration_ms",
        ("beep", "volume"): "beep_volume",
        ("hud", "meter_hz"): "meter_hz",
        ("debug", "audio_dir"): "debug_audio_dir",
        ("debug", "preroll_s"): "debug_preroll_s",
        ("debug", "wake_score_dump"): "debug_wake_score_dump",
        ("intent", "enabled"): "intent_enabled",
        ("intent", "model"): "intent_model",
        ("intent", "threads"): "intent_threads",
        ("intent", "idle_unload_s"): "intent_idle_unload_s",
        ("intent", "fastpath_ratio"): "intent_fastpath_ratio",
        ("intent", "reply_language"): "reply_language",
        ("tts", "enabled"): "tts_enabled",
        ("tts", "voice"): "tts_voice",
        ("tts", "length_scale"): "tts_length_scale",
        ("tts", "volume"): "tts_volume",
        ("tts", "speaker_id"): "tts_speaker_id",
        ("tts", "idle_unload_s"): "tts_idle_unload_s",
        ("phrase", "enabled"): "phrase_enabled",
        ("phrase", "persona"): "phrase_persona",
        ("phrase", "temperature"): "phrase_temperature",
        ("phrase", "top_p"): "phrase_top_p",
        ("phrase", "max_tokens"): "phrase_max_tokens",
        ("phrase", "max_sentences"): "phrase_max_sentences",
        ("phrase", "first_sentence_timeout_s"): "phrase_first_sentence_timeout_s",
        ("phrase", "intents"): "phrase_intents",
    }
    for keys, field_name in mapping.items():
        if _dig(data, *keys, default=_MISSING) is not _MISSING:
            out[field_name] = _dig(data, *keys)

    if isinstance(data.get("commands"), Mapping):
        out["commands"] = dict(data["commands"])

    # Flat keys also accepted for convenience.
    valid = {f.name for f in fields(Settings)} - {"extra", "config_path"}
    for key, value in data.items():
        if key in valid and not isinstance(value, Mapping):
            out[key] = value
    return out


_MISSING = object()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"config root must be a mapping: {path}")
    return _flatten_yaml(raw)


def load_settings(
    config: Optional[str | Path] = None,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Settings:
    """Load Settings from discovered YAML, then apply non-None *overrides*."""
    path = discover_config_path(config)
    values: dict[str, Any] = {}
    if path is not None:
        values.update(load_yaml(path))
        values["config_path"] = str(path)
        logger.info("loaded config %s", path)
    else:
        logger.info("no config file found — using defaults")

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                values[key] = value

    # Normalize empty language string → None (autodetect).
    lang = values.get("asr_language", _MISSING)
    if lang is not _MISSING and (lang is None or str(lang).strip() in ("", "null", "auto", "none")):
        values["asr_language"] = None
    elif isinstance(lang, str):
        values["asr_language"] = lang.strip().lower()

    known = {f.name for f in fields(Settings)}
    filtered = {k: v for k, v in values.items() if k in known}
    return Settings(**filtered)


def settings_summary(settings: Settings) -> dict[str, Any]:
    data = asdict(settings)
    data.pop("extra", None)
    return data
