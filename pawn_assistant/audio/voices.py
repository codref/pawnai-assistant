"""Piper voice catalog + helpers for the ``voice`` CLI (list / test / pick / use)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO = "rhasspy/piper-voices"

QUALITY_ORDER = {"x_low": 0, "low": 1, "medium": 2, "high": 3}

# Short audition sentences per language (fallback: English).
SAMPLE_TEXT = {
    "it": "Ciao, sono Pawn. Sono le nove e quindici, e ho impostato un timer di cinque minuti.",
    "en": "Hello, I am Pawn. It is nine fifteen, and I have set a timer for five minutes.",
    "de": "Hallo, ich bin Pawn. Es ist Viertel nach neun, der Timer läuft fünf Minuten.",
    "fr": "Bonjour, je suis Pawn. Il est neuf heures quinze, minuteur réglé sur cinq minutes.",
    "es": "Hola, soy Pawn. Son las nueve y cuarto, y he puesto un temporizador de cinco minutos.",
}
LONG_SAMPLE_TEXT = {
    "it": (
        "Buongiorno. Oggi è giovedì diciannove novembre. Hai tre timer attivi: il primo "
        "scade tra dodici minuti, gli altri due tra un'ora. Vuoi che li annulli?"
    ),
    "en": (
        "Good morning. Today is Thursday, the nineteenth of November. You have three active "
        "timers: the first expires in twelve minutes, the other two in an hour."
    ),
}


def sample_text(lang: str, *, long: bool = False) -> str:
    lang = (lang or "en").split("_")[0].lower()
    table = LONG_SAMPLE_TEXT if long else SAMPLE_TEXT
    return table.get(lang) or SAMPLE_TEXT.get(lang) or SAMPLE_TEXT["en"]


@dataclass
class VoiceInfo:
    id: str
    locale: str  # e.g. it_IT
    name: str
    quality: str
    speakers: int = 1
    sample_rate: int = 22_050
    size_mb: float = 0.0
    installed: bool = False

    @property
    def lang(self) -> str:
        return self.locale.split("_")[0]


def _entry_to_info(key: str, entry: dict[str, Any]) -> VoiceInfo:
    lang = entry.get("language") or {}
    files = entry.get("files") or {}
    size = sum(
        meta.get("size_bytes", 0) for path, meta in files.items() if path.endswith(".onnx")
    )
    return VoiceInfo(
        id=key,
        locale=lang.get("code") or key.split("-")[0],
        name=entry.get("name") or key.split("-")[1],
        quality=entry.get("quality") or key.rsplit("-", 1)[-1],
        speakers=int(entry.get("num_speakers") or 1),
        sample_rate=int((entry.get("audio") or {}).get("sample_rate") or 22_050),
        size_mb=size / 1_000_000.0,
    )


def parse_catalog(data: dict[str, Any]) -> list[VoiceInfo]:
    voices = [_entry_to_info(k, v) for k, v in data.items() if isinstance(v, dict)]
    voices.sort(key=lambda v: (v.locale, v.name, QUALITY_ORDER.get(v.quality, 9)))
    return voices


def is_installed(voice_id: str) -> bool:
    """True when the voice's .onnx is already in the local HF cache."""
    from pawn_assistant.audio.tts import _VOICE_RE

    m = _VOICE_RE.match(voice_id)
    if not m:
        return False
    from huggingface_hub import hf_hub_download

    lang, region, name, q = m["lang"], m["region"], m["name"], m["q"]
    fname = f"{lang}/{lang}_{region}/{name}/{q}/{voice_id}.onnx"
    try:
        hf_hub_download(repo_id=REPO, filename=fname, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _cached_voice_ids() -> list[str]:
    """Offline fallback: voice ids found in the local HF cache."""
    from huggingface_hub import scan_cache_dir

    ids: list[str] = []
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_id != REPO:
                continue
            for rev in repo.revisions:
                for f in rev.files:
                    if f.file_name.endswith(".onnx"):
                        ids.append(f.file_name.removesuffix(".onnx"))
    except Exception:  # noqa: BLE001
        pass
    return sorted(set(ids))


def load_catalog() -> tuple[list[VoiceInfo], bool]:
    """Return (voices, online). Falls back to cached voices when offline."""
    from pawn_assistant.commands.llm import hf_cached_download

    try:
        path = hf_cached_download(REPO, "voices.json")
        voices = parse_catalog(json.loads(path.read_text(encoding="utf-8")))
        online = True
    except Exception:  # noqa: BLE001
        voices = []
        for vid in _cached_voice_ids():
            parts = vid.split("-")
            if len(parts) == 3:
                voices.append(VoiceInfo(vid, parts[0], parts[1], parts[2]))
        online = False
    for v in voices:
        v.installed = is_installed(v.id)
    return voices, online


def filter_voices(
    voices: list[VoiceInfo],
    *,
    lang: Optional[str] = None,
    quality: Optional[str] = None,
    installed: bool = False,
) -> list[VoiceInfo]:
    out = voices
    if lang:
        prefix = lang.lower()
        out = [v for v in out if v.locale.lower().startswith(prefix)]
    if quality:
        out = [v for v in out if v.quality == quality]
    if installed:
        out = [v for v in out if v.installed]
    return out


# ------------------------------------------------------------------ YAML in-place edit


def _fmt_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, 4))
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if re.fullmatch(r"[\w./~-][\w ./~-]*", text) and not re.fullmatch(r"[\d.eE+-]+", text):
        return text
    return json.dumps(text)


def update_yaml_section(path: Path, section: str, values: dict[str, Any]) -> list[str]:
    """Set ``section.key`` scalars in a YAML file, keeping comments and layout.

    Edits an existing ``key:`` line (preserving any trailing comment), otherwise inserts the
    key at the end of the section; creates the section when missing. Returns changed keys.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    sec_re = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    start = next((i for i, ln in enumerate(lines) if sec_re.match(ln)), None)
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{section}:")
        start = len(lines) - 1
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t#"):
        end += 1
    # Trim trailing blank lines from the section body.
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1

    changed: list[str] = []
    for key, value in values.items():
        rendered = _fmt_scalar(value)
        key_re = re.compile(rf"^(\s+){re.escape(key)}:\s*([^#\n]*?)(\s+#.*)?$")
        for i in range(start + 1, end):
            m = key_re.match(lines[i])
            if m:
                lines[i] = f"{m[1]}{key}: {rendered}{m[3] or ''}"
                break
        else:
            lines.insert(end, f"  {key}: {rendered}")
            end += 1
        changed.append(key)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed
