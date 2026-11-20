"""Text helpers shared by the fast-path and command plugins.

Normalisation, filler stripping and number/duration extraction — no intent
knowledge here, so plugins can import it without cycles.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from pawn_assistant.commands.numbers import number_pattern, parse_number

# Leading / trailing words that carry no intent (wake-word residue, politeness).
_PREFIX_FILLERS = (
    "ok", "okay", "ehi", "ehm", "hey", "senti", "scusa", "allora", "dai",
    "pawn", "jarvis", "alexa", "per favore", "please", "mi",
)
_SUFFIX_FILLERS = ("per favore", "grazie", "please", "thanks", "thank you")


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # apostrophes, ?, !, ., …
    text = re.sub(r"_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_fillers(norm: str) -> str:
    changed = True
    while changed and norm:
        changed = False
        for word in _PREFIX_FILLERS:
            if norm == word:
                break
            if norm.startswith(word + " "):
                norm = norm[len(word) + 1 :]
                changed = True
        for word in _SUFFIX_FILLERS:
            if norm.endswith(" " + word):
                norm = norm[: -len(word) - 1]
                changed = True
    return norm.strip()


@dataclass
class FastMatch:
    intent: str
    slots: dict[str, Any]
    score: float


_NUM = number_pattern()
_UNIT_WORDS = (
    r"ore|ora|hours|hour|h|minuti|minuto|min|minutes|minute"
    r"|secondi|secondo|sec|seconds|second"
)
_DURATION_RE = re.compile(
    rf"\b(?P<num>{_NUM})\s+(?P<unit>{_UNIT_WORDS})\b"
    r"(?:(?P<half>\s+e\s+mezz[oa]|\s+and\s+a\s+half)"
    # "tre minuti e venti" → +20 s; not when a unit follows ("e venti secondi").
    rf"|\s+e\s+(?P<tail>{_NUM})\b(?!\s+(?:{_UNIT_WORDS})\b))?"
)
_HALF_HOUR_RE = re.compile(r"\bmezz\s?ora\b|\bhalf\s+an\s+hour\b")
_QUARTER_RE = re.compile(r"\b(un|tre) quart[oi] d\s?ora\b|\ba quarter of an hour\b")


def _unit_key(unit: str) -> str:
    if unit.startswith(("or", "h")):
        return "hours"
    if unit.startswith("min"):
        return "minutes"
    return "seconds"


def extract_duration(norm: str) -> dict[str, int]:
    """Parse '5 minuti', 'un ora e mezza', 'mezz ora', '90 seconds' into slots."""
    slots: dict[str, int] = {}
    for m in _DURATION_RE.finditer(norm):
        value = parse_number(m.group("num"))
        if value is None:
            continue
        key = _unit_key(m.group("unit"))
        slots[key] = slots.get(key, 0) + value
        smaller = {"hours": "minutes", "minutes": "seconds"}.get(key)
        if smaller is None:
            continue
        if m.group("half"):
            slots[smaller] = slots.get(smaller, 0) + 30
        elif m.group("tail"):
            tail = parse_number(m.group("tail"))
            if tail is not None:
                slots[smaller] = slots.get(smaller, 0) + tail
    if _HALF_HOUR_RE.search(norm):
        slots["minutes"] = slots.get("minutes", 0) + 30
    q = _QUARTER_RE.search(norm)
    if q:
        slots["minutes"] = slots.get("minutes", 0) + (45 if q.group(1) == "tre" else 15)
    return slots
