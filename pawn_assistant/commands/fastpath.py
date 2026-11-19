"""Zero-LLM matcher: normalised phrase lookup + regex slot extraction.

Handles the everyday commands in a few ms; anything it is unsure about
returns ``None`` and goes to the local LLM.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

from pawn_assistant.commands.intents import INTENTS
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
_TIMER_HINT_RE = re.compile(
    r"\b(timer|conto alla rovescia|countdown|avvisami|avvisa mi|sveglia tra|remind me in)\b"
)


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


class FastPath:
    def __init__(self, min_ratio: float = 0.85) -> None:
        self.min_ratio = min_ratio
        self._phrases: dict[str, str] = {}
        for intent in INTENTS:
            for example in intent.examples:
                self._phrases[normalize(example)] = intent.name

    def match(self, text: str) -> Optional[FastMatch]:
        norm = strip_fillers(normalize(text))
        if not norm:
            return None

        # Slot intents first (regex).
        if _TIMER_HINT_RE.search(norm):
            duration = extract_duration(norm)
            if duration:
                return FastMatch("timer.set", duration, 1.0)

        exact = self._phrases.get(norm)
        if exact is not None:
            return FastMatch(exact, {}, 1.0)

        best_name, best_score = None, 0.0
        for phrase, name in self._phrases.items():
            score = difflib.SequenceMatcher(None, norm, phrase).ratio()
            if score > best_score:
                best_name, best_score = name, score
        if best_name is not None and best_score >= self.min_ratio:
            return FastMatch(best_name, {}, round(best_score, 3))
        return None
