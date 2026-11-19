"""Tiny number-word parser (Italian + English, 0–99) for timer durations."""

from __future__ import annotations

from typing import Optional

_IT_UNITS = ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"]
_IT_TEENS = [
    "dieci", "undici", "dodici", "tredici", "quattordici",
    "quindici", "sedici", "diciassette", "diciotto", "diciannove",
]
_IT_TENS = {
    20: "venti", 30: "trenta", 40: "quaranta", 50: "cinquanta",
    60: "sessanta", 70: "settanta", 80: "ottanta", 90: "novanta",
}
_EN_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_EN_TEENS = [
    "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_EN_TENS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}


def _build() -> dict[str, int]:
    words: dict[str, int] = {}
    for i, w in enumerate(_IT_UNITS):
        words[w] = i
    for i, w in enumerate(_IT_TEENS):
        words[w] = 10 + i
    for base, w in _IT_TENS.items():
        words[w] = base
        for u in range(1, 10):
            unit = _IT_UNITS[u]
            # Italian elides the tens vowel before "uno" / "otto": ventuno, ventotto.
            stem = w[:-1] if unit[0] in "uo" else w
            words[stem + unit] = base + u
    for i, w in enumerate(_EN_UNITS):
        words[w] = i
    for i, w in enumerate(_EN_TEENS):
        words[w] = 10 + i
    for base, w in _EN_TENS.items():
        words[w] = base
        for u in range(1, 10):
            words[f"{w} {_EN_UNITS[u]}"] = base + u
            words[f"{w}-{_EN_UNITS[u]}"] = base + u
    # Articles used as "one".
    words.update({"un": 1, "una": 1, "un a": 1, "a": 1, "an": 1})
    return words


NUMBER_WORDS: dict[str, int] = _build()


def parse_number(token: str) -> Optional[int]:
    token = token.strip()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def number_pattern() -> str:
    """Regex alternation matching digits or any known number word (longest first)."""
    import re

    words = sorted(NUMBER_WORDS, key=len, reverse=True)
    return r"\d+|" + "|".join(re.escape(w) for w in words)
