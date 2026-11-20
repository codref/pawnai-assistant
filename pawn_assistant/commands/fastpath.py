"""Zero-LLM matcher: plugin ``fast_match`` hooks + normalised phrase lookup.

Handles the everyday commands in a few ms; anything it is unsure about
returns ``None`` and goes to the local LLM. The phrase table is rebuilt from
the registry's enabled intents whenever a command is toggled.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Optional

from pawn_assistant.commands.text import (  # re-exported for callers/tests
    FastMatch,
    extract_duration,
    normalize,
    strip_fillers,
)

if TYPE_CHECKING:  # pragma: no cover
    from pawn_assistant.commands.registry import CommandRegistry

__all__ = ["FastMatch", "FastPath", "extract_duration", "normalize", "strip_fillers"]


class FastPath:
    def __init__(self, registry: Optional["CommandRegistry"] = None, min_ratio: float = 0.85) -> None:
        if registry is None:
            from pawn_assistant.commands.registry import default_registry

            registry = default_registry()
        self.registry = registry
        self.min_ratio = min_ratio
        self._version = -1
        self._phrases: dict[str, str] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        phrases: dict[str, str] = {}
        for intent in self.registry.intents():
            for example in intent.examples:
                phrases.setdefault(normalize(example), intent.name)
        self._phrases = phrases
        self._version = self.registry.version

    def match(self, text: str) -> Optional[FastMatch]:
        if self._version != self.registry.version:
            self._rebuild()
        norm = strip_fillers(normalize(text))
        if not norm:
            return None

        # Slot intents first (plugin regexes).
        hit = self.registry.fast_match(norm)
        if hit is not None:
            return hit

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
