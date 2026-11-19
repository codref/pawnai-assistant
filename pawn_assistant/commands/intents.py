"""Intent registry — the single source for fast-path phrases, LLM prompt and grammar.

Adding a command = one ``Intent`` entry here + one handler in ``handlers.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Slot types understood by the grammar builder (llm.py).
SLOT_INT = "integer"
SLOT_STR = "string"


@dataclass(frozen=True)
class Intent:
    name: str
    description: str
    # Canonical phrasings (it + en). Used verbatim by the fast-path matcher.
    examples: tuple[str, ...] = ()
    # slot name → SLOT_INT | SLOT_STR
    slots: dict[str, str] = field(default_factory=dict)
    # (utterance, slots) pairs shown to the LLM as few-shot examples.
    fewshot: tuple[tuple[str, dict[str, Any]], ...] = ()


UNKNOWN = "unknown"

INTENTS: tuple[Intent, ...] = (
    Intent(
        name="time.now",
        description="the user asks what time it is",
        examples=(
            "che ora e",
            "che ore sono",
            "mi dici che ore sono",
            "sai che ore sono",
            "dimmi l ora",
            "ora attuale",
            "what time is it",
            "what s the time",
        ),
        fewshot=(("Scusa, mi sai dire l'orario?", {}),),
    ),
    Intent(
        name="date.today",
        description="the user asks today's date or day of the week",
        examples=(
            "che giorno e",
            "che giorno e oggi",
            "quanti ne abbiamo oggi",
            "che data e oggi",
            "che data e",
            "what day is it",
            "what s the date",
            "what s the date today",
        ),
        fewshot=(("oggi che giorno della settimana è?", {}),),
    ),
    Intent(
        name="timer.set",
        description="the user wants a countdown timer for a duration",
        slots={"hours": SLOT_INT, "minutes": SLOT_INT, "seconds": SLOT_INT},
        fewshot=(
            ("metti un timer di cinque minuti", {"minutes": 5}),
            ("avvisami tra un'ora e mezza", {"hours": 1, "minutes": 30}),
            ("tra un quarto d'ora avvisami", {"minutes": 15}),
            ("set a timer for 90 seconds", {"seconds": 90}),
        ),
    ),
    Intent(
        name="cancel",
        description="the user wants to stop, cancel or dismiss (timers, current action)",
        examples=(
            "annulla",
            "annulla timer",
            "annulla il timer",
            "cancella il timer",
            "ferma il timer",
            "ferma il conto alla rovescia",
            "annulla il conto alla rovescia",
            "stop",
            "basta",
            "lascia stare",
            "niente",
            "cancel",
            "cancel the timer",
            "never mind",
        ),
        fewshot=(
            ("togli il conto alla rovescia", {}),
            ("spegni quel timer", {}),
        ),
    ),
    Intent(
        name="greeting",
        description="the user greets or asks how the assistant is",
        examples=(
            "ciao",
            "buongiorno",
            "buonasera",
            "come stai",
            "ciao come stai",
            "hello",
            "hi",
            "how are you",
        ),
        fewshot=(("ehi tutto bene?", {}),),
    ),
    Intent(
        name=UNKNOWN,
        description="anything else, off-topic, or unclear",
        fewshot=(
            ("che tempo fa domani a Milano", {}),
            ("raccontami una barzelletta", {}),
        ),
    ),
)

BY_NAME: dict[str, Intent] = {i.name: i for i in INTENTS}


def all_slot_names() -> dict[str, str]:
    """Union of every slot across intents (name → type), in registry order."""
    out: dict[str, str] = {}
    for intent in INTENTS:
        for name, typ in intent.slots.items():
            out.setdefault(name, typ)
    return out
