"""Command plugin API.

A command is one class: its intents are decorated handler methods, and every
other hook is optional. The registry (``registry.py``) discovers commands,
builds the fast-path table, LLM prompt/grammar and phrasing few-shots from the
enabled ones, and routes UI actions and loop state changes back to them.

```python
from pawn_assistant.commands.base import SLOT_STR, Command, intent


class Coin(Command):
    name = "coin"
    title = "Coin flip"
    description = "Heads or tails."

    @intent("coin.flip", "the user wants to flip a coin",
            examples=("lancia una moneta", "flip a coin"))
    def flip(self, slots, ctx) -> str:
        return ctx.t("Testa.", "Heads.")
```

Handlers return the canonical reply text (the ground truth the phraser may
reword). They run on the loop's worker thread.
"""

from __future__ import annotations

import inspect
import logging
import os
import queue
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from pawn_assistant.commands.text import FastMatch
from pawn_assistant.commands.ui import UiBridge

if TYPE_CHECKING:  # pragma: no cover
    from pawn_assistant.commands.registry import CommandRegistry

logger = logging.getLogger(__name__)

# Slot types understood by the grammar builder (llm.py).
SLOT_INT = "integer"
SLOT_STR = "string"

UNKNOWN = "unknown"

# Phrasing few-shot row: (user said, fact, mood, now, reply).
Shot = tuple[str, str, str, str, str]

Handler = Callable[[dict[str, Any], "CommandContext"], str]


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
    # False = speak the canonical reply verbatim (the phrasing guard only
    # protects numbers/dates, not e.g. station names).
    phrase: bool = True


def intent(
    name: str,
    description: str,
    *,
    examples: tuple[str, ...] = (),
    slots: Optional[dict[str, str]] = None,
    fewshot: tuple[tuple[str, dict[str, Any]], ...] = (),
    phrase: bool = True,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Mark a ``Command`` method as the handler of intent ``name``."""

    spec = Intent(
        name=name,
        description=description,
        examples=tuple(examples),
        slots=dict(slots or {}),
        fewshot=tuple(fewshot),
        phrase=phrase,
    )

    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        fn.__pawn_intent__ = spec  # type: ignore[attr-defined]
        return fn

    return deco


# --------------------------------------------------------------------- context


@dataclass
class Announcement:
    """A spoken alert queued by a plugin; played only when the loop is idle."""

    text: str
    intent: str = "alert"  # phrasing few-shot key, e.g. "timer.alert"
    beeps: int = 3
    command: str = ""


def default_state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "pawn-assistant"


@dataclass
class CommandContext:
    lang: str = "it"
    now: Callable[[], datetime] = datetime.now
    ui: UiBridge = field(default_factory=UiBridge)
    # Spoken alerts; the loop drains this between turns.
    alerts: "queue.SimpleQueue[Announcement]" = field(default_factory=queue.SimpleQueue)
    # Set by the registry / router.
    registry: Optional["CommandRegistry"] = None
    utterance: str = ""  # raw text of the command being handled
    norm: str = ""  # normalised text (fillers stripped)
    last_active: str = ""  # command that most recently started something

    def t(self, it: str, en: str) -> str:
        return it if self.lang == "it" else en

    def announce(self, text: str, *, intent: str = "alert", beeps: int = 3, command: str = "") -> None:
        self.alerts.put(Announcement(text, intent, beeps, command))


# --------------------------------------------------------------------- command


class Command:
    """Base class for command plugins. Override what you need."""

    name: str = ""  # unique id, also the config key under ``commands:``
    title: str = ""
    description: str = ""
    core: bool = False  # core commands cannot be disabled
    default_enabled: bool = True
    order: int = 100  # registry order (prompt / grammar / UI listing)
    # Words that make a generic "stop"/"annulla" target this command.
    cancel_words: tuple[str, ...] = ()
    # lang → intent → few-shot rows for reply phrasing.
    phrasing: dict[str, dict[str, tuple[Shot, ...]]] = {}
    # Widget files next to the plugin module, served at /plugins/<name>/<file>.
    assets: dict[str, tuple[str, ...]] = {}

    def __init__(self) -> None:
        self.ctx: Optional[CommandContext] = None
        self.config: dict[str, Any] = {}

    # -- introspection -----------------------------------------------------

    @classmethod
    def intents(cls) -> tuple[Intent, ...]:
        """Decorated handlers in definition order (base classes first)."""
        seen: dict[str, Intent] = {}
        for klass in reversed(cls.__mro__):
            for attr in vars(klass).values():
                spec = getattr(attr, "__pawn_intent__", None)
                if isinstance(spec, Intent):
                    seen[spec.name] = spec
        return tuple(seen.values())

    def handlers(self) -> dict[str, Handler]:
        out: dict[str, Handler] = {}
        for klass in reversed(type(self).__mro__):
            for attr_name, attr in vars(klass).items():
                spec = getattr(attr, "__pawn_intent__", None)
                if isinstance(spec, Intent):
                    out[spec.name] = getattr(self, attr_name)
        return out

    @classmethod
    def base_dir(cls) -> Path:
        return Path(inspect.getfile(cls)).resolve().parent

    # -- state helpers -----------------------------------------------------

    def state_get(self, key: str, default: Any = None) -> Any:
        reg = self.ctx.registry if self.ctx else None
        return reg.state_get(self.name, key, default) if reg else default

    def state_set(self, key: str, value: Any) -> None:
        reg = self.ctx.registry if self.ctx else None
        if reg:
            reg.state_set(self.name, key, value)

    def mark_active(self) -> None:
        """Record this command as the latest to start something (for 'stop')."""
        if self.ctx is not None:
            self.ctx.last_active = self.name

    def widget_id(self, suffix: str = "main") -> str:
        return f"{self.name}:{suffix}"

    # -- hooks (all optional) ---------------------------------------------

    def configure(self, cfg: dict[str, Any]) -> None:
        """Receives ``commands.<name>`` from the YAML config."""
        self.config = dict(cfg or {})

    def setup(self, ctx: CommandContext) -> None:
        """Called once with the shared context."""

    def fast_match(self, norm: str) -> Optional[FastMatch]:
        """Zero-LLM match on normalised text (slot-bearing intents)."""
        return None

    def is_active(self) -> bool:
        """True while something is running that a generic 'stop' could end."""
        return False

    def on_cancel(self, ctx: CommandContext) -> Optional[str]:
        """Stop what is running; return the reply, or None if nothing to say."""
        return None

    def on_state(self, state: str) -> None:
        """Loop state change (IDLE, AWAKE, LISTEN, THINK, SPEAK, …). Must not block."""

    def on_ui_action(self, widget_id: str, action: str, data: dict[str, Any]) -> None:
        """A button/slider in one of this command's widgets was used."""

    def on_disable(self) -> None:
        """The user switched the command off: stop anything running."""
        if self.is_active() and self.ctx is not None:
            self.on_cancel(self.ctx)

    def shutdown(self) -> None:
        """Process exit."""
