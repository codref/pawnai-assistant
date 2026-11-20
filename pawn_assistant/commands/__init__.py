"""Point 2 — local command intents (fast-path + small llama.cpp model) as plugins."""

from pawn_assistant.commands.base import (
    SLOT_INT,
    SLOT_STR,
    Command,
    CommandContext,
    Intent,
    intent,
)
from pawn_assistant.commands.registry import CommandRegistry
from pawn_assistant.commands.router import CommandResult, CommandRouter

__all__ = [
    "SLOT_INT",
    "SLOT_STR",
    "Command",
    "CommandContext",
    "CommandRegistry",
    "CommandResult",
    "CommandRouter",
    "Intent",
    "intent",
]
