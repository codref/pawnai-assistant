"""Transcript → intent → reply. Fast-path first, local LLM on a miss."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from pawn_assistant.commands.base import UNKNOWN, CommandContext
from pawn_assistant.commands.fastpath import FastPath
from pawn_assistant.commands.registry import CommandRegistry
from pawn_assistant.commands.text import normalize, strip_fillers

logger = logging.getLogger(__name__)


class IntentParser(Protocol):
    resident: bool

    def parse(self, text: str) -> Any: ...  # returns object with .intent / .slots


@dataclass
class CommandResult:
    text: str
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    source: str = "fastpath"  # fastpath | llm | none
    parse_ms: float = 0.0
    score: Optional[float] = None

    def as_event(self) -> dict[str, Any]:
        return {
            "type": "event",
            "name": "intent",
            "intent": self.intent,
            "slots": self.slots,
            "source": self.source,
            "parse_ms": round(self.parse_ms, 1),
        }


class CommandRouter:
    def __init__(
        self,
        ctx: Optional[CommandContext] = None,
        llm: Optional[IntentParser] = None,
        fastpath_ratio: float = 0.85,
        registry: Optional[CommandRegistry] = None,
    ) -> None:
        self.ctx = ctx or CommandContext()
        if registry is None:
            registry = self.ctx.registry or CommandRegistry(entry_points=False)
        self.registry = registry
        if self.ctx.registry is not registry:
            registry.setup(self.ctx)
        self.llm = llm
        self.fastpath = FastPath(registry, min_ratio=fastpath_ratio)

    def classify(self, text: str) -> CommandResult:
        t0 = time.perf_counter()
        fast = self.fastpath.match(text)
        if fast is not None:
            return CommandResult(
                text=text,
                intent=fast.intent,
                slots=fast.slots,
                source="fastpath",
                parse_ms=(time.perf_counter() - t0) * 1000.0,
                score=fast.score,
            )
        if self.llm is None:
            return CommandResult(
                text=text, intent=UNKNOWN, source="none",
                parse_ms=(time.perf_counter() - t0) * 1000.0,
            )
        try:
            parsed = self.llm.parse(text)
            intent, slots = parsed.intent, parsed.slots
        except Exception:  # noqa: BLE001
            logger.exception("intent LLM failed")
            intent, slots = UNKNOWN, {}
        return CommandResult(
            text=text,
            intent=intent,
            slots=slots,
            source="llm",
            parse_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def handle(self, text: str) -> CommandResult:
        result = self.classify(text)
        handler = self.registry.handler_for(result.intent)
        if handler is None:  # disabled meanwhile, or a stale LLM answer
            result.intent, result.slots = UNKNOWN, {}
            handler = self.registry.handler_for(UNKNOWN)
        assert handler is not None
        self.ctx.utterance = text
        self.ctx.norm = strip_fillers(normalize(text))
        try:
            result.reply = handler(result.slots, self.ctx)
        except Exception:  # noqa: BLE001
            logger.exception("handler %s failed", result.intent)
            fallback = self.registry.handler_for(UNKNOWN)
            result.reply = fallback({}, self.ctx) if fallback else ""
        logger.info(
            "command %r → %s %s via %s (%.0f ms) → %r",
            text, result.intent, result.slots, result.source, result.parse_ms, result.reply,
        )
        return result
