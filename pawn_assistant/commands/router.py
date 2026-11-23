"""Transcript → intent → reply. Fast-path first, local LLM on a miss."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from pawn_assistant.commands.base import UNKNOWN, Command, CommandContext
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
    source: str = "fastpath"  # fastpath | llm | none | floor
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

    def floor_owner(self) -> Optional[Command]:
        """The enabled command holding the floor, if any (stale floors are cleared)."""
        name = self.ctx.floor
        if not name:
            return None
        cmd = self.registry.get(name)
        if cmd is None or not self.registry.is_enabled(name):
            self.ctx.floor = None
            return None
        return cmd

    def _classify_floor(self, text: str, owner: Command) -> Optional[CommandResult]:
        """Fast-path only: the owner's own intents, or an exact generic stop."""
        t0 = time.perf_counter()
        norm = strip_fillers(normalize(text))
        fast = None
        try:
            fast = owner.fast_match(norm) if norm else None
        except Exception:  # noqa: BLE001
            logger.exception("floor owner %s fast_match failed", owner.name)
        if fast is None:
            fast = self.fastpath.match(text)
        if fast is None:
            return None
        own = {i.name for i in owner.intents()}
        exact_cancel = fast.intent == "cancel" and (fast.score or 0.0) >= 1.0
        if fast.intent not in own and not exact_cancel:
            return None
        if exact_cancel:
            self.ctx.last_active = owner.name  # "stop" ends the conversation
        return CommandResult(
            text=text,
            intent=fast.intent,
            slots=fast.slots,
            source="fastpath",
            parse_ms=(time.perf_counter() - t0) * 1000.0,
            score=fast.score,
        )

    def handle(self, text: str) -> CommandResult:
        owner = self.floor_owner()
        if owner is not None:
            routed = self._classify_floor(text, owner)
            if routed is None:
                return self._handle_floor(text, owner)
            result = routed
        else:
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

    def _handle_floor(self, text: str, owner: Command) -> CommandResult:
        self.ctx.utterance = text
        self.ctx.norm = strip_fillers(normalize(text))
        result = CommandResult(text=text, intent=f"{owner.name}.converse", source="floor")
        t0 = time.perf_counter()
        try:
            result.reply = owner.converse(text, self.ctx)
        except Exception:  # noqa: BLE001
            logger.exception("converse %s failed", owner.name)
            result.reply = self.ctx.t("Qualcosa è andato storto.", "Something went wrong.")
        result.parse_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("floor %s %r → %r (%.0f ms)", owner.name, text, result.reply, result.parse_ms)
        return result
