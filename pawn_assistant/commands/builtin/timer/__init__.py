"""Countdown timers, each with a live countdown window on the HUD."""

from __future__ import annotations

import itertools
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pawn_assistant.commands.base import SLOT_INT, Command, CommandContext, intent
from pawn_assistant.commands.text import FastMatch, extract_duration

logger = logging.getLogger(__name__)

MAX_TIMER_S = 24 * 3600
ALERT_INTENT = "timer.alert"
# How long an expired countdown stays on screen.
EXPIRED_LINGER_S = 8.0

_TIMER_HINT_RE = re.compile(
    r"\b(timer|conto alla rovescia|countdown|avvisami|avvisa mi|sveglia tra|remind me in)\b"
)


# --------------------------------------------------------------------- manager


@dataclass
class _Timer:
    id: int
    seconds: int
    label: str
    due: float
    handle: threading.Timer


class TimerManager:
    """Countdown timers. ``on_fire(id, label)`` runs on the timer thread."""

    def __init__(self, on_fire: Optional[Callable[[int, str], None]] = None) -> None:
        self.on_fire = on_fire or (lambda _id, _label: None)
        self._timers: dict[int, _Timer] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def start(self, seconds: int, label: str) -> int:
        tid = next(self._ids)

        def _fire() -> None:
            with self._lock:
                timer = self._timers.pop(tid, None)
            if timer is not None:
                logger.info("timer %d fired (%s)", tid, label)
                self.on_fire(tid, label)

        handle = threading.Timer(seconds, _fire)
        handle.daemon = True
        with self._lock:
            self._timers[tid] = _Timer(tid, seconds, label, time.monotonic() + seconds, handle)
        handle.start()
        return tid

    def cancel(self, tid: int) -> bool:
        with self._lock:
            timer = self._timers.pop(tid, None)
        if timer is None:
            return False
        timer.handle.cancel()
        return True

    def cancel_all(self) -> list[int]:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for t in timers:
            t.handle.cancel()
        return [t.id for t in timers]

    @property
    def active(self) -> int:
        with self._lock:
            return len(self._timers)


# --------------------------------------------------------------------- formatting


def format_duration(seconds: int, lang: str = "it") -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    parts: list[str] = []
    if lang == "it":
        if h:
            parts.append("un'ora" if h == 1 else f"{h} ore")
        if m:
            parts.append("un minuto" if m == 1 else f"{m} minuti")
        if s:
            parts.append("un secondo" if s == 1 else f"{s} secondi")
        joiner = " e "
    else:
        if h:
            parts.append("1 hour" if h == 1 else f"{h} hours")
        if m:
            parts.append("1 minute" if m == 1 else f"{m} minutes")
        if s:
            parts.append("1 second" if s == 1 else f"{s} seconds")
        joiner = " and "
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + joiner + parts[-1]


def timer_alert_text(label: str, lang: str = "it") -> str:
    return f"Il timer di {label} è scaduto." if lang == "it" else f"Your {label} timer is done."


def _slot_int(slots: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(slots.get(key) or 0))
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------- command


class TimerCommand(Command):
    name = "timer"
    title = "Timer"
    description = "Countdown timers with a spoken alert and an on-screen countdown."
    order = 20
    cancel_words = ("timer", "conto alla rovescia", "countdown", "sveglia")
    assets = {"js": ("widget.js",), "css": ("widget.css",)}

    phrasing = {
        "it": {
            "timer.set": (
                ("metti un timer di dieci minuti", "Timer di 10 minuti avviato.", "conciso", "sera", "10 minuti, partiti."),
                ("avvisami tra un'ora", "Timer di un'ora avviato.", "cordiale", "mattina", "D'accordo, ti avviso tra un'ora."),
                ("timer di 30 secondi", "Timer di 30 secondi avviato.", "ironico", "pomeriggio", "30 secondi. Non sbattere le palpebre."),
            ),
            ALERT_INTENT: (
                ("-", "Il timer di 5 minuti è scaduto.", "tecnico", "pomeriggio", "Conto alla rovescia di 5 minuti completato."),
                ("-", "Il timer di un'ora è scaduto.", "complice", "sera", "Ehi, l'ora è passata. Il timer è scaduto."),
            ),
        },
        "en": {
            "timer.set": (
                ("set a timer for ten minutes", "Timer for 10 minutes started.", "terse", "evening", "10 minutes, running."),
                ("timer for 30 seconds", "Timer for 30 seconds started.", "ironic", "afternoon", "30 seconds. Don't blink."),
            ),
            ALERT_INTENT: (
                ("-", "Your 5 minutes timer is done.", "technical", "afternoon", "5 minute countdown complete."),
            ),
        },
    }

    def __init__(self) -> None:
        super().__init__()
        self.timers = TimerManager(on_fire=self._fired)

    # -- matching ----------------------------------------------------------

    def fast_match(self, norm: str) -> Optional[FastMatch]:
        if _TIMER_HINT_RE.search(norm):
            duration = extract_duration(norm)
            if duration:
                return FastMatch("timer.set", duration, 1.0)
        return None

    # -- handlers ----------------------------------------------------------

    @intent(
        "timer.set",
        "the user wants a countdown timer for a duration",
        slots={"hours": SLOT_INT, "minutes": SLOT_INT, "seconds": SLOT_INT},
        fewshot=(
            ("metti un timer di cinque minuti", {"minutes": 5}),
            ("avvisami tra un'ora e mezza", {"hours": 1, "minutes": 30}),
            ("tra un quarto d'ora avvisami", {"minutes": 15}),
            ("set a timer for 90 seconds", {"seconds": 90}),
        ),
    )
    def set_timer(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        seconds = (
            _slot_int(slots, "hours") * 3600
            + _slot_int(slots, "minutes") * 60
            + _slot_int(slots, "seconds")
        )
        if seconds <= 0:
            return ctx.t("Non ho capito la durata del timer.", "I didn't get the timer length.")
        if seconds > MAX_TIMER_S:
            return ctx.t("Il massimo è ventiquattro ore.", "The maximum is twenty-four hours.")
        label = format_duration(seconds, ctx.lang)
        tid = self.timers.start(seconds, label)
        self.mark_active()
        now_ms = time.time() * 1000.0
        ctx.ui.open(
            self.widget_id(str(tid)),
            "countdown",
            command=self.name,
            title="TIMER",
            anchor="tr",
            data={
                "label": label,
                "total_s": seconds,
                "started_at_ms": round(now_ms),
                "due_at_ms": round(now_ms + seconds * 1000.0),
                "state": "running",
            },
        )
        return ctx.t(f"Timer di {label} avviato.", f"Timer for {label} started.")

    # -- lifecycle ---------------------------------------------------------

    def _fired(self, tid: int, label: str) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        wid = self.widget_id(str(tid))
        ctx.ui.update(wid, {"state": "expired"})
        ctx.announce(timer_alert_text(label, ctx.lang), intent=ALERT_INTENT, command=self.name)
        closer = threading.Timer(EXPIRED_LINGER_S, ctx.ui.close, args=(wid,))
        closer.daemon = True
        closer.start()

    def is_active(self) -> bool:
        return self.timers.active > 0

    def on_cancel(self, ctx: CommandContext) -> Optional[str]:
        ids = self.timers.cancel_all()
        for tid in ids:
            ctx.ui.close(self.widget_id(str(tid)))
        n = len(ids)
        if n == 1:
            return ctx.t("Timer annullato.", "Timer cancelled.")
        if n > 1:
            return ctx.t(f"{n} timer annullati.", f"{n} timers cancelled.")
        return ctx.t("Ok, nessun timer attivo.", "Okay, no timer running.")

    def on_ui_action(self, widget_id: str, action: str, data: dict[str, Any]) -> None:
        if self.ctx is None:
            return
        if action in ("cancel", "dismiss", "close"):
            try:
                tid = int(widget_id.split(":", 1)[1])
            except (IndexError, ValueError):
                return
            self.timers.cancel(tid)
            self.ctx.ui.close(widget_id)

    def on_disable(self) -> None:
        if self.ctx is not None:
            self.on_cancel(self.ctx)

    def shutdown(self) -> None:
        self.timers.cancel_all()
