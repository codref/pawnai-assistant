"""Deterministic reply handlers — the ground truth for every reply.

The LLM never computes a reply. ``phrasing.py`` may reword the text these
return, but only after a guard checks that its numbers/dates survive.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_IT_WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
_IT_MONTHS = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
]


# --------------------------------------------------------------------- timers


@dataclass
class _Timer:
    id: int
    seconds: int
    label: str
    due: float
    handle: threading.Timer


class TimerManager:
    """Countdown timers. ``on_fire(label)`` runs on the timer thread."""

    def __init__(self, on_fire: Optional[Callable[[str], None]] = None) -> None:
        self.on_fire = on_fire or (lambda _label: None)
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
                self.on_fire(label)

        handle = threading.Timer(seconds, _fire)
        handle.daemon = True
        with self._lock:
            self._timers[tid] = _Timer(tid, seconds, label, time.monotonic() + seconds, handle)
        handle.start()
        return tid

    def cancel_all(self) -> int:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for t in timers:
            t.handle.cancel()
        return len(timers)

    @property
    def active(self) -> int:
        with self._lock:
            return len(self._timers)


# --------------------------------------------------------------------- context


@dataclass
class CommandContext:
    lang: str = "it"
    now: Callable[[], datetime] = datetime.now
    timers: TimerManager = field(default_factory=TimerManager)


# --------------------------------------------------------------------- formatting


def format_time(dt: datetime, lang: str = "it") -> str:
    h, m = dt.hour, dt.minute
    if lang != "it":
        return f"It's {h}:{m:02d}."
    if h == 0:
        head = "È mezzanotte"
    elif h == 12:
        head = "È mezzogiorno"
    elif h == 1:  # 24h clock: 13 stays "Sono le 13"
        head = "È l'una"
    else:
        head = f"Sono le {h}"
    if m == 0:
        return f"{head} in punto."
    return f"{head} e {m}."


def format_date(dt: datetime, lang: str = "it") -> str:
    if lang != "it":
        return dt.strftime("Today is %A, %B ") + f"{dt.day}, {dt.year}."
    day = "primo" if dt.day == 1 else str(dt.day)
    return f"Oggi è {_IT_WEEKDAYS[dt.weekday()]} {day} {_IT_MONTHS[dt.month - 1]} {dt.year}."


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


def _t(ctx: CommandContext, it: str, en: str) -> str:
    return it if ctx.lang == "it" else en


# --------------------------------------------------------------------- handlers

MAX_TIMER_S = 24 * 3600


def _slot_int(slots: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(slots.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def handle_time(slots: dict[str, Any], ctx: CommandContext) -> str:
    return format_time(ctx.now(), ctx.lang)


def handle_date(slots: dict[str, Any], ctx: CommandContext) -> str:
    return format_date(ctx.now(), ctx.lang)


def handle_timer_set(slots: dict[str, Any], ctx: CommandContext) -> str:
    seconds = (
        _slot_int(slots, "hours") * 3600
        + _slot_int(slots, "minutes") * 60
        + _slot_int(slots, "seconds")
    )
    if seconds <= 0:
        return _t(ctx, "Non ho capito la durata del timer.", "I didn't get the timer length.")
    if seconds > MAX_TIMER_S:
        return _t(ctx, "Il massimo è ventiquattro ore.", "The maximum is twenty-four hours.")
    label = format_duration(seconds, ctx.lang)
    ctx.timers.start(seconds, label)
    return _t(ctx, f"Timer di {label} avviato.", f"Timer for {label} started.")


def handle_cancel(slots: dict[str, Any], ctx: CommandContext) -> str:
    n = ctx.timers.cancel_all()
    if n == 1:
        return _t(ctx, "Timer annullato.", "Timer cancelled.")
    if n > 1:
        return _t(ctx, f"{n} timer annullati.", f"{n} timers cancelled.")
    return _t(ctx, "Ok, nessun timer attivo.", "Okay, no timer running.")


def handle_greeting(slots: dict[str, Any], ctx: CommandContext) -> str:
    return _t(ctx, "Ciao. Tutto operativo.", "Hello. All systems nominal.")


def handle_unknown(slots: dict[str, Any], ctx: CommandContext) -> str:
    return _t(ctx, "Non ho capito.", "I didn't understand.")


def timer_alert_text(label: str, lang: str = "it") -> str:
    return f"Il timer di {label} è scaduto." if lang == "it" else f"Your {label} timer is done."


HANDLERS: dict[str, Callable[[dict[str, Any], CommandContext], str]] = {
    "time.now": handle_time,
    "date.today": handle_date,
    "timer.set": handle_timer_set,
    "cancel": handle_cancel,
    "greeting": handle_greeting,
    "unknown": handle_unknown,
}
