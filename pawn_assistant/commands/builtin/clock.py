"""Time and date."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pawn_assistant.commands.base import Command, CommandContext, intent

_IT_WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
_IT_MONTHS = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
]


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


class ClockCommand(Command):
    name = "clock"
    title = "Clock"
    description = "Current time and today's date."
    order = 10

    phrasing = {
        "it": {
            "time.now": (
                ("che ore sono", "Sono le 9 e 10.", "cordiale", "mattina", "Sono le 9 e 10, la mattina è ancora giovane."),
                ("mi dici l'ora", "Sono le 23 e 47.", "ironico", "sera", "Le 23 e 47. Tardi anche per un HUD."),
                ("che ora è", "Sono le 15 in punto.", "tecnico", "pomeriggio", "Orologio di sistema: 15 in punto."),
            ),
            "date.today": (
                ("che giorno è", "Oggi è lunedì 3 marzo 2025.", "tecnico", "pomeriggio", "Data corrente: lunedì 3 marzo 2025."),
                ("quanti ne abbiamo", "Oggi è sabato 12 luglio 2025.", "complice", "mattina", "Siamo a sabato 12 luglio 2025, si respira aria di weekend."),
                ("che data è", "Oggi è mercoledì primo ottobre 2025.", "conciso", "sera", "Mercoledì primo ottobre 2025."),
            ),
        },
        "en": {
            "time.now": (
                ("what time is it", "It's 9:10.", "warm", "morning", "It's 9:10, the morning is still young."),
                ("what's the time", "It's 23:47.", "ironic", "evening", "23:47. Late, even for a HUD."),
            ),
            "date.today": (
                ("what day is it", "Today is Monday, March 3, 2025.", "technical", "afternoon", "Current date: Monday, March 3, 2025."),
                ("what's the date", "Today is Saturday, July 12, 2025.", "playful", "morning", "Saturday, July 12, 2025. Weekend mode."),
            ),
        },
    }

    @intent(
        "time.now",
        "the user asks what time it is",
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
    )
    def time_now(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        return format_time(ctx.now(), ctx.lang)

    @intent(
        "date.today",
        "the user asks today's date or day of the week",
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
    )
    def date_today(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        return format_date(ctx.now(), ctx.lang)
