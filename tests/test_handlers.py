from datetime import datetime

import pytest

from pawn_assistant.commands import CommandContext, CommandRegistry
from pawn_assistant.commands.builtin.clock import format_date, format_time
from pawn_assistant.commands.builtin.core import CoreCommand
from pawn_assistant.commands.builtin.timer import TimerCommand, TimerManager, format_duration


def _timer_ctx():
    ctx = CommandContext()
    reg = CommandRegistry([CoreCommand, TimerCommand])
    reg.setup(ctx)
    return ctx, reg.get("timer"), reg.get("core")


@pytest.mark.parametrize(
    "hour,minute,expected",
    [
        (14, 35, "Sono le 14 e 35."),
        (9, 0, "Sono le 9 in punto."),
        (1, 5, "È l'una e 5."),
        (13, 10, "Sono le 13 e 10."),
        (0, 20, "È mezzanotte e 20."),
        (12, 0, "È mezzogiorno in punto."),
    ],
)
def test_format_time_it(hour, minute, expected):
    assert format_time(datetime(2026, 11, 19, hour, minute), "it") == expected


def test_format_time_en():
    assert format_time(datetime(2026, 11, 19, 7, 5), "en") == "It's 7:05."


def test_format_date_it():
    assert format_date(datetime(2026, 11, 19), "it") == "Oggi è giovedì 19 novembre 2026."
    assert format_date(datetime(2026, 1, 1), "it") == "Oggi è giovedì primo gennaio 2026."


def test_format_duration():
    assert format_duration(300) == "5 minuti"
    assert format_duration(90) == "un minuto e 30 secondi"
    assert format_duration(3600 + 1800 + 5) == "un'ora, 30 minuti e 5 secondi"
    assert format_duration(120, "en") == "2 minutes"


def test_timer_set_and_cancel():
    ctx, timer, core = _timer_ctx()
    assert timer.set_timer({"minutes": 5}, ctx) == "Timer di 5 minuti avviato."
    assert timer.timers.active == 1
    assert core.cancel({}, ctx) == "Timer annullato."
    assert timer.timers.active == 0
    assert core.cancel({}, ctx) == "Ok, non c'è niente da fermare."
    ctx.norm = "annulla il timer"
    assert core.cancel({}, ctx) == "Ok, nessun timer attivo."


def test_timer_rejects_bad_durations():
    ctx, timer, _ = _timer_ctx()
    assert "durata" in timer.set_timer({}, ctx)
    assert "massimo" in timer.set_timer({"hours": 30}, ctx)
    assert timer.timers.active == 0


def test_timer_fires():
    import threading

    done = threading.Event()
    labels = []

    def on_fire(tid, label):
        labels.append(label)
        done.set()

    timers = TimerManager(on_fire=on_fire)
    timers.start(0, "0 secondi")
    assert done.wait(2.0)
    assert labels == ["0 secondi"]
    assert timers.active == 0
