import threading
from datetime import datetime

from pawn_assistant.commands import CommandContext, CommandRegistry, CommandRouter
from pawn_assistant.commands.builtin import timer as timer_mod
from pawn_assistant.commands.builtin.core import CoreCommand
from pawn_assistant.commands.builtin.timer import TimerCommand
from pawn_assistant.commands.ui import UiBridge


def test_bridge_snapshot_merges_updates():
    events = []
    ui = UiBridge(events.append)
    ui.open("x:1", "kv", command="x", title="X", data={"a": 1})
    ui.update("x:1", {"b": 2})
    ui.update("missing", {"c": 3})  # ignored
    snap = ui.snapshot()
    assert snap[0]["data"] == {"a": 1, "b": 2} and snap[0]["op"] == "open"
    assert ui.owner("x:1") == "x"
    ui.close("x:1")
    ui.close("x:1")  # idempotent
    assert ui.snapshot() == []
    assert [e["op"] for e in events] == ["open", "update", "close"]


def _router():
    events = []
    ctx = CommandContext(now=lambda: datetime(2026, 11, 19, 14, 35))
    ctx.ui.emit = events.append
    reg = CommandRegistry([CoreCommand, TimerCommand])
    return CommandRouter(ctx=ctx, registry=reg), events


def test_timer_opens_and_cancel_closes_countdown():
    router, events = _router()
    router.handle("metti un timer di 5 minuti")
    opened = events[-1]
    assert opened["op"] == "open" and opened["kind"] == "countdown"
    data = opened["data"]
    assert data["total_s"] == 300 and data["due_at_ms"] - data["started_at_ms"] == 300_000
    router.handle("annulla il timer")
    assert events[-1] == {"type": "ui", "op": "close", "id": opened["id"]}


def test_widget_cancel_action_routes_to_timer():
    router, events = _router()
    router.handle("metti un timer di 5 minuti")
    wid = events[-1]["id"]
    assert router.registry.ui_action(wid, "cancel", {})
    assert router.registry.get("timer").timers.active == 0
    assert not router.ctx.ui.is_open(wid)


def test_timer_fire_announces():
    router, _ = _router()
    router.registry.get("timer").timers.start(0, "1 secondo")  # fires at once
    alert = router.ctx.alerts.get(timeout=2)
    assert alert.intent == "timer.alert" and alert.text == "Il timer di 1 secondo è scaduto."
    assert alert.beeps == 3 and alert.command == "timer"


def test_expired_widget_closes_after_linger(monkeypatch):
    monkeypatch.setattr(timer_mod, "EXPIRED_LINGER_S", 0.01)
    router, events = _router()
    done = threading.Event()
    base = router.ctx.ui.emit

    def spy(p):
        base(p)
        if p["op"] == "close":
            done.set()

    router.ctx.ui.emit = spy
    timer = router.registry.get("timer")
    timer.set_timer({"seconds": 1}, router.ctx)
    assert done.wait(3.0)
    ops = [e["op"] for e in events]
    assert ops == ["open", "update", "close"]
    assert events[1]["data"] == {"state": "expired"}
