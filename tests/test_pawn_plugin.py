"""Pawn agent plugin: conversation floor, keypad unlock, relock, SSE client."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from pawn_assistant.commands import CommandContext, CommandRouter
from pawn_assistant.commands.builtin.pawn import CHAT_WIDGET, KEYPAD_WIDGET, PawnCommand
from pawn_assistant.commands.builtin.pawn.client import PawnError, VoiceAnswer, iter_sse
from pawn_assistant.commands.registry import CommandRegistry
from pawn_assistant.pipeline.loop import State, WakeLoop


class FakeClient:
    configured = True

    def __init__(self, reply: str = "Ciao, eccomi.", fail: bool = False) -> None:
        self.reply, self.fail = reply, fail
        self.controls: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def voice(self, conversation, message, *, access="chat", lang="it", on_progress=None):
        self.calls.append({"conversation": conversation, "message": message, "access": access})
        if on_progress:
            on_progress("Using skill sessions…")
        if self.fail:
            raise PawnError("unreachable")
        return VoiceAnswer(self.reply, access=access, controls=list(self.controls))


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def env():
    ctx = CommandContext(now=lambda: datetime(2026, 11, 20, 20, 0))
    reg = CommandRegistry(
        config={"pawn": {"server_url": "http://x", "unlock_code": "4321"}},
        entry_points=False,
    )
    router = CommandRouter(ctx=ctx, registry=reg)
    pawn = reg.get("pawn")
    assert isinstance(pawn, PawnCommand)
    pawn.client = FakeClient()
    pawn.clock = Clock()
    pawn.cue = lambda: None
    yield ctx, router, pawn
    reg.shutdown()


def _alerts(ctx: CommandContext) -> list[str]:
    out = []
    while not ctx.alerts.empty():
        out.append(ctx.alerts.get_nowait().text)
    return out


def test_start_takes_floor_and_routes_free_text(env):
    ctx, router, pawn = env
    res = router.handle("Pawn, iniziamo una conversazione")
    assert res.intent == "pawn.start" and res.reply == "Dimmi."
    assert ctx.floor == "pawn" and ctx.ui.is_open(CHAT_WIDGET)
    assert ctx.floor_listen_ms == 8000.0

    # Would normally be the clock intent — inside a conversation it goes to pawn.
    res = router.handle("che ore sono secondo te in Giappone")
    assert res.source == "floor" and res.reply == "Ciao, eccomi."
    call = pawn.client.calls[-1]
    assert call["access"] == "chat" and call["conversation"].startswith("voice:")
    turns = ctx.ui.data(CHAT_WIDGET)["turns"]
    assert [t["who"] for t in turns] == ["you", "pawn"]

    res = router.handle("fine conversazione")
    assert res.intent == "pawn.end" and ctx.floor is None
    assert not ctx.ui.is_open(CHAT_WIDGET)


def test_english_start_phrase(env):
    ctx, router, _ = env
    assert router.handle("let's start a conversation").intent == "pawn.start"


def test_exact_stop_ends_conversation(env):
    ctx, router, _ = env
    router.handle("parliamo")
    res = router.handle("stop")
    assert res.intent == "pawn.end" and ctx.floor is None
    assert res.reply == "Ok, a dopo."


def test_server_error_keeps_floor(env):
    ctx, router, pawn = env
    pawn.client.fail = True
    router.handle("parliamo")
    res = router.handle("come va")
    assert res.reply == "Pawn non risponde." and ctx.floor == "pawn"


def test_unlock_wrong_then_right_code(env):
    ctx, router, pawn = env
    router.handle("parliamo")
    res = router.handle("sblocca le sessioni")
    assert res.intent == "pawn.unlock" and ctx.ui.is_open(KEYPAD_WIDGET)
    assert "4321" not in str(ctx.ui.snapshot())  # code never reaches the HUD

    pawn.on_ui_action(KEYPAD_WIDGET, "submit", {"code": "0000"})
    assert ctx.ui.data(KEYPAD_WIDGET)["result"] == "wrong"
    assert _alerts(ctx) == ["Codice errato."]
    router.handle("dimmi le sessioni di ieri")
    assert pawn.client.calls[-1]["access"] == "chat"

    pawn.on_ui_action(KEYPAD_WIDGET, "submit", {"code": "4321"})
    assert not ctx.ui.is_open(KEYPAD_WIDGET)
    assert _alerts(ctx) == ["Accesso completo."]
    assert ctx.ui.data(CHAT_WIDGET)["access"] == "full"
    router.handle("dimmi le sessioni di ieri")
    assert pawn.client.calls[-1]["access"] == "full"

    res = router.handle("blocca")
    assert res.intent == "pawn.lock"
    router.handle("e adesso?")
    assert pawn.client.calls[-1]["access"] == "chat"


def test_unlock_outside_conversation_starts_one(env):
    ctx, router, _ = env
    res = router.handle("unlock the sessions skills")
    assert res.intent == "pawn.unlock"
    assert ctx.floor == "pawn" and ctx.ui.is_open(KEYPAD_WIDGET)


def test_unlock_expires_and_resets_on_end(env):
    ctx, router, pawn = env
    router.handle("parliamo")
    router.handle("sblocca le sessioni")
    pawn.on_ui_action(KEYPAD_WIDGET, "submit", {"code": "4321"})
    pawn.clock.t += 15 * 60 + 1
    router.handle("ciao")
    assert pawn.client.calls[-1]["access"] == "chat"

    router.handle("sblocca le sessioni")
    pawn.on_ui_action(KEYPAD_WIDGET, "submit", {"code": "4321"})
    router.handle("fine conversazione")
    router.handle("parliamo")
    router.handle("ciao")
    assert pawn.client.calls[-1]["access"] == "chat"


def test_lockout_after_three_failures(env):
    ctx, router, pawn = env
    router.handle("parliamo")
    router.handle("sblocca le sessioni")
    for _ in range(3):
        pawn.on_ui_action(KEYPAD_WIDGET, "submit", {"code": "1"})
    assert not ctx.ui.is_open(KEYPAD_WIDGET)
    res = router.handle("sblocca le sessioni")
    assert "riprova" in res.reply.lower() and not ctx.ui.is_open(KEYPAD_WIDGET)
    pawn.clock.t += 61
    router.handle("sblocca le sessioni")
    pawn.on_ui_action(KEYPAD_WIDGET, "submit", {"code": "4321"})
    assert pawn.access == "full"


def test_floor_timeout_ends_unless_keypad_open(env):
    ctx, router, pawn = env
    router.handle("parliamo")
    router.handle("sblocca le sessioni")
    pawn.on_floor_timeout(ctx)
    assert ctx.floor == "pawn"  # paused while the code is being typed
    ctx.ui.close(KEYPAD_WIDGET)
    pawn.on_floor_timeout(ctx)
    assert ctx.floor is None and _alerts(ctx)[-1] == "Ok, a dopo."


def test_disable_releases_floor(env):
    ctx, router, _ = env
    router.handle("parliamo")
    router.registry.set_enabled("pawn", False)
    assert ctx.floor is None


def test_iter_sse_parses_events_and_skips_keepalive():
    raw = [
        b": keep-alive\n", b"\n",
        b"event: progress\n", b'data: {"kind": "skill", "text": "Using skill"}\n', b"\n",
        b"event: answer\n", b'data: {"content": "Ciao."}\n', b"\n",
        b"event: done\n", b'data: {"run_id": "r1"}\n', b"\n",
    ]
    assert [e for e, _ in iter_sse(iter(raw))] == ["progress", "answer", "done"]


# ------------------------------------------------------------------ loop


def _loop(router: CommandRouter) -> WakeLoop:
    loop = WakeLoop.__new__(WakeLoop)
    loop.router = router
    loop.ctx = router.ctx
    loop.events = []
    loop.calls = []
    loop._stop = type("E", (), {"is_set": staticmethod(lambda: False)})()
    loop._return_to_idle = lambda **k: loop.calls.append("idle")
    loop._follow_up_listen = lambda: loop.calls.append("followup")
    return loop


def test_loop_reopens_mic_while_floor_held(env):
    ctx, router, _ = env
    loop = _loop(router)
    loop._after_turn()
    router.handle("parliamo")
    loop._after_turn()
    router.handle("fine conversazione")
    loop._after_turn()
    assert loop.calls == ["idle", "followup", "idle"]


def test_followup_timeout_asks_owner(env):
    ctx, router, _ = env
    loop = _loop(router)
    router.handle("parliamo")
    loop._on_followup_timeout()
    assert ctx.floor is None and loop.calls == ["idle"]
    assert _alerts(ctx)[-1] == "Ok, a dopo."


def test_floor_reply_spoken_verbatim_by_sentence(env):
    ctx, router, pawn = env
    pawn.client.reply = "Hai tre sessioni. Vuoi il riassunto?"
    router.handle("parliamo")
    loop = _loop(router)
    loop.tts = None
    loop.phraser = object()  # would explode if used
    loop._set_state = lambda s: setattr(loop, "_state", s)
    loop._emit_event = loop.events.append
    loop._state = State.IDLE
    loop._run_command("quante sessioni ho")
    replies = [e for e in loop.events if e.get("name") == "reply"]
    assert [(e["seq"], e["text"]) for e in replies] == [
        (0, "Hai tre sessioni."),
        (1, "Hai tre sessioni. Vuoi il riassunto?"),
    ]


# ------------------------------------------------------------------ stop / lock phrasing

# Real transcripts from a test session that previously leaked to the agent.
@pytest.mark.parametrize(
    "text",
    [
        "Possiamo smettere di conversare.",
        "Fa ermati.",
        "Fermati.",
        "Basta così, grazie.",
        "Smettila di ascoltarmi.",
        "Ok, a dopo!",
        "Ciao ciao.",
        "Chiudi la conversazione per favore.",
        "Stop listening.",
        "End the conversation.",
    ],
)
def test_end_phrasings(env, text):
    ctx, router, _ = env
    router.handle("parliamo")
    res = router.handle(text)
    assert res.intent in ("pawn.end", "cancel"), (text, res.intent)
    assert ctx.floor is None


@pytest.mark.parametrize(
    "text",
    [
        "Bene, adesso puoi bloccare l'accesso, per favore.",
        "Blocca di nuovo le sessioni.",
        "Richiudi l'accesso.",
        "Revoca l'accesso completo.",
        "Blocca.",
        "Lock the sessions again.",
    ],
)
def test_lock_phrasings(env, text):
    ctx, router, pawn = env
    router.handle("parliamo")
    router.handle("sblocca le sessioni")
    pawn.on_ui_action(KEYPAD_WIDGET, "submit", {"code": "4321"})
    res = router.handle(text)
    assert res.intent == "pawn.lock", (text, res.intent)
    assert pawn.access == "chat" and ctx.floor == "pawn"


@pytest.mark.parametrize(
    "text",
    [
        "Raccontami una barzelletta.",
        "Chi è Warren? Cerca nelle sessioni.",
        "Di cosa parlava la sessione cve nginx di ieri?",
        "Come si ferma un attacco a nginx?",
        "Sblocca le sessioni.",
        "Ciao, come stai?",
    ],
)
def test_normal_turns_not_mistaken_for_control(env, text):
    ctx, router, _ = env
    router.handle("parliamo")
    res = router.handle(text)
    assert res.intent not in ("pawn.end", "pawn.lock", "cancel"), (text, res.intent)
    assert ctx.floor == "pawn"


def test_agent_end_control_releases_floor(env):
    ctx, router, pawn = env
    router.handle("parliamo")
    pawn.client.controls = ["end"]
    res = router.handle("ok direi che per stasera abbiamo chiacchierato abbastanza")
    assert res.source == "floor" and ctx.floor is None
    assert not ctx.ui.is_open(CHAT_WIDGET)


def test_agent_lock_control_relocks(env):
    ctx, router, pawn = env
    router.handle("parliamo")
    router.handle("sblocca le sessioni")
    pawn.on_ui_action(KEYPAD_WIDGET, "submit", {"code": "4321"})
    pawn.client.controls = ["lock"]
    router.handle("niente più dati riservati per oggi")
    assert pawn.access == "chat" and ctx.floor == "pawn"
    pawn.client.controls = []
    router.handle("e ora?")
    assert pawn.client.calls[-1]["access"] == "chat"
