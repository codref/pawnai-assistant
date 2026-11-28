"""pawn-server link: device plugin commands/presence, hand-off, phone unlock, wake sync."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from pawn_assistant.commands import CommandContext, CommandRouter
from pawn_assistant.commands.builtin.device import DeviceCommand, parse_expiry
from pawn_assistant.commands.builtin.pawn import CHAT_WIDGET, PawnCommand
from pawn_assistant.commands.builtin.pawn.client import PawnError, VoiceAnswer
from pawn_assistant.commands.registry import CommandRegistry
from pawn_assistant.server_api import DeviceIdentity, load_identity
from pawn_assistant.wake_train import sync


def _alerts(ctx: CommandContext) -> list[str]:
    out = []
    while not ctx.alerts.empty():
        out.append(ctx.alerts.get_nowait().text)
    return out


# ------------------------------------------------------------------ identity


def test_identity_roundtrip_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "device.json"
    DeviceIdentity("http://pawn:8000/", "dev1", "tok", "laptop").save(path)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    ident = load_identity(path=path)
    assert ident is not None and ident.device_id == "dev1"
    over = load_identity({"device_id": "dev2"}, path=path)
    assert over is not None and over.device_id == "dev2" and over.device_token == "tok"
    assert load_identity(path=tmp_path / "missing.json") is None


# ------------------------------------------------------------------ device plugin


class FakeDeviceClient:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, dict, str]] = []
        self.presences: list[tuple[str, dict]] = []
        self.queue: list[dict] = []
        self.ident = DeviceIdentity("http://x", "dev1", "tok", "laptop")

    def post_result(self, cid, status, result=None, message=""):
        self.results.append((cid, status, result or {}, message))
        return True

    def presence(self, state, **flags):
        self.presences.append((state, flags))
        return {}

    def pending(self):
        out, self.queue = self.queue, []
        return out


@pytest.fixture
def linked():
    ctx = CommandContext()
    reg = CommandRegistry(config={"device": {"start_threads": False}}, entry_points=False, state_path=None)
    CommandRouter(ctx=ctx, registry=reg)
    dev = reg.get("device")
    assert isinstance(dev, DeviceCommand)
    dev.client = FakeDeviceClient()
    yield ctx, reg, dev
    reg.shutdown()


def _future(seconds: float = 30) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).replace(tzinfo=None).isoformat()


def test_speak_and_notify_queue_alerts(linked) -> None:
    ctx, _reg, dev = linked
    assert dev.run({"id": "c1", "tool": "speak", "args": {"text": "La pasta è pronta."}, "expires_at": _future()}) == "ok"
    assert dev.run({"id": "c2", "tool": "notify", "args": {"text": "Riunione", "title": "agenda"}, "expires_at": _future()}) == "ok"
    assert _alerts(ctx) == ["La pasta è pronta.", "Riunione"]
    assert [r[:2] for r in dev.client.results] == [("c1", "ok"), ("c2", "ok")]


def test_timer_set_starts_a_countdown(linked) -> None:
    ctx, reg, dev = linked
    assert dev.run({"id": "c3", "tool": "timer_set", "args": {"seconds": "600", "label": "pasta"}, "expires_at": _future()}) == "ok"
    timer = reg.get("timer")
    assert timer.timers.active == 1
    assert dev.client.results[-1][2]["timer_id"] >= 1
    timer.timers.cancel_all()


def test_expired_and_unknown_commands(linked) -> None:
    _ctx, _reg, dev = linked
    assert dev.run({"id": "old", "tool": "speak", "args": {"text": "x"}, "expires_at": _future(-5)}) == "skipped"
    assert dev.client.results == []
    assert dev.run({"id": "c4", "tool": "sms_send", "args": {}, "expires_at": _future()}) == "error"
    assert dev.client.results[-1][1] == "error"
    assert parse_expiry("2026-11-27T10:00:00") == datetime(2026, 11, 27, 10, tzinfo=timezone.utc).timestamp()


def test_drain_runs_pending(linked) -> None:
    ctx, _reg, dev = linked
    dev.client.queue = [{"id": "p1", "tool": "speak", "args": {"text": "ciao"}, "expires_at": _future()}]
    assert dev.drain() == 1
    assert _alerts(ctx) == ["ciao"]


def test_presence_follows_wake_and_idle(linked) -> None:
    _ctx, _reg, dev = linked
    t = [1000.0]
    dev.mono = lambda: t[0]
    dev.wall = lambda: 1_764_000_000.0
    assert dev.presence_state() == "idle"
    dev.on_state("AWAKE")
    assert dev.presence_state() == "active"
    assert dev.report_presence()
    assert dev.client.presences[-1] == ("active", {"last_wake_at": 1_764_000_000})
    t[0] += dev.idle_after_s + 1
    assert dev.presence_state() == "idle"


# ------------------------------------------------------------------ hand-off / phone unlock


class FakeVoice:
    configured = True
    server_url = "http://x"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.thread: Any = None
        self.claims: list[tuple[str, str]] = []
        self.statuses: list[dict] = []
        self.requests: list[dict] = []

    def voice(self, conversation, message, *, access="chat", lang="it", on_progress=None, device_id=None):
        self.calls.append({"conversation": conversation, "access": access, "device_id": device_id})
        return VoiceAnswer("Ok.", access=access)

    def active_thread(self, exclude_device=None):
        self.exclude = exclude_device
        return self.thread

    def claim(self, conversation, device_id):
        self.claims.append((conversation, device_id))
        return {**self.thread, "from_device": "Pixel 9"}

    def request_access(self, device_id, *, conversation, minutes):
        self.requests.append({"device_id": device_id, "conversation": conversation, "minutes": minutes})
        if getattr(self, "no_phone", False):
            raise PawnError("HTTP 409", status=409)
        return {"id": "ar1", "status": "pending"}

    def access_status(self, request_id):
        return self.statuses.pop(0) if self.statuses else {"status": "pending"}


@pytest.fixture
def pawn():
    ctx = CommandContext()
    reg = CommandRegistry(
        config={"pawn": {"server_url": "http://x", "device_id": "dev-laptop"}},
        entry_points=False,
        state_path=None,
    )
    router = CommandRouter(ctx=ctx, registry=reg)
    cmd = reg.get("pawn")
    assert isinstance(cmd, PawnCommand)
    cmd.client = FakeVoice()
    cmd.sleep = lambda _s: None
    cmd.cue = lambda: None
    jobs: list = []
    cmd.spawn = jobs.append
    yield ctx, router, cmd, jobs
    reg.shutdown()


def test_turns_carry_device_id(pawn) -> None:
    ctx, router, cmd, _ = pawn
    router.handle("parliamo")
    router.handle("come va")
    assert cmd.client.calls[-1]["device_id"] == "dev-laptop"


def test_resume_claims_the_phone_thread(pawn) -> None:
    ctx, router, cmd, _ = pawn
    cmd.client.thread = {
        "conversation": "voice:phone1",
        "device_name": "Pixel 9",
        "last_answer": "Alle nove hai la riunione con Sean. Poi niente.",
    }
    res = router.handle("continua qui")
    assert res.intent == "pawn.resume"
    assert res.reply == "Riprendo da Pixel 9. Ero rimasto a: Alle nove hai la riunione con Sean."
    assert cmd.client.exclude == "dev-laptop"
    assert cmd.client.claims == [("voice:phone1", "dev-laptop")]
    assert cmd.conversation == "voice:phone1" and ctx.floor == "pawn"
    assert ctx.ui.is_open(CHAT_WIDGET)
    router.handle("e dopo pranzo?")
    assert cmd.client.calls[-1]["conversation"] == "voice:phone1"


def test_resume_without_thread(pawn) -> None:
    _ctx, router, cmd, _ = pawn
    res = router.handle("riprendi la conversazione")
    assert res.reply == "Non c'è una conversazione da riprendere."
    assert cmd.conversation is None


def test_unlock_with_phone_approved(pawn) -> None:
    ctx, router, cmd, jobs = pawn
    res = router.handle("sblocca col telefono")
    assert res.intent == "pawn.unlock_phone" and res.reply == "Conferma sul telefono."
    assert cmd.conversation is not None and cmd.access_request == "ar1"
    assert cmd.client.requests[0]["conversation"] == cmd.conversation
    # Waiting for the phone keeps the floor open.
    cmd.on_floor_timeout(ctx)
    assert ctx.floor == "pawn"
    cmd.client.statuses = [{"status": "pending"}, {"status": "approved", "minutes": 15}]
    jobs.pop()()
    assert cmd.access == "full" and cmd._is_unlocked() and cmd.access_request is None
    assert _alerts(ctx)[-1] == "Accesso completo."
    router.handle("quante sessioni ho")
    assert cmd.client.calls[-1]["access"] == "full"


def test_unlock_with_phone_denied_or_missing(pawn) -> None:
    ctx, router, cmd, jobs = pawn
    router.handle("sblocca col telefono")
    cmd.client.statuses = [{"status": "denied"}]
    assert cmd.wait_access(ctx, "ar1", cmd.conversation) == "denied"
    assert cmd.access == "chat"
    assert _alerts(ctx)[-1] == "Il telefono ha rifiutato."
    jobs.clear()
    cmd.client.no_phone = True
    assert router.handle("sblocca col telefono").reply == "Nessun telefono può confermare."


def test_unlock_ignored_after_conversation_ended(pawn) -> None:
    ctx, router, cmd, jobs = pawn
    router.handle("sblocca col telefono")
    conv = cmd.conversation
    router.handle("fine conversazione")
    cmd.client.statuses = [{"status": "approved", "minutes": 15}]
    assert cmd.wait_access(ctx, "ar1", conv) == "approved"
    assert cmd.access == "chat"


# ------------------------------------------------------------------ wake sync


class FakeAdmin:
    def __init__(self, clips: list[dict]) -> None:
        self.clips = clips
        self.downloads = 0
        self.published: list[dict] = []

    def list_clips(self):
        return self.clips

    def download_clip(self, cid):
        self.downloads += 1
        return b"RIFF....WAVE"

    def publish_model(self, data, **kw):
        self.published.append({"size": len(data), **kw})
        return {"version": kw.get("version") or "v1", "sha256": "0" * 64}


def test_pull_is_idempotent(tmp_path: Path) -> None:
    admin = FakeAdmin([
        {"id": "a" * 32, "label": "confirmed", "score": 0.8, "created_at": "2026-11-27T10:00:00", "device_id": "pix"},
        {"id": "b" * 32, "label": "weird", "score": 0.1},
    ])
    rep = sync.pull_clips(admin, tmp_path)
    assert (rep.added, rep.skipped) == (1, 1)
    wavs = list((tmp_path / "inbox" / "confirmed").glob("*.wav"))
    meta = json.loads(wavs[0].with_suffix(".json").read_text())
    assert meta["source"] == "server" and meta["device_id"] == "pix"
    rep2 = sync.pull_clips(admin, tmp_path)
    assert rep2.added == 0 and admin.downloads == 1


def test_push_skips_server_and_uploaded(tmp_path: Path) -> None:
    folder = tmp_path / "inbox" / "barge"
    folder.mkdir(parents=True)
    for name, meta in (("a", {"score": 0.7}), ("b", {"score": 0.6, "source": "server"}), ("c", {"uploaded": True})):
        (folder / f"{name}.wav").write_bytes(b"RIFF")
        (folder / f"{name}.json").write_text(json.dumps(meta))
    uploads: list[dict] = []

    class Dev:
        def upload_clip(self, wav, **kw):
            uploads.append(kw)

    rep = sync.push_clips(Dev(), tmp_path)
    assert rep.added == 1 and uploads[0]["label"] == "barge"
    assert json.loads((folder / "a.json").read_text())["uploaded"] is True
    assert sync.push_clips(Dev(), tmp_path).added == 0


def test_ensure_model_downloads_once(tmp_path: Path) -> None:
    import hashlib

    blob = b"onnx-model"

    class Dev:
        downloads = 0

        def current_model(self):
            return {"name": "hey_pawn", "version": "v2", "sha256": hashlib.sha256(blob).hexdigest()}

        def download_model(self, version):
            Dev.downloads += 1
            return blob

    path, model = sync.ensure_model(Dev(), tmp_path)
    assert path.read_bytes() == blob and model["version"] == "v2"
    sync.ensure_model(Dev(), tmp_path)
    assert Dev.downloads == 1
