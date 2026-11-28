"""pawn-server voice contract (tests/contracts, synced from pawnai — never edit there).

Replays the golden SSE transcripts through the real client parser and checks
the shared utterances route to the same intents as on Android.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from pawn_assistant.commands import CommandContext, CommandRouter
from pawn_assistant.commands.builtin.pawn.client import PawnError, iter_sse, read_voice_stream
from pawn_assistant.commands.registry import CommandRegistry

CONTRACTS = Path(__file__).parent / "contracts"
CASES = json.loads((CONTRACTS / "voice" / "cases.json").read_text(encoding="utf-8"))


def _lines(name: str):
    return iter(io.BytesIO((CONTRACTS / "voice" / name).read_bytes()).readline, b"")


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["file"])
def test_golden_voice_stream(case: dict) -> None:
    progress: list[str] = []
    if "error" in case:
        with pytest.raises(PawnError, match=case["error"]):
            read_voice_stream(_lines(case["file"]), on_progress=progress.append)
        return
    answer = read_voice_stream(_lines(case["file"]), on_progress=progress.append)
    want = case["answer"]
    assert progress == case["progress"]
    assert answer.text == want["text"]
    assert answer.access == want["access"]
    assert answer.controls == want["controls"]
    assert answer.display == want["display"]
    assert answer.run_id == want["run_id"]


def test_golden_device_stream() -> None:
    spec = CASES["device_stream"]
    events = list(iter_sse(_lines(spec["file"])))
    assert events[0][0] == "ready"
    cmds = [c for ev, data in events if ev == "cmd" for c in data["commands"]]
    assert [{k: c[k] for k in ("id", "tool", "args")} for c in cmds] == spec["commands"]


def _aliases() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (CONTRACTS / "utterances" / "linux-aliases.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            canon, local = (part.strip() for part in line.split("=", 1))
            out[canon] = local
    return out


def _fixtures() -> list[tuple[str, str, str]]:
    rows = []
    for line in (CONTRACTS / "utterances" / "shared.fixtures").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        lang, text, intent = (part.strip() for part in line.split("|")[:3])
        rows.append((lang, text, intent))
    return rows


@pytest.fixture(scope="module")
def router():
    reg = CommandRegistry(config={"pawn": {"server_url": "http://x"}}, entry_points=False, state_path=None)
    r = CommandRouter(ctx=CommandContext(), registry=reg)
    yield r
    reg.shutdown()


@pytest.mark.parametrize("lang,text,intent", _fixtures(), ids=lambda v: str(v))
def test_shared_utterances(router, lang: str, text: str, intent: str) -> None:
    router.ctx.lang = lang
    got = router.classify(text).intent
    want = _aliases().get(intent, intent)
    if intent == "-":
        assert got == "unknown"
    else:
        assert got == want, f"{text!r} → {got}, want {want}"
