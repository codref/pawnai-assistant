from dataclasses import dataclass
from datetime import datetime

from pawn_assistant.commands import CommandContext, CommandRouter
from pawn_assistant.commands.intents import INTENTS
from pawn_assistant.commands.llm import build_grammar, build_system_prompt, parse_output


@dataclass
class _Parse:
    intent: str
    slots: dict


class FakeLLM:
    resident = True

    def __init__(self, intent="unknown", slots=None, fail=False):
        self.intent, self.slots, self.fail = intent, slots or {}, fail
        self.calls = []

    def parse(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("boom")
        return _Parse(self.intent, self.slots)


def _router(llm=None):
    ctx = CommandContext(now=lambda: datetime(2026, 11, 19, 14, 35))
    return CommandRouter(ctx=ctx, llm=llm)


def test_fastpath_skips_llm():
    llm = FakeLLM()
    res = _router(llm).handle("che ora è")
    assert res.source == "fastpath"
    assert res.reply == "Sono le 14 e 35."
    assert llm.calls == []


def test_miss_goes_to_llm():
    llm = FakeLLM("timer.set", {"minutes": 3, "seconds": 20})
    router = _router(llm)
    res = router.handle("fammi partire il conto alla rovescia per la pasta")
    assert res.source == "llm"
    assert res.intent == "timer.set"
    assert res.reply == "Timer di 3 minuti e 20 secondi avviato."
    router.ctx.timers.cancel_all()


def test_no_llm_is_unknown():
    res = _router(None).handle("che tempo fa domani")
    assert res.intent == "unknown" and res.source == "none"
    assert res.reply == "Non ho capito."


def test_llm_failure_degrades_to_unknown():
    res = _router(FakeLLM(fail=True)).handle("che tempo fa domani")
    assert res.intent == "unknown"
    assert res.reply == "Non ho capito."


def test_intent_event_shape():
    ev = _router().handle("ciao").as_event()
    assert ev["name"] == "intent" and ev["intent"] == "greeting" and ev["source"] == "fastpath"


def test_grammar_covers_registry():
    grammar = build_grammar()
    for intent in INTENTS:
        assert f'\\"{intent.name}\\"' in grammar
    assert grammar.count('"\\"minutes\\""') == 1  # only timer.set carries slots


def test_prompt_is_static():
    assert build_system_prompt() == build_system_prompt()


def test_parse_output_validates():
    assert parse_output('{"intent": "timer.set", "minutes": 5}') == ("timer.set", {"minutes": 5})
    assert parse_output('{"intent": "bogus"}') == ("unknown", {})
    assert parse_output("not json") == ("unknown", {})


def test_grammar_compiles():
    import pytest

    llama_cpp = pytest.importorskip("llama_cpp")
    assert llama_cpp.LlamaGrammar.from_string(build_grammar(), verbose=False) is not None
