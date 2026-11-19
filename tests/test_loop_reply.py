"""WakeLoop._reply: streamed phrased sentences vs canonical fallback (no audio)."""

from pawn_assistant.commands.phrasing import PhraseSettings, ReplyPhraser
from pawn_assistant.pipeline.loop import State, WakeLoop


class _LLM:
    def __init__(self, text):
        self.text = text

    def stream_chat(self, messages, **kwargs):
        yield from (self.text[i : i + 4] for i in range(0, len(self.text), 4))


def _loop(text, **phrase_kw):
    loop = WakeLoop.__new__(WakeLoop)  # skip audio/model setup
    loop.phraser = ReplyPhraser(_LLM(text), PhraseSettings(**phrase_kw))
    loop.tts = None
    loop._state = State.IDLE
    loop.events = []
    loop._set_state = lambda s: setattr(loop, "_state", s)
    loop._emit_event = loop.events.append
    return loop


def test_phrased_sentences_stream_as_growing_events():
    loop = _loop("Le 14 e 35, pomeriggio pieno. Buon lavoro.\n")
    loop._reply("time.now", "che ore sono", "Sono le 14 e 35.")
    assert [(e["seq"], e["text"], e["phrased"]) for e in loop.events] == [
        (0, "Le 14 e 35, pomeriggio pieno.", True),
        (1, "Le 14 e 35, pomeriggio pieno. Buon lavoro.", True),
    ]
    assert loop._state == State.SPEAK


def test_guard_failure_speaks_canonical():
    loop = _loop("Sono quasi le tre.\n")
    loop._reply("time.now", "che ore sono", "Sono le 14 e 35.")
    assert [(e["text"], e["phrased"]) for e in loop.events] == [("Sono le 14 e 35.", False)]


def test_intent_not_enabled_skips_llm():
    loop = _loop("whatever", intents=frozenset({"greeting"}))
    loop._reply("time.now", "che ore sono", "Sono le 14 e 35.", kind="alert")
    assert [(e["text"], e["kind"]) for e in loop.events] == [("Sono le 14 e 35.", "alert")]
    assert "phrased" not in loop.events[0]
