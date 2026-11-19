import time
from types import SimpleNamespace

import pytest

from pawn_assistant.commands.phrasing import (
    PhraseFailed,
    PhraseSettings,
    PhraseStats,
    ReplyPhraser,
    build_phrase_prompt,
    clean_sentence,
    facts_kept,
    no_new_facts,
    split_sentences,
)


class FakeLLM:
    """Yields scripted pieces; optional per-piece delay / mid-stream error."""

    def __init__(self, pieces, delay=0.0, fail_at=None):
        self.pieces, self.delay, self.fail_at = list(pieces), delay, fail_at
        self.calls = []
        self.consumed = 0

    def stream_chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        for i, piece in enumerate(self.pieces):
            if self.fail_at == i:
                raise RuntimeError("boom")
            if self.delay:
                time.sleep(self.delay)
            self.consumed += 1
            yield piece


def _phraser(pieces, **kw):
    llm = FakeLLM(pieces, delay=kw.pop("delay", 0.0), fail_at=kw.pop("fail_at", None))
    return ReplyPhraser(llm, PhraseSettings(**kw)), llm


def _tokens(text):
    return [text[i : i + 3] for i in range(0, len(text), 3)]


# ------------------------------------------------------------------ pure helpers


def test_split_sentences_keeps_tail():
    done, tail = split_sentences("Sono le 14 e 35. Buon pomeriggio")
    assert done == ["Sono le 14 e 35."]
    assert tail == " Buon pomeriggio"


def test_split_sentences_ignores_time_separators():
    done, tail = split_sentences("Sono le 14.35 adesso")
    assert done == [] and tail == "Sono le 14.35 adesso"


def test_clean_sentence_strips_prefix_and_quotes():
    assert clean_sentence('REPLY: "Le 14 e 35."') == "Le 14 e 35."
    assert clean_sentence("**Ciao**") is None
    assert clean_sentence("FACT: x") is None
    assert clean_sentence("...") is None


def test_facts_kept_numbers_and_calendar_words():
    canonical = "Oggi è giovedì 19 novembre 2026."
    assert facts_kept("Siamo a giovedì 19 novembre 2026.", canonical)
    assert not facts_kept("Siamo a giovedì 19 novembre.", canonical)
    assert not facts_kept("Siamo a venerdì 19 novembre 2026.", canonical)


def test_no_new_facts():
    assert no_new_facts("Le 14 e 35, pomeriggio pieno.", "Sono le 14 e 35.")
    assert not no_new_facts("Le 14 e 35, mancano 25 minuti alle 15.", "Sono le 14 e 35.")
    assert not no_new_facts("Ed è pure venerdì.", "Sono le 14 e 35.")
    # "un minuto" may legitimately become "1 minuto"
    assert no_new_facts("Timer di 1 minuto partito.", "Timer di un minuto avviato.")


def test_prompt_is_static_and_has_persona():
    a = build_phrase_prompt("Sei PAWN.", "it")
    assert a == build_phrase_prompt("Sei PAWN.", "it")
    assert a.startswith("Sei PAWN.")
    assert "Always reply in Italian." in a


# ------------------------------------------------------------------ phraser


def test_streams_valid_sentences():
    phraser, llm = _phraser(_tokens("Sono le 14 e 35. Pomeriggio inoltrato.\n"))
    stats = PhraseStats()
    out = list(phraser.sentences("time.now", "che ore sono", "Sono le 14 e 35.", stats))
    assert out == ["Sono le 14 e 35.", "Pomeriggio inoltrato."]
    assert stats.phrased and stats.first_ms > 0
    assert llm.calls[0][1]["stop"] == ["\n"]


def test_missing_number_falls_back():
    phraser, _ = _phraser(_tokens("Sono quasi le tre. "))
    text, stats = phraser.phrase("time.now", "che ore sono", "Sono le 14 e 35.")
    assert text == "Sono le 14 e 35."
    assert not stats.phrased and stats.reason.startswith("guard")


def test_hallucinated_number_in_second_sentence_is_dropped():
    phraser, _ = _phraser(_tokens("Le 14 e 35. Tra 25 minuti sono le 15. "))
    text, stats = phraser.phrase("time.now", "che ore sono", "Sono le 14 e 35.")
    assert text == "Le 14 e 35."
    assert stats.phrased and stats.reason.startswith("dropped")


def test_max_sentences_caps_and_stops_generation():
    pieces = _tokens("Fatto. Uno. Due. Tre. Quattro. Cinque. ")
    phraser, llm = _phraser(pieces, max_sentences=2, delay=0.01)
    text, _ = phraser.phrase("cancel", "annulla", "Ok.")
    assert text == "Fatto. Uno."
    time.sleep(0.1)
    assert llm.consumed < len(pieces)


def test_timeout_falls_back():
    phraser, _ = _phraser(["Ciao", ". ", "x"], delay=0.3, first_sentence_timeout_s=0.2)
    text, stats = phraser.phrase("greeting", "ciao", "Ciao. Tutto operativo.")
    assert text == "Ciao. Tutto operativo."
    assert stats.reason == "timeout"


def test_error_before_first_sentence_falls_back():
    phraser, _ = _phraser(["Cia"], fail_at=0)
    with pytest.raises(PhraseFailed):
        list(phraser.sentences("greeting", "ciao", "Ciao."))


def test_error_after_first_sentence_keeps_it():
    phraser, _ = _phraser(["Ciao a te. ", "Tutto"], fail_at=1)
    text, stats = phraser.phrase("greeting", "ciao", "Ciao.")
    assert text == "Ciao a te." and stats.phrased


def test_last_sentence_without_punctuation_is_flushed():
    phraser, _ = _phraser(_tokens("Timer annullato, come non detto"))
    text, _ = phraser.phrase("cancel", "annulla il timer", "Timer annullato.")
    assert text == "Timer annullato, come non detto"


def test_applies_and_from_settings():
    app = SimpleNamespace(
        reply_language="it",
        phrase_persona="",
        phrase_temperature=0.7,
        phrase_top_p=0.9,
        phrase_max_tokens=40,
        phrase_max_sentences=1,
        phrase_first_sentence_timeout_s=2.0,
        phrase_intents=["greeting", "timer.alert"],
    )
    s = PhraseSettings.from_settings(app)
    assert s.intents == frozenset({"greeting", "timer.alert"})
    assert s.persona.startswith("Sei PAWN")
    phraser = ReplyPhraser(FakeLLM([]), s)
    assert phraser.applies("greeting") and not phraser.applies("time.now")
    app.phrase_intents = "all"
    assert PhraseSettings.from_settings(app).intents is None


def test_unknown_long_reply_falls_back():
    joke = "Un uomo entra in un bar e chiede un caffè, ma il barista gli risponde con un indovinello lungo. "
    phraser, _ = _phraser(_tokens(joke))
    text, stats = phraser.phrase("unknown", "raccontami una barzelletta", "Non ho capito.")
    assert text == "Non ho capito." and not stats.phrased


def test_unknown_is_capped_to_one_sentence():
    phraser, _ = _phraser(_tokens("Non è il mio reparto. Ma posso provarci! "))
    text, _ = phraser.phrase("unknown", "cantami una canzone", "Non ho capito.")
    assert text == "Non è il mio reparto."


def test_mood_rotates_and_part_of_day_in_message():
    from datetime import datetime

    phraser = ReplyPhraser(FakeLLM([]), PhraseSettings(), now=lambda: datetime(2026, 11, 19, 22, 43))
    moods = set()
    for _ in range(20):
        msg = phraser.messages("time.now", "che ore sono", "Sono le 22 e 43.")[-1]["content"]
        assert "NOW: sera" in msg
        moods.add(msg.split("MOOD: ")[1].split("\n")[0])
    assert len(moods) > 3


def test_only_current_intent_fewshots_are_sent():
    phraser = ReplyPhraser(FakeLLM([]), PhraseSettings())
    msgs = phraser.messages("time.now", "che ore sono", "Sono le 22 e 43.")
    assert msgs[0]["role"] == "system"
    shots = [m["content"] for m in msgs[1:-1]]
    assert shots and all("INTENT: time.now" in c for c in shots[::2])
    assert [m["role"] for m in msgs[1:-1]] == ["user", "assistant"] * (len(shots) // 2)


def test_meaning_kept_units_and_negation():
    from pawn_assistant.commands.phrasing import meaning_kept

    timer = "Timer di un minuto e 30 secondi avviato."
    assert meaning_kept("Un minuto e 30 secondi, partiti.", timer)
    assert not meaning_kept("30 secondi, partiti.", timer)
    assert meaning_kept("Ti avviso tra un'ora.", "Timer di un'ora avviato.")
    assert meaning_kept("2 ore, partite.", "Timer di 2 ore avviato.")
    assert not meaning_kept("2 minuti, partiti.", "Timer di 2 ore avviato.")
    none = "Ok, nessun timer attivo."
    assert meaning_kept("Niente da fermare.", none)
    assert not meaning_kept("Timer annullato.", none)
    assert not meaning_kept("Barzelletta?", "Non ho capito.")
    assert meaning_kept("That isn't my thing.", "I didn't understand.")
    # no units / negation in the fact → nothing required
    assert meaning_kept("Le 14 e 35.", "Sono le 14 e 35.")
