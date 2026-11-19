import pytest

from pawn_assistant.commands.fastpath import FastPath, extract_duration, normalize, strip_fillers


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Che ora è?") == "che ora e"
    assert normalize("  Che  ore e'  ") == "che ore e"
    assert normalize("Mezz'ora!") == "mezz ora"


def test_strip_fillers():
    assert strip_fillers("ok senti che ora e per favore") == "che ora e"
    assert strip_fillers("ok") == "ok"


@pytest.mark.parametrize(
    "text,intent",
    [
        ("Che ora è?", "time.now"),
        ("che ore e'", "time.now"),  # Whisper variant
        ("Che ore sono.", "time.now"),
        ("Jarvis, che ore sono?", "time.now"),  # wake residue
        ("Che giorno è oggi?", "date.today"),
        ("Annulla il timer.", "cancel"),
        ("Stop.", "cancel"),
        ("Ciao!", "greeting"),
        ("what time is it", "time.now"),
    ],
)
def test_phrase_matches(text, intent):
    match = FastPath().match(text)
    assert match is not None and match.intent == intent


@pytest.mark.parametrize(
    "text",
    ["che tempo fa domani", "raccontami una barzelletta", "accendi la luce in cucina", ""],
)
def test_misses_go_to_llm(text):
    assert FastPath().match(text) is None


@pytest.mark.parametrize(
    "text,slots",
    [
        ("Metti un timer di cinque minuti.", {"minutes": 5}),
        ("timer di 10 secondi", {"seconds": 10}),
        ("timer di un'ora e mezza", {"hours": 1, "minutes": 30}),
        ("avvisami tra un minuto e mezzo", {"minutes": 1, "seconds": 30}),
        ("timer di mezz'ora", {"minutes": 30}),
        ("timer di ventotto minuti", {"minutes": 28}),
        ("set a timer for twenty five minutes", {"minutes": 25}),
        ("conto alla rovescia di tre minuti e venti", {"minutes": 3, "seconds": 20}),
        ("timer di un ora e 15", {"hours": 1, "minutes": 15}),
        ("timer di 2 minuti e 10 secondi", {"minutes": 2, "seconds": 10}),
        ("sveglia tra un quarto d'ora", {"minutes": 15}),
        ("timer di tre quarti d'ora", {"minutes": 45}),
    ],
)
def test_timer_slots(text, slots):
    match = FastPath().match(text)
    assert match is not None
    assert match.intent == "timer.set"
    assert match.slots == slots


def test_timer_word_without_duration_is_not_timer_set():
    match = FastPath().match("annulla il timer")
    assert match is not None and match.intent == "cancel"


def test_extract_duration_ignores_unitless_numbers():
    assert extract_duration("timer 5") == {}
