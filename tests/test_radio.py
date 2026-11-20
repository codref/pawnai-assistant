import json
from pathlib import Path

import pytest

from pawn_assistant.commands import CommandContext, CommandRegistry, CommandRouter
from pawn_assistant.commands.builtin.core import CoreCommand
from pawn_assistant.commands.builtin.radio import RadioCommand, name_score
from pawn_assistant.commands.builtin.radio.player import icy_title
from pawn_assistant.commands.builtin.radio.tunein import (
    TuneIn,
    parse_describe,
    parse_search,
    parse_tune,
)

DATA = Path(__file__).parent / "data"


def _json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


class FakeFetch:
    """Search → fixture; Tune → playable except for ids listed in ``blocked``."""

    def __init__(self, blocked=(), locations=None):
        self.blocked = set(blocked)
        self.locations = locations or {}
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        if "Search.ashx" in url:
            return _json("tunein_search.json")
        gid = url.split("id=")[1].split("&")[0]
        if "Tune.ashx" in url:
            return _json("tunein_tune_notcompatible.json" if gid in self.blocked else "tunein_tune.json")
        if "Describe.ashx" in url:
            data = _json("tunein_describe.json")
            data["body"][0]["guide_id"] = gid
            data["body"][0]["location"] = self.locations.get(gid, "Somewhere")
            data["body"][0]["language"] = "Italian" if gid == "s69185" else "English"
            return data
        raise AssertionError(url)


class FakePlayer:
    def __init__(self):
        self.running = False
        self.calls = []
        self.binary = "mpv"

    def available(self, _binary="mpv"):
        return True

    def play(self, url, *, volume):
        self.running = True
        self.calls.append(("play", url, volume))

    def stop(self):
        self.running = False
        self.calls.append(("stop",))

    def set_volume(self, v):
        self.calls.append(("set_volume", v))

    def ramp_to(self, v, ms=300):
        self.calls.append(("ramp", v))


def _radio(cfg=None, fetch=None):
    radio = RadioCommand()
    radio.tunein = TuneIn(fetch=fetch or FakeFetch())
    radio.player = FakePlayer()
    reg = CommandRegistry([CoreCommand, radio], config={"radio": cfg or {}})
    ctx = CommandContext()
    events = []
    ctx.ui.emit = events.append
    router = CommandRouter(ctx=ctx, registry=reg)
    return radio, router, events


# --------------------------------------------------------------------- tunein parsing


def test_parse_search_keeps_stations_only():
    stations = parse_search(_json("tunein_search.json"))
    assert [s.guide_id for s in stations] == ["s307295", "s69185"]
    assert stations[0].now_playing == "Queen - Bohemian Rhapsody"
    assert stations[0].logo.startswith("https://")  # upgraded from http


def test_parse_tune_prefers_https_and_drops_placeholder():
    assert parse_tune(_json("tunein_tune.json"))[0].startswith("https://")
    assert parse_tune(_json("tunein_tune_notcompatible.json")) == []


def test_parse_describe():
    info = parse_describe(_json("tunein_describe.json"), "s69185")
    assert info.name == "Virgin Radio Italy" and info.genre == "Rock"
    assert info.current_song == ""  # null → empty


def test_icy_title_any_case():
    assert icy_title({"icy-title": "A - B"}) == "A - B"
    assert icy_title({"ICY-TITLE": "C"}) == "C"
    assert icy_title(None) == ""


# --------------------------------------------------------------------- fast path


@pytest.mark.parametrize(
    "text,slots",
    [
        ("Metti Virgin Radio.", {"station": "virgin radio"}),
        ("Jarvis, sintonizzati su radio 105", {"station": "radio 105"}),
        ("accendi la radio", {}),
        ("metti un po' di musica", {}),
        ("Radio Capital!", {"station": "radio capital"}),
        ("play bbc radio 1", {"station": "bbc radio 1"}),
    ],
)
def test_fastpath_radio_play(text, slots):
    _, router, _ = _radio()
    res = router.classify(text)
    assert (res.intent, res.slots, res.source) == ("radio.play", slots, "fastpath")


def test_fastpath_preset_without_radio_word():
    _, router, _ = _radio({"presets": {"deejay": "s114685"}})
    assert router.classify("metti deejay").slots == {"station": "deejay"}
    _, router, _ = _radio()
    assert router.classify("metti deejay").intent != "radio.play"


@pytest.mark.parametrize("text,intent", [("spegni la radio", "radio.stop"), ("cosa sta suonando", "radio.info")])
def test_fastpath_other_radio_intents(text, intent):
    _, router, _ = _radio()
    assert router.classify(text).intent == intent


# --------------------------------------------------------------------- handlers


def test_play_skips_geo_blocked_and_opens_widget():
    radio, router, events = _radio(fetch=FakeFetch(blocked={"s307295"}))
    res = router.handle("metti virgin radio")
    assert res.reply == "Metto Virgin Radio Italy."
    assert radio.player.calls[0] == ("play", "https://icecast.unitedradio.it/Virgin.mp3", 70.0)
    opened = [e for e in events if e["op"] == "open"]
    assert opened and opened[0]["kind"] == "radio" and opened[0]["data"]["name"] == "Virgin Radio Italy"
    assert radio.is_active()


def test_prefer_reorders_hits():
    fetch = FakeFetch(locations={"s69185": "Lombardy"})
    radio, router, _ = _radio({"prefer": ["Italian"]}, fetch=fetch)
    assert router.handle("metti virgin radio").reply == "Metto Virgin Radio Italy."


def test_generic_stop_stops_radio_and_closes_widget():
    radio, router, events = _radio()
    router.handle("metti virgin radio")
    res = router.handle("stop")
    assert res.intent == "cancel" and res.reply == "Radio spenta."
    assert not radio.is_active()
    assert events[-1] == {"type": "ui", "op": "close", "id": "radio:main"}


def test_no_station_and_no_history():
    _, router, _ = _radio()
    assert router.handle("accendi la radio").reply == "Quale stazione?"


def test_replays_last_station(tmp_path):
    radio = RadioCommand()
    radio.tunein = TuneIn(fetch=FakeFetch())
    radio.player = FakePlayer()
    reg = CommandRegistry([CoreCommand, radio], state_path=tmp_path / "s.json")
    router = CommandRouter(ctx=CommandContext(), registry=reg)
    router.handle("metti virgin radio")
    router.handle("spegni la radio")
    assert router.handle("accendi la radio").reply == "Metto Virgin Radio Legends."


def test_ducking_follows_loop_state():
    radio, router, events = _radio({"volume": 60, "duck_volume": 10})
    radio.on_state("AWAKE")  # before playback: remembered, no player call
    assert radio.player.calls == []
    router.handle("metti virgin radio")
    assert radio.player.calls[-1][2] == 10  # started ducked (still in a turn)
    radio.on_state("IDLE")
    assert radio.player.calls[-1] == ("ramp", 60.0)
    radio.on_state("STIR")  # no duck on STIR: the radio itself nudges the wake score
    assert radio.player.calls[-1] == ("ramp", 60.0)
    radio.on_state("LISTEN")
    assert radio.player.calls[-1] == ("ramp", 10.0)
    assert events[-1]["data"] == {"ducked": True}


def test_ui_volume_and_stop():
    radio, router, events = _radio()
    router.handle("metti virgin radio")
    radio.on_ui_action("radio:main", "set_volume", {"value": 33})
    assert radio.player.calls[-1] == ("set_volume", 33.0)
    radio.on_ui_action("radio:main", "stop", {})
    assert not radio.is_active()


# --------------------------------------------------------------------- name matching


@pytest.mark.parametrize(
    "query,name,ok",
    [
        ("bbc radio 2", "BBC World Service West Africa", False),  # TuneIn's top hit
        ("bbc radio 2", "BBC Radio 1", False),  # numbers must match
        ("bbc radio 2", "BBC Radio 2", True),
        ("bbc radio two", "BBC Radio 2", True),  # number words
        ("virgin radio", "Virgin Radio Italy", True),
        ("deejay", "Radio Deejay", True),
        ("radio capital", "Capital Anthems", False),  # only the generic word left
        ("rtl 102.5", "bel RTL", False),
    ],
)
def test_name_score(query, name, ok):
    assert (name_score(query, name) >= 0.8) is ok


def test_unmatched_search_is_refused_not_played():
    radio, router, events = _radio()
    res = router.handle("metti bbc radio 2")  # fixture only has Virgin stations
    assert res.reply == "Non trovo quella stazione."
    assert radio.player.calls == [] and events == []


def test_preset_bypasses_search():
    fetch = FakeFetch()
    radio, router, _ = _radio({"presets": {"bbc radio 2": "s24940"}}, fetch=fetch)
    router.handle("metti bbc radio 2")
    assert radio.player.calls and radio.player.calls[0][0] == "play"
    assert not any("Search.ashx" in u for u in fetch.urls)
