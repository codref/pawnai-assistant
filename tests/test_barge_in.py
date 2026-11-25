"""Barge-in (wake word over PAWN's reply) and echo-cancel routing — no audio."""

import threading
import time

import numpy as np
import pytest

from pawn_assistant.audio import aec, routing
from pawn_assistant.config import Settings, load_settings
from pawn_assistant.pipeline.loop import State, WakeLoop


class _Wake:
    def __init__(self, scores):
        self.scores = list(scores)
        self.resets = 0

    def score(self, frame):
        return self.scores.pop(0) if self.scores else 0.0

    def reset(self):
        self.resets += 1


class _Aec:
    active = True


def _loop(scores, **settings):
    loop = WakeLoop.__new__(WakeLoop)  # skip audio/model setup
    loop.settings = Settings(**{"wake_threshold": 0.5, "barge_in_arm_ms": 0, **settings})
    loop.aec = _Aec()
    loop.wake = _Wake(scores)
    loop.debug = None
    loop.events = []
    loop._emit_event = loop.events.append
    loop._state = State.SPEAK
    loop._barge = threading.Event()
    loop._barge_hits = 0
    loop._speak_since = None
    loop._wake_score = 0.0
    loop._wake_mute_until = 0.0
    return loop


FRAME = np.zeros(1280, dtype=np.float32)


def _feed(loop, n):
    for _ in range(n):
        loop._handle_frame(FRAME, 80.0)


def test_two_frames_over_threshold_barge_in():
    loop = _loop([0.7, 0.7])
    _feed(loop, 2)
    assert loop._barge.is_set()
    assert loop.events[-1]["name"] == "barge_in"
    assert loop.wake.resets == 1  # stale features flushed when the reply began


def test_default_threshold_is_above_wake_threshold():
    loop = _loop([0.55, 0.55, 0.55])
    _feed(loop, 3)
    assert loop.settings.barge_threshold == pytest.approx(0.6)
    assert not loop._barge.is_set()


def test_isolated_spike_does_not_barge_in():
    loop = _loop([0.9, 0.1, 0.9, 0.1])
    _feed(loop, 4)
    assert not loop._barge.is_set()


def test_no_aec_means_no_scoring_while_speaking():
    loop = _loop([0.99, 0.99])
    loop.aec = None
    _feed(loop, 2)
    assert not loop._barge.is_set()
    assert loop.wake.scores == [0.99, 0.99]  # wake model never fed PAWN's voice


def test_require_aec_false_allows_barge_in_without_aec():
    loop = _loop([0.99, 0.99], barge_in_require_aec=False)
    loop.aec = None
    _feed(loop, 2)
    assert loop._barge.is_set()


def test_arm_window_ignores_reply_onset():
    loop = _loop([0.99, 0.99, 0.99, 0.99], barge_in_arm_ms=10_000)
    _feed(loop, 4)
    assert not loop._barge.is_set()


def test_barged_turn_stops_remaining_sentences_and_reopens_mic():
    loop = _loop([])
    loop.tts = None
    loop._set_state = lambda s: setattr(loop, "_state", s)
    spoken = []
    real_say = WakeLoop._say

    def _say(self, text, **kw):
        spoken.append(text)
        loop._barge.set()  # wake word heard during the first sentence
        return real_say(self, text, **kw)

    loop._say = _say.__get__(loop)
    loop._say_sentences("Prima frase. Seconda frase. Terza.")
    assert spoken == ["Prima frase."]

    reopened = []
    loop._stop = threading.Event()
    loop._barge_listen = lambda: reopened.append(True)
    loop._after_turn()
    assert reopened == [True]
    assert not loop._barge.is_set()


# ------------------------------------------------------------------ routing


def test_routing_targets_echo_cancel_sink():
    routing.set_output_sink("pawn_ec_sink")
    try:
        assert routing.player_command("/x.wav", "paplay") == ["paplay", "--device=pawn_ec_sink", "/x.wav"]
        assert routing.player_command("/x.wav", "pw-play") == ["pw-play", "--target", "pawn_ec_sink", "/x.wav"]
        assert routing.mpv_audio_args() == ["--ao=pulse", "--audio-device=pulse/pawn_ec_sink"]
    finally:
        routing.set_output_sink(None)
    assert routing.player_command("/x.wav", "paplay") == ["paplay", "/x.wav"]
    assert routing.mpv_audio_args() == []


def test_play_pcm_stop_kills_player(monkeypatch, tmp_path):
    from pawn_assistant.audio import beep

    monkeypatch.setattr(beep.shutil, "which", lambda name: name == "paplay")
    monkeypatch.setattr(routing, "player_command", lambda path, player: ["sleep", "5"])
    stop = threading.Event()
    threading.Timer(0.1, stop.set).start()
    t0 = time.monotonic()
    beep.play_pcm(b"\0\0" * 160, 16_000, block=True, stop=stop)
    assert time.monotonic() - t0 < 2.0


# ------------------------------------------------------------------ pactl parsing


def test_pick_source_master_skips_monitors():
    state = aec.PulseState(
        sources=["out.monitor", "alsa_input.analog", "pawn_ec_source"],
        default_source="out.monitor",
    )
    assert aec.pick_source_master(state, aec.AECSettings()) == "alsa_input.analog"


def test_no_mic_source_explains_profile():
    state = aec.PulseState(sources=["out.monitor"], default_source="out.monitor")
    with pytest.raises(aec.AECUnavailable, match="Duplex"):
        aec.pick_source_master(state, aec.AECSettings())


CARDS = """Card #49
\tName: alsa_card.pci-0000_00_1f.3
\tDriver: alsa
\tProfiles:
\t\toff: Off (sinks: 0, sources: 0, priority: 0, available: yes)
\t\toutput:analog-stereo: Analog Stereo Output (sinks: 1, sources: 0, priority: 6500, available: yes)
\t\toutput:analog-stereo+input:analog-stereo: Analog Stereo Duplex (sinks: 1, sources: 1, priority: 6565, available: yes)
\t\toutput:hdmi-stereo+input:analog-stereo: Digital Stereo (HDMI) Output + Analog Stereo Input (sinks: 1, sources: 1, priority: 5965, available: yes)
\t\toutput:hdmi-stereo-extra1+input:analog-stereo: HDMI 2 (sinks: 1, sources: 1, priority: 5765, available: no)
\tActive Profile: output:analog-stereo
\tPorts:
"""


def test_duplex_profiles_prefers_analog(monkeypatch):
    monkeypatch.setattr(aec, "_pactl", lambda *a, **k: CARDS)
    found = aec.duplex_profiles()
    assert [p for _c, p, _d in found] == [
        "output:analog-stereo+input:analog-stereo",
        "output:hdmi-stereo+input:analog-stereo",
    ]
    assert found[0][2] == "Analog Stereo Duplex"
    assert aec.active_profile("alsa_card.pci-0000_00_1f.3") == "output:analog-stereo"


def test_yaml_sections(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "aec:\n  enabled: true\n  method: webrtc\n"
        "barge_in:\n  threshold: 0.8\n  frames: 3\n"
    )
    s = load_settings(cfg)
    assert aec.AECSettings.from_mapping(s.aec).enabled is True
    assert s.barge_threshold == 0.8
    assert s.barge_in_frames == 3
