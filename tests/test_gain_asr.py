"""Gain backends, ASR prompt-echo guard, utterance head trim (no audio/models)."""

import numpy as np

from pawn_assistant.audio import gain as gain_mod
from pawn_assistant.audio.capture import AudioCapture
from pawn_assistant.audio.gain import InputGain, alsa_card_from_name, parse_sget
from pawn_assistant.config import Settings
from pawn_assistant.pipeline.asr import _is_prompt_echo
from pawn_assistant.pipeline.loop import WakeLoop

SGET = """Simple mixer control 'Capture',0
  Capabilities: cvolume cswitch
  Limits: Capture 0 - 63
  Front Left: Capture 63 [100%] [30.00dB] [on]
  Front Right: Capture 63 [100%] [30.00dB] [on]
"""


def test_card_from_name():
    assert alsa_card_from_name("HDA Intel PCH: ALC3204 Analog (hw:0,0)") == 0
    assert alsa_card_from_name("hw:2,1") == 2
    assert alsa_card_from_name("pulse") is None


def test_parse_sget():
    assert parse_sget(SGET) == (100.0, 30.0)
    assert parse_sget("nothing here") is None


class _Cap:
    gain_db = 0.0

    class device_info:
        name = "pulse"

    def set_gain_db(self, db):
        self.gain_db = db


def test_software_backend_clamps_and_reports():
    g = InputGain(_Cap())
    assert g.backend == "software"
    st = g.set(99)
    assert st["value"] == gain_mod.SOFT_MAX_DB and st["unit"] == "dB"
    assert g.set(-99)["value"] == gain_mod.SOFT_MIN_DB


def test_alsa_backend_sets_percent(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)

        class P:
            returncode = 0
            stdout = SGET
            stderr = ""

        return P()

    monkeypatch.setattr(gain_mod.shutil, "which", lambda _n: "/usr/bin/amixer")
    monkeypatch.setattr(gain_mod.subprocess, "run", fake_run)
    cap = _Cap()
    cap.device_info = type("D", (), {"name": "X (hw:1,0)"})()
    g = InputGain(cap)
    assert g.backend == "alsa"
    g.set(250)  # clamped to 100
    assert ["amixer", "-c", "1", "-q", "sset", "Capture", "100%"] in calls
    assert g.state()["db"] == 30.0


def test_capture_gain_multiplies_and_clips():
    cap = AudioCapture.__new__(AudioCapture)
    cap._gain_lin = 1.0
    cap.set_gain_db(6.0)
    assert abs(cap._gain_lin - 1.995) < 0.01
    cap.set_gain_db(-6.0)
    assert abs(cap._gain_lin - 0.501) < 0.01


def test_prompt_echo_guard():
    prompt = "Comandi vocali: che ore sono, che giorno è oggi, imposta un timer"
    assert _is_prompt_echo("Che giorno è oggi, imposta un timer", prompt)
    assert not _is_prompt_echo("Che ore sono?", prompt)  # short, real command
    assert not _is_prompt_echo("metti un timer a tre secondi", prompt)


def test_head_trim_drops_beep_region():
    audio = np.arange(16_000 * 2, dtype=np.float32)  # 2 s ramp
    seen = {}
    loop = WakeLoop.__new__(WakeLoop)
    loop.settings = Settings(head_trim_ms=400, silence_end_ms=0)
    loop._utt_audio_16k = lambda: audio
    loop._utterance = []
    loop._silence_ms = 0.0
    loop._set_state = lambda s: None
    loop.debug = None
    loop._emit_event = lambda e: None
    loop._return_to_idle = lambda **k: None
    loop.router = None

    class _ASR:
        def transcribe(self, a, sample_rate):
            seen["first"] = float(a[0])
            seen["n"] = a.size
            raise RuntimeError("stop")

    loop.asr = _ASR()
    import pawn_assistant.pipeline.loop as L

    class _Sync:
        def __init__(self, target, **kw):
            self.target = target

        def start(self):
            self.target()

    orig = L.threading.Thread
    L.threading.Thread = _Sync
    try:
        loop._begin_transcribe()
    finally:
        L.threading.Thread = orig
    assert seen["first"] == 6400.0  # 0.4 s * 16 kHz dropped
    assert seen["n"] == 32_000 - 6_400
