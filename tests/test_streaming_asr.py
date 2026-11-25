"""Streaming Zipformer backend and its wake-loop wiring (fake sherpa recognizer)."""

import threading

import numpy as np
import pytest

from pawn_assistant.config import Settings, load_settings
from pawn_assistant.pipeline import streaming_asr as S
from pawn_assistant.pipeline.asr import TranscriptResult
from pawn_assistant.pipeline.loop import State, WakeLoop

WORDS = ["CHE", "ORE", "SONO"]


class _Stream:
    def __init__(self):
        self.samples = 0
        self.finished = False

    def accept_waveform(self, sr, samples):
        assert sr == 16_000
        self.samples += len(samples)

    def input_finished(self):
        self.finished = True


class _Recognizer:
    """One word per 0.4 s of audio; decodes whenever a chunk is buffered."""

    def __init__(self):
        self.decoded = {}

    def create_stream(self):
        return _Stream()

    def is_ready(self, stream):
        return self.decoded.get(id(stream), 0) < stream.samples // 6400

    def decode_stream(self, stream):
        self.decoded[id(stream)] = self.decoded.get(id(stream), 0) + 1

    def get_result(self, stream):
        return " ".join(WORDS[: self.decoded.get(id(stream), 0)])


def _asr(**cfg):
    asr = S.ZipformerASR(S.StreamingSettings(**cfg))
    asr._recognizer = _Recognizer()
    return asr


FRAME = np.zeros(1280, dtype=np.float32)


def test_partials_grow_and_final_is_lowercased():
    asr = _asr()
    asr.begin()
    partials = [p for p in (asr.feed(FRAME) for _ in range(15)) if p is not None]
    assert partials == ["che", "che ore", "che ore sono"]
    result = asr.finish()
    assert result.text == "che ore sono"
    assert result.duration_ms == pytest.approx(15 * 80)
    assert not asr.active


def test_model_files_prefer_int8(tmp_path):
    for name in ("encoder-epoch-99.onnx", "encoder-epoch-99.int8.onnx", "decoder.onnx",
                 "joiner.int8.onnx", "joiner.onnx", "tokens.txt"):
        (tmp_path / name).write_text("x")
    files = S.resolve_model_files(S.StreamingSettings(model_dir=str(tmp_path)))
    assert files["encoder"].endswith("encoder-epoch-99.int8.onnx")
    assert files["decoder"].endswith("decoder.onnx")
    assert files["joiner"].endswith("joiner.int8.onnx")


def test_missing_model_files_are_named(tmp_path):
    (tmp_path / "tokens.txt").write_text("x")
    with pytest.raises(FileNotFoundError, match="encoder, decoder, joiner"):
        S.resolve_model_files(S.StreamingSettings(model_dir=str(tmp_path)))


def test_yaml_streaming_section(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("asr:\n  engine: zipformer\n  streaming:\n    model_dir: /m\n    silence_end_ms: 500\n")
    s = load_settings(cfg)
    assert s.asr_engine == "zipformer"
    assert S.StreamingSettings.from_mapping(s.asr_streaming).silence_end_ms == 500


# ------------------------------------------------------------------ loop wiring


class _Whisper:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        return TranscriptResult(self.text, 1000.0, 300.0)


def _loop(stream_text_words=WORDS, **stream_cfg):
    loop = WakeLoop.__new__(WakeLoop)
    loop.settings = Settings(head_trim_ms=160)
    loop.stream_asr = _asr(**stream_cfg)
    loop.asr = _Whisper("whisper text")
    loop.events = []
    loop._emit_event = loop.events.append
    loop._streaming = False
    loop._stream_skip = 0
    return loop


def test_loop_feeds_after_beep_window_and_emits_partials():
    loop = _loop()
    loop._stream_begin()
    assert loop._stream_skip == 2  # 160 ms of beep leakage never decoded
    for _ in range(17):
        loop._stream_feed(FRAME)
    assert loop.stream_asr._stream.samples == 15 * 1280
    assert [e["text"] for e in loop.events if e["name"] == "partial"] == ["che", "che ore", "che ore sono"]
    assert loop._silence_end_ms() == 600
    result, engine = loop._transcribe(np.zeros(16_000, np.float32), streamed=True)
    assert (result.text, engine) == ("che ore sono", "zipformer")
    assert loop.asr.calls == 0


def test_empty_stream_falls_back_to_whisper():
    loop = _loop()
    loop._stream_begin()
    result, engine = loop._transcribe(np.zeros(16_000, np.float32), streamed=True)
    assert (result.text, engine) == ("whisper text", "whisper")


def test_no_fallback_keeps_empty_result():
    loop = _loop(fallback_whisper=False)
    loop._stream_begin()
    result, engine = loop._transcribe(np.zeros(16_000, np.float32), streamed=True)
    assert (result.text, engine) == ("", "zipformer")


def test_load_failure_means_whisper_turn():
    loop = _loop()

    def boom():
        raise RuntimeError("no model")

    loop.stream_asr.begin = boom
    loop._stream_begin()
    assert loop._streaming is False
    assert loop._silence_end_ms() == loop.settings.silence_end_ms
    _, engine = loop._transcribe(np.zeros(16_000, np.float32), streamed=False)
    assert engine == "whisper"
