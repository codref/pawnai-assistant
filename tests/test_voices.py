from pawn_assistant.audio.voices import (
    VoiceInfo,
    filter_voices,
    parse_catalog,
    sample_text,
    update_yaml_section,
)

CATALOG = {
    "it_IT-paola-medium": {
        "name": "paola",
        "quality": "medium",
        "num_speakers": 1,
        "language": {"code": "it_IT"},
        "audio": {"sample_rate": 22050},
        "files": {"it/x.onnx": {"size_bytes": 63_000_000}, "it/x.onnx.json": {"size_bytes": 5}},
    },
    "en_US-libritts_r-medium": {
        "name": "libritts_r",
        "quality": "medium",
        "num_speakers": 904,
        "language": {"code": "en_US"},
        "files": {},
    },
}


def test_parse_and_filter():
    voices = parse_catalog(CATALOG)
    assert [v.id for v in voices] == ["en_US-libritts_r-medium", "it_IT-paola-medium"]
    it = voices[1]
    assert it.size_mb == 63.0 and it.sample_rate == 22050 and it.lang == "it"
    assert voices[0].speakers == 904
    assert [v.id for v in filter_voices(voices, lang="it")] == ["it_IT-paola-medium"]
    assert filter_voices(voices, lang="en_GB") == []
    voices[0].installed = True
    assert [v.id for v in filter_voices(voices, installed=True)] == ["en_US-libritts_r-medium"]
    assert isinstance(voices[0], VoiceInfo)


def test_sample_text_fallback():
    assert "Pawn" in sample_text("it_IT")
    assert sample_text("xx") == sample_text("en")


YAML = """\
server:
  port: 8765

tts:
  # Piper local voice for replies.
  enabled: true
  voice: it_IT-paola-medium   # default
  length_scale: 1.0

phrase:
  enabled: true
"""


def test_update_in_place(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML)
    changed = update_yaml_section(p, "tts", {"voice": "it_IT-riccardo-x_low", "speaker_id": 2})
    out = p.read_text()
    assert changed == ["voice", "speaker_id"]
    assert "  voice: it_IT-riccardo-x_low   # default\n" in out
    assert "# Piper local voice for replies." in out
    # New key lands inside tts, before the blank line and `phrase:`.
    assert "  length_scale: 1.0\n  speaker_id: 2\n\nphrase:" in out


def test_update_creates_section(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("server:\n  port: 1\n")
    update_yaml_section(p, "tts", {"voice": "en_US-lessac-medium", "length_scale": 1.1})
    import yaml

    assert yaml.safe_load(p.read_text())["tts"] == {
        "voice": "en_US-lessac-medium",
        "length_scale": 1.1,
    }
