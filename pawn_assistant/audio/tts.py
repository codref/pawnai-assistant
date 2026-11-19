"""Piper local TTS with lazy load / idle unload. Playback via ``beep.play_pcm``."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from pawn_assistant.audio.beep import play_pcm

logger = logging.getLogger(__name__)

_VOICE_RE = re.compile(
    r"^(?P<lang>[a-z]{2,3})_(?P<region>[A-Z]{2})-(?P<name>\w+)-(?P<q>x_low|low|medium|high)$"
)


def resolve_voice(voice: str) -> tuple[Path, Path]:
    """Local .onnx path (config alongside), or download a voice id from rhasspy/piper-voices."""
    path = Path(voice).expanduser()
    if path.suffix == ".onnx" and path.is_file():
        return path, Path(str(path) + ".json")
    m = _VOICE_RE.match(voice)
    if not m:
        raise ValueError(
            f"bad Piper voice {voice!r}: expected e.g. it_IT-paola-medium or a .onnx path"
        )
    from pawn_assistant.commands.llm import hf_cached_download

    lang, region, name, quality = m["lang"], m["region"], m["name"], m["q"]
    base = f"{lang}/{lang}_{region}/{name}/{quality}/{voice}"
    model = hf_cached_download("rhasspy/piper-voices", base + ".onnx")
    config = hf_cached_download("rhasspy/piper-voices", base + ".onnx.json")
    return model, config


@dataclass
class Speech:
    pcm: bytes
    sample_rate: int
    synth_ms: float

    @property
    def duration_ms(self) -> float:
        return 1000.0 * (len(self.pcm) // 2) / float(self.sample_rate)


class PiperTTS:
    def __init__(
        self,
        voice: str = "it_IT-paola-medium",
        length_scale: float = 1.0,
        volume: float = 1.0,
        idle_unload_s: float = 300.0,
        speaker_id: Optional[int] = None,
    ) -> None:
        self.voice = voice
        self.length_scale = length_scale
        self.volume = volume
        self.speaker_id = speaker_id
        self.idle_unload_s = idle_unload_s
        self._voice = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._unload_timer: Optional[threading.Timer] = None

    @property
    def resident(self) -> bool:
        return self._voice is not None

    def load(self) -> None:
        with self._lock:
            if self._voice is not None:
                return
            from piper import PiperVoice

            model, config = resolve_voice(self.voice)
            logger.info("loading Piper voice %s", model.name)
            self._voice = PiperVoice.load(model, config_path=config)
            self._last_used = time.monotonic()

    def unload(self) -> None:
        with self._lock:
            if self._voice is None:
                return
            logger.info("unloading Piper voice")
            self._voice = None

    def _schedule_unload(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        if self.idle_unload_s <= 0:
            return

        def _fire() -> None:
            if time.monotonic() - self._last_used >= self.idle_unload_s:
                self.unload()

        self._unload_timer = threading.Timer(self.idle_unload_s, _fire)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def synthesize(self, text: str) -> Speech:
        self.load()
        from piper import SynthesisConfig

        t0 = time.perf_counter()
        with self._lock:
            assert self._voice is not None
            cfg = SynthesisConfig(
                speaker_id=self.speaker_id,
                length_scale=self.length_scale,
                volume=self.volume,
            )
            chunks = list(self._voice.synthesize(text, syn_config=cfg))
            self._last_used = time.monotonic()
        if not chunks:
            return Speech(b"", 22_050, 0.0)
        rate = chunks[0].sample_rate
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        synth_ms = (time.perf_counter() - t0) * 1000.0
        self._schedule_unload()
        return Speech(pcm, rate, synth_ms)

    def speak(self, text: str) -> Speech:
        """Synthesize and play (blocking until playback ends)."""
        speech = self.synthesize(text)
        if speech.pcm:
            play_pcm(speech.pcm, speech.sample_rate, block=True)
        return speech
