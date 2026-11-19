"""Synthesize speech negatives with Piper so the wake model learns "talking ≠ wake".

A model trained only on silence / keyboard / ambient negatives fires on any
voice. Recording hundreds of speech clips by hand is tedious; Piper can produce
them offline in a minute across several voices and speaking rates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pawn_assistant.wake_train.io_wav import resample_mono, write_wav
from pawn_assistant.wake_train.project import WakeProject

logger = logging.getLogger(__name__)

# (voice id, language). en_US-libritts_r-medium is multi-speaker (~900 voices).
DEFAULT_VOICES: tuple[tuple[str, str], ...] = (
    ("it_IT-paola-medium", "it"),
    ("it_IT-riccardo-x_low", "it"),
    ("en_US-libritts_r-medium", "en"),
    ("en_GB-alan-medium", "en"),
)

IT_SENTENCES = (
    "che ora è", "che ore sono", "che giorno è oggi", "metti un timer di cinque minuti",
    "annulla il timer", "stop", "basta così", "ciao come stai", "buongiorno a tutti",
    "accendi la luce in cucina", "che tempo fa domani", "raccontami una barzelletta",
    "hai visto le chiavi della macchina", "passami il pane per favore", "la pasta è pronta",
    "domani andiamo al mare", "ho una riunione alle tre", "dove hai messo il telefono",
    "mi sa che piove", "non trovo il caricabatterie", "ehi", "ehi tu", "ehi ciao",
    "ehi ragazzi", "ehi Paolo", "ehi Paola", "hey", "hey ciao", "pane", "panino",
    "panna", "pomodoro", "paura", "paolo", "ehi mamma", "ehi papà", "e poi", "eppure",
    "ok", "va bene", "allora", "senti un attimo", "scusa", "grazie mille",
    "facciamo una pausa", "pronto chi parla", "aspetta un attimo", "andiamo",
    "ok Google", "Alexa", "hey Siri", "hey Jarvis", "computer",
    "il gatto dorme sul divano", "stasera guardiamo un film", "quanto costa questo",
    "ho fame", "che cosa facciamo oggi", "il treno è in ritardo", "chiudi la finestra",
    "metti un po' di musica", "abbassa il volume", "alza il volume", "ricordami di comprare il latte",
    "uno due tre quattro cinque", "sei sette otto nove dieci", "mamma mia", "dai su",
)

EN_SENTENCES = (
    "what time is it", "set a timer for five minutes", "cancel the timer", "hello there",
    "how are you doing", "turn on the kitchen lights", "what's the weather tomorrow",
    "tell me a joke", "have you seen my keys", "pass me the bread please",
    "dinner is ready", "I have a meeting at three", "where did you put my phone",
    "hey", "hey there", "hey you", "hey guys", "hey Paul", "hey mom", "hey dad",
    "pan", "panda", "pause", "paw", "pen", "pony", "porn", "phone", "pound", "bone",
    "okay", "alright", "listen", "sorry", "thanks a lot", "let's take a break",
    "hold on a second", "let's go", "okay Google", "Alexa", "hey Siri", "hey Jarvis",
    "the cat is sleeping on the couch", "let's watch a movie tonight", "how much is this",
    "I'm hungry", "what are we doing today", "the train is late", "close the window",
    "play some music", "turn the volume down", "remind me to buy milk",
    "one two three four five", "six seven eight nine ten",
)


def _speech_sets(lang: str) -> tuple[str, ...]:
    return IT_SENTENCES if lang == "it" else EN_SENTENCES


def generate_speech_negatives(
    project: WakeProject,
    *,
    count: int = 600,
    voices: tuple[tuple[str, str], ...] = DEFAULT_VOICES,
    category: str = "speech_tts",
    seed: int = 0,
) -> Path:
    """Write ``count`` synthetic speech clips into ``negatives/<category>/``."""
    from piper import PiperVoice, SynthesisConfig

    from pawn_assistant.audio.tts import resolve_voice

    rng = np.random.default_rng(seed)
    out_dir = project.negatives_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for voice_id, lang in voices:
        try:
            model, config = resolve_voice(voice_id)
            voice = PiperVoice.load(model, config_path=config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip voice %s: %s", voice_id, exc)
            continue
        n_speakers = int(getattr(voice.config, "num_speakers", 1) or 1)
        loaded.append((voice_id, lang, voice, n_speakers))
        logger.info("voice %s (%d speakers)", voice_id, n_speakers)
    if not loaded:
        raise RuntimeError("no Piper voices could be loaded")

    phrase_words = {w.lower() for w in project.phrase.split()}
    written = 0
    while written < count:
        voice_id, lang, voice, n_speakers = loaded[written % len(loaded)]
        text = str(rng.choice(_speech_sets(lang)))
        # Never synthesize the wake phrase itself (or its distinctive word).
        if any(w in text.lower().split() for w in phrase_words if w not in {"hey", "ehi", "ok"}):
            continue
        cfg = SynthesisConfig(
            speaker_id=int(rng.integers(0, n_speakers)) if n_speakers > 1 else None,
            length_scale=float(rng.uniform(0.8, 1.3)),
            noise_scale=float(rng.uniform(0.5, 0.9)),
            noise_w_scale=float(rng.uniform(0.6, 1.0)),
        )
        chunks = list(voice.synthesize(text, syn_config=cfg))
        if not chunks:
            continue
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        audio = resample_mono(audio, chunks[0].sample_rate, project.sample_rate)
        # Room-ish level and noise: mic speech peaks well below full scale.
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = audio / peak * float(rng.uniform(0.1, 0.7))
        audio = audio + rng.normal(0.0, float(rng.uniform(1e-4, 2e-3)), audio.size)
        name = f"{voice_id}_{written:04d}.wav"
        write_wav(out_dir / name, np.clip(audio, -1.0, 1.0), project.sample_rate)
        written += 1
    logger.info("wrote %d speech negatives to %s", written, out_dir)
    return out_dir
