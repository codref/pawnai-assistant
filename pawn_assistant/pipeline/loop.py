"""Wake → beep → VAD → ASR state machine with meter/event fan-out."""

from __future__ import annotations

import logging
import queue
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import psutil

from pawn_assistant.audio.beep import play_beep
from pawn_assistant.audio.tts import PiperTTS
from pawn_assistant.commands import CommandContext, CommandRouter, TimerManager
from pawn_assistant.commands.handlers import timer_alert_text
from pawn_assistant.commands.llm import IntentLLM
from pawn_assistant.commands.phrasing import (
    ALERT_INTENT,
    PhraseFailed,
    PhraseSettings,
    PhraseStats,
    ReplyPhraser,
)
from pawn_assistant.audio.capture import AudioCapture, resample_mono
from pawn_assistant.audio.gain import InputGain
from pawn_assistant.audio.debug_dump import DebugAudioDump
from pawn_assistant.config import Settings
from pawn_assistant.pipeline.asr import WhisperASR
from pawn_assistant.pipeline.vad import SileroVAD
from pawn_assistant.pipeline.wake import WakeDetector

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]


class State(str, Enum):
    IDLE = "IDLE"
    STIR = "STIR"
    AWAKE = "AWAKE"
    LISTEN = "LISTEN"
    HOLD = "HOLD"
    BUSY = "BUSY"
    THINK = "THINK"
    SPEAK = "SPEAK"
    FAULT = "FAULT"


def _dbfs(samples: np.ndarray) -> tuple[float, float]:
    if samples.size == 0:
        return -120.0, -120.0
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak_db = 20.0 * np.log10(peak) if peak > 1e-9 else -120.0
    rms_db = 20.0 * np.log10(rms) if rms > 1e-9 else -120.0
    return rms_db, peak_db


# States in which the mic is ignored (worker thread owns the turn).
_BUSY_STATES = (State.BUSY, State.THINK, State.SPEAK)


class WakeLoop:
    """Background audio pipeline that publishes meters and events."""

    def __init__(
        self,
        settings: Settings,
        on_event: Optional[EventCallback] = None,
        on_meter: Optional[EventCallback] = None,
    ) -> None:
        self.settings = settings
        self.on_event = on_event or (lambda _e: None)
        self.on_meter = on_meter or (lambda _m: None)

        self.capture = AudioCapture(
            sample_rate=settings.sample_rate,
            frame_samples=settings.frame_samples,
            device=settings.device,
        )
        self.gain = InputGain(self.capture, control=settings.gain_control)
        self.wake = WakeDetector(settings.wake_model)
        self.vad = SileroVAD(threshold=settings.vad_threshold)
        self.asr = WhisperASR(
            model_size=settings.whisper_model,
            compute_type=settings.whisper_compute_type,
            idle_unload_s=settings.whisper_idle_unload_s,
            language=settings.asr_language,
            initial_prompt=settings.asr_initial_prompt,
            vad_filter=settings.asr_vad_filter,
        )
        # Point 2 — commands + spoken replies.
        self._alerts: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self.llm: Optional[IntentLLM] = None
        self.router: Optional[CommandRouter] = None
        if settings.intent_enabled:
            self.llm = IntentLLM(
                model=settings.intent_model,
                n_threads=settings.intent_threads,
                idle_unload_s=settings.intent_idle_unload_s,
            )
            self.router = CommandRouter(
                ctx=CommandContext(
                    lang=settings.reply_language,
                    timers=TimerManager(on_fire=self._alerts.put),
                ),
                llm=self.llm,
                fastpath_ratio=settings.intent_fastpath_ratio,
            )
        self.phraser: Optional[ReplyPhraser] = None
        if self.llm is not None and settings.phrase_enabled:
            self.phraser = ReplyPhraser(self.llm, PhraseSettings.from_settings(settings))
            self.llm.add_warmup(self.phraser.warmup)
        self.tts: Optional[PiperTTS] = None
        if settings.tts_enabled:
            self.tts = PiperTTS(
                voice=settings.tts_voice,
                length_scale=settings.tts_length_scale,
                volume=settings.tts_volume,
                idle_unload_s=settings.tts_idle_unload_s,
                speaker_id=settings.tts_speaker_id,
            )

        self.debug: Optional[DebugAudioDump] = None
        if settings.debug_audio_dir:
            self.debug = DebugAudioDump(
                Path(settings.debug_audio_dir),
                sample_rate=settings.sample_rate,
                preroll_s=settings.debug_preroll_s,
            )

        self._state = State.IDLE
        self._state_since = time.monotonic()
        self._thread: Optional[threading.Thread] = None
        self._asr_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake_mute_until = 0.0
        self._wake_score = 0.0
        self._vad_prob = 0.0
        self._silence_ms = 0.0
        self._heard_speech = False
        self._speech_run_ms = 0.0
        self._speech_total_ms = 0.0
        self._listen_since = 0.0
        self._utterance: list[np.ndarray] = []
        self._last_rms = -120.0
        self._last_peak = -120.0
        self._last_frame_at = 0.0
        self._last_near_miss_dump = 0.0
        self._process = psutil.Process()
        self._meter_interval = 1.0 / settings.meter_hz
        self._next_meter = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        return self._state

    def snapshot_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "device_name": self.capture.device_info.name,
                "sample_rate": self.settings.sample_rate,
                "wake_model": self.settings.wake_model,
                "wake_threshold": self.settings.wake_threshold,
                "whisper_resident": self.asr.resident,
                "llm_resident": self.llm_resident,
                "tts_resident": self.tts_resident,
                "overruns": self.capture.overruns,
                "rss_mb": self._rss_mb(),
                "debug_audio": str(self.debug.directory) if self.debug else None,
            }

    @property
    def llm_resident(self) -> bool:
        return bool(self.llm and self.llm.resident)

    @property
    def tts_resident(self) -> bool:
        return bool(self.tts and self.tts.resident)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        try:
            self.wake.load()
            self.vad.load()
            self.capture.start()
        except Exception as exc:  # noqa: BLE001
            self._set_state(State.FAULT)
            self._emit_event({"type": "event", "name": "fault", "message": str(exc)})
            raise
        self._apply_startup_gain()
        self._last_frame_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="pawn-wake-loop", daemon=True)
        self._thread.start()
        # Prefill the intent prompt once at boot; its KV state then survives
        # idle unloads, so later reloads cost ~1 s instead of several.
        self._prefetch_models()
        logger.info(
            "wake loop started device=%r model=%s threshold=%.2f debug=%s",
            self.capture.device_info.name,
            self.settings.wake_model,
            self.settings.wake_threshold,
            self.settings.debug_audio_dir,
        )

    def _apply_startup_gain(self) -> None:
        cfg = self.settings
        if self.gain.backend == "alsa":
            if cfg.input_gain_percent is not None:
                self.gain.set(cfg.input_gain_percent)
        elif cfg.software_gain_db:
            self.gain.set(cfg.software_gain_db)
        logger.info("input gain %s", self.gain.state())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._asr_thread and self._asr_thread.is_alive():
            self._asr_thread.join(timeout=5.0)
        self.capture.stop()

    def _rss_mb(self) -> float:
        try:
            return self._process.memory_info().rss / (1024.0 * 1024.0)
        except Exception:  # noqa: BLE001
            return 0.0

    def _set_state(self, state: State) -> None:
        if state == self._state:
            return
        prev = self._state
        self._state = state
        self._state_since = time.monotonic()
        logger.info("state %s → %s", prev.value, state.value)
        self._emit_event(
            {
                "type": "event",
                "name": "state",
                "state": state.value,
                "since_ms": 0,
            }
        )

    def _emit_event(self, payload: dict[str, Any]) -> None:
        try:
            self.on_event(payload)
        except Exception:  # noqa: BLE001
            logger.exception("on_event failed")

    def _emit_meter(self) -> None:
        now = time.monotonic()
        if now < self._next_meter:
            return
        self._next_meter = now + self._meter_interval
        since_ms = int((now - self._state_since) * 1000)
        payload = {
            "type": "meter",
            "rms_db": round(self._last_rms, 1),
            "peak_db": round(self._last_peak, 1),
            "wake_score": round(self._wake_score, 3),
            "wake_threshold": self.settings.wake_threshold,
            "vad_prob": round(self._vad_prob, 3),
            "silence_ms": int(self._silence_ms),
            "state": self._state.value,
            "since_ms": since_ms,
            "rss_mb": round(self._rss_mb(), 1),
            "overruns": self.capture.overruns,
            "queue_depth": self.capture.queue_depth,
            "whisper_resident": self.asr.resident,
            "llm_resident": self.llm_resident,
            "tts_resident": self.tts_resident,
            "device_name": self.capture.device_info.name,
            "sample_rate": self.settings.sample_rate,
            "stream_rate": self.capture.device_info.stream_rate,
        }
        try:
            self.on_meter(payload)
        except Exception:  # noqa: BLE001
            logger.exception("on_meter failed")

    def _is_speech_frame(self, frame: np.ndarray, *, allow_energy: bool = True) -> bool:
        self._vad_prob = self.vad.prob(frame)
        if self._vad_prob >= self.settings.vad_threshold:
            return True
        if not allow_energy:
            return False
        # Energy fallback only to *start* an utterance (Silero shy on hot mics).
        # Do not use it to keep LISTEN open — room noise would never end the turn.
        _rms, peak_db = _dbfs(frame)
        return peak_db >= self.settings.energy_speech_db

    def _return_to_idle(self, *, debounce_ms: Optional[float] = None) -> None:
        self._utterance = []
        self._silence_ms = 0.0
        self._heard_speech = False
        self._speech_run_ms = 0.0
        self._speech_total_ms = 0.0
        self._listen_since = 0.0
        self._vad_prob = 0.0
        self._wake_score = 0.0
        self.capture.stop_utt_record()  # discard any in-flight native buffer
        self.capture.drain()
        self.wake.reset()
        self.vad.reset()
        wait = (
            self.settings.wake_debounce_ms if debounce_ms is None else debounce_ms
        )
        self._wake_mute_until = time.monotonic() + (wait / 1000.0)
        self._set_state(State.IDLE)

    def _utt_audio_16k(self) -> np.ndarray:
        """Prefer native-rate capture (no callback drops), resample once for ASR."""
        native, native_rate = self.capture.stop_utt_record()
        if native.size > 0:
            audio = resample_mono(native, native_rate, self.settings.sample_rate)
            if audio.size > 0:
                return audio
        if self._utterance:
            return np.concatenate(self._utterance)
        return np.zeros(0, dtype=np.float32)

    def _run(self) -> None:
        frame_ms = self.settings.frame_ms
        while not self._stop.is_set():
            frame = self.capture.read(timeout=0.5)
            if frame is None:
                if (
                    self._last_frame_at
                    and time.monotonic() - self._last_frame_at > 2.0
                    and self._state not in _BUSY_STATES + (State.FAULT,)
                ):
                    logger.warning("no audio frames for 2s — restarting capture")
                    try:
                        self.capture.stop()
                        self.capture.start()
                        self._last_frame_at = time.monotonic()
                        self.wake.reset()
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("capture restart failed: %s", exc)
                        self._set_state(State.FAULT)
                        self._emit_event(
                            {"type": "event", "name": "fault", "message": str(exc)}
                        )
                self._emit_meter()
                continue
            self._last_frame_at = time.monotonic()
            if self.debug is not None:
                self.debug.push(frame)
            try:
                self._handle_frame(frame, frame_ms)
            except Exception as exc:  # noqa: BLE001
                logger.exception("frame handler failed")
                self._set_state(State.FAULT)
                self._emit_event({"type": "event", "name": "fault", "message": str(exc)})
                self._utterance.clear()
                time.sleep(0.5)
                self._return_to_idle()
            self._maybe_play_alert()
            self._emit_meter()

    def _handle_frame(self, frame: np.ndarray, frame_ms: float) -> None:
        self._last_rms, self._last_peak = _dbfs(frame)
        muted = time.monotonic() < self._wake_mute_until

        if self._state in _BUSY_STATES:
            # Includes SPEAK: no AEC yet, so never score PAWN's own voice.
            self._wake_score = 0.0
            return

        if self._state in (State.IDLE, State.STIR):
            if muted:
                self._wake_score = 0.0
            else:
                self._wake_score = self.wake.score(frame)
                self._maybe_dump_near_miss()
            if self._wake_score >= self.settings.wake_threshold:
                self._on_wake()
            elif self._wake_score >= self.settings.stir_threshold:
                self._set_state(State.STIR)
            else:
                self._set_state(State.IDLE)
            return

        if self._state == State.AWAKE:
            self._set_state(State.LISTEN)
            self.vad.reset()
            self._utterance = [frame]
            self._silence_ms = 0.0
            self._speech_run_ms = 0.0
            self._speech_total_ms = 0.0
            self._heard_speech = False
            self._listen_since = time.monotonic()
            return

        if self._state in (State.LISTEN, State.HOLD):
            armed = (
                time.monotonic() - self._listen_since
                >= self.settings.post_wake_arm_ms / 1000.0
            )
            # Before arm: Silero only (skip energy — wake-word echo is loud).
            # After speech latched: VAD-only so room noise cannot prolong the turn.
            allow_energy = armed and not self._heard_speech
            speech = self._is_speech_frame(frame, allow_energy=allow_energy)
            if not armed:
                # Still fill VAD meters, but never latch speech on the wake tail.
                speech = False
            self._utterance.append(frame)
            utterance_ms = len(self._utterance) * frame_ms

            if speech:
                self._speech_run_ms += frame_ms
                self._silence_ms = 0.0
                if (
                    not self._heard_speech
                    and self._speech_run_ms >= self.settings.speech_start_ms
                ):
                    self._heard_speech = True
                    logger.info(
                        "speech start after %.0f ms listen",
                        utterance_ms,
                    )
                if self._heard_speech:
                    self._speech_total_ms += frame_ms
                    self._set_state(State.LISTEN)
            elif self._heard_speech:
                self._speech_run_ms = 0.0
                self._silence_ms += frame_ms
                self._set_state(State.HOLD)
            else:
                self._speech_run_ms = 0.0
                self._silence_ms = 0.0
                self._set_state(State.LISTEN)

            if not self._heard_speech and utterance_ms >= self.settings.speech_wait_ms:
                logger.info("no speech after wake — back to idle")
                if self.debug is not None:
                    audio = self._utt_audio_16k()
                    self.debug.save(
                        "empty_utt",
                        audio,
                        meta=(
                            f"reason=no_speech_wait vad={self._vad_prob:.3f} "
                            f"peak_db={self._last_peak:.1f} "
                            f"overruns={self.capture.overruns}"
                        ),
                    )
                    self.debug.save(
                        "empty_preroll",
                        meta=f"reason=no_speech_wait score_at_wake see wake_*.wav",
                    )
                self._emit_event(
                    {
                        "type": "event",
                        "name": "transcript",
                        "text": "",
                        "duration_ms": 0,
                        "infer_ms": 0,
                        "rtf": 0,
                        "reason": "no_speech",
                    }
                )
                self._return_to_idle()
                return

            can_end = (
                self._heard_speech
                and self._speech_total_ms >= self.settings.min_speech_ms
                and self._silence_ms >= self.settings.silence_end_ms
            )
            if can_end or utterance_ms >= self.settings.max_utterance_ms:
                self._begin_transcribe()
            return

        self._wake_score = 0.0

    def _maybe_dump_near_miss(self) -> None:
        if self.debug is None:
            return
        thr = self.settings.debug_wake_score_dump
        if self._wake_score < thr:
            return
        now = time.monotonic()
        if now - self._last_near_miss_dump < 2.0:
            return
        self._last_near_miss_dump = now
        self.debug.save(
            f"near_miss_{self._wake_score:.2f}",
            meta=(
                f"wake_score={self._wake_score:.3f} "
                f"threshold={self.settings.wake_threshold}"
            ),
        )

    def _on_wake(self) -> None:
        logger.info("wake detected score=%.3f", self._wake_score)
        if self.debug is not None:
            self.debug.save(
                f"wake_{self._wake_score:.2f}",
                meta=(
                    f"wake_score={self._wake_score:.3f} "
                    f"threshold={self.settings.wake_threshold} "
                    f"device={self.capture.device_info.name}"
                ),
            )
        self._set_state(State.AWAKE)
        self._emit_event({"type": "event", "name": "beep"})
        play_beep(
            sample_rate=self.settings.sample_rate,
            freq_hz=self.settings.beep_freq_hz,
            duration_ms=self.settings.beep_duration_ms,
            volume=self.settings.beep_volume,
        )
        self._wake_mute_until = time.monotonic() + (
            self.settings.beep_mute_ms / 1000.0
        )
        self.wake.reset()
        self._utterance = []
        self._silence_ms = 0.0
        self._speech_run_ms = 0.0
        self._speech_total_ms = 0.0
        self._heard_speech = False
        self._vad_prob = 0.0
        # Drop wake-word audio still sitting in the convert queue so it cannot
        # latch "heard speech" and burn the silence budget before you talk.
        self.capture.drain()
        self.capture.start_utt_record()
        self._prefetch_models()

    def _prefetch_models(self) -> None:
        """Load LLM (+ prompt prefix) and voice while the user is still talking."""
        jobs = []
        if self.llm is not None and not self.llm.resident:
            jobs.append(self.llm.warmup)
        if self.tts is not None and not self.tts.resident:
            jobs.append(self.tts.load)
        if not jobs:
            return

        def _run() -> None:
            for job in jobs:
                try:
                    job()
                except Exception:  # noqa: BLE001
                    logger.exception("model prefetch failed")

        threading.Thread(target=_run, name="pawn-prefetch", daemon=True).start()

    def _begin_transcribe(self) -> None:
        audio = self._utt_audio_16k()
        self._utterance = []
        self._silence_ms = 0.0
        self._set_state(State.BUSY)

        trim = int(
            self.settings.sample_rate * (self.settings.silence_end_ms / 1000.0) * 0.5
        )
        if trim > 0 and audio.size > trim:
            audio = audio[:-trim]

        # Head: wake-beep leakage from speaker to mic.
        head = int(self.settings.sample_rate * self.settings.head_trim_ms / 1000.0)
        min_keep = int(self.settings.sample_rate * 0.3)
        if head > 0 and audio.size > head + min_keep:
            audio = audio[head:]

        dur_s = audio.size / float(self.settings.sample_rate) if audio.size else 0.0
        if self.debug is not None:
            self.debug.save(
                "utt",
                audio,
                meta=(
                    f"samples={audio.size} dur_s={dur_s:.2f} "
                    f"peak={float(np.max(np.abs(audio))) if audio.size else 0:.3f} "
                    f"overruns={self.capture.overruns} "
                    f"stream_rate={self.capture.device_info.stream_rate}"
                ),
            )

        def _worker() -> None:
            try:
                if audio.size < self.settings.sample_rate * 0.15:
                    self._emit_event(
                        {
                            "type": "event",
                            "name": "transcript",
                            "text": "",
                            "duration_ms": 0,
                            "infer_ms": 0,
                            "rtf": 0,
                            "reason": "too_short",
                        }
                    )
                else:
                    result = self.asr.transcribe(
                        audio, sample_rate=self.settings.sample_rate
                    )
                    self._emit_event(
                        {
                            "type": "event",
                            "name": "transcript",
                            "text": result.text,
                            "duration_ms": round(result.duration_ms, 1),
                            "infer_ms": round(result.infer_ms, 1),
                            "rtf": round(result.rtf, 3),
                        }
                    )
                    if result.text:
                        self._run_command(result.text)
            except Exception as exc:  # noqa: BLE001
                logger.exception("ASR failed")
                self._emit_event(
                    {"type": "event", "name": "fault", "message": f"ASR: {exc}"}
                )
            finally:
                self._return_to_idle()

        self._asr_thread = threading.Thread(
            target=_worker, name="pawn-asr", daemon=True
        )
        self._asr_thread.start()

    # ------------------------------------------------------------------ point 2

    def _run_command(self, text: str) -> None:
        """THINK → intent → SPEAK reply. Runs on the ASR worker thread."""
        if self.router is None:
            return
        self._set_state(State.THINK)
        result = self.router.handle(text)
        self._emit_event(result.as_event())
        self._reply(result.intent, text, result.reply)

    def _reply(self, intent: str, user_text: str, canonical: str, *, kind: str = "reply") -> None:
        """Speak ``canonical``, reworded by the LLM when phrasing applies.

        Sentences are synthesized and played as they arrive while the model
        keeps generating; if no valid first sentence comes in time, the
        canonical text is spoken instead.
        """
        if self.phraser is None or not self.phraser.applies(intent) or not canonical:
            self._say(canonical, kind=kind)
            return
        self._set_state(State.SPEAK)
        stats = PhraseStats()
        spoken: list[str] = []
        try:
            for sentence in self.phraser.sentences(intent, user_text, canonical, stats):
                spoken.append(sentence)
                self._say(
                    sentence,
                    kind=kind,
                    full=" ".join(spoken),
                    seq=len(spoken) - 1,
                    phrased=True,
                    first_ms=round(stats.first_ms, 1),
                )
        except PhraseFailed as exc:
            stats.reason = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("reply phrasing failed")
            stats.reason = f"error: {exc}"
        if not spoken:
            logger.info("phrasing fell back to canonical (%s)", stats.reason)
            self._say(canonical, kind=kind, phrased=False)
        else:
            logger.info(
                "phrased %r → %r (first %.0f ms, total %.0f ms%s)",
                canonical, " ".join(spoken), stats.first_ms, stats.gen_ms,
                f", {stats.reason}" if stats.reason else "",
            )

    def _say(
        self,
        text: str,
        *,
        kind: str = "reply",
        full: Optional[str] = None,
        seq: int = 0,
        **extra: Any,
    ) -> None:
        """Synthesize + play ``text``; the event carries ``full`` (reply so far)."""
        if not text:
            return
        self._set_state(State.SPEAK)
        payload = {
            "type": "event",
            "name": "reply",
            "kind": kind,
            "text": full if full is not None else text,
            "seq": seq,
            "tts_ms": 0.0,
            **extra,
        }
        if self.tts is None:
            self._emit_event(payload)
            return
        try:
            speech = self.tts.synthesize(text)
            payload["tts_ms"] = round(speech.synth_ms, 1)
            self._emit_event(payload)
            from pawn_assistant.audio.beep import play_pcm

            play_pcm(speech.pcm, speech.sample_rate, block=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("TTS failed")
            self._emit_event(payload)
            self._emit_event({"type": "event", "name": "fault", "message": f"TTS: {exc}"})

    def _maybe_play_alert(self) -> None:
        """Speak a fired timer only between turns, never over an active one."""
        if self._state != State.IDLE or self._alerts.empty():
            return
        if self._asr_thread is not None and self._asr_thread.is_alive():
            return
        try:
            label = self._alerts.get_nowait()
        except queue.Empty:
            return
        self._set_state(State.SPEAK)

        def _alert() -> None:
            try:
                for _ in range(3):
                    play_beep(
                        sample_rate=self.settings.sample_rate,
                        freq_hz=self.settings.beep_freq_hz * 1.5,
                        duration_ms=180,
                        volume=self.settings.beep_volume,
                        block=True,
                    )
                self._reply(
                    ALERT_INTENT,
                    "",
                    timer_alert_text(label, self.settings.reply_language),
                    kind="alert",
                )
            finally:
                self._return_to_idle()

        self._asr_thread = threading.Thread(target=_alert, name="pawn-alert", daemon=True)
        self._asr_thread.start()
