"""Wake → beep → VAD → ASR state machine with meter/event fan-out."""

from __future__ import annotations

import logging
import queue
import re
import subprocess
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import psutil

from pawn_assistant.audio.aec import AECSettings, AECUnavailable, EchoCancel
from pawn_assistant.audio.beep import play_beep, play_pcm
from pawn_assistant.audio.tts import PiperTTS
from pawn_assistant.commands import CommandContext, CommandRegistry, CommandRouter
from pawn_assistant.commands.base import Announcement
from pawn_assistant.commands.llm import IntentLLM
from pawn_assistant.commands.phrasing import (
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
from pawn_assistant.wake_train.harvest import WakeHarvester, resolve_harvest_root

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
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


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

        self.aec_cfg = AECSettings.from_mapping(settings.aec)
        self.aec: Optional[EchoCancel] = None
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
        # Point 2 — command plugins + spoken replies.
        self.ctx = CommandContext(lang=settings.reply_language)
        self.ctx.ui.emit = self._emit_event
        self._alerts = self.ctx.alerts
        self.commands = CommandRegistry(
            config=settings.commands,
            state_path=settings.commands_state_path(),
        )
        self.commands.setup(self.ctx)
        self.llm: Optional[IntentLLM] = None
        self.router: Optional[CommandRouter] = None
        if settings.intent_enabled:
            self.llm = IntentLLM(
                model=settings.intent_model,
                n_threads=settings.intent_threads,
                idle_unload_s=settings.intent_idle_unload_s,
                intents=self.commands.intents(),
            )
            self.router = CommandRouter(
                ctx=self.ctx,
                llm=self.llm,
                fastpath_ratio=settings.intent_fastpath_ratio,
                registry=self.commands,
            )
            llm = self.llm
            self.commands.add_listener(lambda: llm.set_intents(self.commands.intents()))
        self.phraser: Optional[ReplyPhraser] = None
        if self.llm is not None and settings.phrase_enabled:
            self.phraser = ReplyPhraser(
                self.llm, PhraseSettings.from_settings(settings), registry=self.commands
            )
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

        self.harvest: Optional[WakeHarvester] = None
        harvest_root = resolve_harvest_root(settings.wake_harvest, settings.wake_model)
        if harvest_root is not None:
            self.harvest = WakeHarvester(
                harvest_root,
                sample_rate=settings.sample_rate,
                max_per_label=settings.wake_harvest_max,
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
        self._listen_wait_ms = float(settings.speech_wait_ms)
        self._followup = False  # LISTEN opened by a floor holder, not the wake word
        self._utterance: list[np.ndarray] = []
        self._last_rms = -120.0
        self._last_peak = -120.0
        self._last_frame_at = 0.0
        self._last_near_miss_dump = 0.0
        self._process = psutil.Process()
        self._meter_interval = 1.0 / settings.meter_hz
        self._next_meter = 0.0
        self._lock = threading.Lock()
        # Barge-in: set by the audio thread when the wake word cuts PAWN off;
        # doubles as the stop flag for playback on the worker thread.
        self._barge = threading.Event()
        self._barge_hits = 0
        self._speak_since: Optional[float] = None

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
                "aec": self.aec_active,
                "barge_in": self.barge_in_ready,
            }

    @property
    def aec_active(self) -> bool:
        aec = getattr(self, "aec", None)
        return bool(aec and aec.active)

    @property
    def barge_in_ready(self) -> bool:
        s = self.settings
        return bool(s.barge_in_enabled and (self.aec_active or not s.barge_in_require_aec))

    def _harvest(self) -> Optional[WakeHarvester]:
        return getattr(self, "harvest", None)

    def _harvest_resolve(self, label: str, **meta: Any) -> None:
        harvest = self._harvest()
        if harvest is not None:
            harvest.resolve(label, **meta)

    def _clear_barge(self) -> None:
        barge = getattr(self, "_barge", None)
        if barge is not None:
            barge.clear()

    def _barged(self) -> bool:
        barge = getattr(self, "_barge", None)
        return bool(barge is not None and barge.is_set())

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
        self._start_aec()
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

    def _start_aec(self) -> None:
        """Route mic + playback through PipeWire echo-cancel when configured."""
        if not self.aec_cfg.enabled or self.aec_active:
            return
        aec = EchoCancel(self.aec_cfg)
        try:
            aec.start()
        except (AECUnavailable, OSError, subprocess.SubprocessError) as exc:
            logger.warning("echo cancellation unavailable: %s", exc)
            self._emit_event({"type": "event", "name": "aec", "active": False, "message": str(exc)})
            return
        self.aec = aec
        # PULSE_SOURCE now names the echo-cancel source: capture via "pulse".
        self.capture = AudioCapture(
            sample_rate=self.settings.sample_rate,
            frame_samples=self.settings.frame_samples,
            device="pulse",
        )
        self.gain = InputGain(self.capture, control=self.settings.gain_control)
        logger.info("AEC active: capture=%s playback=%s", aec.cfg.source_name, aec.cfg.sink_name)
        self._emit_event({"type": "event", "name": "aec", "active": True, "message": aec.cfg.source_name})

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
        self.commands.shutdown()
        if self.aec is not None:
            self.aec.stop()
            self.aec = None

    # ------------------------------------------------------------------ commands API (server)

    def set_command_enabled(self, name: str, enabled: bool) -> bool:
        return self.commands.set_enabled(name, enabled)

    def ui_action(self, widget_id: str, action: str, data: Optional[dict[str, Any]] = None) -> bool:
        return self.commands.ui_action(widget_id, action, data)

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
        commands = getattr(self, "commands", None)
        if commands is not None:
            commands.on_state(state.value)  # e.g. radio ducking; hooks must not block

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
        self._reset_turn(debounce_ms=debounce_ms)
        self._set_state(State.IDLE)

    def _reset_turn(self, *, debounce_ms: Optional[float] = None) -> None:
        """Clear per-utterance state and audio buffers (no state change)."""
        self._followup = False
        self._listen_wait_ms = float(self.settings.speech_wait_ms)
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
            if self.harvest is not None:
                self.harvest.push(frame)
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
            if self._state == State.SPEAK and self.barge_in_ready:
                self._score_barge_in(frame)
            else:
                # No AEC: never score PAWN's own voice.
                self._speak_since = None
                self._wake_score = 0.0
            return
        self._speak_since = None

        if self._state in (State.IDLE, State.STIR):
            if muted:
                self._wake_score = 0.0
            else:
                self._wake_score = self.wake.score(frame)
                self._maybe_dump_near_miss()
                harvest = self._harvest()
                if harvest is not None:
                    harvest.on_score(
                        self._wake_score,
                        self.settings.stir_threshold,
                        self.settings.wake_threshold,
                    )
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

            if not self._heard_speech and utterance_ms >= self._listen_wait_ms:
                if self._followup:
                    self._on_followup_timeout()
                    return
                logger.info("no speech after wake — back to idle")
                self._harvest_resolve("false_wake", reason="no_speech")
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

    def _score_barge_in(self, frame: np.ndarray) -> None:
        """While PAWN speaks (echo-cancelled), listen for the wake word."""
        now = time.monotonic()
        if self._barge.is_set():
            self._wake_score = 0.0
            return
        if self._speak_since is None:
            # Fresh reply: wake features still hold pre-turn audio (often the
            # wake word itself) — flush them before scoring.
            self._speak_since = now
            self._barge_hits = 0
            self.wake.reset()
        score = self.wake.score(frame)
        armed = now - self._speak_since >= self.settings.barge_in_arm_ms / 1000.0
        self._wake_score = score if armed else 0.0
        if not armed:
            return
        if score >= self.settings.barge_threshold:
            self._barge_hits += 1
        else:
            self._barge_hits = 0
        if self._barge_hits >= max(1, int(self.settings.barge_in_frames)):
            self._on_barge_in(score)

    def _on_barge_in(self, score: float) -> None:
        """Audio thread: cut PAWN off; the worker re-opens the mic when it unwinds."""
        logger.info("barge-in score=%.3f — stopping reply", score)
        self._barge.set()
        self._barge_hits = 0
        if self.debug is not None:
            self.debug.save(
                f"barge_{score:.2f}",
                meta=f"wake_score={score:.3f} threshold={self.settings.barge_threshold}",
            )
        self._emit_event({"type": "event", "name": "barge_in", "score": round(score, 3)})
        harvest = self._harvest()
        if harvest is not None:
            harvest.on_barge(score)

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
        harvest = self._harvest()
        if harvest is not None:
            harvest.on_wake(self._wake_score)
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
                    self._harvest_resolve("false_wake", reason="too_short")
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
                        intent = self._run_command(result.text)
                        label = "unsure" if intent == "unknown" else "confirmed"
                        self._harvest_resolve(label, text=result.text, intent=intent)
                    else:
                        self._harvest_resolve("false_wake", reason="empty_transcript")
            except Exception as exc:  # noqa: BLE001
                logger.exception("ASR failed")
                self._emit_event(
                    {"type": "event", "name": "fault", "message": f"ASR: {exc}"}
                )
            finally:
                self._after_turn()

        self._clear_barge()  # a late barge from the previous reply must not mute this one
        self._asr_thread = threading.Thread(
            target=_worker, name="pawn-asr", daemon=True
        )
        self._asr_thread.start()

    # ------------------------------------------------------------------ point 2

    def _run_command(self, text: str) -> Optional[str]:
        """THINK → intent → SPEAK reply. Runs on the ASR worker thread."""
        if self.router is None:
            return None
        self._set_state(State.THINK)
        result = self.router.handle(text)
        self._emit_event(result.as_event())
        if result.source == "floor":
            self._say_sentences(result.reply)
            return "floor"
        self._reply(result.intent, text, result.reply)
        return result.intent

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
                if self._barged():
                    stats.reason = "barge-in"
                    break
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
        if self._barged():
            return
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
        if not text or self._barged():
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
            if self._barged():
                return
            play_pcm(
                speech.pcm,
                speech.sample_rate,
                block=True,
                stop=getattr(self, "_barge", None),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("TTS failed")
            self._emit_event(payload)
            self._emit_event({"type": "event", "name": "fault", "message": f"TTS: {exc}"})

    def _say_sentences(self, text: str, *, kind: str = "reply") -> None:
        """Speak a remote (already concise) answer one sentence at a time."""
        parts = [p for p in _SENTENCE_SPLIT.split((text or "").strip()) if p.strip()]
        spoken: list[str] = []
        for part in parts:
            if self._barged():
                break
            spoken.append(part.strip())
            self._say(part.strip(), kind=kind, full=" ".join(spoken), seq=len(spoken) - 1)

    # ------------------------------------------------------------ conversation

    def _floor_active(self) -> bool:
        return bool(self.router is not None and self.router.floor_owner() is not None)

    def _after_turn(self) -> None:
        """End of a spoken turn: keep the mic open while a command holds the floor."""
        if self._barged() and not self._stop.is_set():
            self._barge.clear()
            self._barge_listen()
            return
        if not self._floor_active() or self._stop.is_set():
            self._return_to_idle()
            return
        self._follow_up_listen()

    def _follow_up_listen(self) -> None:
        """Re-open the mic without the wake word (conversation mode)."""
        self._reset_turn(debounce_ms=0)
        self._followup = True
        self._listen_wait_ms = max(1000.0, float(self.ctx.floor_listen_ms))
        self._set_state(State.AWAKE)
        self._emit_event({"type": "event", "name": "beep", "kind": "followup"})
        play_beep(
            sample_rate=self.settings.sample_rate,
            freq_hz=self.settings.beep_freq_hz * 1.25,
            duration_ms=max(60, int(self.settings.beep_duration_ms * 0.6)),
            volume=self.settings.beep_volume * 0.6,
        )
        self._wake_mute_until = time.monotonic() + (self.settings.beep_mute_ms / 1000.0)
        self.capture.drain()
        self.capture.start_utt_record()

    def _barge_listen(self) -> None:
        """After a barge-in: same as a fresh wake (beep, LISTEN), minus the score."""
        self._reset_turn(debounce_ms=0)
        if self._floor_active():
            # Mid-conversation: silence after the interruption hands the floor
            # holder its usual timeout instead of dropping to IDLE.
            self._followup = True
            self._listen_wait_ms = max(1000.0, float(self.ctx.floor_listen_ms))
        self._set_state(State.AWAKE)
        self._emit_event({"type": "event", "name": "beep", "kind": "barge"})
        play_beep(
            sample_rate=self.settings.sample_rate,
            freq_hz=self.settings.beep_freq_hz,
            duration_ms=self.settings.beep_duration_ms,
            volume=self.settings.beep_volume,
        )
        self._wake_mute_until = time.monotonic() + (self.settings.beep_mute_ms / 1000.0)
        self.capture.drain()
        self.capture.start_utt_record()
        self._prefetch_models()

    def _on_followup_timeout(self) -> None:
        """No reply in the follow-up window: let the floor holder decide."""
        owner = self.router.floor_owner() if self.router is not None else None
        logger.info("conversation follow-up timed out (owner=%s)", owner and owner.name)
        if owner is not None:
            try:
                owner.on_floor_timeout(self.ctx)
            except Exception:  # noqa: BLE001
                logger.exception("on_floor_timeout failed")
                self.ctx.floor = None
        self._return_to_idle()

    def _maybe_play_alert(self) -> None:
        """Speak a queued plugin alert only between turns, never over an active one."""
        if self._alerts.empty():
            return
        in_followup = (
            self._followup
            and self._state == State.LISTEN
            and not self._heard_speech
        )
        if self._state != State.IDLE and not in_followup:
            return
        if self._asr_thread is not None and self._asr_thread.is_alive():
            return
        if in_followup:
            self.capture.stop_utt_record()
        try:
            alert: Announcement = self._alerts.get_nowait()
        except queue.Empty:
            return
        self._set_state(State.SPEAK)

        def _alert() -> None:
            try:
                for _ in range(max(0, alert.beeps)):
                    if self._barged():
                        break
                    play_beep(
                        sample_rate=self.settings.sample_rate,
                        freq_hz=self.settings.beep_freq_hz * 1.5,
                        duration_ms=180,
                        volume=self.settings.beep_volume,
                        block=True,
                    )
                self._reply(alert.intent, "", alert.text, kind="alert")
            finally:
                self._after_turn()

        self._clear_barge()
        self._asr_thread = threading.Thread(target=_alert, name="pawn-alert", daemon=True)
        self._asr_thread.start()
