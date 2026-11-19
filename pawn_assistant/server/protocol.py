"""WebSocket payload shapes (documented for clients; runtime uses dicts)."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class MeterMessage(BaseModel):
    type: Literal["meter"] = "meter"
    rms_db: float
    peak_db: float
    wake_score: float
    wake_threshold: float
    vad_prob: float
    silence_ms: int
    state: str
    since_ms: int = 0
    rss_mb: float
    overruns: int = 0
    whisper_resident: bool = False
    device_name: str = ""
    sample_rate: int = 16_000


class StateEvent(BaseModel):
    type: Literal["event"] = "event"
    name: Literal["state"] = "state"
    state: str
    since_ms: int = 0


class BeepEvent(BaseModel):
    type: Literal["event"] = "event"
    name: Literal["beep"] = "beep"


class TranscriptEvent(BaseModel):
    type: Literal["event"] = "event"
    name: Literal["transcript"] = "transcript"
    text: str
    duration_ms: float
    infer_ms: float
    rtf: float


class FaultEvent(BaseModel):
    type: Literal["event"] = "event"
    name: Literal["fault"] = "fault"
    message: str


EventMessage = Union[StateEvent, BeepEvent, TranscriptEvent, FaultEvent]


class HelloMessage(BaseModel):
    type: Literal["hello"] = "hello"
    wake_model: str
    wake_threshold: float
    device_name: str
    sample_rate: int
    status: dict = Field(default_factory=dict)
