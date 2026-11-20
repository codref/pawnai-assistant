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
    llm_resident: bool = False
    tts_resident: bool = False
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


class IntentEvent(BaseModel):
    type: Literal["event"] = "event"
    name: Literal["intent"] = "intent"
    intent: str
    slots: dict = Field(default_factory=dict)
    source: Literal["fastpath", "llm", "none"]
    parse_ms: float


class ReplyEvent(BaseModel):
    type: Literal["event"] = "event"
    name: Literal["reply"] = "reply"
    kind: Literal["reply", "alert"] = "reply"
    text: str
    tts_ms: float = 0.0


class FaultEvent(BaseModel):
    type: Literal["event"] = "event"
    name: Literal["fault"] = "fault"
    message: str


EventMessage = Union[
    StateEvent, BeepEvent, TranscriptEvent, IntentEvent, ReplyEvent, FaultEvent
]


class GainMessage(BaseModel):
    """Server → client on hello and after every ``set_gain``."""

    type: Literal["gain"] = "gain"
    backend: Literal["alsa", "software"]
    label: str
    value: float
    min: float
    max: float
    step: float
    unit: str
    db: Optional[float] = None


class SetGainMessage(BaseModel):
    """Client → server: percent for ``alsa``, dB for ``software``."""

    type: Literal["set_gain"] = "set_gain"
    value: float


class HelloMessage(BaseModel):
    type: Literal["hello"] = "hello"
    wake_model: str
    wake_threshold: float
    device_name: str
    sample_rate: int
    status: dict = Field(default_factory=dict)
    gain: Optional[dict] = None
    commands: list[dict] = Field(default_factory=list)
    assets: dict = Field(default_factory=dict)  # {"js": [...], "css": [...]} plugin widget files
    ui: list[dict] = Field(default_factory=list)  # open windows (UiMessage op=open)
    now_ms: Optional[int] = None  # server clock, for countdown skew


# --------------------------------------------------------------------- command plugins / windows


class UiMessage(BaseModel):
    """Server → client: a command opens/updates/closes a HUD window.

    ``update`` carries only the changed ``data`` keys; ``hello.ui`` replays every
    open window (``op: open``) to late clients.
    """

    type: Literal["ui"] = "ui"
    op: Literal["open", "update", "close"]
    id: str
    kind: Optional[str] = None  # widget renderer, e.g. "countdown", "radio"
    command: Optional[str] = None
    title: Optional[str] = None
    anchor: Optional[str] = None  # tl | tr | bl | br | center
    data: dict = Field(default_factory=dict)


class CommandInfo(BaseModel):
    name: str
    title: str
    description: str
    core: bool
    enabled: bool
    active: bool
    intents: list[dict] = Field(default_factory=list)


class CommandsMessage(BaseModel):
    """Server → all clients after a toggle (also ``hello.commands``)."""

    type: Literal["commands"] = "commands"
    commands: list[CommandInfo]


class SetCommandMessage(BaseModel):
    """Client → server: enable/disable a command plugin (persisted)."""

    type: Literal["set_command"] = "set_command"
    name: str
    enabled: bool


class UiActionMessage(BaseModel):
    """Client → server: a widget control was used (routed to the owning command)."""

    type: Literal["ui_action"] = "ui_action"
    id: str
    action: str
    data: dict = Field(default_factory=dict)
