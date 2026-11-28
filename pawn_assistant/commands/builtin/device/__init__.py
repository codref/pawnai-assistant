"""Link to pawn-server as a registered assistant: live commands + presence.

When ``pawn-assistant device register`` has written ``device.json`` (or the
YAML ``commands.device`` block names an identity), this plugin:

* keeps ``GET /v1/devices/{id}/commands/stream`` open and runs what arrives:
  ``speak`` (queued spoken alert), ``notify`` (HUD window + speech) and
  ``timer_set`` (a normal countdown through the timer plugin); results are
  posted back so the agent's tool call returns;
* reports presence: ``active`` on each wake/turn, ``idle`` after
  ``idle_after_s``, ``away`` on shutdown, plus a heartbeat.

pawn-server then routes ``assistant_speak`` / ``assistant_timer`` to whichever
assistant (laptop or phone) the user was last near. Protocol: pawnai
``docs/VOICE_PROTOCOL.md``. No intents: nothing here is triggered by voice.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pawn_assistant.commands.base import Command, CommandContext

logger = logging.getLogger(__name__)

SPEAK_INTENT = "device.speak"
NOTIFY_INTENT = "device.notify"
_WAKE_STATES = {"AWAKE", "LISTEN", "THINK", "SPEAK"}


def parse_expiry(raw: Any) -> Optional[float]:
    """Server ``expires_at`` (naive UTC ISO) → epoch seconds."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class DeviceCommand(Command):
    name = "device"
    title = "pawn-server link"
    description = "Registered assistant: runs pawn-server commands (speak, notify, timer) and reports presence."
    order = 90

    def __init__(self) -> None:
        super().__init__()
        self.client: Any = None  # server_api.DeviceClient
        self.wall: Callable[[], float] = time.time
        self.mono: Callable[[], float] = time.monotonic
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._threads: list[threading.Thread] = []
        self.last_activity = 0.0
        self.last_wake_at: Optional[float] = None
        self.reported = ""
        self.connected = False

    # ------------------------------------------------------------------ config

    @property
    def idle_after_s(self) -> float:
        return float(self.config.get("idle_after_s") or 300.0)

    @property
    def heartbeat_s(self) -> float:
        return float(self.config.get("heartbeat_s") or 120.0)

    def setup(self, ctx: CommandContext) -> None:
        if self.config.get("enabled") is False:
            return
        from pawn_assistant.server_api import DeviceClient, load_identity

        ident = load_identity(self.config)
        if ident is None:
            logger.info("device: not registered with pawn-server (pawn-assistant device register)")
            return
        self.client = DeviceClient(ident)
        if self.config.get("start_threads", True):
            self.start()

    def start(self) -> None:
        self._stop.clear()
        for target, name in ((self._stream_loop, "pawn-device-stream"), (self._presence_loop, "pawn-device-presence")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info("device: linked to %s as %s", self.client.ident.server_url, self.client.ident.name)

    # ------------------------------------------------------------------ presence

    def on_state(self, state: str) -> None:
        if state.upper() not in _WAKE_STATES:
            return
        now = self.mono()
        first = now - self.last_activity > 5.0
        self.last_activity = now
        if state.upper() == "AWAKE":
            self.last_wake_at = self.wall()
        if first:
            self._kick.set()  # report "active" soon, without blocking the loop

    def presence_state(self) -> str:
        if self.last_activity and self.mono() - self.last_activity < self.idle_after_s:
            return "active"
        return "idle"

    def report_presence(self, state: Optional[str] = None) -> bool:
        if self.client is None:
            return False
        state = state or self.presence_state()
        flags: dict[str, Any] = {}
        if self.last_wake_at:
            flags["last_wake_at"] = int(self.last_wake_at)
        try:
            self.client.presence(state, **flags)
        except Exception as exc:  # noqa: BLE001 — server down; next beat retries
            logger.debug("device: presence failed: %s", exc)
            return False
        self.reported = state
        return True

    def _presence_loop(self) -> None:
        self.report_presence()
        while not self._stop.is_set():
            self._kick.wait(timeout=min(self.heartbeat_s, 30.0))
            self._kick.clear()
            if self._stop.is_set():
                break
            state = self.presence_state()
            due = self.mono() - getattr(self, "_last_beat", 0.0) >= self.heartbeat_s
            if state != self.reported or due:
                if self.report_presence(state):
                    self._last_beat = self.mono()

    # ------------------------------------------------------------------ commands

    def _stream_loop(self) -> None:
        from pawn_assistant.server_api import ServerError

        backoff = 2.0
        while not self._stop.is_set():
            try:
                self.drain()  # anything queued while we were offline
                for event, data in self.client.stream():
                    if self._stop.is_set():
                        return
                    if event == "ready":
                        self.connected = True
                        backoff = 2.0
                    elif event == "cmd":
                        for cmd in data.get("commands") or []:
                            self.run(cmd)
            except ServerError as exc:
                if exc.status in (401, 403, 404):
                    logger.warning("device: pawn-server rejected this device (%s); re-register it", exc)
                    backoff = 300.0
                else:
                    logger.info("device: command stream down (%s); retry in %.0fs", exc, backoff)
            except Exception:  # noqa: BLE001
                logger.exception("device: command stream failed")
            self.connected = False
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, 60.0)

    def drain(self) -> int:
        cmds = self.client.pending()
        for cmd in cmds:
            self.run(cmd)
        return len(cmds)

    def run(self, cmd: dict[str, Any]) -> str:
        """Run one command and post its result. Returns the status sent."""
        cid = str(cmd.get("id") or "")
        expires = parse_expiry(cmd.get("expires_at"))
        if expires is not None and self.wall() > expires:
            logger.info("device: skipping expired %s %s", cmd.get("tool"), cid)
            return "skipped"
        try:
            status, result, message = self.execute(str(cmd.get("tool") or ""), dict(cmd.get("args") or {}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("device: %s failed", cmd.get("tool"))
            status, result, message = "error", {}, str(exc)[:300]
        try:
            self.client.post_result(cid, status, result, message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("device: could not post result for %s: %s", cid, exc)
        return status

    def execute(self, tool: str, args: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        ctx = self.ctx
        if ctx is None:
            return "error", {}, "assistant not ready"
        if tool == "speak":
            text = str(args.get("text") or "").strip()
            if not text:
                return "error", {}, "empty text"
            ctx.announce(text, intent=SPEAK_INTENT, beeps=1, command=self.name)
            return "ok", {"queued": True}, ""
        if tool == "notify":
            text = str(args.get("text") or "").strip()
            title = str(args.get("title") or "PAWN").strip()[:40]
            if not text:
                return "error", {}, "empty text"
            wid = self.widget_id(f"notify-{int(self.wall() * 1000)}")
            ctx.ui.open(wid, "text", command=self.name, title=title.upper(), anchor="tr", data={"text": text})
            ctx.announce(text, intent=NOTIFY_INTENT, beeps=2, command=self.name)
            return "ok", {"shown": True}, ""
        if tool == "timer_set":
            try:
                seconds = int(float(args.get("seconds") or 0))
            except (TypeError, ValueError):
                return "error", {}, "seconds must be a number"
            if seconds <= 0:
                return "error", {}, "seconds must be positive"
            timer = ctx.registry.get("timer") if ctx.registry else None
            if timer is None or not ctx.registry.is_enabled("timer") or not hasattr(timer, "start_timer"):
                return "error", {}, "the timer module is off"
            from pawn_assistant.commands.builtin.timer import format_duration

            label = str(args.get("label") or "").strip() or format_duration(seconds, ctx.lang)
            tid = timer.start_timer(seconds, label, ctx)
            return "ok", {"timer_id": tid}, ""
        return "error", {}, f"unsupported tool {tool}"

    # ------------------------------------------------------------------ lifecycle

    def on_disable(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if self.client is None or self._stop.is_set():
            return
        self._stop.set()
        self._kick.set()
        self.report_presence("away")
