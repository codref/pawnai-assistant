"""Minimal client for pawn-server's voice endpoints (stdlib only).

``POST /v1/pawn/voice`` (SSE turn), plus the hand-off and phone-approved
access calls. Protocol: pawnai ``docs/VOICE_PROTOCOL.md``; golden transcripts
in ``tests/contracts/voice`` (synced from pawnai, never edit them here).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

USER_AGENT = "pawn-assistant/0.1"


class PawnError(RuntimeError):
    """The server was unreachable or the turn failed."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class VoiceAnswer:
    text: str
    access: str = "chat"
    run_id: Optional[str] = None
    # Server-side intent detection ("end", "lock"): the agent asked us to act.
    controls: list[str] = field(default_factory=list)
    # Markdown detail for the screen only; never spoken.
    display: Optional[str] = None


def iter_sse(lines: Iterator[bytes]) -> Iterator[tuple[str, dict]]:
    """Yield ``(event, data)`` from an SSE byte-line stream (comments skipped)."""
    event, data_lines = "message", []
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = {"raw": payload}
                yield event, parsed if isinstance(parsed, dict) else {"value": parsed}
            event, data_lines = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        try:
            yield event, json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            pass


class PawnClient:
    def __init__(self, server_url: str, token: str = "", timeout_s: float = 90.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    @property
    def configured(self) -> bool:
        return bool(self.server_url)

    def voice(
        self,
        conversation: str,
        message: str,
        *,
        access: str = "chat",
        lang: str = "it",
        on_progress: Optional[Callable[[str], None]] = None,
        device_id: Optional[str] = None,
    ) -> VoiceAnswer:
        payload = {"conversation": conversation, "message": message, "access": access, "lang": lang}
        if device_id:
            payload["device_id"] = device_id
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": USER_AGENT,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            f"{self.server_url}/v1/pawn/voice", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                return read_voice_stream(iter(resp.readline, b""), access=access, on_progress=on_progress)
        except urllib.error.HTTPError as exc:
            raise PawnError(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PawnError(f"unreachable: {exc}") from exc

    # ------------------------------------------------------------ hand-off / access

    def _json(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(f"{self.server_url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8", "replace")).get("detail") or ""
            except Exception:  # noqa: BLE001
                pass
            raise PawnError(f"HTTP {exc.code}" + (f": {detail}" if detail else ""), status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PawnError(f"unreachable: {exc}") from exc

    def active_thread(self, exclude_device: Optional[str] = None) -> Optional[dict]:
        """The newest voice conversation another assistant could hand over (or None)."""
        path = "/v1/pawn/voice/active"
        if exclude_device:
            path += "?" + urllib.parse.urlencode({"exclude_device": exclude_device})
        return (self._json("GET", path) or {}).get("thread")

    def claim(self, conversation: str, device_id: str) -> dict:
        path = f"/v1/pawn/voice/{urllib.parse.quote(conversation, safe=':')}/claim"
        return self._json("POST", path, {"device_id": device_id})

    def request_access(self, device_id: str, *, conversation: Optional[str], minutes: int) -> dict:
        body = {"device_id": device_id, "conversation": conversation, "minutes": minutes}
        return self._json("POST", "/v1/pawn/voice/access-requests", body)

    def access_status(self, request_id: str) -> dict:
        return self._json("GET", f"/v1/pawn/voice/access-requests/{urllib.parse.quote(request_id)}")


def read_voice_stream(
    lines: Iterator[bytes],
    *,
    access: str = "chat",
    on_progress: Optional[Callable[[str], None]] = None,
) -> VoiceAnswer:
    """Fold a ``/v1/pawn/voice`` SSE stream into a :class:`VoiceAnswer`."""
    answer: Optional[VoiceAnswer] = None
    for event, data in iter_sse(lines):
        if event == "progress" and on_progress is not None:
            text = str(data.get("text") or "").strip()
            if text:
                try:
                    on_progress(text)
                except Exception:  # noqa: BLE001
                    logger.exception("progress callback failed")
        elif event == "answer":
            display = str(data.get("display") or "").strip() or None
            answer = VoiceAnswer(
                text=str(data.get("content") or ""),
                access=str(data.get("access") or access),
                controls=[str(c) for c in data.get("controls") or []],
                display=display,
            )
        elif event == "error":
            raise PawnError(str(data.get("message") or "error"))
        elif event == "done":
            if answer is not None:
                answer.run_id = data.get("run_id")
            break
    if answer is None:
        raise PawnError("no answer")
    return answer
