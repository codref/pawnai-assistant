"""Minimal SSE client for pawn-server's ``POST /v1/pawn/voice`` (stdlib only)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

USER_AGENT = "pawn-assistant/0.1"


class PawnError(RuntimeError):
    """The server was unreachable or the turn failed."""


@dataclass
class VoiceAnswer:
    text: str
    access: str = "chat"
    run_id: Optional[str] = None
    # Server-side intent detection ("end", "lock"): the agent asked us to act.
    controls: list[str] = field(default_factory=list)


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
    ) -> VoiceAnswer:
        body = json.dumps(
            {"conversation": conversation, "message": message, "access": access, "lang": lang}
        ).encode("utf-8")
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
        answer: Optional[VoiceAnswer] = None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                for event, data in iter_sse(iter(resp.readline, b"")):
                    if event == "progress" and on_progress is not None:
                        text = str(data.get("text") or "").strip()
                        if text:
                            try:
                                on_progress(text)
                            except Exception:  # noqa: BLE001
                                logger.exception("progress callback failed")
                    elif event == "answer":
                        answer = VoiceAnswer(
                            text=str(data.get("content") or ""),
                            access=str(data.get("access") or access),
                            controls=[str(c) for c in data.get("controls") or []],
                        )
                    elif event == "error":
                        raise PawnError(str(data.get("message") or "error"))
                    elif event == "done":
                        if answer is not None:
                            answer.run_id = data.get("run_id")
                        break
        except urllib.error.HTTPError as exc:
            raise PawnError(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PawnError(f"unreachable: {exc}") from exc
        if answer is None:
            raise PawnError("no answer")
        return answer
