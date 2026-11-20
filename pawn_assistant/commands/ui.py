"""Command → HUD widget bridge.

Plugins open/update/close windows on the HUD; the bridge keeps the open set so
a client that connects later gets a snapshot in ``hello``. Messages:

    {"type": "ui", "op": "open",   "id": "timer:3", "kind": "countdown",
     "command": "timer", "title": "TIMER", "data": {...}, "anchor": "tr"}
    {"type": "ui", "op": "update", "id": "timer:3", "data": {...partial...}}
    {"type": "ui", "op": "close",  "id": "timer:3"}
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

Emit = Callable[[dict[str, Any]], None]


class UiBridge:
    def __init__(self, emit: Optional[Emit] = None) -> None:
        self.emit: Emit = emit or (lambda _p: None)
        self._open: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _send(self, payload: dict[str, Any]) -> None:
        try:
            self.emit(payload)
        except Exception:  # noqa: BLE001
            logger.exception("ui emit failed")

    def open(
        self,
        id: str,
        kind: str,
        *,
        command: str,
        title: str = "",
        data: Optional[dict[str, Any]] = None,
        anchor: str = "tr",
    ) -> None:
        payload = {
            "type": "ui",
            "op": "open",
            "id": id,
            "kind": kind,
            "command": command,
            "title": title or kind.upper(),
            "data": dict(data or {}),
            "anchor": anchor,
        }
        with self._lock:
            self._open[id] = payload
        self._send(copy.deepcopy(payload))

    def update(self, id: str, data: dict[str, Any], *, title: Optional[str] = None) -> None:
        with self._lock:
            cur = self._open.get(id)
            if cur is None:
                return
            cur["data"].update(data)
            if title:
                cur["title"] = title
        payload: dict[str, Any] = {"type": "ui", "op": "update", "id": id, "data": dict(data)}
        if title:
            payload["title"] = title
        self._send(payload)

    def close(self, id: str) -> None:
        with self._lock:
            if self._open.pop(id, None) is None:
                return
        self._send({"type": "ui", "op": "close", "id": id})

    def is_open(self, id: str) -> bool:
        with self._lock:
            return id in self._open

    def owner(self, id: str) -> Optional[str]:
        with self._lock:
            cur = self._open.get(id)
            return cur["command"] if cur else None

    def data(self, id: str) -> dict[str, Any]:
        with self._lock:
            cur = self._open.get(id)
            return copy.deepcopy(cur["data"]) if cur else {}

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(p) for p in self._open.values()]
