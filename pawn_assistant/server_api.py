"""pawn-server endpoints for a registered assistant (stdlib only).

The laptop registers in pawn-server's device registry (``kind: desktop``),
keeps ``GET /v1/devices/{id}/commands/stream`` open for live commands, reports
presence, and shares wake-word clips and models. The protocol is in pawnai's
``docs/VOICE_PROTOCOL.md`` and ``docs/ASSISTANTS.md``.

Identity (device id + per-device token) lives in
``~/.local/state/pawn-assistant/device.json`` (0600), written by
``pawn-assistant device register``. YAML ``device:`` keys override it.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from pawn_assistant.commands.base import default_state_dir
from pawn_assistant.commands.builtin.pawn.client import USER_AGENT, iter_sse

logger = logging.getLogger(__name__)


class ServerError(RuntimeError):
    """pawn-server unreachable or the request failed."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status

    @property
    def missing(self) -> bool:
        """404: the server predates this feature (or the device was revoked)."""
        return self.status == 404


# ------------------------------------------------------------------ identity


def identity_path() -> Path:
    return default_state_dir() / "device.json"


@dataclass
class DeviceIdentity:
    server_url: str
    device_id: str
    device_token: str
    name: str = "laptop"

    @property
    def configured(self) -> bool:
        return bool(self.server_url and self.device_id and self.device_token)

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or identity_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        tmp.replace(path)
        return path


def load_identity(
    overrides: Optional[dict[str, Any]] = None, path: Optional[Path] = None
) -> Optional[DeviceIdentity]:
    """device.json, with YAML ``device:`` keys (server_url / device_id / device_token) on top."""
    data: dict[str, Any] = {}
    path = path or identity_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable %s — ignored", path)
    for key in ("server_url", "device_id", "device_token", "name"):
        value = (overrides or {}).get(key)
        if value:
            data[key] = str(value)
    ident = DeviceIdentity(
        server_url=str(data.get("server_url") or "").rstrip("/"),
        device_id=str(data.get("device_id") or ""),
        device_token=str(data.get("device_token") or ""),
        name=str(data.get("name") or "laptop"),
    )
    return ident if ident.configured else None


# ------------------------------------------------------------------ HTTP


def _request(
    url: str,
    *,
    token: str = "",
    method: str = "GET",
    body: Any = None,
    raw: Optional[bytes] = None,
    content_type: str = "application/json",
    timeout: float = 15.0,
    accept: str = "application/json",
):
    data = raw
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    if data is not None:
        headers["Content-Type"] = content_type
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace")).get("detail") or ""
        except Exception:  # noqa: BLE001
            pass
        raise ServerError(f"HTTP {exc.code}{': ' + str(detail) if detail else ''}", exc.code) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ServerError(f"unreachable: {exc}") from exc


def _json(resp) -> Any:
    with resp:
        return json.loads(resp.read().decode("utf-8") or "null")


def _multipart(fields: dict[str, str], file_field: str, filename: str, payload: bytes, mime: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode()
        + payload
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def register_device(server_url: str, shared_token: str, name: str) -> DeviceIdentity:
    """Register this laptop with the shared API token (used once)."""
    url = server_url.rstrip("/") + "/v1/devices/register"
    body = {"name": name, "kind": "desktop", "platform": platform.system().lower() or "linux"}
    data = _json(_request(url, token=shared_token, method="POST", body=body))
    ident = DeviceIdentity(server_url.rstrip("/"), data["device_id"], data["device_token"], data.get("name") or name)
    client = DeviceClient(ident)
    client.update(capabilities=list(DESKTOP_CAPABILITIES))
    return ident


#: Live tools the Linux assistant runs (see the device plugin).
DESKTOP_CAPABILITIES = ("speak", "notify", "timer_set")


class DeviceClient:
    """Calls authenticated with the per-device token."""

    def __init__(self, ident: DeviceIdentity, timeout_s: float = 15.0) -> None:
        self.ident = ident
        self.timeout_s = timeout_s

    def _url(self, path: str, **query: Any) -> str:
        q = {k: v for k, v in query.items() if v is not None}
        tail = ("?" + urllib.parse.urlencode(q)) if q else ""
        return f"{self.ident.server_url}{path}{tail}"

    def _call(self, method: str, path: str, body: Any = None, **query: Any) -> Any:
        return _json(
            _request(
                self._url(path, **query),
                token=self.ident.device_token,
                method=method,
                body=body,
                timeout=self.timeout_s,
            )
        )

    @property
    def base(self) -> str:
        return f"/v1/devices/{self.ident.device_id}"

    def info(self) -> dict[str, Any]:
        return self._call("GET", self.base)

    def update(self, **fields: Any) -> dict[str, Any]:
        return self._call("PUT", self.base, {k: v for k, v in fields.items() if v is not None})

    def forget(self) -> dict[str, Any]:
        return self._call("DELETE", self.base)

    def presence(self, state: str, **flags: Any) -> dict[str, Any]:
        return self._call("POST", f"{self.base}/presence", {"state": state, **flags})

    def pending(self) -> list[dict[str, Any]]:
        return list(self._call("GET", f"{self.base}/commands/pending").get("commands") or [])

    def post_result(self, command_id: str, status: str, result: Optional[dict] = None, message: str = "") -> bool:
        """False when the server already finalised the command (409)."""
        try:
            self._call(
                "POST",
                f"{self.base}/commands/{command_id}/result",
                {"status": status, "result": result or {}, "message": message},
            )
        except ServerError as exc:
            if exc.status == 409:
                return False
            raise
        return True

    def stream(self, *, read_timeout_s: float = 60.0) -> Iterator[tuple[str, dict]]:
        """``(event, data)`` from the command stream until the server closes it.

        The server sends a keep-alive at least every ``api.stream_keepalive_seconds``
        (default 10 s), so a read timeout means the connection is dead.
        """
        resp = _request(
            self._url(f"{self.base}/commands/stream"),
            token=self.ident.device_token,
            timeout=read_timeout_s,
            accept="text/event-stream",
        )
        with resp:
            yield from iter_sse(iter(resp.readline, b""))

    # -- wake word ------------------------------------------------------------

    def upload_clip(self, wav: bytes, *, label: str, score: float, ts: str = "", meta: Optional[dict] = None) -> dict:
        fields = {"label": label, "device_id": self.ident.device_id, "score": f"{score:.4f}"}
        if ts:
            fields["ts"] = ts
        if meta:
            fields["meta"] = json.dumps(meta, ensure_ascii=False)
        body, ctype = _multipart(fields, "file", "clip.wav", wav, "audio/wav")
        return _json(
            _request(
                self._url("/v1/assistants/wake/clips"),
                token=self.ident.device_token,
                method="POST",
                raw=body,
                content_type=ctype,
                timeout=60.0,
            )
        )

    def current_model(self) -> Optional[dict[str, Any]]:
        data = self._call("GET", "/v1/assistants/wake/models/current", device_id=self.ident.device_id)
        return data.get("model")

    def download_model(self, version: str) -> bytes:
        resp = _request(
            self._url(f"/v1/assistants/wake/models/{urllib.parse.quote(version)}", device_id=self.ident.device_id),
            token=self.ident.device_token,
            timeout=120.0,
            accept="application/octet-stream",
        )
        with resp:
            return resp.read()


class AdminClient:
    """Calls that need the shared API token (training laptop only)."""

    def __init__(self, server_url: str, token: str, timeout_s: float = 60.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def list_clips(self, *, since: str = "", label: str = "", limit: int = 5000) -> list[dict[str, Any]]:
        q = {"limit": limit}
        if since:
            q["since"] = since
        if label:
            q["label"] = label
        url = f"{self.server_url}/v1/assistants/wake/clips?{urllib.parse.urlencode(q)}"
        return list(_json(_request(url, token=self.token, timeout=self.timeout_s)).get("clips") or [])

    def download_clip(self, clip_id: str) -> bytes:
        url = f"{self.server_url}/v1/assistants/wake/clips/{urllib.parse.quote(clip_id)}"
        with _request(url, token=self.token, timeout=self.timeout_s, accept="audio/wav") as resp:
            return resp.read()

    def publish_model(self, onnx: bytes, *, version: str = "", notes: str = "", thresholds: Optional[dict] = None, name: str = "hey_pawn") -> dict:
        fields = {"name": name}
        if version:
            fields["version"] = version
        if notes:
            fields["notes"] = notes
        if thresholds:
            fields["thresholds"] = json.dumps(thresholds)
        body, ctype = _multipart(fields, "file", f"{name}.onnx", onnx, "application/octet-stream")
        url = f"{self.server_url}/v1/assistants/wake/models"
        return _json(_request(url, token=self.token, method="POST", raw=body, content_type=ctype, timeout=120.0))
