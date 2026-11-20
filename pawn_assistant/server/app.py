"""FastAPI app: static HUD + WebSocket meter/event fan-out."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pawn_assistant.config import Settings
from pawn_assistant.pipeline.loop import WakeLoop

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class Hub:
    """Broadcast JSON messages to connected HUD clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    def publish_threadsafe(self, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload)
        async with self._lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


def create_app(settings: Settings, wake_loop: Optional[WakeLoop] = None) -> FastAPI:
    hub = Hub()
    state: dict[str, Any] = {"settings": settings, "loop": wake_loop, "hub": hub}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        hub.bind_loop(loop)
        wl = state["loop"]
        if wl is None:
            wl = WakeLoop(
                settings,
                on_event=hub.publish_threadsafe,
                on_meter=hub.publish_threadsafe,
            )
            state["loop"] = wl
        try:
            wl.start()
        except Exception:
            logger.exception("wake loop failed to start")
        yield
        wl.stop()

    app = FastAPI(title="Pawn Assistant", lifespan=lifespan)
    app.state.pawn = state

    @app.get("/health")
    async def health() -> dict:
        wl: WakeLoop = app.state.pawn["loop"]
        return {"ok": True, **wl.snapshot_status()}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        wl: WakeLoop = app.state.pawn["loop"]
        settings = app.state.pawn["settings"]
        hello = {
            "type": "hello",
            "wake_model": settings.wake_model,
            "wake_threshold": settings.wake_threshold,
            "device_name": wl.capture.device_info.name,
            "sample_rate": settings.sample_rate,
            "status": wl.snapshot_status(),
            "gain": await asyncio.to_thread(wl.gain.state),
            "commands": wl.commands.describe(),
            "assets": wl.commands.assets(),
            "ui": wl.ctx.ui.snapshot(),
            "now_ms": round(time.time() * 1000),
        }
        await ws.send_text(json.dumps(hello))
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(msg, dict) and msg.get("type") == "set_gain":
                    try:
                        value = float(msg["value"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    state = await asyncio.to_thread(wl.gain.set, value)
                    await hub._broadcast({"type": "gain", **state})
                elif isinstance(msg, dict) and msg.get("type") == "set_command":
                    name = str(msg.get("name") or "")
                    enabled = bool(msg.get("enabled"))
                    await asyncio.to_thread(wl.set_command_enabled, name, enabled)
                    await hub._broadcast({"type": "commands", "commands": wl.commands.describe()})
                elif isinstance(msg, dict) and msg.get("type") == "ui_action":
                    wid = str(msg.get("id") or "")
                    action = str(msg.get("action") or "")
                    data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
                    if wid and action:
                        await asyncio.to_thread(wl.ui_action, wid, action, data)
        except WebSocketDisconnect:
            pass
        finally:
            await hub.disconnect(ws)

    @app.get("/plugins/{command}/{path:path}")
    async def plugin_asset(command: str, path: str) -> FileResponse:
        wl: WakeLoop = app.state.pawn["loop"]
        target = wl.commands.asset_path(command, path)
        if target is None:
            raise HTTPException(status_code=404)
        return FileResponse(target, headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
