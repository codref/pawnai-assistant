"""mpv wrapper over JSON IPC: play a stream, live volume (ducking), ICY titles.

mpv outputs through PipeWire/Pulse, so it never touches the sounddevice
capture stream (same reason beeps/TTS go through paplay/pw-play).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MetaCallback = Callable[[str], None]
ExitCallback = Callable[[str], None]


def _socket_dir() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    return Path(runtime) if runtime and Path(runtime).is_dir() else Path(tempfile.gettempdir())


def icy_title(metadata: Any) -> str:
    """``icy-title`` from mpv's ``metadata`` property (key case varies)."""
    if not isinstance(metadata, dict):
        return ""
    for key, value in metadata.items():
        if str(key).lower() in ("icy-title", "streamtitle", "title"):
            return str(value).strip()
    return ""


class MpvPlayer:
    def __init__(
        self,
        *,
        on_meta: Optional[MetaCallback] = None,
        on_exit: Optional[ExitCallback] = None,
        binary: str = "mpv",
    ) -> None:
        self.binary = binary
        self.on_meta = on_meta or (lambda _t: None)
        self.on_exit = on_exit or (lambda _r: None)
        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self._sock_path = _socket_dir() / f"pawn-mpv-{os.getpid()}.sock"
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._gen = 0  # bumps on every start/stop: stale threads bail out
        self._volume = 70.0
        self._ramp_gen = 0

    # ------------------------------------------------------------------ status

    @staticmethod
    def available(binary: str = "mpv") -> bool:
        return shutil.which(binary) is not None

    @property
    def running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def volume(self) -> float:
        return self._volume

    # ------------------------------------------------------------------ control

    def play(self, url: str, *, volume: float) -> None:
        self.stop()
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._volume = float(volume)
            try:
                self._sock_path.unlink(missing_ok=True)
            except OSError:
                pass
            cmd = [
                self.binary,
                "--no-video",
                "--no-terminal",
                "--idle=no",
                "--force-window=no",
                "--audio-display=no",
                "--cache=yes",
                f"--volume={int(volume)}",
                f"--input-ipc-server={self._sock_path}",
                url,
            ]
            logger.info("radio: mpv %s", url)
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        threading.Thread(target=self._ipc_loop, args=(gen,), name="pawn-mpv-ipc", daemon=True).start()
        threading.Thread(target=self._watch, args=(gen,), name="pawn-mpv-watch", daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
            proc, self._proc = self._proc, None
            sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            self._sock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def set_volume(self, volume: float) -> None:
        self._ramp_gen += 1  # cancel any ramp in flight
        self._apply_volume(volume)

    def ramp_to(self, target: float, ms: float = 300.0, steps: int = 10) -> None:
        """Non-blocking volume fade (used for ducking)."""
        self._ramp_gen += 1
        gen = self._ramp_gen
        start = self._volume

        def _run() -> None:
            for i in range(1, steps + 1):
                if gen != self._ramp_gen:
                    return
                self._apply_volume(start + (target - start) * i / steps)
                time.sleep(ms / 1000.0 / steps)

        if ms <= 0:
            self._apply_volume(target)
        else:
            threading.Thread(target=_run, name="pawn-mpv-ramp", daemon=True).start()

    def _apply_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(130.0, float(volume)))
        self._send({"command": ["set_property", "volume", round(self._volume, 1)]})

    # ------------------------------------------------------------------ IPC

    def _send(self, payload: dict[str, Any]) -> None:
        sock = self._sock
        if sock is None:
            return
        data = (json.dumps(payload) + "\n").encode()
        with self._send_lock:
            try:
                sock.sendall(data)
            except OSError:
                pass

    def _connect(self, gen: int) -> Optional[socket.socket]:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and gen == self._gen:
            if self._sock_path.exists():
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    s.connect(str(self._sock_path))
                    return s
                except OSError:
                    s.close()
            time.sleep(0.05)
        return None

    def _ipc_loop(self, gen: int) -> None:
        sock = self._connect(gen)
        if sock is None:
            return
        with self._lock:
            if gen != self._gen:
                sock.close()
                return
            self._sock = sock
        self._send({"command": ["observe_property", 1, "metadata"]})
        self._apply_volume(self._volume)  # a duck may have happened before connect
        buf = b""
        while gen == self._gen:
            try:
                chunk = sock.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("event") == "property-change" and msg.get("name") == "metadata":
                    title = icy_title(msg.get("data"))
                    if title:
                        try:
                            self.on_meta(title)
                        except Exception:  # noqa: BLE001
                            logger.exception("radio meta callback failed")

    def _watch(self, gen: int) -> None:
        proc = self._proc
        if proc is None:
            return
        code = proc.wait()
        if gen != self._gen:
            return  # stopped on purpose / replaced
        reason = "ended" if code == 0 else f"mpv exited ({code})"
        logger.warning("radio: %s", reason)
        try:
            self.on_exit(reason)
        except Exception:  # noqa: BLE001
            logger.exception("radio exit callback failed")
