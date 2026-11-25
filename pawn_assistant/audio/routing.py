"""Where PAWN's own audio goes (beeps, Piper, radio).

``None`` = the desktop default sink. With echo cancellation on, this is the
echo-cancel sink, so everything PAWN plays is the AEC reference signal.
"""

from __future__ import annotations

import threading
from typing import Optional

_lock = threading.Lock()
_output_sink: Optional[str] = None


def set_output_sink(name: Optional[str]) -> None:
    global _output_sink
    with _lock:
        _output_sink = name or None


def output_sink() -> Optional[str]:
    return _output_sink


def player_command(path: str, player: str) -> list[str]:
    """Command line for ``paplay`` / ``pw-play`` / ``aplay`` aimed at the sink."""
    sink = _output_sink
    if player == "paplay":
        return ["paplay", f"--device={sink}", path] if sink else ["paplay", path]
    if player == "pw-play":
        return ["pw-play", "--target", sink, path] if sink else ["pw-play", path]
    # aplay bypasses PipeWire routing entirely (no AEC reference).
    return ["aplay", "-q", path]


def mpv_audio_args() -> list[str]:
    sink = _output_sink
    if not sink:
        return []
    return ["--ao=pulse", f"--audio-device=pulse/{sink}"]
