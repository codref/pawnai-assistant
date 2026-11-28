"""Keep tests away from the real ~/.local/state/pawn-assistant (device.json, toggles)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))
