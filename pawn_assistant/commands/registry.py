"""Command discovery, enable/disable and the views the pipeline needs.

Sources: every module in ``pawn_assistant/commands/builtin/`` plus any
installed package exposing an entry point in group ``pawn_assistant.commands``
(the entry point may name a ``Command`` subclass or a module containing some).

Enabled state: YAML ``commands.disabled`` gives the defaults; toggles made at
runtime (HUD → MODULES) are saved in the state file and win over the YAML.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import pkgutil
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Optional

from pawn_assistant.commands.base import (
    UNKNOWN,
    Command,
    CommandContext,
    Handler,
    Intent,
    Shot,
    default_state_dir,
)
from pawn_assistant.commands.text import FastMatch

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "pawn_assistant.commands"


def _commands_in_module(mod: ModuleType) -> list[type[Command]]:
    out = []
    for obj in vars(mod).values():
        if (
            inspect.isclass(obj)
            and issubclass(obj, Command)
            and obj is not Command
            and obj.name
            and obj.__module__.startswith(mod.__name__)
        ):
            out.append(obj)
    return out


def discover_builtin() -> list[type[Command]]:
    from pawn_assistant.commands import builtin

    found: list[type[Command]] = []
    for info in pkgutil.iter_modules(builtin.__path__):
        name = f"{builtin.__name__}.{info.name}"
        try:
            found.extend(_commands_in_module(importlib.import_module(name)))
        except Exception:  # noqa: BLE001
            logger.exception("command plugin %s failed to import", name)
    return found


def discover_entry_points() -> list[type[Command]]:
    from importlib.metadata import entry_points

    found: list[type[Command]] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
        except Exception:  # noqa: BLE001
            logger.exception("command entry point %s failed to load", ep.name)
            continue
        if inspect.isclass(obj) and issubclass(obj, Command):
            found.append(obj)
        elif inspect.ismodule(obj):
            found.extend(_commands_in_module(obj))
        else:
            logger.warning("entry point %s is not a Command or module", ep.name)
    return found


class CommandRegistry:
    def __init__(
        self,
        commands: Optional[Iterable[Command | type[Command]]] = None,
        *,
        config: Optional[dict[str, Any]] = None,
        state_path: Optional[Path] = None,
        entry_points: bool = True,
    ) -> None:
        self.config = dict(config or {})
        self.state_path = state_path
        if commands is None:
            classes: list[Any] = discover_builtin()
            if entry_points:
                classes += discover_entry_points()
        else:
            classes = list(commands)
        self._commands: dict[str, Command] = {}
        for item in classes:
            cmd = item() if inspect.isclass(item) else item
            if not cmd.name:
                continue
            if cmd.name in self._commands:
                logger.warning("duplicate command %r ignored", cmd.name)
                continue
            self._commands[cmd.name] = cmd
        self._commands = dict(
            sorted(self._commands.items(), key=lambda kv: (kv[1].order, kv[0]))
        )
        self._lock = threading.RLock()
        self._listeners: list[Callable[[], None]] = []
        self.version = 0
        self.ctx: Optional[CommandContext] = None
        self._state = self._load_state()
        self._enabled = self._initial_enabled()
        for name, cmd in self._commands.items():
            try:
                cmd.configure(self.config.get(name) or {})
            except Exception:  # noqa: BLE001
                logger.exception("command %s configure failed", name)
        self._rebuild()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.is_file():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            logger.warning("unreadable command state %s — ignored", self.state_path)
            return {}

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.state_path)
        except Exception:  # noqa: BLE001
            logger.exception("could not save command state %s", self.state_path)

    def _initial_enabled(self) -> set[str]:
        disabled_cfg = set(self.config.get("disabled") or ())
        overrides: dict[str, bool] = self._state.get("enabled") or {}
        out = set()
        for name, cmd in self._commands.items():
            on = cmd.default_enabled and name not in disabled_cfg
            if name in overrides:
                on = bool(overrides[name])
            if cmd.core or on:
                out.add(name)
        return out

    def state_get(self, command: str, key: str, default: Any = None) -> Any:
        with self._lock:
            return (self._state.get("data") or {}).get(command, {}).get(key, default)

    def state_set(self, command: str, key: str, value: Any) -> None:
        with self._lock:
            self._state.setdefault("data", {}).setdefault(command, {})[key] = value
            self._save_state()

    # ------------------------------------------------------------------ lifecycle

    def setup(self, ctx: CommandContext) -> None:
        self.ctx = ctx
        ctx.registry = self
        for name, cmd in self._commands.items():
            cmd.ctx = ctx
            try:
                cmd.setup(ctx)
            except Exception:  # noqa: BLE001
                logger.exception("command %s setup failed", name)

    def add_listener(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def shutdown(self) -> None:
        for name, cmd in self._commands.items():
            try:
                cmd.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("command %s shutdown failed", name)

    # ------------------------------------------------------------------ enable

    def get(self, name: str) -> Optional[Command]:
        return self._commands.get(name)

    def is_enabled(self, name: str) -> bool:
        return name in self._enabled

    def set_enabled(self, name: str, on: bool) -> bool:
        """Toggle a command. Returns False for unknown or core commands."""
        cmd = self._commands.get(name)
        if cmd is None or cmd.core:
            return False
        with self._lock:
            if (name in self._enabled) == on:
                return True
            if on:
                self._enabled.add(name)
            else:
                self._enabled.discard(name)
            self._state.setdefault("enabled", {})[name] = on
            self._save_state()
            self._rebuild()
        if not on:
            try:
                cmd.on_disable()
            except Exception:  # noqa: BLE001
                logger.exception("command %s on_disable failed", name)
        logger.info("command %s %s", name, "enabled" if on else "disabled")
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:  # noqa: BLE001
                logger.exception("registry listener failed")
        return True

    def commands(self, *, enabled_only: bool = True) -> list[Command]:
        return [
            c for n, c in self._commands.items() if not enabled_only or n in self._enabled
        ]

    # ------------------------------------------------------------------ views

    def _rebuild(self) -> None:
        intents: list[Intent] = []
        handlers: dict[str, Handler] = {}
        unknown: Optional[Intent] = None
        for cmd in self.commands():
            specs = {i.name: i for i in cmd.intents()}
            for name, fn in cmd.handlers().items():
                spec = specs[name]
                if name in handlers:
                    logger.warning("intent %s defined twice; keeping the first", name)
                    continue
                handlers[name] = fn
                if name == UNKNOWN:
                    unknown = spec
                else:
                    intents.append(spec)
        if unknown is not None:
            intents.append(unknown)  # last in prompt + grammar
        self._intents = tuple(intents)
        self._handlers = handlers
        self._all_intents = {i.name: i for c in self._commands.values() for i in c.intents()}
        self.version += 1

    def intents(self) -> tuple[Intent, ...]:
        """Enabled intents, ``unknown`` last."""
        return self._intents

    def intent(self, name: str) -> Optional[Intent]:
        return self._all_intents.get(name)

    def handler_for(self, name: str) -> Optional[Handler]:
        return self._handlers.get(name)

    def fast_match(self, norm: str) -> Optional[FastMatch]:
        for cmd in self.commands():
            try:
                hit = cmd.fast_match(norm)
            except Exception:  # noqa: BLE001
                logger.exception("command %s fast_match failed", cmd.name)
                continue
            if hit is not None and hit.intent in self._handlers:
                return hit
        return None

    def fewshot_rows(self, intent: str, lang: str) -> tuple[Shot, ...]:
        for cmd in self._commands.values():
            rows = (cmd.phrasing.get(lang) or {}).get(intent)
            if rows:
                return tuple(rows)
        return ()

    def phrase_intents(self, lang: str) -> list[str]:
        """Intents (incl. alert pseudo-intents) with phrasing rows, enabled commands only."""
        out: list[str] = []
        for cmd in self.commands():
            for name in (cmd.phrasing.get(lang) or {}):
                if self.phrase_allowed(name) and name not in out:
                    out.append(name)
        return out

    def phrase_allowed(self, intent: str) -> bool:
        spec = self._all_intents.get(intent)
        return spec.phrase if spec is not None else True

    # ------------------------------------------------------------------ cancel

    def cancel(self, ctx: CommandContext) -> str:
        """Generic stop: commands named in the text, else the latest active one, else all."""
        enabled = self.commands()
        norm = f" {ctx.norm} "
        named = [c for c in enabled if any(f" {w} " in norm for w in c.cancel_words)]
        active = [c for c in enabled if c.is_active()]
        if named:
            targets = named
        elif any(c.name == ctx.last_active for c in active):
            targets = [c for c in active if c.name == ctx.last_active]
        else:
            targets = active
        replies = []
        for cmd in targets:
            try:
                reply = cmd.on_cancel(ctx)
            except Exception:  # noqa: BLE001
                logger.exception("command %s on_cancel failed", cmd.name)
                continue
            if reply:
                replies.append(reply)
        if not replies:
            return ctx.t("Ok, non c'è niente da fermare.", "Okay, nothing to stop.")
        return " ".join(replies)

    # ------------------------------------------------------------------ loop / UI

    def on_state(self, state: str) -> None:
        for cmd in self.commands(enabled_only=False):
            try:
                cmd.on_state(state)
            except Exception:  # noqa: BLE001
                logger.exception("command %s on_state failed", cmd.name)

    def ui_action(self, widget_id: str, action: str, data: Optional[dict[str, Any]] = None) -> bool:
        owner = self.ctx.ui.owner(widget_id) if self.ctx else None
        if owner is None:
            owner = widget_id.split(":", 1)[0]
        cmd = self._commands.get(owner)
        if cmd is None:
            return False
        try:
            cmd.on_ui_action(widget_id, action, dict(data or {}))
        except Exception:  # noqa: BLE001
            logger.exception("command %s ui action %s failed", owner, action)
            return False
        return True

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for name, cmd in self._commands.items():
            out.append(
                {
                    "name": name,
                    "title": cmd.title or name,
                    "description": cmd.description,
                    "core": cmd.core,
                    "enabled": name in self._enabled,
                    "active": bool(cmd.is_active()),
                    "intents": [
                        {
                            "name": i.name,
                            "description": i.description,
                            "example": (i.examples[0] if i.examples else (i.fewshot[0][0] if i.fewshot else "")),
                            "slots": list(i.slots),
                        }
                        for i in cmd.intents()
                    ],
                }
            )
        return out

    def assets(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"js": [], "css": []}
        for name, cmd in self._commands.items():
            for kind in ("css", "js"):
                for rel in cmd.assets.get(kind, ()):
                    out[kind].append(f"/plugins/{name}/{rel}")
        return out

    def asset_path(self, command: str, rel: str) -> Optional[Path]:
        """Only files a command declares in ``assets`` are served."""
        cmd = self._commands.get(command)
        if cmd is None:
            return None
        declared = {r for files in cmd.assets.values() for r in files}
        if rel not in declared:
            return None
        base = cmd.base_dir()
        path = (base / rel).resolve()
        if base not in path.parents or not path.is_file():
            return None
        return path


_DEFAULT: Optional[CommandRegistry] = None
_DEFAULT_LOCK = threading.Lock()


def default_registry() -> CommandRegistry:
    """Static view over all built-in commands (prompt/grammar/few-shot defaults)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = CommandRegistry(entry_points=False)
        return _DEFAULT


def default_state_path() -> Path:
    return default_state_dir() / "commands.json"
