import json

from pawn_assistant.commands import (
    SLOT_STR,
    Command,
    CommandContext,
    CommandRegistry,
    CommandRouter,
    intent,
)
from pawn_assistant.commands.builtin.core import CoreCommand
from pawn_assistant.commands.llm import build_grammar, build_system_prompt, parse_output
from pawn_assistant.commands.phrasing import ReplyPhraser


class Echo(Command):
    name = "echo"
    title = "Echo"
    description = "Repeats a word."
    phrasing = {"it": {"echo.say": (("di ciao", "Ciao.", "asciutto", "sera", "Ciao a te."),)}}

    @intent(
        "echo.say",
        "the user wants a word repeated",
        slots={"word": SLOT_STR},
        examples=("ripeti",),
    )
    def say(self, slots, ctx):
        return f"{slots.get('word', 'niente')}."


def test_builtin_discovery():
    reg = CommandRegistry(entry_points=False)
    names = [c.name for c in reg.commands(enabled_only=False)]
    assert {"core", "clock", "timer", "radio"} <= set(names)
    assert reg.intents()[-1].name == "unknown"


def test_decorated_intents_and_handlers():
    assert [i.name for i in Echo.intents()] == ["echo.say"]
    reg = CommandRegistry([CoreCommand, Echo])
    router = CommandRouter(ctx=CommandContext(), registry=reg)
    assert router.handle("ripeti").reply == "niente."


def test_toggle_rebuilds_views_and_persists(tmp_path):
    state = tmp_path / "commands.json"
    reg = CommandRegistry([CoreCommand, Echo], state_path=state)
    router = CommandRouter(ctx=CommandContext(), registry=reg)
    seen = []
    reg.add_listener(lambda: seen.append(reg.version))

    assert reg.set_enabled("echo", False)
    assert seen and "echo.say" not in {i.name for i in reg.intents()}
    assert "echo.say" not in build_grammar(reg.intents())
    assert "echo.say" not in build_system_prompt(reg.intents())
    assert parse_output('{"intent": "echo.say"}', [i.name for i in reg.intents()]) == ("unknown", {})
    assert router.handle("ripeti").intent == "unknown"
    assert json.loads(state.read_text())["enabled"] == {"echo": False}

    # Survives a restart, and wins over the YAML default.
    reg2 = CommandRegistry([CoreCommand, Echo], state_path=state)
    assert not reg2.is_enabled("echo")
    assert reg2.set_enabled("echo", True)
    assert CommandRegistry([CoreCommand, Echo], config={"disabled": ["echo"]}, state_path=state).is_enabled("echo")


def test_yaml_disabled_and_core_locked():
    reg = CommandRegistry([CoreCommand, Echo], config={"disabled": ["echo", "core"]})
    assert not reg.is_enabled("echo")
    assert reg.is_enabled("core")
    assert reg.set_enabled("core", False) is False
    assert reg.set_enabled("nope", True) is False


def test_disable_stops_running_work():
    class Busy(Command):
        name = "busy"
        stopped = False

        def is_active(self):
            return not self.stopped

        def on_cancel(self, ctx):
            self.stopped = True
            return "fermato"

    busy = Busy()
    reg = CommandRegistry([CoreCommand, busy])
    reg.setup(CommandContext())
    reg.set_enabled("busy", False)
    assert busy.stopped


def test_phrasing_rows_come_from_plugins():
    reg = CommandRegistry([CoreCommand, Echo])
    phraser = ReplyPhraser(llm=None, registry=reg)
    msgs = phraser.messages("echo.say", "di ciao", "Ciao.", mood="x")
    assert any(m["content"] == "Ciao a te." for m in msgs)
    assert "echo.say" in reg.phrase_intents("it")
    reg.set_enabled("echo", False)
    assert "echo.say" not in reg.phrase_intents("it")


def test_phrase_false_skips_llm():
    reg = CommandRegistry(entry_points=False)
    phraser = ReplyPhraser(llm=None, registry=reg)
    assert not phraser.applies("radio.play")
    assert phraser.applies("time.now")
    assert phraser.applies("timer.alert")


def test_describe_and_assets():
    reg = CommandRegistry(entry_points=False)
    info = {c["name"]: c for c in reg.describe()}
    assert info["core"]["core"] is True
    assert any(i["name"] == "radio.play" for i in info["radio"]["intents"])
    assets = reg.assets()
    assert "/plugins/timer/widget.js" in assets["js"]
    assert reg.asset_path("timer", "widget.js").is_file()
    assert reg.asset_path("timer", "__init__.py") is None  # not declared
    assert reg.asset_path("timer", "../radio/widget.js") is None


def test_plugin_state(tmp_path):
    state = tmp_path / "s.json"
    echo = Echo()
    reg = CommandRegistry([CoreCommand, echo], state_path=state)
    reg.setup(CommandContext())
    echo.state_set("last", {"a": 1})
    reg2 = CommandRegistry([CoreCommand, Echo], state_path=state)
    assert reg2.state_get("echo", "last") == {"a": 1}
