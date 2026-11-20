"""Core intents: greeting, generic cancel/stop, unknown. Always enabled."""

from __future__ import annotations

from typing import Any

from pawn_assistant.commands.base import UNKNOWN, Command, CommandContext, intent


class CoreCommand(Command):
    name = "core"
    title = "Core"
    description = "Greeting, generic stop/cancel and the fallback reply."
    core = True
    order = 90  # after the feature commands: keeps the prompt order stable

    phrasing = {
        "it": {
            "cancel": (
                ("annulla il timer", "Timer annullato.", "complice", "pomeriggio", "Timer annullato, facciamo finta di niente."),
                ("ferma tutto", "2 timer annullati.", "tecnico", "sera", "2 timer annullati, conto alla rovescia azzerato."),
                ("stop", "Ok, non c'è niente da fermare.", "asciutto", "mattina", "Niente da fermare, qui è tutto fermo."),
            ),
            "greeting": (
                ("ciao come stai", "Ciao. Tutto operativo.", "solenne", "mattina", "Buongiorno. Ogni sistema risponde all'appello."),
                ("ehi", "Ciao. Tutto operativo.", "pigro", "notte", "Ehi. Sveglio, più o meno."),
                ("buonasera", "Ciao. Tutto operativo.", "cordiale", "sera", "Buonasera. Qui tutto in ordine."),
            ),
            UNKNOWN: (
                ("che tempo fa domani", "Non ho capito.", "asciutto", "pomeriggio", "Non è roba per me."),
                ("raccontami una storia", "Non ho capito.", "ironico", "sera", "Storie? Non è il mio reparto."),
                ("ordina una pizza", "Non ho capito.", "cordiale", "mattina", "Mi dispiace, questo non lo so fare."),
            ),
        },
        "en": {
            "cancel": (
                ("cancel the timer", "Timer cancelled.", "playful", "afternoon", "Timer cancelled. It never happened."),
                ("stop", "Okay, nothing to stop.", "dry", "morning", "Nothing to stop, all quiet here."),
            ),
            "greeting": (
                ("hi how are you", "Hello. All systems nominal.", "solemn", "morning", "Good morning. Every system reports for duty."),
                ("hey", "Hello. All systems nominal.", "laid-back", "night", "Hey. Awake, more or less."),
            ),
            UNKNOWN: (
                ("what's the weather tomorrow", "I didn't understand.", "dry", "afternoon", "Not my department."),
                ("tell me a story", "I didn't understand.", "ironic", "evening", "Stories? Not my department."),
            ),
        },
    }

    @intent(
        "cancel",
        "the user wants to stop, cancel or dismiss (timers, radio, current action)",
        examples=(
            "annulla",
            "annulla timer",
            "annulla il timer",
            "cancella il timer",
            "ferma il timer",
            "ferma il conto alla rovescia",
            "annulla il conto alla rovescia",
            "stop",
            "basta",
            "lascia stare",
            "niente",
            "cancel",
            "cancel the timer",
            "never mind",
        ),
        fewshot=(
            ("togli il conto alla rovescia", {}),
            ("spegni quel timer", {}),
        ),
    )
    def cancel(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        assert ctx.registry is not None
        return ctx.registry.cancel(ctx)

    @intent(
        "greeting",
        "the user greets or asks how the assistant is",
        examples=(
            "ciao",
            "buongiorno",
            "buonasera",
            "come stai",
            "ciao come stai",
            "hello",
            "hi",
            "how are you",
        ),
        fewshot=(("ehi tutto bene?", {}),),
    )
    def greeting(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        return ctx.t("Ciao. Tutto operativo.", "Hello. All systems nominal.")

    @intent(
        UNKNOWN,
        "anything else: weather, news, lights, jokes, chit-chat, off-topic or unclear",
        fewshot=(
            ("che tempo fa domani a Milano", {}),
            ("raccontami una barzelletta", {}),
        ),
    )
    def unknown(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        return ctx.t("Non ho capito.", "I didn't understand.")
