"""Talk to the remote pawn agent; unlock its full toolset with an on-screen keypad.

"Parliamo" / "let's start a conversation" opens a conversation: the plugin
takes the floor, so every utterance goes to pawn-server ``/v1/pawn/voice``
and the loop keeps listening after each reply (no wake word) until silence or
"fine conversazione".

Access starts as ``chat`` (no tools). "Sblocca le sessioni" opens a keypad;
the right code (``commands.pawn.unlock_code``) switches the conversation to
``full`` until it ends, the user says "blocca", or ``unlock_max_minutes``.
"Sblocca col telefono" asks the phone instead: pawn-server sends it an
``access_approve`` command, the user confirms with a fingerprint, and the
conversation is ``full`` for the granted minutes.

"Continua qui" takes over the conversation the phone (or another assistant)
was having in the last few minutes (pawn-server ``/v1/pawn/voice/active`` +
``claim``). Turns carry this laptop's ``device_id`` once it is registered
(``pawn-assistant device register``). See pawnai docs/VOICE_PROTOCOL.md.
"""

from __future__ import annotations

import hmac
import logging
import re
import threading
import time
import uuid
from typing import Any, Callable, Optional, Protocol

from pawn_assistant.commands.base import Command, CommandContext, intent
from pawn_assistant.commands.builtin.pawn.client import PawnClient, PawnError, VoiceAnswer
from pawn_assistant.commands.text import FastMatch

logger = logging.getLogger(__name__)

CHAT_WIDGET = "pawn:chat"
KEYPAD_WIDGET = "pawn:keypad"
MAX_TURNS_SHOWN = 8
MAX_FAILURES = 3

_START_RE = re.compile(
    r"\b(?:parliamo|chiacchieriamo"
    r"|(?:inizia\w*|avvia\w*|comincia\w*|facciamo|apri|start|begin|open|have)"
    r"\s+(?:una\s+|la\s+|a\s+|the\s+)?(?:conversazione|chiacchierata|conversation|chat)"
    r"|let\s?s\s+(?:talk|chat)|voglio parlare)\b"
)
# Inside a conversation these are checked before anything goes to the agent,
# so they are deliberately loose: a missed "stop" is far worse than ending a
# chat one turn early. Text is already normalize()d (lowercase, no accents).
_END_VERB = (
    r"(?:smett\w*|fine|finisc\w*|finiamo|finire|chiud\w*|termin\w*|interromp\w*"
    r"|ferm\w*|stop\w*|basta|esc\w*|end|quit|exit|close|finish)"
)
_END_OBJECT = (
    r"(?:conversa\w*|parlare|parliamo|ascolt\w*|chiacchier\w*"
    r"|conversation|chat|talking|listening)"
)
_END_RE = re.compile(
    rf"\b{_END_VERB}\b.*\b{_END_OBJECT}\b"
    r"|\b(?:abbiamo finito|ho finito|basta cosi|that s all|we re done|ciao ciao)\b"
)
# Short utterances (≤ 4 words) that are a stop on their own — incl. Whisper
# mishearings like "fa ermati" for "fermati". Not "ciao": it opens chats too.
_END_SHORT_RE = re.compile(
    r"\b(?:\w*ermat[aio]|ferm\w*|stop\w*|basta|esci|smettila|zitt[oa]|silenzio"
    r"|arrivederci|a dopo|bye|goodbye|shut up|quiet)\b"
)
_UNLOCK_RE = re.compile(r"\b(?:sblocca\w*|unlock\w*)\b")
_PHONE_RE = re.compile(r"\b(?:telefon\w*|cellulare|smartphone|phone)\b")
_RESUME_RE = re.compile(
    r"\b(?:continu\w*|riprend\w*|prosegu\w*)\s+(?:qui|qua|da qui|la conversazione|la chiacchierata|il discorso)\b"
    r"|\b(?:continue|resume|pick up)\s+(?:here|the conversation|where we left off)\b"
)
# "\bbloc" never matches inside "sblocca" (no word boundary after the s).
_LOCK_RE = re.compile(
    r"\b(?:(?:ri)?bloc\w*|richiud\w*|revoc\w*|togli\w*|disattiv\w*|lock\w*|revoke|disable)\b"
    r".*\b(?:access\w*|session\w*|sblocc\w*|strumenti|tool\w*|pawn|tutto|permess\w*|it)\b"
    r"|^(?:(?:ri)?blocc[ao](?:lo|la|le)?|lock(?: it)?)$"
)


class VoiceBackend(Protocol):
    configured: bool

    def voice(
        self,
        conversation: str,
        message: str,
        *,
        access: str = ...,
        lang: str = ...,
        on_progress: Optional[Callable[[str], None]] = ...,
    ) -> VoiceAnswer: ...


ACCESS_POLL_S = 1.0

class PawnCommand(Command):
    name = "pawn"
    title = "Pawn agent"
    description = "Spoken conversation with the pawn agent; keypad unlock for full access."
    order = 40
    cancel_words = ("conversazione", "conversation", "pawn")
    assets = {"js": ("widget.js",), "css": ("widget.css",)}

    def __init__(self) -> None:
        super().__init__()
        self.client: VoiceBackend = PawnClient("")
        self.clock: Callable[[], float] = time.monotonic
        self.cue: Callable[[], None] = _thinking_cue
        self._lock = threading.RLock()
        self.conversation: Optional[str] = None
        self.access = "chat"
        self.unlocked_until = 0.0
        self.failures = 0
        self.lockout_until = 0.0
        self.turns: list[dict[str, str]] = []
        # Registered device id (pawn-server); None until `device register`.
        self.device_id: Optional[str] = None
        self.access_request: Optional[str] = None  # pending phone approval id
        self.sleep: Callable[[float], None] = time.sleep
        self.spawn: Callable[[Callable[[], None]], None] = _spawn

    # ------------------------------------------------------------------ config

    def configure(self, cfg: dict[str, Any]) -> None:
        super().configure(cfg)
        self.client = PawnClient(
            str(self.config.get("server_url") or ""),
            token=str(self.config.get("token") or ""),
            timeout_s=float(self.config.get("timeout_s") or 90.0),
        )

    def setup(self, ctx: CommandContext) -> None:
        self.device_id = str(self.config.get("device_id") or "") or None
        if self.device_id is None and self.config.get("link_device", True):
            try:
                from pawn_assistant.server_api import load_identity

                ident = load_identity()
            except Exception:  # noqa: BLE001
                ident = None
            if ident is not None and ident.server_url.rstrip("/") == self.client.server_url:
                self.device_id = ident.device_id

    @property
    def unlock_code(self) -> str:
        return str(self.config.get("unlock_code") or "")

    @property
    def unlock_max_s(self) -> float:
        return float(self.config.get("unlock_max_minutes") or 15) * 60.0

    @property
    def followup_ms(self) -> float:
        return float(self.config.get("followup_timeout_s") or 8) * 1000.0

    @property
    def thinking_cue_s(self) -> float:
        return float(self.config.get("thinking_cue_s") or 4.0)

    @property
    def lockout_s(self) -> float:
        return float(self.config.get("lockout_s") or 60.0)

    # ------------------------------------------------------------------ matching

    @property
    def in_conversation(self) -> bool:
        return self.conversation is not None

    def fast_match(self, norm: str) -> Optional[FastMatch]:
        if self.in_conversation:
            if _LOCK_RE.search(norm):
                return FastMatch("pawn.lock", {}, 1.0)
            if _END_RE.search(norm) or (
                len(norm.split()) <= 4 and _END_SHORT_RE.search(norm)
            ):
                return FastMatch("pawn.end", {}, 1.0)
        if _UNLOCK_RE.search(norm) and _PHONE_RE.search(norm):
            return FastMatch("pawn.unlock_phone", {}, 1.0)
        if _UNLOCK_RE.search(norm) and (
            self.in_conversation or re.search(r"session|pawn|accesso|access", norm)
        ):
            return FastMatch("pawn.unlock", {}, 1.0)
        if _RESUME_RE.search(norm):
            return FastMatch("pawn.resume", {}, 1.0)
        if not self.in_conversation and _START_RE.search(norm):
            return FastMatch("pawn.start", {}, 1.0)
        return None

    def is_active(self) -> bool:
        return self.in_conversation

    # ------------------------------------------------------------------ intents

    @intent(
        "pawn.start",
        "the user wants to start a free conversation with the pawn agent",
        examples=(
            "parliamo",
            "iniziamo una conversazione",
            "facciamo una chiacchierata",
            "let's start a conversation",
            "let's talk",
        ),
        fewshot=(("ho voglia di fare due chiacchiere", {}),),
        phrase=False,
    )
    def start(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        if not self.client.configured:
            return ctx.t("Il server di pawn non è configurato.", "The pawn server is not configured.")
        self._open(ctx, f"voice:{uuid.uuid4().hex[:12]}")
        return ctx.t("Dimmi.", "I'm listening.")

    def _open(self, ctx: CommandContext, conversation: str) -> None:
        with self._lock:
            self.conversation = conversation
            self._relock()
            self.turns = []
            ctx.floor = self.name
            ctx.floor_listen_ms = self.followup_ms
            self.mark_active()
        ctx.ui.open(
            CHAT_WIDGET,
            "pawn-chat",
            command=self.name,
            title="PAWN",
            data=self._chat_data(status=ctx.t("in ascolto", "listening")),
            anchor="tr",
        )

    @intent(
        "pawn.resume",
        "the user wants to continue here the conversation started on another device",
        examples=("continua qui", "riprendi la conversazione", "continue here"),
        phrase=False,
    )
    def resume(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        if not self.client.configured:
            return ctx.t("Il server di pawn non è configurato.", "The pawn server is not configured.")
        try:
            thread = self.client.active_thread(exclude_device=self.device_id)
        except PawnError as exc:
            logger.warning("pawn resume: %s", exc)
            return ctx.t("Pawn non risponde.", "Pawn is not answering.")
        if not thread:
            return ctx.t("Non c'è una conversazione da riprendere.", "There's no conversation to pick up.")
        conversation = str(thread["conversation"])
        origin = thread.get("device_name")
        if self.device_id:
            try:
                claimed = self.client.claim(conversation, self.device_id)
                origin = claimed.get("from_device") or origin
            except PawnError as exc:
                logger.warning("pawn resume: claim failed: %s", exc)
                return ctx.t("Non riesco a riprendere quella conversazione.", "I can't pick that conversation up.")
        self._open(ctx, conversation)
        last = _first_sentence(str(thread.get("last_answer") or ""))
        if last:
            self.turns = [{"who": "pawn", "text": str(thread.get("last_answer"))}]
            self._push_chat(ctx, status=ctx.t("in ascolto", "listening"))
        where = f" da {origin}" if origin and ctx.lang == "it" else (f" from {origin}" if origin else "")
        head = ctx.t(f"Riprendo{where}.", f"Picking up{where}.")
        return f"{head} {ctx.t('Ero rimasto a:', 'Last thing I said:')} {last}" if last else head

    @intent(
        "pawn.unlock_phone",
        "the user wants to unlock full access by confirming on the phone",
        examples=("sblocca col telefono", "unlock with the phone"),
        phrase=False,
    )
    def unlock_phone(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        if not self.client.configured:
            return ctx.t("Il server di pawn non è configurato.", "The pawn server is not configured.")
        if not self.device_id:
            return ctx.t(
                "Questo computer non è registrato con pawn.", "This computer isn't registered with pawn."
            )
        if not self.in_conversation:
            self.start({}, ctx)
        if self._is_unlocked():
            return ctx.t("È già sbloccato.", "It's already unlocked.")
        if self.access_request:
            return ctx.t("Sto già aspettando il telefono.", "Already waiting for the phone.")
        minutes = max(1, int(self.unlock_max_s // 60))
        try:
            req = self.client.request_access(
                self.device_id, conversation=self.conversation, minutes=minutes
            )
        except PawnError as exc:
            logger.warning("pawn unlock_phone: %s", exc)
            if exc.status == 409:
                return ctx.t("Nessun telefono può confermare.", "No phone can confirm.")
            return ctx.t("Pawn non risponde.", "Pawn is not answering.")
        self.access_request = str(req["id"])
        conversation = self.conversation
        self._push_chat(ctx, status=ctx.t("conferma sul telefono…", "confirm on the phone…"))
        self.spawn(lambda: self.wait_access(ctx, self.access_request or "", conversation))
        return ctx.t("Conferma sul telefono.", "Confirm on your phone.")

    def wait_access(self, ctx: CommandContext, request_id: str, conversation: Optional[str]) -> str:
        """Poll the approval until it is final; announce the outcome. Returns the status."""
        status = "pending"
        deadline = self.clock() + 120.0
        req: dict[str, Any] = {}
        while status == "pending" and self.clock() < deadline:
            self.sleep(ACCESS_POLL_S)
            try:
                req = self.client.access_status(request_id)
            except PawnError as exc:
                logger.debug("pawn access status: %s", exc)
                continue
            status = str(req.get("status") or "pending")
        with self._lock:
            if self.access_request == request_id:
                self.access_request = None
            if self.conversation != conversation or conversation is None:
                return status  # the conversation ended meanwhile: grant nothing
            if status == "approved":
                minutes = float(req.get("minutes") or self.unlock_max_s / 60)
                self.access = "full"
                self.unlocked_until = self.clock() + min(minutes * 60.0, self.unlock_max_s)
                self.failures = 0
                text = ctx.t("Accesso completo.", "Full access unlocked.")
            elif status == "denied":
                text = ctx.t("Il telefono ha rifiutato.", "The phone declined.")
            else:
                text = ctx.t("Il telefono non ha risposto.", "The phone didn't answer.")
        logger.info("pawn phone unlock %s: %s", request_id, status)
        self._push_chat(ctx)
        ctx.announce(text, intent="pawn.unlock", beeps=0, command=self.name)
        return status

    @intent(
        "pawn.end",
        "the user wants to end the conversation with the pawn agent",
        examples=("fine conversazione", "chiudi la conversazione", "end conversation"),
        phrase=False,
    )
    def end(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        if not self.in_conversation:
            return ctx.t("Non stiamo parlando.", "We're not talking.")
        self._end(ctx)
        return ctx.t("Ok, a dopo.", "Okay, talk later.")

    @intent(
        "pawn.unlock",
        "the user wants to unlock full access (sessions skills) of the pawn agent",
        examples=("sblocca le sessioni", "unlock the sessions skills", "sblocca pawn"),
        phrase=False,
    )
    def unlock(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        if not self.in_conversation:
            if not self.client.configured:
                return ctx.t("Il server di pawn non è configurato.", "The pawn server is not configured.")
            self.start({}, ctx)  # unlocking implies talking: open the conversation too
        if not self.unlock_code:
            return ctx.t("Nessun codice configurato.", "No unlock code is configured.")
        if self._is_unlocked():
            return ctx.t("È già sbloccato.", "It's already unlocked.")
        if self.clock() < self.lockout_until:
            return ctx.t("Tastierino bloccato, riprova tra poco.", "Keypad locked, try again shortly.")
        ctx.ui.open(
            KEYPAD_WIDGET,
            "keypad",
            command=self.name,
            title="UNLOCK",
            data={"length": len(self.unlock_code), "result": "", "nonce": 0},
            anchor="center",
        )
        return ctx.t("Inserisci il codice.", "Enter the code.")

    @intent(
        "pawn.lock",
        "the user wants to lock the pawn agent back to chat only",
        examples=("blocca le sessioni", "lock the sessions"),
        phrase=False,
    )
    def lock(self, slots: dict[str, Any], ctx: CommandContext) -> str:
        if not self._is_unlocked():
            return ctx.t("È già bloccato.", "It's already locked.")
        self._relock()
        self._push_chat(ctx)
        return ctx.t("Accesso completo bloccato.", "Full access locked.")

    # ------------------------------------------------------------------ floor

    def converse(self, text: str, ctx: CommandContext) -> str:
        conversation = self.conversation
        if conversation is None:
            return ""
        self.mark_active()
        access = "full" if self._is_unlocked() else "chat"
        if access == "chat" and self.access == "full":  # deadline passed
            self._relock()
        self.turns.append({"who": "you", "text": text})
        self._push_chat(ctx, status=ctx.t("penso…", "thinking…"))

        cue = threading.Timer(self.thinking_cue_s, self.cue)
        cue.daemon = True
        cue.start()
        controls: list[str] = []
        try:
            extra: dict[str, Any] = {"device_id": self.device_id} if self.device_id else {}
            answer = self.client.voice(
                conversation,
                text,
                access=access,
                lang=ctx.lang,
                on_progress=lambda line: ctx.ui.update(CHAT_WIDGET, {"status": line}),
                **extra,
            )
            reply = answer.text.strip() or ctx.t("Non ho una risposta.", "I have no answer.")
            controls = list(answer.controls)
        except PawnError as exc:
            logger.warning("pawn voice turn failed: %s", exc)
            reply = ctx.t("Pawn non risponde.", "Pawn is not answering.")
        finally:
            cue.cancel()
        if self.conversation != conversation:  # ended meanwhile
            return reply
        if "lock" in controls and self.access == "full":
            logger.info("pawn agent asked to lock full access")
            self._relock()
        if "end" in controls:
            logger.info("pawn agent asked to end the conversation")
            self._end(ctx)
            return reply
        self.turns.append({"who": "pawn", "text": reply})
        self._push_chat(ctx, status=ctx.t("in ascolto", "listening"))
        return reply

    def on_floor_timeout(self, ctx: CommandContext) -> None:
        if ctx.ui.is_open(KEYPAD_WIDGET) or self.access_request:
            # Typing the code / confirming on the phone: keep the floor; the
            # unlock announcement re-opens the mic.
            self._push_chat(ctx, status=ctx.t("in pausa", "paused"))
            return
        if self.in_conversation:
            self._end(ctx)
            ctx.announce(ctx.t("Ok, a dopo.", "Okay, talk later."), intent="pawn.end", beeps=0)

    def on_cancel(self, ctx: CommandContext) -> Optional[str]:
        if not self.in_conversation:
            return None
        self._end(ctx)
        return ctx.t("Conversazione chiusa.", "Conversation closed.")

    def on_disable(self) -> None:
        if self.ctx is not None and self.in_conversation:
            self._end(self.ctx)

    # ------------------------------------------------------------------ UI

    def on_ui_action(self, widget_id: str, action: str, data: dict[str, Any]) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        if widget_id == CHAT_WIDGET and action == "close":
            if self.in_conversation:
                self._end(ctx)
            return
        if widget_id != KEYPAD_WIDGET:
            return
        if action == "close":
            ctx.ui.close(KEYPAD_WIDGET)
            return
        if action == "submit":
            self._submit(ctx, str(data.get("code") or ""))

    def _submit(self, ctx: CommandContext, code: str) -> None:
        with self._lock:
            now = self.clock()
            if not self.in_conversation:
                ctx.ui.close(KEYPAD_WIDGET)
                return
            if now < self.lockout_until:
                self._keypad(ctx, "locked")
                return
            expected = self.unlock_code
            ok = bool(expected) and hmac.compare_digest(
                code.encode("utf-8"), expected.encode("utf-8")
            )
            if ok:
                self.failures = 0
                self.access = "full"
                self.unlocked_until = now + self.unlock_max_s
                ctx.ui.close(KEYPAD_WIDGET)
                self._push_chat(ctx)
                logger.info("pawn full access unlocked for %s", self.conversation)
                ctx.announce(
                    ctx.t("Accesso completo.", "Full access unlocked."),
                    intent="pawn.unlock",
                    beeps=0,
                    command=self.name,
                )
                return
            self.failures += 1
            logger.warning("pawn unlock: wrong code (%d)", self.failures)
            if self.failures >= MAX_FAILURES:
                self.failures = 0
                self.lockout_until = now + self.lockout_s
                self._keypad(ctx, "locked")
                ctx.ui.close(KEYPAD_WIDGET)
                ctx.announce(
                    ctx.t("Troppi tentativi. Riprova più tardi.", "Too many attempts. Try later."),
                    intent="pawn.unlock",
                    beeps=0,
                    command=self.name,
                )
                return
            self._keypad(ctx, "wrong")
            ctx.announce(
                ctx.t("Codice errato.", "Wrong code."), intent="pawn.unlock", beeps=0, command=self.name
            )

    # ------------------------------------------------------------------ helpers

    def _is_unlocked(self) -> bool:
        return self.access == "full" and self.clock() < self.unlocked_until

    def _relock(self) -> None:
        self.access = "chat"
        self.unlocked_until = 0.0

    def _end(self, ctx: CommandContext) -> None:
        with self._lock:
            self.conversation = None
            self.access_request = None
            self._relock()
            self.turns = []
            if ctx.floor == self.name:
                ctx.floor = None
        ctx.ui.close(KEYPAD_WIDGET)
        ctx.ui.close(CHAT_WIDGET)

    def _keypad(self, ctx: CommandContext, result: str) -> None:
        nonce = int(ctx.ui.data(KEYPAD_WIDGET).get("nonce") or 0) + 1
        ctx.ui.update(KEYPAD_WIDGET, {"result": result, "nonce": nonce})

    def _chat_data(self, status: str = "") -> dict[str, Any]:
        data: dict[str, Any] = {
            "turns": list(self.turns[-MAX_TURNS_SHOWN:]),
            "access": "full" if self._is_unlocked() else "chat",
            "unlocked_until_s": max(0, int(self.unlocked_until - self.clock()))
            if self._is_unlocked()
            else 0,
        }
        if status:
            data["status"] = status
        return data

    def _push_chat(self, ctx: CommandContext, status: str = "") -> None:
        ctx.ui.update(CHAT_WIDGET, self._chat_data(status))


def _first_sentence(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", text)
    out = match.group(1) if match else text
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _spawn(fn: Callable[[], None]) -> None:
    thread = threading.Thread(target=fn, name="pawn-access-wait", daemon=True)
    thread.start()


def _thinking_cue() -> None:
    """Soft double blip while pawn is still thinking."""
    try:
        from pawn_assistant.audio.beep import play_beep

        play_beep(freq_hz=660.0, duration_ms=70, volume=0.12, block=True)
        play_beep(freq_hz=880.0, duration_ms=70, volume=0.12)
    except Exception:  # noqa: BLE001
        logger.debug("thinking cue failed", exc_info=True)
