"""LLM rephrasing of deterministic replies, for less robotic speech.

Handlers stay the ground truth: their canonical text is passed to the local LLM
as the *fact* to restate. The model only rewords it (sampling on, no grammar);
a guard checks every number / weekday / month survives and nothing new is
invented, and a deadline caps the wait. Any failure before the first sentence
falls back to the canonical text, so the user always hears a correct reply.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterator, Optional, Protocol

logger = logging.getLogger(__name__)

DEFAULT_PERSONA_IT = "Sei PAWN, un assistente vocale da HUD: asciutto, preciso, con un filo di ironia."
DEFAULT_PERSONA_EN = "You are PAWN, a HUD voice assistant: dry, precise, with a hint of irony."

# Pseudo-intent for spoken timer alerts (not in the intent registry).
ALERT_INTENT = "timer.alert"


class ChatStreamer(Protocol):
    def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]: ...


class PhraseFailed(Exception):
    """No usable phrased reply — caller speaks the canonical text instead."""


@dataclass
class PhraseSettings:
    lang: str = "it"
    persona: str = DEFAULT_PERSONA_IT
    temperature: float = 0.6
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_tokens: int = 60
    max_sentences: int = 2
    first_sentence_timeout_s: float = 2.5
    # Extra wait for each later sentence; on timeout we just stop talking.
    next_sentence_timeout_s: float = 3.0
    # None = every intent; otherwise only these (plus ALERT_INTENT if listed).
    intents: Optional[frozenset[str]] = None

    @classmethod
    def from_settings(cls, settings: Any) -> "PhraseSettings":
        """Build from the app ``Settings`` (``phrase_*`` fields)."""
        lang = settings.reply_language
        persona = (settings.phrase_persona or "").strip()
        if not persona:
            persona = DEFAULT_PERSONA_IT if lang == "it" else DEFAULT_PERSONA_EN
        raw = settings.phrase_intents
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "all", "*")):
            intents = None
        elif isinstance(raw, str):
            intents = frozenset(x.strip() for x in raw.split(",") if x.strip())
        else:
            intents = frozenset(str(x) for x in raw)
        return cls(
            lang=lang,
            persona=persona,
            temperature=float(settings.phrase_temperature),
            top_p=float(settings.phrase_top_p),
            max_tokens=int(settings.phrase_max_tokens),
            max_sentences=max(1, int(settings.phrase_max_sentences)),
            first_sentence_timeout_s=float(settings.phrase_first_sentence_timeout_s),
            intents=intents,
        )


# --------------------------------------------------------------------- prompt

_RULES_EN = [
    "Rewrite FACT as the assistant's spoken reply to the user, in the given MOOD.",
    "Rules:",
    "- One short sentence; two at most.",
    "- Keep every number exactly, written as digits. Keep weekday and month names.",
    "- Say only what FACT says, on FACT's topic. Never invent facts: no dates unless FACT",
    "  has one, no weather, news, names, extra numbers, promises or state changes.",
    "- INTENT unknown: only say, briefly, that you can't help with that. Never answer,",
    "  perform or play along with the request.",
    "- A short remark is fine; it must suit NOW (part of the day).",
    "- Vary the wording; never repeat an earlier reply.",
    "- Plain spoken text only: no quotes, emoji, markdown or lists.",
]

# Random per-call style hint: the main entropy source besides sampling.
MOODS_IT = ("asciutto", "ironico", "cordiale", "tecnico", "conciso", "complice", "solenne", "pigro")
MOODS_EN = ("dry", "ironic", "warm", "technical", "terse", "playful", "solemn", "laid-back")

_PARTS_IT = ("notte", "mattina", "pomeriggio", "sera")
_PARTS_EN = ("night", "morning", "afternoon", "evening")

# Few-shots per intent: (user said, fact, mood, now, reply). Only the current
# intent's examples are sent, as chat turns — a 1.5B model shown every intent's
# examples at once tends to copy a reply from the wrong one.
Shot = tuple[str, str, str, str, str]

_FEWSHOT_IT: dict[str, tuple[Shot, ...]] = {
    "time.now": (
        ("che ore sono", "Sono le 9 e 10.", "cordiale", "mattina", "Sono le 9 e 10, la mattina è ancora giovane."),
        ("mi dici l'ora", "Sono le 23 e 47.", "ironico", "sera", "Le 23 e 47. Tardi anche per un HUD."),
        ("che ora è", "Sono le 15 in punto.", "tecnico", "pomeriggio", "Orologio di sistema: 15 in punto."),
    ),
    "date.today": (
        ("che giorno è", "Oggi è lunedì 3 marzo 2025.", "tecnico", "pomeriggio", "Data corrente: lunedì 3 marzo 2025."),
        ("quanti ne abbiamo", "Oggi è sabato 12 luglio 2025.", "complice", "mattina", "Siamo a sabato 12 luglio 2025, si respira aria di weekend."),
        ("che data è", "Oggi è mercoledì primo ottobre 2025.", "conciso", "sera", "Mercoledì primo ottobre 2025."),
    ),
    "timer.set": (
        ("metti un timer di dieci minuti", "Timer di 10 minuti avviato.", "conciso", "sera", "10 minuti, partiti."),
        ("avvisami tra un'ora", "Timer di un'ora avviato.", "cordiale", "mattina", "D'accordo, ti avviso tra un'ora."),
        ("timer di 30 secondi", "Timer di 30 secondi avviato.", "ironico", "pomeriggio", "30 secondi. Non sbattere le palpebre."),
    ),
    "cancel": (
        ("annulla il timer", "Timer annullato.", "complice", "pomeriggio", "Timer annullato, facciamo finta di niente."),
        ("ferma tutto", "2 timer annullati.", "tecnico", "sera", "2 timer annullati, conto alla rovescia azzerato."),
        ("stop", "Ok, nessun timer attivo.", "asciutto", "mattina", "Niente da fermare, nessun timer attivo."),
    ),
    "greeting": (
        ("ciao come stai", "Ciao. Tutto operativo.", "solenne", "mattina", "Buongiorno. Ogni sistema risponde all'appello."),
        ("ehi", "Ciao. Tutto operativo.", "pigro", "notte", "Ehi. Sveglio, più o meno."),
        ("buonasera", "Ciao. Tutto operativo.", "cordiale", "sera", "Buonasera. Qui tutto in ordine."),
    ),
    "unknown": (
        ("che tempo fa domani", "Non ho capito.", "asciutto", "pomeriggio", "Non è roba per me."),
        ("raccontami una storia", "Non ho capito.", "ironico", "sera", "Storie? Non è il mio reparto."),
        ("ordina una pizza", "Non ho capito.", "cordiale", "mattina", "Mi dispiace, questo non lo so fare."),
    ),
    ALERT_INTENT: (
        ("-", "Il timer di 5 minuti è scaduto.", "tecnico", "pomeriggio", "Conto alla rovescia di 5 minuti completato."),
        ("-", "Il timer di un'ora è scaduto.", "complice", "sera", "Ehi, l'ora è passata. Il timer è scaduto."),
    ),
}

_FEWSHOT_EN: dict[str, tuple[Shot, ...]] = {
    "time.now": (
        ("what time is it", "It's 9:10.", "warm", "morning", "It's 9:10, the morning is still young."),
        ("what's the time", "It's 23:47.", "ironic", "evening", "23:47. Late, even for a HUD."),
    ),
    "date.today": (
        ("what day is it", "Today is Monday, March 3, 2025.", "technical", "afternoon", "Current date: Monday, March 3, 2025."),
        ("what's the date", "Today is Saturday, July 12, 2025.", "playful", "morning", "Saturday, July 12, 2025. Weekend mode."),
    ),
    "timer.set": (
        ("set a timer for ten minutes", "Timer for 10 minutes started.", "terse", "evening", "10 minutes, running."),
        ("timer for 30 seconds", "Timer for 30 seconds started.", "ironic", "afternoon", "30 seconds. Don't blink."),
    ),
    "cancel": (
        ("cancel the timer", "Timer cancelled.", "playful", "afternoon", "Timer cancelled. It never happened."),
        ("stop", "Okay, no timer running.", "dry", "morning", "Nothing to stop, no timer running."),
    ),
    "greeting": (
        ("hi how are you", "Hello. All systems nominal.", "solemn", "morning", "Good morning. Every system reports for duty."),
        ("hey", "Hello. All systems nominal.", "laid-back", "night", "Hey. Awake, more or less."),
    ),
    "unknown": (
        ("what's the weather tomorrow", "I didn't understand.", "dry", "afternoon", "Not my department."),
        ("tell me a story", "I didn't understand.", "ironic", "evening", "Stories? Not my department."),
    ),
    ALERT_INTENT: (
        ("-", "Your 5 minutes timer is done.", "technical", "afternoon", "5 minute countdown complete."),
    ),
}


def fewshot_turns(intent: str, lang: str = "it") -> list[dict[str, str]]:
    """Chat turns demonstrating ``intent`` (static per intent → cacheable prefix)."""
    table = _FEWSHOT_IT if lang == "it" else _FEWSHOT_EN
    turns: list[dict[str, str]] = []
    for said, fact, mood, now, reply in table.get(intent, ()):
        turns.append({"role": "user", "content": user_message(intent, said, fact, mood, now)})
        turns.append({"role": "assistant", "content": reply})
    return turns


def part_of_day(hour: int, lang: str = "it") -> str:
    parts = _PARTS_IT if lang == "it" else _PARTS_EN
    if hour < 6:
        return parts[0]
    if hour < 13:
        return parts[1]
    if hour < 19:
        return parts[2]
    return parts[3]


def user_message(intent: str, user_text: str, canonical: str, mood: str, now: str) -> str:
    said = user_text.strip()[:200] or "-"
    return f'INTENT: {intent}\nUSER: "{said}"\nFACT: "{canonical}"\nMOOD: {mood}\nNOW: {now}'


def build_phrase_prompt(persona: str, lang: str = "it") -> str:
    """Static system prompt — byte-identical across calls so its KV prefix is cached."""
    language = "Italian" if lang == "it" else "English"
    lines = [persona.strip(), "", *_RULES_EN, f"- Always reply in {language}."]
    return "\n".join(lines)


# --------------------------------------------------------------------- guard

_DIGITS = re.compile(r"\d+")
_WORD = re.compile(r"[a-zà-ÿ]+", re.IGNORECASE)
_PREFIX = re.compile(r"^\s*(?:REPLY|PAWN|Risposta)\s*:\s*", re.IGNORECASE)
_BAD = re.compile(r"[{}*#<>\[\]|`_\"“”«»]|INTENT:|FACT:|USER:")
_ONE = re.compile(r"\b(?:un|uno|una|primo|one|first|an?)\b|\bun'", re.IGNORECASE)

_CALENDAR_WORDS = frozenset(
    "lunedì martedì mercoledì giovedì venerdì sabato domenica "
    "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre "
    "monday tuesday wednesday thursday friday saturday sunday "
    "january february march april may june july august september october november december".split()
)

MAX_SENTENCE_CHARS = 200
# Refusals stay short: a long "unknown" reply means the model played along.
_INTENT_LIMITS: dict[str, tuple[int, int]] = {"unknown": (1, 80)}  # (sentences, chars)


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Complete sentences in ``buffer`` and the unfinished tail.

    A sentence ends at ``.?!…`` followed by whitespace; the end of the text is
    left in the tail (more tokens may follow); the caller flushes it at stream end.
    """
    out: list[str] = []
    start = 0
    for m in re.finditer(r"[.!?…]+(?=\s)", buffer):
        piece = buffer[start : m.end()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    return out, buffer[start:]


def clean_sentence(text: str) -> Optional[str]:
    s = _PREFIX.sub("", text).strip().strip('"“”«»\'').strip()
    if not s or len(s) > MAX_SENTENCE_CHARS or _BAD.search(s):
        return None
    if not re.search(r"[A-Za-zÀ-ÿ]", s):
        return None
    return s


# Meaning that must survive besides numbers: duration units and negation.
# (stems in the canonical text, stems accepted in the reply)
_UNIT_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("minut",), ("minut",)),
    (("second",), ("second",)),
    (("ore", "un'ora", "hour"), ("ora", "ore", "un'ora", "l'ora", "hour")),
)
_NEGATION = ("non", "nessun", "niente", "nulla", "no", "not", "nothing", "didn")


def _words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zà-ÿ']+", text, re.IGNORECASE)}


def _stems_present(words: set[str], stems: tuple[str, ...]) -> bool:
    """Exact match for short stems, prefix match for longer ones."""
    return any(w == s or (len(s) > 3 and w.startswith(s)) for w in words for s in stems)


def _negated(words: set[str]) -> bool:
    return _stems_present(words, _NEGATION) or any(w.endswith("n't") for w in words)


def required_facts(canonical: str) -> tuple[set[str], set[str]]:
    """(digit tokens, calendar words) that every phrased reply must keep."""
    digits = set(_DIGITS.findall(canonical))
    words = {w.lower() for w in _WORD.findall(canonical)} & _CALENDAR_WORDS
    return digits, words


def meaning_kept(text: str, canonical: str) -> bool:
    """Duration units and negation of the canonical text are still there.

    Catches "30 secondi" for "un minuto e 30 secondi" and "Timer annullato"
    for "nessun timer attivo" — errors the number check cannot see.
    """
    have, want = _words(text), _words(canonical)
    for in_canonical, accepted in _UNIT_GROUPS:
        if _stems_present(want, in_canonical) and not _stems_present(have, accepted):
            return False
    return not _negated(want) or _negated(have)


def allowed_numbers(canonical: str) -> set[str]:
    allowed = set(_DIGITS.findall(canonical))
    if _ONE.search(canonical):  # "un minuto" / "primo" → "1 minuto" is fine
        allowed.add("1")
    return allowed


def facts_kept(text: str, canonical: str) -> bool:
    digits, words = required_facts(canonical)
    found_digits = set(_DIGITS.findall(text))
    found_words = {w.lower() for w in _WORD.findall(text)}
    return digits <= found_digits and words <= found_words


def no_new_facts(text: str, canonical: str) -> bool:
    if not set(_DIGITS.findall(text)) <= allowed_numbers(canonical):
        return False
    words = {w.lower() for w in _WORD.findall(text)} & _CALENDAR_WORDS
    _, required_words = required_facts(canonical)
    return words <= required_words


# --------------------------------------------------------------------- phraser


@dataclass
class PhraseStats:
    phrased: bool = False
    first_ms: float = 0.0
    gen_ms: float = 0.0
    reason: str = ""
    sentences: list[str] = field(default_factory=list)


_END = object()


class ReplyPhraser:
    def __init__(
        self,
        llm: ChatStreamer,
        settings: Optional[PhraseSettings] = None,
        *,
        now: Callable[[], datetime] = datetime.now,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or PhraseSettings()
        self.system_prompt = build_phrase_prompt(self.settings.persona, self.settings.lang)
        self.now = now
        self.rng = rng or random.Random()
        self._last_mood = ""

    def _pick_mood(self) -> str:
        moods = MOODS_IT if self.settings.lang == "it" else MOODS_EN
        mood = self.rng.choice([m for m in moods if m != self._last_mood])
        self._last_mood = mood
        return mood

    def applies(self, intent: str) -> bool:
        allowed = self.settings.intents
        return allowed is None or intent in allowed

    def messages(
        self, intent: str, user_text: str, canonical: str, mood: Optional[str] = None
    ) -> list[dict[str, str]]:
        mood = mood or self._pick_mood()
        now = part_of_day(self.now().hour, self.settings.lang)
        return [
            {"role": "system", "content": self.system_prompt},
            *fewshot_turns(intent, self.settings.lang),
            {"role": "user", "content": user_message(intent, user_text, canonical, mood, now)},
        ]

    def warmup(self) -> None:
        """Prime each enabled intent's prompt prefix (system + its few-shots).

        A cold prefix costs ~2 s of prefill on 4 cores — most of the first-
        sentence budget. The shared LLM's RAM cache keeps them (~11 MB each).
        """
        table = _FEWSHOT_IT if self.settings.lang == "it" else _FEWSHOT_EN
        for intent in table:
            if not self.applies(intent):
                continue
            for _ in self.llm.stream_chat(
                self.messages(intent, "-", "Ok.", mood="-"),
                max_tokens=1,
                temperature=0.0,
            ):
                pass

    # -- generation ------------------------------------------------------

    def _raw_sentences(
        self, intent: str, user_text: str, canonical: str, abort: threading.Event
    ) -> Iterator[str]:
        s = self.settings
        stream = self.llm.stream_chat(
            self.messages(intent, user_text, canonical),
            temperature=s.temperature,
            top_p=s.top_p,
            repeat_penalty=s.repeat_penalty,
            max_tokens=s.max_tokens,
            stop=["\n"],
        )
        buf = ""
        try:
            for piece in stream:
                if abort.is_set():  # stop decoding at once: frees the model lock
                    return
                buf += piece
                done, buf = split_sentences(buf)
                yield from done
            if buf.strip():
                yield buf.strip()
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    def sentences(
        self, intent: str, user_text: str, canonical: str, stats: Optional[PhraseStats] = None
    ) -> Iterator[str]:
        """Validated sentences, generated on a worker thread under a deadline.

        Raises ``PhraseFailed`` if nothing usable arrives for the first sentence
        (guard, timeout, error). Later problems just end the reply early.
        """
        s = self.settings
        stats = stats if stats is not None else PhraseStats()
        out: "queue.Queue[Any]" = queue.Queue()
        abort = threading.Event()
        t0 = time.perf_counter()

        def _produce() -> None:
            gen = self._raw_sentences(intent, user_text, canonical, abort)
            try:
                for sent in gen:
                    if abort.is_set():
                        break
                    out.put(sent)
            except Exception as exc:  # noqa: BLE001
                out.put(exc)
            finally:
                gen.close()
                out.put(_END)

        worker = threading.Thread(target=_produce, name="pawn-phrase", daemon=True)
        worker.start()
        try:
            max_sentences, max_chars = _INTENT_LIMITS.get(intent, (s.max_sentences, MAX_SENTENCE_CHARS))
            max_sentences = min(max_sentences, s.max_sentences)
            count = 0
            while count < max_sentences:
                timeout = s.first_sentence_timeout_s if count == 0 else s.next_sentence_timeout_s
                remaining = timeout - (time.perf_counter() - t0) if count == 0 else timeout
                try:
                    item = out.get(timeout=max(0.0, remaining))
                except queue.Empty:
                    if count == 0:
                        raise PhraseFailed("timeout")
                    stats.reason = "next-timeout"
                    return
                if item is _END:
                    if count == 0:
                        raise PhraseFailed("empty")
                    return
                if isinstance(item, Exception):
                    if count == 0:
                        raise PhraseFailed(f"error: {item}") from item
                    logger.warning("phrasing stopped: %s", item)
                    return
                sent = clean_sentence(item)
                if sent is not None and len(sent) > max_chars:
                    sent = None
                if count == 0:
                    if (
                        sent is None
                        or not facts_kept(sent, canonical)
                        or not no_new_facts(sent, canonical)
                        or not meaning_kept(sent, canonical)
                    ):
                        raise PhraseFailed(f"guard: {item!r}")
                    stats.first_ms = (time.perf_counter() - t0) * 1000.0
                    stats.phrased = True
                elif sent is None or not no_new_facts(sent, canonical):
                    stats.reason = f"dropped: {item!r}"
                    return
                count += 1
                stats.sentences.append(sent)
                yield sent
        finally:
            abort.set()
            stats.gen_ms = (time.perf_counter() - t0) * 1000.0

    def phrase(self, intent: str, user_text: str, canonical: str) -> tuple[str, PhraseStats]:
        """Whole phrased reply (blocking), or the canonical text on failure."""
        stats = PhraseStats()
        try:
            text = " ".join(self.sentences(intent, user_text, canonical, stats))
        except PhraseFailed as exc:
            stats.reason = str(exc)
            return canonical, stats
        return text, stats
