"""Small local LLM (llama.cpp) that maps an utterance to ``{"intent", "slots"}``.

The output is forced by a GBNF grammar built from the enabled intents, so the
model can only emit a known intent name and integer/string slots — no free text.
Toggling a command rebuilds prompt + grammar (``IntentLLM.set_intents``).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from pawn_assistant.commands.base import SLOT_INT, UNKNOWN, Intent

logger = logging.getLogger(__name__)

# Alias → (HF repo, GGUF filename).
MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
}


def hf_cached_download(repo_id: str, filename: str) -> Path:
    """Use the local HF cache when present (no network); download otherwise."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
    except Exception:  # noqa: BLE001 — not cached yet
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def resolve_model_path(model: str) -> Path:
    """Local .gguf path, or download an alias into the HF cache."""
    path = Path(model).expanduser()
    if path.suffix == ".gguf" and path.is_file():
        return path
    key = model.lower().removesuffix("-instruct").removesuffix("-q4_k_m")
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"unknown intent model {model!r}: use {sorted(MODEL_ALIASES)} or a .gguf path"
        )
    repo, filename = MODEL_ALIASES[key]
    return hf_cached_download(repo, filename)


# --------------------------------------------------------------------- prompt / grammar


def _resolve(intents: Optional[Iterable[Intent]]) -> tuple[Intent, ...]:
    if intents is None:
        from pawn_assistant.commands.registry import default_registry

        return default_registry().intents()
    return tuple(intents)


def all_slot_names(intents: Optional[Iterable[Intent]] = None) -> dict[str, str]:
    """Union of every slot across intents (name → type), in registry order."""
    out: dict[str, str] = {}
    for intent in _resolve(intents):
        for name, typ in intent.slots.items():
            out.setdefault(name, typ)
    return out


def build_grammar(intents: Optional[Iterable[Intent]] = None) -> str:
    """GBNF: one alternative per intent; only that intent's slots, all optional.

    ``{"intent": "time.now"}`` / ``{"intent": "timer.set", "minutes": 5}``.
    Omitting unmentioned slots keeps the output to a handful of tokens, and
    slot-less intents cannot hallucinate slots.
    """
    intents = _resolve(intents)
    slot_order = all_slot_names(intents)
    alts, rules = [], []
    for idx, intent in enumerate(intents):
        rule = f"i{idx}"
        name = json.dumps(json.dumps(intent.name))
        parts = [f'"{{" ws "\\"intent\\"" ws ":" ws {name}']
        for slot in slot_order:  # registry order == _example_json order
            if slot not in intent.slots:
                continue
            value = "int" if intent.slots[slot] == SLOT_INT else "str"
            parts.append(f'("," ws {json.dumps(json.dumps(slot))} ws ":" ws {value})?')
        parts.append('ws "}"')
        rules.append(f"{rule} ::= " + " ".join(parts))
        alts.append(rule)
    return "\n".join(
        [
            "root ::= " + " | ".join(alts),
            *rules,
            "int ::= [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9] | [1-9] [0-9] [0-9] [0-9]",
            'str ::= "\\"" [^"\\\\\\n]{0,40} "\\""',
            "ws ::= [ ]?",
        ]
    )


def _example_json(intent: str, slots: dict[str, Any], slot_order: Sequence[str]) -> str:
    out: dict[str, Any] = {"intent": intent}
    for slot in slot_order:  # registry order == grammar order
        if slots.get(slot) is not None:
            out[slot] = slots[slot]
    return json.dumps(out, ensure_ascii=False)


def build_system_prompt(intents: Optional[Iterable[Intent]] = None) -> str:
    """Static prompt — keep byte-identical across calls so llama.cpp reuses the KV prefix."""
    intents = _resolve(intents)
    slot_order = list(all_slot_names(intents))
    lines = [
        "You are the command parser of a voice assistant. The user speaks Italian or English;",
        "the text comes from speech recognition and may contain mistakes.",
        "Reply with ONE JSON object: the intent plus only the slots the user mentioned.",
        "",
        "Intents:",
    ]
    for intent in intents:
        slots = f" (slots: {', '.join(intent.slots)})" if intent.slots else ""
        lines.append(f"- {intent.name}: {intent.description}{slots}")
    lines += ["", "Examples:"]
    for intent in intents:
        for utterance, slots in intent.fewshot:
            lines.append(f"{utterance} => {_example_json(intent.name, slots, slot_order)}")
        for example in intent.examples[:2]:
            lines.append(f"{example} => {_example_json(intent.name, {}, slot_order)}")
    return "\n".join(lines)


@dataclass
class LLMParse:
    intent: str
    slots: dict[str, Any]
    infer_ms: float
    raw: str


def parse_output(raw: str, names: Optional[Iterable[str]] = None) -> tuple[str, dict[str, Any]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON: %r", raw)
        return UNKNOWN, {}
    intent = data.pop("intent", UNKNOWN)
    known = set(names) if names is not None else {i.name for i in _resolve(None)}
    if intent not in known:
        intent = UNKNOWN
    slots = {k: v for k, v in data.items() if v is not None}
    return intent, slots


# --------------------------------------------------------------------- runtime


class IntentLLM:
    """Lazy llama.cpp wrapper with idle unload (mirrors ``WhisperASR``)."""

    def __init__(
        self,
        model: str = "qwen2.5-1.5b",
        n_threads: int = 4,
        idle_unload_s: float = 300.0,
        n_ctx: int = 1024,  # ~430-token prompt + utterance + short JSON
        # ~11 MB per cached prompt state: intent parse + one per phrasing intent.
        cache_bytes: int = 128 * 1024 * 1024,
        intents: Optional[Iterable[Intent]] = None,
    ) -> None:
        self.model = model
        self.n_threads = n_threads
        self.idle_unload_s = idle_unload_s
        self.n_ctx = n_ctx
        self._intents = _resolve(intents)
        self._names = frozenset(i.name for i in self._intents)
        self.system_prompt = build_system_prompt(self._intents)
        self._grammar_text = build_grammar(self._intents)
        self._llm = None
        self._grammar = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._unload_timer: Optional[threading.Timer] = None
        # Prompt KV states (~11 MB each for 1.5B), keyed by token prefix. Two
        # static system prompts share the model (intent parse + reply phrasing),
        # so a single "last prompt" reuse would re-prefill on every switch; the
        # RAM cache restores the longest matching prefix for either. Kept across
        # idle unloads so a reload skips the multi-second prefill.
        self.cache_bytes = cache_bytes
        self._cache = None
        self._warmups: list[Any] = []

    @property
    def resident(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        with self._lock:
            if self._llm is not None:
                return
            from llama_cpp import Llama, LlamaGrammar

            path = resolve_model_path(self.model)
            logger.info("loading intent LLM %s (threads=%d)", path.name, self.n_threads)
            t0 = time.perf_counter()
            self._llm = Llama(
                model_path=str(path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=512,
                # mmap + llama.cpp's CPU weight repacking keeps two copies
                # resident (~1.9 GB vs ~1.3 GB for the 1.5B model).
                use_mmap=False,
                verbose=False,
            )
            self._grammar = LlamaGrammar.from_string(self._grammar_text, verbose=False)
            if self.cache_bytes > 0:
                from llama_cpp import LlamaRAMCache

                if self._cache is None:
                    self._cache = LlamaRAMCache(capacity_bytes=self.cache_bytes)
                self._llm.set_cache(self._cache)
            logger.info("intent LLM loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
            self._last_used = time.monotonic()

    def unload(self) -> None:
        with self._lock:
            if self._llm is None:
                return
            logger.info("unloading intent LLM")
            close = getattr(self._llm, "close", None)
            if close is not None:
                close()
            self._llm = None
            self._grammar = None

    def _schedule_unload(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        if self.idle_unload_s <= 0:
            return

        def _fire() -> None:
            if time.monotonic() - self._last_used >= self.idle_unload_s:
                self.unload()

        self._unload_timer = threading.Timer(self.idle_unload_s, _fire)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def set_intents(self, intents: Iterable[Intent], *, warm: bool = True) -> None:
        """Swap prompt + grammar after a command was toggled.

        The KV RAM cache keeps the old prefix too, so toggling back is cheap.
        If the model is resident the new prefix is primed in the background.
        """
        intents = tuple(intents)
        prompt, grammar_text = build_system_prompt(intents), build_grammar(intents)
        with self._lock:
            if prompt == self.system_prompt and grammar_text == self._grammar_text:
                return
            self._intents = intents
            self._names = frozenset(i.name for i in intents)
            self.system_prompt = prompt
            self._grammar_text = grammar_text
            if self._llm is not None:
                from llama_cpp import LlamaGrammar

                self._grammar = LlamaGrammar.from_string(grammar_text, verbose=False)
            resident = self._llm is not None
        logger.info("intent LLM prompt rebuilt for %d intents", len(intents))
        if warm and resident:
            threading.Thread(target=self.warmup, name="pawn-llm-rewarm", daemon=True).start()

    def add_warmup(self, fn: Any) -> None:
        """Extra prefix to prime in ``warmup`` (e.g. the reply-phrasing prompt)."""
        self._warmups.append(fn)

    def warmup(self) -> None:
        """Load and evaluate the static prompt(s) once so the first command is fast."""
        self.parse("ciao")
        for fn in self._warmups:
            try:
                fn()
            except Exception:  # noqa: BLE001
                logger.exception("LLM warmup hook failed")

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        max_tokens: int = 60,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        """Free-text sampling (no grammar), yielding content pieces as they decode.

        Holds the model lock for the whole generation; closing the iterator early
        aborts it. ``seed=None`` draws a fresh one — llama-cpp would otherwise
        reuse the same derived seed every call and kill the variety.
        """
        import random

        self.load()
        assert self._llm is not None
        if seed is None:
            seed = random.SystemRandom().randrange(2**31)
        with self._lock:
            try:
                stream = self._llm.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=seed,
                    stream=True,
                )
                try:
                    for chunk in stream:
                        piece = chunk["choices"][0]["delta"].get("content")
                        if piece:
                            yield piece
                finally:
                    close = getattr(stream, "close", None)
                    if close is not None:
                        close()
            finally:
                self._last_used = time.monotonic()
                self._schedule_unload()

    def parse(self, text: str) -> LLMParse:
        self.load()
        assert self._llm is not None
        t0 = time.perf_counter()
        with self._lock:
            out = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text.strip()[:300]},
                ],
                grammar=self._grammar,
                temperature=0.0,
                max_tokens=64,
            )
            self._last_used = time.monotonic()
        raw = (out["choices"][0]["message"]["content"] or "").strip()
        infer_ms = (time.perf_counter() - t0) * 1000.0
        intent, slots = parse_output(raw, self._names)
        self._schedule_unload()
        return LLMParse(intent=intent, slots=slots, infer_ms=infer_ms, raw=raw)
