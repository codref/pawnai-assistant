# Point 2 — local command words

Turn a transcript into an action, offline, on the i5 / 8 GB box. The LLM **classifies and extracts slots**; replies come from deterministic Python handlers. The same LLM then **rewords** each reply (sampling on) so PAWN does not sound canned — a guard keeps the facts exact — and a local Piper voice speaks it sentence by sentence.

## Pipeline

```
transcript ─▶ fast-path ──hit──▶ handler ─▶ canonical ─▶ phraser ─▶ Piper ─▶ speaker
                 │ miss              ▲                       │ guard/timeout fail
                 └─▶ IntentLLM ──────┘   (GBNF JSON)         └─▶ canonical ─▶ Piper
```

| Stage | Module | Cost (4 cores) |
|-------|--------|----------------|
| Fast-path: normalise + phrase / regex match | `commands/fastpath.py` | < 5 ms |
| Intent LLM: Qwen2.5-1.5B-Instruct Q4_K_M, llama.cpp | `commands/llm.py` | p50 ≈ 0.5 s, p95 ≈ 0.85 s |
| Handler (time, date, timers, …) | `commands/handlers.py` | ~0 |
| Reply phrasing: same Qwen, free text, streamed | `commands/phrasing.py` | first sentence p50 ≈ 0.5–0.8 s, p95 ≈ 1.2 s |
| Piper `it_IT-paola-medium` | `audio/tts.py` | ~60 ms synth / 2 s audio |

About half of the eval set never touches the LLM.

## Intents

Registry in `commands/intents.py` — the single source for fast-path phrases, LLM prompt, few-shots and grammar.

| Intent | Slots | Example |
|--------|-------|---------|
| `time.now` | — | "che ora è" |
| `date.today` | — | "che giorno è oggi" |
| `timer.set` | `hours`, `minutes`, `seconds` | "metti un timer di cinque minuti" |
| `cancel` | — | "annulla il timer", "stop" |
| `greeting` | — | "ciao" |
| `unknown` | — | anything else → "Non ho capito." |

**Adding a command:** add an `Intent` (description, examples, slots, few-shots) and a handler in `HANDLERS`. Prompt and grammar rebuild automatically. Add cases to `tests/data/commands_it.yaml` and run the eval.

## LLM details

- Grammar: one alternative per intent; only that intent's slots, all optional. Output is typically `{"intent": "time.now"}` — a handful of tokens.
- Static system prompt (~430 tokens) → llama.cpp reuses the KV prefix; only the utterance is prefilled.
- The prompt KV state (~13 MB) is saved on idle unload and restored on reload: a cold reload + parse costs ~1.3 s instead of ~5–9 s.
- Warmed up at `serve` start and prefetched on wake (while the user is still speaking).
- `use_mmap=False`: with mmap, llama.cpp's CPU weight repacking keeps two copies resident (~1.9 GB vs ~1.3 GB).
- `qwen2.5-0.5b` is selectable but scored poorly on slots / off-domain in testing — not recommended.

## Reply phrasing

The handler output is the **fact**; the LLM only rewords it (`temperature` 0.6, fresh seed per call, no grammar, `stop="\n"`, ≤ 2 sentences). Each call also gets a random **MOOD** (asciutto, ironico, cordiale, …) and the real **NOW** (part of day) — the mood is the main entropy source; above ~0.7 the 1.5B model's Italian gets odd.

- **Prompt:** static system prompt (persona + rules) and only the **current intent's** few-shots as chat turns. Showing every intent's examples made the 1.5B model copy replies from the wrong intent.
- **Guard:** the first sentence must keep every digit, weekday/month, duration unit and negation of the canonical text and add no number or date (`un`/`primo` may become `1`). Later sentences that add a number or date are dropped. `unknown` is capped at one sentence ≤ 80 chars (a longer one means the model played along). Markup, quotes, prompt echoes → rejected.
- **Deadline:** no valid first sentence within `first_sentence_timeout_s` (2.5 s) → the canonical text is spoken. A cold LLM reload counts against it, so timer alerts after a long idle usually fall back.
- **Streaming:** the model runs on a worker thread; sentence 1 is synthesized and played while sentence 2 is still decoding. Aborting stops decoding at the next token and releases the model lock.
- **KV cache:** intent parsing and phrasing share one model. A `LlamaRAMCache` (128 MB, ~11 MB per prefix) restores the longest cached prefix, so switching prompts never re-prefills; it survives idle unloads. Warmup primes the parse prompt and every enabled intent's phrasing prefix (a cold one costs ~2 s of prefill — most of the budget; ~15 s total at `serve` start, in the background).
- **Persona / scope:** `phrase.persona` (system prompt) and `phrase.intents` (`all` or a list; `timer.alert` = spoken alerts) in the YAML.

```bash
pawn-assistant ask --phrase -n 10 "che ora è"     # variants + first-sentence p50/p95 + fallback rate
pawn-assistant ask --no-phrase "che ora è"        # canonical only
```

Measured on 4 cores (`taskset -c 0-3`), 8 samples per command, temperature 0.55–0.6:

| Command | Phrased | First sentence p50 / p95 |
|---------|---------|--------------------------|
| che ora è | 7/8 | 0.67 / 1.2 s |
| timer 90 s | 6/8 | 0.68 / 1.1 s |
| barzelletta (unknown) | 8/8 | 0.54 / 0.9 s |
| ciao / annulla / data | 8/8 | 0.5–0.8 / 1.0–1.3 s |

Fallbacks are guard rejections (e.g. "30 secondi" for 1 min 30 s); the user hears the canonical text, so facts stay exact. Remarks are still sometimes clumsy — a 1.5B model's limit.

## States / events

New states: `THINK` (intent parsing), `SPEAK` (TTS playing). Mic frames are ignored in `BUSY`/`THINK`/`SPEAK` (no AEC yet), and the wake debounce after `SPEAK` covers the speaker tail.

```json
{"type": "event", "name": "intent", "intent": "timer.set", "slots": {"minutes": 5}, "source": "fastpath", "parse_ms": 0.4}
{"type": "event", "name": "reply", "kind": "reply", "seq": 0, "text": "Fatto, 5 minuti sul conto alla rovescia.", "phrased": true, "first_ms": 980.0, "tts_ms": 61.2}
{"type": "event", "name": "reply", "kind": "reply", "seq": 1, "text": "Fatto, 5 minuti sul conto alla rovescia. Ti avviso io.", "phrased": true, "first_ms": 980.0, "tts_ms": 40.3}
{"type": "event", "name": "reply", "kind": "alert", "text": "Il timer di 5 minuti è scaduto.", "tts_ms": 70.0}
```

Meter adds `llm_resident`, `tts_resident`.

`text` is the reply so far; the HUD replaces the line on `seq > 0`. A fallback reply has `phrased: false`.

Timer alerts queue up and play (3 beeps + speech) only when the loop is `IDLE`, never over an active turn.

## Memory (measured)

| Resident | RSS |
|----------|-----|
| Always-on loop (wake + VAD + server) | ~0.6 GB |
| + Whisper `base` int8 | +0.35 GB |
| + intent LLM 1.5B | +1.3 GB |
| + Piper medium | +0.15 GB |

All three on-demand models unload after idle (`idle_unload_s`).

## Dev loop

```bash
pawn-assistant models pull
pawn-assistant ask "che ora è"
pawn-assistant ask --eval tests/data/commands_it.yaml
pawn-assistant say "Sono le 14 e 35."
taskset -c 0-3 pawn-assistant ask --eval tests/data/commands_it.yaml   # approximate the i5
```

## Licensing note

`piper-tts` ≥ 1.3 is **GPL-3.0-or-later**. Fine for local/personal use; revisit before distributing this package under MIT.
