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
| Handler (plugin method) | `commands/builtin/*` | ~0 |
| Reply phrasing: same Qwen, free text, streamed | `commands/phrasing.py` | first sentence p50 ≈ 0.5–0.8 s, p95 ≈ 1.2 s |
| Piper `it_IT-paola-medium` | `audio/tts.py` | ~60 ms synth / 2 s audio |

About half of the eval set never touches the LLM.

## Command plugins

Every command is a plugin: one `Command` subclass whose decorated methods are the intents. The registry (`commands/registry.py`) builds the fast-path phrase table, the LLM prompt + grammar and the phrasing few-shots from the **enabled** plugins.

| Plugin | Intents | Slots | Example |
|--------|---------|-------|---------|
| `core` (locked) | `greeting`, `cancel`, `unknown` | — | "ciao", "stop" |
| `clock` | `time.now`, `date.today` | — | "che ora è" |
| `timer` | `timer.set` | `hours`, `minutes`, `seconds` | "metti un timer di cinque minuti" |
| `radio` | `radio.play`, `radio.stop`, `radio.info` | `station` (str) | "metti virgin radio" |

Sources: modules in `pawn_assistant/commands/builtin/`, plus installed packages exposing an entry point in group `pawn_assistant.commands` (a `Command` class or a module).

### Writing one

```python
# pawn_assistant/commands/builtin/coin.py  (or your own package + entry point)
import random
from pawn_assistant.commands.base import Command, intent


class CoinCommand(Command):
    name = "coin"                      # id + YAML key under commands:
    title = "Coin flip"
    description = "Heads or tails."
    phrasing = {"it": {"coin.flip": (  # optional: rows for the reply rewording
        ("lancia una moneta", "Testa.", "ironico", "sera", "Testa. La sorte ha parlato."),
    )}}

    @intent("coin.flip", "the user wants to flip a coin",
            examples=("lancia una moneta", "testa o croce", "flip a coin"),
            fewshot=(("tira una monetina", {}),))
    def flip(self, slots, ctx) -> str:
        return ctx.t("Testa.", "Heads.") if random.random() < 0.5 else ctx.t("Croce.", "Tails.")
```

`examples` feed the fast-path verbatim; `fewshot` goes into the LLM prompt; `slots` (`SLOT_INT` / `SLOT_STR`) shape the grammar. `phrase=False` on `@intent` speaks the canonical reply verbatim (the guard protects digits and dates, not names). Add eval cases to `tests/data/commands_it.yaml`.

Optional hooks on `Command`:

| Hook | Use |
|------|-----|
| `configure(cfg)` | `commands.<name>` from the YAML |
| `setup(ctx)` | once, with the shared `CommandContext` |
| `fast_match(norm) → FastMatch` | regex slot extraction before the phrase table (timer durations, radio station) |
| `is_active()` / `on_cancel(ctx)` | generic "stop": the command named in the text (`cancel_words`), else the latest active one (`mark_active()`), else all active |
| `on_state(state)` | loop state changes (radio ducking). Called on the capture thread — must not block |
| `on_ui_action(id, action, data)` | buttons/sliders in the command's windows |
| `on_disable()` / `shutdown()` | stop what is running |

`ctx` gives `ctx.t(it, en)`, `ctx.now()`, `ctx.announce(text, intent=…, beeps=3)` (spoken alert, queued until the loop is idle), `ctx.ui` (windows, below) and `self.state_get/state_set` (persisted JSON).

### Enable / disable

HUD → **MODULES** (or `M`) toggles a plugin live: fast-path table, LLM grammar + prompt (`IntentLLM.set_intents`, re-warmed in the background; the KV cache keeps the old prefix) and phrasing warmup all follow. Toggles persist in `~/.local/state/pawn-assistant/commands.json` (`commands.state_file`), overriding `commands.disabled` from the YAML. `core` cannot be disabled.

### HUD windows

A plugin opens windows through `ctx.ui`:

```python
ctx.ui.open("timer:3", "countdown", command="timer", title="TIMER", anchor="tr",
            data={"label": "5 minuti", "total_s": 300, "due_at_ms": …, "state": "running"})
ctx.ui.update("timer:3", {"state": "expired"})   # partial data
ctx.ui.close("timer:3")
```

and ships the renderer as an asset next to the module (`assets = {"js": ("widget.js",), "css": ("widget.css",)}`, served at `/plugins/<name>/<file>`, loaded by the HUD from `hello.assets`):

```js
PawnWidgets.register("countdown", {
  mount(body, data, api) {   // api.action(name, data) → on_ui_action; api.now(); api.setStatus(); api.flash(); api.close()
    …
    return { update(data, changed) { … }, unmount() { … } };
  },
});
```

Kinds without a renderer fall back to a key/value list. The window manager (`static/js/wm.js`, `PawnWM.open({...})`) is generic — the MODULES panel uses it too.

## Web radio

`radio` searches TuneIn (`opml.radiotime.com`, JSON render), tunes the first playable hit (geo-blocked stations return a `notcompatible` placeholder → next hit) and plays it with **mpv** over JSON IPC: live volume, ICY now-playing titles. Station details (`Describe.ashx`: slogan, genre, location, FM frequency) fill the window in the background.

- Search hits must match the spoken name (`name_score` ≥ 0.8: numbers must match, "radio"/"fm" count less), else the reply is "Non trovo quella stazione." — TuneIn search hides some stations entirely (BBC Radio 1/2/4/6, RTL 102.5) and would otherwise hand back an unrelated top hit.
- TuneIn search is not localised and misses some Italian stations; `commands.radio.prefer: [Italian, Italy]` ranks hits by language/location, and `presets` map spoken names to guide ids.
- Ducking: AWAKE → IDLE at `duck_volume` (80 ms down, `duck_ramp_ms` back up). Not on STIR — the radio itself nudges the wake score and would pump.
- No AEC: the wake word still has to beat the radio at full volume.

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

New states: `THINK` (intent parsing), `SPEAK` (TTS playing). Plugins see every change via `on_state`. Mic frames are ignored in `BUSY`/`THINK`/`SPEAK` (no AEC yet), and the wake debounce after `SPEAK` covers the speaker tail.

```json
{"type": "event", "name": "intent", "intent": "timer.set", "slots": {"minutes": 5}, "source": "fastpath", "parse_ms": 0.4}
{"type": "event", "name": "reply", "kind": "reply", "seq": 0, "text": "Fatto, 5 minuti sul conto alla rovescia.", "phrased": true, "first_ms": 980.0, "tts_ms": 61.2}
{"type": "event", "name": "reply", "kind": "reply", "seq": 1, "text": "Fatto, 5 minuti sul conto alla rovescia. Ti avviso io.", "phrased": true, "first_ms": 980.0, "tts_ms": 40.3}
{"type": "event", "name": "reply", "kind": "alert", "text": "Il timer di 5 minuti è scaduto.", "tts_ms": 70.0}
```

Meter adds `llm_resident`, `tts_resident`.

`text` is the reply so far; the HUD replaces the line on `seq > 0`. A fallback reply has `phrased: false`.

Plugin alerts (`ctx.announce`, e.g. timer expiry) queue up and play (3 beeps + speech) only when the loop is `IDLE`, never over an active turn.

Plugin/window messages:

```json
{"type": "ui", "op": "open", "id": "radio:main", "kind": "radio", "command": "radio", "title": "RADIO", "anchor": "tl", "data": {"name": "Virgin Radio Italy", "volume": 70, "ducked": false}}
{"type": "ui", "op": "update", "id": "radio:main", "data": {"now_playing": "Muse - Uprising"}}
{"type": "ui", "op": "close", "id": "radio:main"}
{"type": "commands", "commands": [{"name": "radio", "title": "Web radio", "enabled": true, "core": false, "active": true, "intents": [...]}]}
```

`hello` adds `commands`, `assets`, `ui` (open windows) and `now_ms` (server clock for countdowns). Client → server: `{"type": "set_command", "name": "radio", "enabled": false}`, `{"type": "ui_action", "id": "radio:main", "action": "set_volume", "data": {"value": 40}}` (closing a window by hand sends `action: "close"`).

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
