# Pawn Assistant

Always-listening wake loop for the PawnAI family: detect a stock wake phrase, beep, capture an utterance with Silero VAD, transcribe with faster-whisper `tiny` (int8), and show a cyberpunk HUD.

Point 2 adds local **command words**: the transcript goes through a fast phrase matcher, then (on a miss) a small in-process LLM (Qwen2.5-1.5B, llama.cpp) that outputs grammar-forced intent JSON; Python handlers answer and a local Piper voice speaks. Recorder session control and pawn-server chat come later — see [`draft/plan.md`](draft/plan.md), [`draft/01-wake-ui.md`](draft/01-wake-ui.md), [`draft/02-commands.md`](draft/02-commands.md).

## Requirements

- Linux (PipeWire / PulseAudio)
- **Python 3.11** (openWakeWord’s `tflite-runtime` has no cp312 wheels; `.python-version` pins 3.11)
- Mic access
- ~2.5 GB free disk for Whisper, the intent GGUF and Piper voice on first run

Target hardware: CPU laptop (i5 / 8 GB). Do not co-locate pyannote or a 4B Ollama model in this process.

## Install

```bash
cd pawnai-assistant
uv sync          # uses Python 3.11 from .python-version
# or: python3.11 -m venv .venv && .venv/bin/pip install -e .
pawn-assistant models pull   # Whisper + intent LLM + Piper voice (then fully offline)
```

`llama-cpp-python` builds from source by default (a few minutes). To use a prebuilt CPU wheel instead:

```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```


## Run

```bash
pawn-assistant list-devices
pawn-assistant probe              # speak while it runs — picks a LIVE device
pawn-assistant monitor --device 0 # live dB meter; bar should move when you talk
pawn-assistant serve
# or: pawn-assistant serve -c ./pawn-assistant.yaml --debug-audio ./debug-audio
```

Copy [`pawn-assistant.example.yaml`](pawn-assistant.example.yaml) → `pawn-assistant.yaml` (also checked in `~/.config/pawn-assistant/config.yaml`). Set `asr.language: it` (or `en`) — autodetection is messy on short commands. Same file holds wake model/threshold, mic `device`, VAD timings, beep, and debug dumps. CLI flags override the file when passed.

`--debug-audio` writes WAVs: `wake_*`, `near_miss_*`, `utt_*`, `empty_*`.

## Commands

Say the wake word, then e.g. “che ora è”, “che giorno è oggi”, “metti un timer di cinque minuti”, “annulla il timer”. Test without a mic:

```bash
pawn-assistant ask "che ore sono"                     # intent + reply, spoken
pawn-assistant ask --no-speak "fra due minuti dimmi che la pasta è pronta"
pawn-assistant ask --eval tests/data/commands_it.yaml # accuracy + latency
pawn-assistant say "Sono le 14 e 35."
```

Configure under `intent:` / `tts:` in the YAML. Add a command in `pawn_assistant/commands/intents.py` + `handlers.py` — see [`draft/02-commands.md`](draft/02-commands.md).

## Choosing a voice

```bash
pawn-assistant voice list                      # Piper voices for reply_language (✓ = downloaded)
pawn-assistant voice test it_IT-paola-medium it_IT-riccardo-x_low   # A/B the same sentence
pawn-assistant voice test --all --lang it      # audition every Italian voice
pawn-assistant voice pick                      # interactive: n/p browse, +/- speed, enter = keep
pawn-assistant voice use it_IT-riccardo-x_low --speed 0.95   # validate + write tts: in your YAML
pawn-assistant voice current
```

`voice use` edits the existing config in place (comments are kept); with no config file it prints the snippet. Multi-speaker voices take `--speaker N` (`tts.speaker_id`).

Note: `piper-tts` is GPL-3.0.

## Custom wake word

Stock models: `hey_jarvis`, `hey_mycroft`, `alexa`, `hey_rhasspy`.

To train **hey pawn** (or any phrase) from your mic:

```bash
uv sync --extra wake-train
# CPU Torch (recommended on this laptop — skips multi‑GB CUDA wheels):
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

pawn-assistant wake-train init "hey pawn"
# → wakewords/hey_pawn/

pawn-assistant wake-train record-pos --project wakewords/hey_pawn --device 0 --count 40
pawn-assistant wake-train record-neg --project wakewords/hey_pawn --device 0 --category speech --count 60
pawn-assistant wake-train record-neg --project wakewords/hey_pawn --device 0 --category keyboard --count 40
pawn-assistant wake-train record-neg --project wakewords/hey_pawn --device 0 --category silence --count 30
pawn-assistant wake-train record-neg --project wakewords/hey_pawn --device 0 --category ambient --count 40

# Speech negatives are essential: without them the model fires on any voice
# (and on PAWN's own replies). Piper generates ~600 it/en clips offline:
pawn-assistant wake-train gen-neg --project wakewords/hey_pawn --count 600

pawn-assistant wake-train train --project wakewords/hey_pawn
pawn-assistant wake-train test --project wakewords/hey_pawn --device 0

pawn-assistant serve --device 0 --wake-model wakewords/hey_pawn/hey_pawn.onnx
```

Keep the mic out of clipping (peaks well below 0 dBFS in `monitor`) — clipped audio distorts the wake features.

Record positives hands-free (countdown cadence — don’t type between takes). Record negatives **one category at a time**. Silence/ambient negatives matter so the model does not learn “quiet room = wake”.


Open [http://127.0.0.1:8787/](http://127.0.0.1:8787/).

If the HUD stays silent: PipeWire often has **no mic Source** (output-only profile), so `pulse` opens but captures nothing and `pactl get-default-source` ends with `.monitor`. Enable **Analog Stereo Duplex** (or any profile with an input) in System Settings → Audio, or pin the ALSA device from `probe` (often `hw:0,0` at 48 kHz — the assistant resamples to 16 kHz).

### Wake phrase

Default openWakeWord model: **`hey_jarvis`**.

Say “hey jarvis”, wait for the beep, then speak. After ~700 ms of silence the dock shows the transcript.

```bash
pawn-assistant serve --wake-model hey_jarvis --wake-threshold 0.5
pawn-assistant serve --device 0   # sounddevice index from list-devices
```

### HUD

- **PAWN** presence (Phaser) reacts to state / wake score / level.
- HTML debug dock: dBFS, wake sparkline, VAD, silence, Whisper resident flag, RSS, transcript log.
- Reloading the tab reconnects the WebSocket; the mic and models stay in the Python process.

## Architecture

```
mic → openWakeWord → beep → Silero VAD → faster-whisper int8
                                              ↓
                       fast-path ─miss→ Qwen2.5-1.5B (llama.cpp, GBNF JSON)
                                              ↓
                                handler → Piper → speaker
                ↘
            WS meter/events → browser HUD
```

WebSocket: `ws://127.0.0.1:8787/ws` — see `draft/01-wake-ui.md`.

## Out of scope (point 2)

- `pawnai-recorder` `/sessions/*`
- pawn-server chat (open-ended questions land on `unknown`)
- Barge-in / echo cancellation (mic is ignored while PAWN speaks)
