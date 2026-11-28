# Pawn Assistant

Always-listening wake loop for the PawnAI family: detect a stock wake phrase, beep, capture an utterance with Silero VAD, transcribe with faster-whisper `tiny` (int8), and show a cyberpunk HUD.

Point 2 adds local **command words**: the transcript goes through a fast phrase matcher, then (on a miss) a small in-process LLM (Qwen2.5-1.5B, llama.cpp) that outputs grammar-forced intent JSON; Python handlers answer and a local Piper voice speaks. "Parliamo" opens a spoken conversation with the pawn agent on pawn-server (see [pawn-server](#pawn-server) below). Recorder session control comes later — see [`draft/plan.md`](draft/plan.md), [`draft/01-wake-ui.md`](draft/01-wake-ui.md), [`draft/02-commands.md`](draft/02-commands.md).

## Requirements

- Linux (PipeWire / PulseAudio)
- **Python 3.11** (openWakeWord’s `tflite-runtime` has no cp312 wheels; `.python-version` pins 3.11)
- Mic access
- ~2.5 GB free disk for Whisper, the intent GGUF and Piper voice on first run
- `mpv` for the web-radio command (`pacman -S mpv` / `apt install mpv`); everything else works without it

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

Say the wake word, then e.g. “che ora è”, “che giorno è oggi”, “metti un timer di cinque minuti”, “annulla il timer”, “metti virgin radio”, “cosa sta suonando”, “spegni la radio”. Test without a mic:

```bash
pawn-assistant ask "che ore sono"                     # intent + reply, spoken
pawn-assistant ask --no-speak "fra due minuti dimmi che la pasta è pronta"
pawn-assistant ask --eval tests/data/commands_it.yaml # accuracy + latency
pawn-assistant ask --no-speak "metti radio 105"       # plays until Ctrl+C
pawn-assistant say "Sono le 14 e 35."
```

Commands are **plugins** (`pawn_assistant/commands/builtin/`: core, clock, timer, radio — or any installed package with a `pawn_assistant.commands` entry point). Toggle them live from the HUD (**MODULES** button or `M`); toggles persist in `~/.local/state/pawn-assistant/commands.json`, defaults come from `commands.disabled` in the YAML. Writing a command: [`draft/02-commands.md`](draft/02-commands.md).

### Web radio

Stations come from TuneIn and play through `mpv` (never through the capture device). Configure under `commands.radio` — presets for stations TuneIn search ranks badly, a default for “accendi la radio”, volumes. The radio is **ducked** from wake until PAWN is idle again. Without echo cancellation the wake word still has to be heard over the radio at its normal volume; with `aec.enabled` mpv plays into the echo-cancel sink and the radio is subtracted from the mic.

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

### Improving it from real use

While `serve` runs a project's model, every wake saves the 2 s that triggered it into `wakewords/hey_pawn/inbox/`, labelled by what happened next: `confirmed` (a command followed), `false_wake` (silence or empty transcript), `unsure` (speech, no command understood), `near_miss` (scored in the stir band) and `barge` (heard over PAWN's own voice). Nothing is trained on until you file it:

```bash
pawn-assistant wake-train inbox    -p wakewords/hey_pawn   # counts per label
pawn-assistant wake-train review   -p wakewords/hey_pawn -l near_miss -l unsure   # listen: w / n / d
pawn-assistant wake-train accept   -p wakewords/hey_pawn   # confirmed+barge → positives, false_wake → negatives
pawn-assistant wake-train train    -p wakewords/hey_pawn --out wakewords/hey_pawn/hey_pawn.v2.onnx
pawn-assistant wake-train calibrate -p wakewords/hey_pawn --model wakewords/hey_pawn/hey_pawn.v2.onnx
pawn-assistant wake-train calibrate -p wakewords/hey_pawn --write pawn-assistant.yaml   # set thresholds
```

Training augments every positive (speed ±10 %, synthetic room reverb, real negatives mixed underneath at 5–20 dB SNR, far-field levels), and live false wakes (`negatives/live_*`) count three times. `calibrate` streams each clip through openWakeWord the way `serve` does, prints recall / false positives per threshold, the hardest negatives and weakest positives, and suggests `wake.threshold` and `barge_in.threshold`. Run calibrate on inbox clips before accepting them: they are the only clips the model has not been trained on.

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
- Window layer over the stage: commands open live windows (timer countdown, radio station card). Drag by the title bar, resize from the corner, minimize to the tray, `Esc` closes; positions are remembered per window kind. `M` opens the MODULES manager.
- Reloading the tab reconnects the WebSocket; the mic and models stay in the Python process.

## Streaming ASR (Zipformer)

`asr.engine: zipformer` swaps Whisper for a sherpa-onnx streaming transducer: every 80 ms frame is decoded while you talk, the HUD log shows the partial text live, and the final transcript is ready at end of speech, so the turn ends after `asr.streaming.silence_end_ms` (600 ms) instead of 900. Whisper stays loaded on demand as the fallback for an empty streaming result (and when the model cannot load).

```bash
uv sync --extra streaming
# point asr.streaming.model_dir at a model folder (encoder/decoder/joiner .onnx + tokens.txt)
```

There is no official Italian streaming model in sherpa-onnx; community Zipformer2 builds (e.g. Kroko) work if they ship the usual transducer files. English/French official models work as-is for testing.

## Echo cancellation and barge-in

With `aec.enabled: true` PAWN loads PipeWire's `module-echo-cancel` (WebRTC) at startup: the wake loop records from `pawn_ec_source`, and beeps, Piper replies and the radio play into `pawn_ec_sink`, so PAWN hears the room minus itself. That enables **barge-in**: say the wake word while PAWN is talking and it stops mid-sentence, beeps and listens for the new command (also works on timer alerts and inside a conversation).

```bash
pawn-assistant aec check          # what is missing (mic source, profile)
pawn-assistant aec check --fix    # switch the card to its duplex profile
pawn-assistant aec check --test   # load + unload the module once
```

PipeWire needs a real mic *source*. If the card sits on an output-only profile (the mic used as raw `hw:0,0`), switch it to **Analog Stereo Duplex** — the assistant then captures via `pulse` and `audio.device` is ignored. If AEC cannot start, PAWN logs why and keeps the old behaviour (mic ignored while it speaks, radio ducked). Barge-in uses a stricter threshold than a normal wake (`barge_in.threshold`, default `wake.threshold + 0.1`, two consecutive frames).

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

## pawn-server

Set `commands.pawn.server_url` and `token`. "Parliamo" opens a conversation over `POST /v1/pawn/voice`; "sblocca le sessioni" opens the keypad for full agent access. The protocol is pawnai's `docs/VOICE_PROTOCOL.md`; its golden transcripts and the shared utterances live in [`tests/contracts/`](tests/contracts) (copied by pawnai's `scripts/sync-contracts.sh` — never edit them here), and `tests/test_contract.py` replays them.

### Register the laptop

```bash
pawn-assistant device register --name laptop     # uses commands.pawn.server_url + token once
pawn-assistant device status
pawn-assistant device forget                     # revoke + delete device.json
```

Or register it from the portal (**Assistants → Register → Laptop**) and paste the snippet it shows (`--device-id … --device-token …`). The identity is stored in `~/.local/state/pawn-assistant/device.json` (0600). Once registered, `serve`:

- keeps `GET /v1/devices/{id}/commands/stream` open and runs what the agent sends: **speak** (a spoken alert), **notify** (a HUD window and speech) and **timer_set** (a normal countdown). The agent's `assistant_speak` / `assistant_timer` tools pick the assistant you were last near;
- reports presence (`active` on each wake, `idle` after `commands.device.idle_after_s`, `away` on exit).

### Hand-off and phone unlock

- **"Continua qui"** / "riprendi la conversazione" takes over the conversation you were having on the phone (or another assistant) in the last 10 minutes. Access always restarts locked.
- **"Sblocca col telefono"** asks the phone for full access: it shows "Allow laptop full access for N min?" and needs your fingerprint. The keypad still works.

The Android app does the same in the other direction.

### Wake word sync

Every assistant can upload the clips that triggered its wakes. On this laptop:

```bash
pawn-assistant wake-train pull -p wakewords/hey_pawn       # phone clips → inbox/<label>/ (source: server)
pawn-assistant wake-train push -p wakewords/hey_pawn       # this laptop's harvested clips → server
pawn-assistant wake-train review -p wakewords/hey_pawn     # as usual
pawn-assistant wake-train accept -p wakewords/hey_pawn
pawn-assistant wake-train train -p wakewords/hey_pawn --out wakewords/hey_pawn/hey_pawn.v3.onnx
pawn-assistant wake-train calibrate -p wakewords/hey_pawn --model wakewords/hey_pawn/hey_pawn.v3.onnx
pawn-assistant wake-train publish -m wakewords/hey_pawn/hey_pawn.v3.onnx --notes "phone clips" --wake-threshold 0.55
```

With `wake.auto_update: true`, `serve` downloads the active model into `~/.local/state/pawn-assistant/wake/` when its checksum changes and uses its suggested threshold. The phone does the same.

## Out of scope (point 2)

- `pawnai-recorder` `/sessions/*`
