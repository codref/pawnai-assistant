# Pawn Assistant

Always-listening wake loop for the PawnAI family: detect a stock wake phrase, beep, capture an utterance with Silero VAD, transcribe with faster-whisper `tiny` (int8), and show a cyberpunk HUD.

Point 1 ends at the transcript. Recorder session control and pawn-server chat/TTS come later — see [`draft/plan.md`](draft/plan.md) and [`draft/01-wake-ui.md`](draft/01-wake-ui.md).

## Requirements

- Linux (PipeWire / PulseAudio)
- **Python 3.11** (openWakeWord’s `tflite-runtime` has no cp312 wheels; `.python-version` pins 3.11)
- Mic access
- ~1 GB free disk for Whisper + ONNX models on first run

Target hardware: CPU laptop (i5 / 8 GB). Do not co-locate pyannote or a 4B Ollama model in this process.

## Install

```bash
cd pawnai-assistant
uv sync          # uses Python 3.11 from .python-version
# or: python3.11 -m venv .venv && .venv/bin/pip install -e .
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

pawn-assistant wake-train train --project wakewords/hey_pawn
pawn-assistant wake-train test --project wakewords/hey_pawn --device 0

pawn-assistant serve --device 0 --wake-model wakewords/hey_pawn/hey_pawn.onnx
```

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
mic → openWakeWord → beep → Silero VAD → faster-whisper tiny int8
                ↘
            WS meter/events → browser HUD
```

WebSocket: `ws://127.0.0.1:8787/ws` — see `draft/01-wake-ui.md`.

## Out of scope (point 1)

- Custom “hey pawn” model
- `pawnai-recorder` `/sessions/*`
- pawn-server chat / TTS
