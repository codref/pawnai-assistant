# Pawn Assistant — family tech plan

Always-listening companion for the PawnAI family. Runs on a Linux laptop (i5, 8 GB RAM, no discrete GPU). Owns wake word, short command capture, and a local HUD. Recording sessions, diarization, vault, and agent memory stay in existing packages.

## Family roles

| Package | Role for the assistant |
|---------|------------------------|
| **pawnai-assistant** (this repo) | Always-on ear: wake → command ASR → local phrases / agent turns → HUD |
| **pawnai-recorder** | Session capture: chunked FLAC, S3 upload, `transcribe-diarize` queue jobs. Control via `pawnai-recorder serve` on `127.0.0.1:8765` |
| **pawnai** (`pawn-server`) | Agent chat, vault tools, sallm memory, Kokoro TTS, heavy ASR/diarize workers |

Do not run pyannote, NeMo Parakeet, TitaNet, sentence-transformers, or a 4B chat model in the always-on assistant process.

## Runtime note

Point-1 Python is **3.11** only: openWakeWord depends on `tflite-runtime`, which has no cp312 wheels. The assistant process still uses ONNX for wake inference (`inference_framework="onnx"`).

## Hardware budget (8 GB, CPU only)

| Resident | Technology | Approx. cost |
|----------|------------|--------------|
| Always | openWakeWord + ONNX Runtime | &lt; 100 MB, few % of one core |
| Always | Silero VAD (ONNX) | shared with wake runtime |
| After wake | faster-whisper `tiny` int8 | load on demand; unload after idle |
| Reply (later) | Kokoro via `POST /v1/audio/speech` on pawn-server | prefer remote; local only if needed |
| Later | Chat | pawn-server; keep 4B Ollama off this box if diarize also runs here |

Desktop OS already uses ~1.5–2.5 GB. Keep the always-on loop under ~200 MB when Whisper is unloaded.

## Point roadmap

1. **Wake loop (this scaffold)** — wake word, beep, VAD utterance, tiny Whisper transcript, cyberpunk HUD. No Pawn network.
2. **Recorder phrases** — local intent list → `POST /sessions/start|stop|flush|notes` on recorder serve.
3. **Agent turns** — other utterances → `POST /v1/pawn/chat`; speak replies via `/v1/audio/speech`.
4. **Later** — custom “hey pawn”, streaming Zipformer, barge-in / AEC.

## Integration contracts (later points)

### Recorder (`pawnai-recorder serve`)

- `POST /sessions/start`, `/stop`, `/flush`, `/notes`, `/screenshot`
- `GET /status`, `/devices`, `/sessions`
- Same PipeWire/Pulse source as the wake loop (separate clients).

### pawn-server

- Chat / memory: `POST /v1/pawn/chat` or `/v1/chat/completions` with stable `user` session id (sallm SQLite + Lance under `agent.sallm.state_dir`).
- TTS: `POST /v1/audio/speech` (Kokoro).
- Avoid `/v1/audio/transcriptions` for commands — that path uses the configured heavy engine (Parakeet), not tiny Whisper.

## Explicitly out of scope on this laptop (always-on)

- pyannote / Nemotron diarization
- NeMo Parakeet as the command ASR
- Custom wake-word training (until after point 1)
- Full echo cancellation (mute wake during beep/TTS first)
- Running `ollama@gemma4:4b` alongside diarize workers on 8 GB

## Identity

Display name **PAWN**. Agent personality (Bob / `anima.md`) applies to spoken/chat replies later; the HUD uses a dry status lexicon: `IDLE`, `STIR`, `AWAKE`, `LISTEN`, `HOLD`, `BUSY`, `FAULT`.
