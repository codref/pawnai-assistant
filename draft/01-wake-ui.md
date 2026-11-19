# Point 1 — wake UI contract

Runnable scaffold: Python listener owns mic + models; browser HUD is a thin WebSocket client. Phaser draws Pawn presence only; debug numbers stay in HTML.

## Process split

```
mic → capture → wake / VAD / ASR (Python)
                    ↓
              WS meter + events
                    ↓
         HUD (Phaser presence + HTML dock)
```

- Bind `127.0.0.1:8787` by default.
- Reloading the tab must not reload models or drop the mic.
- Server plays the wake beep; HUD only animates it.
- Mute wake scoring while the beep plays (speaker→mic loop).

## States

| State | Meaning |
|-------|---------|
| `IDLE` | Wake model running; score under stir band |
| `STIR` | Score rising toward threshold |
| `AWAKE` | Threshold crossed; beep |
| `LISTEN` | Capturing speech (VAD) |
| `HOLD` | Trailing silence before end-of-utterance |
| `BUSY` | faster-whisper running |
| `THINK` | intent parsing (point 2) |
| `SPEAK` | Piper reply / timer alert playing (point 2) |
| `FAULT` | Device or model failure |

## WebSocket protocol

URL: `ws://127.0.0.1:8787/ws`

### Meter (~20 Hz)

```json
{
  "type": "meter",
  "rms_db": -42.1,
  "peak_db": -28.0,
  "wake_score": 0.12,
  "wake_threshold": 0.5,
  "vad_prob": 0.0,
  "silence_ms": 0,
  "state": "IDLE",
  "rss_mb": 180.5,
  "overruns": 0,
  "whisper_resident": false,
  "device_name": "default",
  "sample_rate": 16000
}
```

### Events

```json
{"type": "event", "name": "state", "state": "LISTEN", "since_ms": 0}
{"type": "event", "name": "beep"}
{"type": "event", "name": "transcript", "text": "…", "duration_ms": 1200, "infer_ms": 340, "rtf": 0.28}
{"type": "event", "name": "intent", "intent": "time.now", "slots": {}, "source": "fastpath", "parse_ms": 0.3}
{"type": "event", "name": "reply", "kind": "reply", "text": "Sono le 14 e 35.", "tts_ms": 60.1}
{"type": "event", "name": "fault", "message": "…"}
```

Meter also carries `llm_resident` and `tts_resident` (point 2).

HUD never drives the microphone.

## Visual direction

- Cyberpunk chrome: scanlines, clipped cyan / amber panels, dark void background.
- Hero nameplate: **PAWN** (brand-first; status words secondary).
- Phaser canvas: glyph / ring driven by state + wake score.
- HTML dock: selectable monospace metrics and transcript log — never Phaser text.

## Dock fields

- Input device, sample rate, callback overruns
- RMS / peak dBFS
- Wake score sparkline vs threshold
- VAD probability, silence countdown
- Current state + dwell time
- Whisper resident flag
- Last utterance: duration, infer_ms, RTF
- Process RSS (MB)
- Rolling transcript log

## Models (point 1)

| Piece | Choice |
|-------|--------|
| Wake | openWakeWord stock model (see README for exact id) |
| VAD | Silero ONNX |
| ASR | faster-whisper `tiny` int8; unload after idle (default 60s) |
| Custom “hey pawn” | out of scope |

## Identity lexicon

Status chips use Bob’s dry register: short uppercase tokens, no filler. No LLM replies in point 1 — the machine face is the product.
