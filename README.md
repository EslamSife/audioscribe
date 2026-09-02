# transcribe

Modular CLI for turning audio/video files, YouTube/web video URLs, or live microphone and computer audio into accurate transcripts, summaries, and subtitles using AssemblyAI or Local Whisper.

## Features

- 🌐 **Direct YouTube & Web Video Transcription**: Pass any YouTube, Vimeo, or web URL directly to transcribe without downloading videos first.
- ⚡ **Dual Engine**: Cloud transcription with **AssemblyAI** or 100% offline, free transcription with **Local Whisper** (`--provider whisper`).
- 🔴 **Live Microphone & System Audio**: Real-time speech-to-text from your mic or internal computer audio (Teams, Zoom, browser tabs). Phrases are split on natural speech pauses using Silero VAD, with a live in-place preview line and crash-safe incremental saving.
- 🎙️ **Meeting Mode**: Merge two devices (`--device "BlackHole" --mix-device "MacBook"`) to capture the other participants and your own mic in one transcript.
- 📝 **AI Notes** (`--notes`): After transcribing, Claude writes a structured Markdown note: summary, key points, action items, and timestamped quotes.
- 🎞️ **In-Flight Audio Preprocessing**: Auto-extracts and downsamples video/audio using FFmpeg for 10× faster uploads.
- 👥 **Speaker Diarization**: Separates and labels speakers. Cloud via AssemblyAI, or fully local via pyannote for the Whisper provider.
- 🧠 **AI Intelligence**: Summaries, auto topic chapters, and custom word boost.
- 🌍 **Multi-Language**: Explicit language selection or automatic spoken language detection.
- 📦 **Rich Output Formats**: `txt`, `md` (Markdown), `srt` (SubRip), `vtt` (WebVTT), and `json`.
- 📁 **Batch Processing & Stdout Streaming**: Handle multiple files or pipe outputs directly to tools like `jq`.

---

## Installation & Setup

### 1. Cloud Provider (AssemblyAI)
Get a free API key at [AssemblyAI](https://www.assemblyai.com) and save it:

```sh
# Via CLI command:
uv run transcribe config set-key "your-key"

# Or via shell profile (~/.zshrc):
export ASSEMBLYAI_API_KEY="your-key"
```

### 2. Optional Extras

```sh
# YouTube & Web URL support:
uv add yt-dlp
# Or: uv sync --extra url

# Local offline Whisper support:
uv add faster-whisper
# Or: uv sync --extra local

# Live microphone & system audio recording:
uv add sounddevice
# Or: uv sync --extra live

# Local speaker labels with the Whisper provider (heavy: downloads pyannote models):
uv sync --extra speakers
# Then create a token at https://huggingface.co/settings/tokens, accept the terms of
# https://huggingface.co/pyannote/speaker-diarization-3.1, and export HF_TOKEN=your_token

# Install everything:
uv sync --extra all
```

### 3. AI Notes (`--notes`)

The `--notes` flag sends the finished transcript to Claude and saves a structured note
(summary, key points, action items, quotes). It needs a Claude API key:

```sh
export ANTHROPIC_API_KEY="your-key"   # get one at https://console.anthropic.com/settings/keys
```

---

## Usage Examples

### 🌐 1. YouTube & Web Video Transcription
```sh
# Transcribe and summarize a YouTube video directly from its URL
uv run transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --format md --summarize --chapters

# Generate WebVTT subtitles directly from a web video link
uv run transcribe "https://www.youtube.com/watch?v=..." --format vtt
```

### 🖥️ 2. Live Microphone & System/Browser Audio
```sh
# List all audio input devices (microphones & virtual loopbacks)
uv run transcribe --list-devices

# Live microphone transcription
uv run transcribe live --format md --out live_notes.md

# Record computer/browser audio (e.g. Teams, Zoom, YouTube tab) via virtual loopback:
uv run transcribe live --device "BlackHole" --format md --out meeting_notes.md

# Offline live transcription with Local Whisper (no API key, phrase-by-phrase output)
uv run transcribe live --provider whisper --model base --device "BlackHole" --format md

# Meeting mode: capture the call audio AND your own microphone together
uv run transcribe live --provider whisper --model base --device "BlackHole" --mix-device "MacBook" --format md

# Automatic meeting minutes: transcribe the meeting, then let Claude write the notes
uv run transcribe live --provider whisper --model base --device "BlackHole" --mix-device "MacBook" --notes
```

Live transcripts are saved line by line while you record, so a crash or sleep never
loses the session. If the output file name is taken, a numbered name is used instead.

### 📁 3. Local Files & Dual Engines
```sh
# Basic transcription (AssemblyAI cloud)
uv run transcribe meeting.m4a

# 100% Offline with Local Whisper (no API key required)
uv run transcribe talk.mp3 --provider whisper --model turbo

# Video subtitle extraction with automatic FFmpeg downsampling
uv run transcribe lecture.mp4 --format vtt

# Batch process multiple recordings into a directory
uv run transcribe rec1.mp3 rec2.wav --out ./transcripts/

# Markdown output with timestamps, speaker labels, summaries & chapters
uv run transcribe interview.mp3 --format md --speakers --summarize --chapters

# Fully local speaker labels (Whisper + pyannote, needs the 'speakers' extra and HF_TOKEN)
uv run transcribe interview.mp3 --provider whisper --model base --speakers --format md

# Transcribe a file and let Claude write structured notes next to it
uv run transcribe lecture.mp4 --provider whisper --model base --notes

# Multi-language & auto-detection
uv run transcribe speech.mp3 --language es
uv run transcribe foreign.mp3 --auto-detect-language

# Custom vocabulary boost
uv run transcribe devops.mp3 --boost "Kubernetes,Docker,AssemblyAI,Prometheus"

# Stream JSON output to stdout for CLI piping
uv run transcribe clip.mp3 --format json --stdout | jq '.summary'
```

### Supported Formats

- `txt`: Plain text or speaker dialogue (`Speaker A: ...`).
- `md`: Rich Markdown with duration, language, confidence, executive summaries, chapters, and timestamps.
- `srt`: SubRip subtitle file with millisecond timecodes.
- `vtt`: WebVTT subtitle file for HTML5 and web video players.
- `json`: Structured JSON containing metadata, summaries, chapters, and time-stamped segments.

---

## Development & CI

```sh
# Run the test suite
uv run pytest -v

# Run linting
uv run ruff check .
```

---

## Architecture

The project adheres to strict protocol isolation:
- Provider implementations ([`AssemblyAIProvider`](file:///Users/eslam.sife/workspace/transcribe/src/transcribe/providers/assemblyai.py) and [`LocalWhisperProvider`](file:///Users/eslam.sife/workspace/transcribe/src/transcribe/providers/whisper.py)) implement the [`TranscriptionProvider`](file:///Users/eslam.sife/workspace/transcribe/src/transcribe/providers/base.py) protocol.
- Vendor types never leak outside their adapter modules.
- New transcription backends can be added in a single self-contained adapter file.



