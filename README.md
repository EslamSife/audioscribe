# audioscribe

A command-line tool that turns any speech into text: meetings, audio books, podcasts, lectures,
YouTube videos, or live sound from your microphone and computer. It writes plain text, Markdown,
subtitles (SRT, VTT) or JSON, and can ask an LLM to write structured notes from the transcript.

It works with two engines:

- **AssemblyAI** (cloud): needs an API key, adds summaries, chapters and speaker labels.
- **Local Whisper** (`--provider whisper`): runs fully on your computer with
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No key, nothing is uploaded.

The package is called `audioscribe`; the command you type is `transcribe`.

## Features

- **Files and links**: audio and video files, plus YouTube and other sites that
  [yt-dlp](https://github.com/yt-dlp/yt-dlp) supports. A link to a video inside a playlist
  transcribes only that video.
- **Live mode**: transcribe your microphone or computer audio (Zoom, Teams, a browser tab) as
  people speak. Each finished line is saved right away, so a crash does not lose the session.
- **Meeting mode**: record two devices (`--device "BlackHole" --mix-device "MacBook"`) to capture
  the other people and your own microphone in one transcript. With local Whisper, each line
  says who spoke: `Me` (your microphone) or `Them` (the call).
- **Final pass**: live text is a fast draft. `--final-model` transcribes the saved recording
  again with a bigger, more accurate model when you stop.
- **AI notes** (`--notes`): summary, key points, action items and quotes, written by Claude,
  GPT, Gemini or a local Ollama model. Can save straight into an Obsidian folder.
- **Speaker labels** (`--speakers`): AssemblyAI in the cloud, or locally with pyannote.
- **Word boost** (`--boost`): tell the engine about names and terms it should expect.
- **Formats**: `txt`, `md`, `srt`, `vtt`, `json`. Batch several files, or print to stdout.

### What works where

| Option | AssemblyAI file | AssemblyAI live | Whisper file | Whisper live |
|---|---|---|---|---|
| `--language`, `--auto-detect-language` | yes | yes | yes | yes |
| `--boost` | yes | yes | yes | yes |
| `--speakers` | yes | yes | yes (needs the `speakers` extra) | only with `--final-model` |
| `Me`/`Them` labels in meeting mode | | no (both devices are mixed) | | yes |
| `--summarize`, `--chapters` | yes | no | no | no |

When an option is not supported, the tool says so instead of ignoring it silently.

## Installation

You need Python 3.11 or newer. The easiest way is [uv](https://docs.astral.sh/uv/) (a fast
Python package manager), which installs the `transcribe` command on your PATH:

```sh
uv tool install audioscribe              # core
uv tool install 'audioscribe[openai]'    # also notes with GPT, Gemini or Ollama
uv tool install 'audioscribe[all]'       # everything, including local speaker labels (large)
transcribe doctor                        # check your setup
```

`pipx install audioscribe` works too. The examples below use `uv run transcribe`, which is how
you run it from a clone; after `uv tool install`, type just `transcribe`.

From a clone of this repository:

```sh
uv sync                  # core: AssemblyAI, local Whisper, live mode, links, Claude notes
uv sync --extra openai   # also notes with GPT, Gemini or Ollama
uv sync --extra speakers # also local speaker labels (large: installs PyTorch)
uv sync --all-extras     # everything
```

Then run it with `uv run transcribe ...`.

**Optional, but recommended:** install [ffmpeg](https://ffmpeg.org/). With it, video files and
large audio files are converted to small mono audio before upload or transcription. Without it,
files are used as they are.

### AssemblyAI key

Get a key at [assemblyai.com](https://www.assemblyai.com), then either:

```sh
uv run transcribe config set-key        # asks for the key, so it stays out of your shell history
export ASSEMBLYAI_API_KEY="your-key"    # or set it in your shell profile or a .env file
```

`config set-key` saves the key in `~/.config/transcribe/config.json`, readable only by you.
Check it with `uv run transcribe config show`.

### Local speaker labels

Local labels use [pyannote](https://github.com/pyannote/pyannote-audio):

1. `uv sync --extra speakers`
2. Create a token at <https://huggingface.co/settings/tokens>.
3. Accept the terms of <https://huggingface.co/pyannote/speaker-diarization-3.1>.
4. `export HF_TOKEN=your_token`

If labelling fails, you still get the transcript, with a warning.

### AI notes (`--notes`)

`--notes` sends the finished transcript to an LLM and saves a structured note. The tool prints
where the transcript is sent. Claude is the default. Choose another provider with environment
variables (or a `.env` file):

| Variable | Meaning |
|---|---|
| `NOTES_PROVIDER` | `claude` (default), `openai`, `gemini` or `ollama` |
| `NOTES_MODEL` | Model name. Required for every provider except `claude` (default `claude-opus-5-5`) |
| `NOTES_BASE_URL` | Server address, to override the preset (e.g. LM Studio, OpenRouter) |
| `NOTES_API_KEY` | Key for that server. Falls back to `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` |

```sh
# Claude (default)
export ANTHROPIC_API_KEY="your-key"   # get one at https://console.anthropic.com/settings/keys

# GPT
export NOTES_PROVIDER=openai NOTES_MODEL="<model>" OPENAI_API_KEY="your-key"

# Gemini (through its OpenAI-compatible endpoint)
export NOTES_PROVIDER=gemini NOTES_MODEL="<model>" GEMINI_API_KEY="your-key"

# Fully local and private: Ollama on your machine, no key, nothing leaves your computer
export NOTES_PROVIDER=ollama NOTES_MODEL="qwen2.5:7b"
```

The notes settings are checked before anything is recorded, so a mistake never costs you a
meeting.

**Ollama and long recordings:** Ollama may keep only the last 4,096 tokens of a long prompt
(about 15 minutes of speech) and still answer normally. `transcribe` warns you when the model
read much less than it was sent. Start Ollama with a bigger window to fix it:
`OLLAMA_CONTEXT_LENGTH=32768 ollama serve`. Small local models also write weaker notes,
especially action items, so check them before you trust them.

**Templates for work meetings:** `--template` picks the note's sections. Action items become
Markdown tasks (`- [ ] Owner: task`), and the template name is added as a tag.

| Template | Sections |
|---|---|
| `general` (default) | Summary, key points, action items, quotes |
| `standup` | Updates per person (done, next, blockers), blockers, action items |
| `design-review` | Context, options with pros and cons, decisions, open questions, action items, a draft ADR (Architecture Decision Record) |
| `incident` | Summary, timeline, impact, root cause, mitigation and fix, follow-ups, open questions (blameless) |

```sh
uv run transcribe live --provider whisper --device "BlackHole" --mix-device "MacBook" \
  --final-model small --notes --template incident
```

**Notes straight into Obsidian:** `--notes-dir` (or the `NOTES_DIR` environment variable) saves
the note into a folder such as your vault inbox, with a name like
`release-planning-2026-10-16.md`. Every note starts with frontmatter (`created`, `updated`,
`type`, `status`, `tags`, `sources`, `related`) and ends with a `## Related` section. An existing
note is never overwritten, and a missing folder stops the run before any recording starts.

## Usage

### Check your setup

```sh
uv run transcribe doctor
```

It checks ffmpeg, your AssemblyAI key, audio devices (and whether one carries computer audio),
local speaker labels, yt-dlp, and the `--notes` setup, including whether Ollama is running and has
your model. Each problem comes with the command that fixes it. It exits with an error only when
something you set up is broken, for example `NOTES_DIR` pointing to a missing folder.

### Files and links

```sh
# AssemblyAI (cloud)
uv run transcribe meeting.m4a

# Fully offline with local Whisper
uv run transcribe audiobook-chapter1.mp3 --provider whisper --model small

# Subtitles from a video
uv run transcribe lecture.mp4 --format vtt

# A YouTube video, with summary and chapters
uv run transcribe "https://www.youtube.com/watch?v=..." --format md --summarize --chapters

# Several files into one folder
uv run transcribe rec1.mp3 rec2.wav --out ./transcripts/

# Speaker labels, language, and names the engine should expect
uv run transcribe interview.mp3 --format md --speakers --language es --boost "Kubernetes,Helm"

# JSON to another tool
uv run transcribe clip.mp3 --format json --stdout | jq '.segments'
```

Whisper models, from fast to accurate: `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`.
The first use of a model downloads it.

### Live mode

```sh
# List input devices (microphones and virtual devices like BlackHole)
uv run transcribe --list-devices

# Your microphone
uv run transcribe live --format md --out notes.md

# Computer audio through a virtual device, fully offline
uv run transcribe live --provider whisper --model base --device "BlackHole" --format md

# Meeting mode: the call audio AND your microphone, then a better final pass and notes
uv run transcribe live --provider whisper --model base --device "BlackHole" --mix-device "MacBook" \
  --final-model small --notes
```

In meeting mode, `--device` is the call and `--mix-device` is your microphone. With
`--provider whisper` the two are not mixed: each line is labeled `Them:` or `Me:`, two people
talking at once are both heard, and `--notes` knows which tasks are yours. The recording is one
stereo `.wav` (the call on the left, your microphone on the right), and `--final-model` keeps the
labels; with `--speakers` it also tells the people on the call apart (`Speaker 1`, `Speaker 2`).

Use headphones if you can. Without them your microphone also hears the call from the speakers.
Lines that only repeat the call are dropped from the saved transcript (one may still flash by on
screen), but when you talk over the call at the same moment, your words can be dropped with them.

To record computer audio on macOS you need a virtual audio device such as
[BlackHole](https://github.com/ExistentialAudio/BlackHole). On Windows, "Stereo Mix" does the
same job. With local Whisper, live mode finds pauses in speech with the Silero voice activity
model that ships with faster-whisper.

Press Ctrl+C to stop. The recording is saved as a `.wav` next to the transcript (turn this off
with `--no-save-audio`). If the output name is taken, a numbered name is used instead. If the
final pass fails or you skip it with a second Ctrl+C, the live transcript is kept. You can also
transcribe a saved recording later, like any file:

```sh
uv run transcribe live_session.wav --provider whisper --model small --format md --notes
```

A meeting recording transcribed this way is heard as one mixed voice, without `Me`/`Them`
labels; use `--final-model` during the session to keep them.

A file that is really named `live`, `doctor` or `config` needs a path, for example `./live`.

### Output formats

- `txt`: plain text, or `Speaker 1: ...` lines when speakers are labelled.
- `md`: Markdown with duration, language, confidence, summary, chapters and timestamps.
- `srt`: SubRip subtitles.
- `vtt`: WebVTT subtitles for web video players.
- `json`: everything above as structured data.

## Privacy and consent

Recording people can be regulated where you live, and many places require everyone's consent.
Ask before you record a meeting.

With AssemblyAI, your audio is uploaded to AssemblyAI. With `--notes` and a cloud provider, the
transcript is sent to that provider. For nothing to leave your computer, use
`--provider whisper` together with `NOTES_PROVIDER=ollama`. Recordings and transcripts are saved
as normal files, so treat them like any private document.

## Development

```sh
uv sync --all-extras --dev
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Architecture

- Each engine is one adapter ([`providers/assemblyai.py`](src/transcribe/providers/assemblyai.py),
  [`providers/whisper.py`](src/transcribe/providers/whisper.py)) behind the small
  [`TranscriptionProvider`](src/transcribe/providers/base.py) protocol. Notes providers follow the
  same shape in [`notes/`](src/transcribe/notes/).
- Vendor types never leave their adapter. Everything else works with the plain data classes in
  [`transcription/model.py`](src/transcribe/transcription/model.py).
- Live mode lives in [`live.py`](src/transcribe/live.py). The logic that cuts audio into phrases
  is a separate, pure class in [`audio/phrases.py`](src/transcribe/audio/phrases.py), so it can be
  tested without audio devices.

## License

[MIT](LICENSE)
