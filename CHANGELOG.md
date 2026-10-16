# Changelog

## 0.4.0

First public release on PyPI.

### Added

- `transcribe doctor` checks the setup (ffmpeg, keys, audio devices, speaker labels, yt-dlp,
  notes and Ollama) and prints the fix for each problem.
- `--template` for `--notes`: `standup`, `design-review` (with a draft ADR) and `incident`
  (timeline, root cause, follow-ups). Action items are Markdown tasks, and the template is
  added as a tag.

### Fixed

- Local Whisper now skips silence in files and in the live final pass (faster-whisper's
  `vad_filter`). Without it, Whisper can invent text on silence, and it lost or changed words
  after long pauses. Live phrases are already cut at pauses, so live mode is unchanged.

## 0.3.0

First release prepared for open source, as `audioscribe` (the command stays `transcribe`).

### Fixed

- Live mode with AssemblyAI crashed on start: the streaming client needs connection parameters
  and registers handlers with `on(event, handler)`. Live lines also had no real timestamps.
- `--summarize` crashed with the current AssemblyAI SDK (renamed summarization types).
- Local speaker labels failed with pyannote 4. A labelling failure now keeps the transcript and
  shows a warning, and a missing `HF_TOKEN` is reported before the transcription starts.
- Meeting mode: a second device that stops no longer slows down the main one or drops audio.
  Device errors are shown instead of giving a silent, empty session.
- A provider bug no longer starts a second transcription without your options.
- Playlist links transcribe only the linked video.
- Pressing Ctrl+C twice can no longer add a line after the transcript was saved.
- Live transcripts report the real audio length, and no longer claim English when the language
  is unknown.

### Changed

- `--boost` now also works with local Whisper (as `hotwords`) and AssemblyAI live mode.
- `--language`, `--auto-detect-language` and `--speakers` now work in AssemblyAI live mode.
- Options a provider can't use (for example `--summarize` with Whisper) print a note.
- `config set-key` asks for the key when it is not given, and the key file is owner-only.
- The notes setup (`NOTES_PROVIDER`, `NOTES_MODEL`, keys) is checked before recording starts.
- Python 3.11 or newer (was 3.14). MIT license.
- Removed the unused PyAudio and openai-whisper fallbacks, and unused audio level helpers.
