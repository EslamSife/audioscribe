from dataclasses import dataclass

from transcribe.transcription.model import Transcript, TranscriptionError

from .base import Completion, NotesProvider
from .factory import get_notes_provider

SYSTEM_PROMPT = """You turn raw audio transcripts into clear, structured Markdown notes.

Rules:
- Write the note in the same language as the transcript.
- Be faithful to the transcript. Never invent facts that are not in it.
- Transcripts come from imperfect speech recognition: fix obvious recognition
  errors from context, but do not change the meaning.
- Keep timestamps in [MM:SS] form when you quote or reference a moment.
- Do not use em dashes; use commas, colons or periods instead.

Produce exactly these sections:
# <A short descriptive title>
## Summary
2-4 sentences: what this recording is about and its main message.
## Key Points
The important ideas as short bullet points, in the order they appear.
## Action Items
Concrete tasks, decisions, or follow-ups mentioned. Write "None" if there are none.
## Notable Quotes
Up to 5 memorable or important quotes, each with its [MM:SS] timestamp.
"""

# English text averages about 4 characters per token. Only used to notice a
# model that read far less than we sent; most other languages use more tokens
# per character, so they can't trigger a false warning.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Notes:
    text: str
    warnings: tuple[str, ...] = ()


def generate_notes(
    transcript: Transcript, title: str, provider: NotesProvider | None = None
) -> Notes:
    """Ask the configured LLM for a structured Markdown note about the transcript."""
    text = _timestamped_text(transcript).strip()
    if not text:
        raise TranscriptionError("Transcript is empty, nothing to make notes from.")

    user = f"Recording name: {title}\n\nTranscript:\n{text}"
    result = (provider or get_notes_provider()).complete(SYSTEM_PROMPT, user)
    note = result.text.strip()
    if not note:
        raise TranscriptionError("The model returned an empty note.")
    return Notes(text=note + "\n", warnings=_warnings(result, sent_chars=len(SYSTEM_PROMPT + user)))


def _warnings(result: Completion, sent_chars: int) -> tuple[str, ...]:
    warnings = []
    expected = sent_chars // CHARS_PER_TOKEN
    if result.input_tokens is not None and result.input_tokens < expected // 2:
        # Ollama, for example, silently drops the start of a prompt that is longer than its context window
        warnings.append(
            f"The model read only {result.input_tokens} of about {expected} tokens, so it probably "
            "dropped the start of the transcript and the notes may miss part of it. "
            "For Ollama, raise the limit (e.g. OLLAMA_CONTEXT_LENGTH=32768) and restart it."
        )
    if result.cut_off:
        warnings.append(
            "The model reached its output limit, so the end of the note may be missing."
        )
    return tuple(warnings)


def _timestamped_text(transcript: Transcript) -> str:
    if not transcript.segments:
        return transcript.text
    lines = []
    for seg in transcript.segments:
        mins, secs = divmod(seg.start_ms // 1000, 60)
        speaker = f"{seg.speaker}: " if seg.speaker else ""
        lines.append(f"[{mins:02d}:{secs:02d}] {speaker}{seg.text}")
    return "\n".join(lines)
