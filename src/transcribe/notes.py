"""Turn a finished transcript into a structured Markdown note using Claude."""

from transcribe.transcription.model import Transcript, TranscriptionError

NOTES_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You turn raw audio transcripts into clear, structured Markdown notes.

Rules:
- Write the note in the same language as the transcript.
- Be faithful to the transcript. Never invent facts that are not in it.
- Transcripts come from imperfect speech recognition: fix obvious recognition
  errors from context, but do not change the meaning.
- Keep timestamps in [MM:SS] form when you quote or reference a moment.

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


def _timestamped_text(transcript: Transcript) -> str:
    if not transcript.segments:
        return transcript.text
    lines = []
    for seg in transcript.segments:
        mins, secs = divmod(seg.start_ms // 1000, 60)
        speaker = f"{seg.speaker}: " if seg.speaker else ""
        lines.append(f"[{mins:02d}:{secs:02d}] {speaker}{seg.text}")
    return "\n".join(lines)


def generate_notes(transcript: Transcript, title: str) -> str:
    """Send the transcript to Claude and return a structured Markdown note."""
    try:
        import anthropic
    except ImportError:
        raise TranscriptionError(
            "The --notes feature requires the 'anthropic' package.\n"
            "Install it with:\n"
            "  uv add anthropic"
        )

    text = _timestamped_text(transcript).strip()
    if not text:
        raise TranscriptionError("Transcript is empty, nothing to make notes from.")

    client = anthropic.Anthropic()
    try:
        with client.messages.stream(
            model=NOTES_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Recording name: {title}\n\nTranscript:\n{text}",
                }
            ],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError, TypeError:
        # The SDK raises TypeError when no key/credentials can be found at all
        raise TranscriptionError(
            "Claude API key is missing or invalid.\n"
            "Set it with:  export ANTHROPIC_API_KEY=your_key\n"
            "Get a key at https://console.anthropic.com/settings/keys"
        )
    except anthropic.APIConnectionError:
        raise TranscriptionError("Could not reach the Claude API. Check your internet connection.")
    except anthropic.APIStatusError as error:
        raise TranscriptionError(f"Claude API error ({error.status_code}): {error.message}")

    note = "".join(block.text for block in message.content if block.type == "text").strip()
    if not note:
        raise TranscriptionError("Claude returned an empty note.")
    return note + "\n"
