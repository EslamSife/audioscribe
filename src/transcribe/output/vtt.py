from transcribe.transcription.model import Transcript, TranscriptionError


def render(transcript: Transcript, title: str) -> str:
    if not transcript.segments:
        raise TranscriptionError("No timing information available, cannot build subtitles")
    blocks = ["WEBVTT"]
    for i, s in enumerate(transcript.segments, start=1):
        text = f"{s.speaker}: {s.text}" if s.speaker else s.text
        blocks.append(f"{i}\n{_stamp(s.start_ms)} --> {_stamp(s.end_ms)}\n{text}")
    return "\n\n".join(blocks) + "\n"


def _stamp(ms: int) -> str:
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, milli = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"
