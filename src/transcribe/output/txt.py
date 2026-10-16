from transcribe.transcription.model import Transcript


def render(transcript: Transcript, title: str) -> str:
    sections = []
    if transcript.summary:
        sections.append(f"Summary:\n{transcript.summary}")
    if any(s.speaker for s in transcript.segments):
        lines = [f"{s.speaker}: {s.text}" for s in transcript.segments]
        sections.append("\n".join(lines))
    elif transcript.text:
        sections.append(transcript.text)
    return ("\n\n".join(filter(None, sections))).strip() + "\n"
