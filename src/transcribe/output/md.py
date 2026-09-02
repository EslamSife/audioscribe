from transcribe.transcription.model import Transcript


def render(transcript: Transcript, title: str) -> str:
    lines = [f"# Transcript: {title}", ""]
    meta = []
    if transcript.audio_duration_s is not None:
        meta.append(f"**Duration:** {_clock(transcript.audio_duration_s * 1000)}")
    if transcript.language_code:
        meta.append(f"**Language:** {transcript.language_code}")
    if transcript.confidence is not None:
        meta.append(f"**Confidence:** {transcript.confidence * 100:.1f}%")
    if meta:
        lines += [" | ".join(meta), ""]

    has_sections = bool(transcript.summary or transcript.chapters)

    if transcript.summary:
        lines += ["## Summary", "", transcript.summary, ""]

    if transcript.chapters:
        lines += ["## Chapters", ""]
        for c in transcript.chapters:
            time_range = f"[{_clock(c.start_ms)} - {_clock(c.end_ms)}]"
            lines.append(f"### {time_range} {c.headline}")
            lines.append("")
            lines.append(c.summary)
            lines.append("")

    if has_sections:
        lines += ["## Dialogue", ""]

    if transcript.segments:
        for s in transcript.segments:
            prefix = f"**[{_clock(s.start_ms)}]**"
            if s.speaker:
                prefix += f" **{s.speaker}:**"
            lines.append(f"{prefix} {s.text}")
            lines.append("")
    else:
        lines += [transcript.text, ""]
    return "\n".join(lines).strip() + "\n"


def _clock(ms: int) -> str:
    total_s = ms // 1000
    h, rest = divmod(total_s, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

