import json

from transcribe.transcription.model import Transcript


def render(transcript: Transcript, title: str) -> str:
    data = {
        "title": title,
        "text": transcript.text,
        "audio_duration_s": transcript.audio_duration_s,
        "language_code": transcript.language_code,
        "confidence": transcript.confidence,
        "summary": transcript.summary,
        "chapters": [
            {
                "headline": c.headline,
                "summary": c.summary,
                "start_ms": c.start_ms,
                "end_ms": c.end_ms,
                "gist": c.gist,
            }
            for c in transcript.chapters
        ],
        "segments": [
            {
                "text": s.text,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "speaker": s.speaker,
            }
            for s in transcript.segments
        ],
    }
    return json.dumps(data, indent=2) + "\n"
