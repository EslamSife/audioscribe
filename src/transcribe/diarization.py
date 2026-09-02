"""Local speaker diarization: label who is speaking in each transcript segment.

Uses pyannote.audio (optional heavy dependency). The transcript segments come
from Whisper; pyannote answers "which voice was active when"; we merge the two
by time overlap.
"""

import os
from pathlib import Path

from transcribe.transcription.model import Segment, Transcript, TranscriptionError

# (start_ms, end_ms, raw_label) of one speaker turn found by the diarization model
SpeakerTurn = tuple[int, int, str]


def assign_speakers(segments: tuple[Segment, ...], turns: list[SpeakerTurn]) -> tuple[Segment, ...]:
    """Give each transcript segment the speaker whose turn overlaps it the most.

    Raw diarization labels (e.g. 'SPEAKER_00') are renamed to 'Speaker 1',
    'Speaker 2', ... in order of first appearance.
    """
    if not turns:
        return segments

    label_names: dict[str, str] = {}

    def display_name(raw: str) -> str:
        if raw not in label_names:
            label_names[raw] = f"Speaker {len(label_names) + 1}"
        return label_names[raw]

    labeled: list[Segment] = []
    for seg in segments:
        best_label: str | None = None
        best_overlap = 0
        for t_start, t_end, raw in turns:
            overlap = min(seg.end_ms, t_end) - max(seg.start_ms, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = raw
        speaker = display_name(best_label) if best_label else None
        labeled.append(Segment(text=seg.text, start_ms=seg.start_ms, end_ms=seg.end_ms, speaker=speaker))
    return tuple(labeled)


def _run_pyannote(path: Path, speakers_expected: int | None) -> list[SpeakerTurn]:
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise TranscriptionError(
            "Speaker labels with local Whisper require 'pyannote.audio'.\n"
            "Install it with:\n"
            "  uv add pyannote.audio\n"
            "Then create a free token at https://huggingface.co/settings/tokens,\n"
            "accept the terms of https://huggingface.co/pyannote/speaker-diarization-3.1,\n"
            "and export it:  export HF_TOKEN=your_token"
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise TranscriptionError(
            "Speaker diarization needs a Hugging Face token to download the model.\n"
            "Create one at https://huggingface.co/settings/tokens,\n"
            "accept the terms of https://huggingface.co/pyannote/speaker-diarization-3.1,\n"
            "then run:  export HF_TOKEN=your_token"
        )

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        kwargs = {"num_speakers": speakers_expected} if speakers_expected else {}
        diarization = pipeline(str(path), **kwargs)
    except Exception as error:
        raise TranscriptionError(f"Speaker diarization failed: {error}")

    turns: list[SpeakerTurn] = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        turns.append((int(turn.start * 1000), int(turn.end * 1000), str(label)))
    return turns


def diarize_transcript(
    path: Path,
    transcript: Transcript,
    speakers_expected: int | None = None,
) -> Transcript:
    """Return a copy of the transcript with speaker labels on every segment."""
    turns = _run_pyannote(path, speakers_expected)
    labeled = assign_speakers(transcript.segments, turns)
    return Transcript(
        text=transcript.text,
        segments=labeled,
        audio_duration_s=transcript.audio_duration_s,
        summary=transcript.summary,
        chapters=transcript.chapters,
        language_code=transcript.language_code,
        confidence=transcript.confidence,
    )
