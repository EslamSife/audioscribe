"""Local speaker diarization: label who is speaking in each transcript segment.

Uses pyannote.audio (optional heavy dependency). The transcript segments come
from Whisper; pyannote answers "which voice was active when"; we merge the two
by time overlap.
"""

import os
from dataclasses import replace
from pathlib import Path

from transcribe.transcription.model import Segment, Transcript, TranscriptionError

PIPELINE = "pyannote/speaker-diarization-3.1"

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
        labeled.append(
            Segment(text=seg.text, start_ms=seg.start_ms, end_ms=seg.end_ms, speaker=speaker)
        )
    return tuple(labeled)


def check_ready() -> None:
    """Fail fast, before a long transcription, when pyannote or its token is missing."""
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        raise TranscriptionError(
            "Speaker labels with local Whisper require 'pyannote.audio'.\n"
            "Install it with:\n"
            "  uv sync --extra speakers\n"
            "Then create a free token at https://huggingface.co/settings/tokens,\n"
            f"accept the terms of https://huggingface.co/{PIPELINE},\n"
            "and export it:  export HF_TOKEN=your_token"
        )
    if not _token():
        raise TranscriptionError(
            "Speaker diarization needs a Hugging Face token to download the model.\n"
            "Create one at https://huggingface.co/settings/tokens,\n"
            f"accept the terms of https://huggingface.co/{PIPELINE},\n"
            "then run:  export HF_TOKEN=your_token"
        )


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _run_pyannote(path: Path, speakers_expected: int | None) -> list[SpeakerTurn]:
    check_ready()
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(PIPELINE, token=_token())
        if pipeline is None:
            raise TranscriptionError(
                f"Could not load {PIPELINE}. Accept its terms at https://huggingface.co/{PIPELINE}"
            )
        kwargs = {"num_speakers": speakers_expected} if speakers_expected else {}
        output = pipeline(str(path), **kwargs)
    except TranscriptionError:
        raise
    except Exception as error:
        raise TranscriptionError(f"Speaker diarization failed: {error}")

    # pyannote 4 returns a DiarizeOutput; its "exclusive" annotation has no
    # overlapping turns, which is what matching against transcript lines needs.
    annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(
        output, "speaker_diarization", output
    )
    return [
        (int(turn.start * 1000), int(turn.end * 1000), str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


def diarize_transcript(
    path: Path,
    transcript: Transcript,
    speakers_expected: int | None = None,
) -> Transcript:
    """Return a copy of the transcript with speaker labels on every segment.

    If diarization fails, the transcript comes back unlabeled with a warning,
    so a long transcription is never thrown away because of the labels.
    """
    try:
        turns = _run_pyannote(path, speakers_expected)
    except TranscriptionError as error:
        return replace(
            transcript,
            warnings=(*transcript.warnings, f"Speaker labels skipped: {error}"),
        )
    return replace(transcript, segments=assign_speakers(transcript.segments, turns))
