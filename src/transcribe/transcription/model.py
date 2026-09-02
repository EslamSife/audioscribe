from dataclasses import dataclass


class TranscriptionError(Exception):
    pass


@dataclass(frozen=True)
class Segment:
    text: str
    start_ms: int
    end_ms: int
    speaker: str | None = None


@dataclass(frozen=True)
class Chapter:
    headline: str
    summary: str
    start_ms: int
    end_ms: int
    gist: str | None = None


@dataclass(frozen=True)
class TranscriptionOptions:
    speakers: bool = False
    speakers_expected: int | None = None
    language: str | None = None
    auto_detect_language: bool = False
    summarize: bool = False
    chapters: bool = False
    boost_words: tuple[str, ...] = ()
    # Text from the previous phrase, given to Whisper as context so live
    # phrases are not transcribed blind (maps to whisper's initial_prompt).
    context_prompt: str | None = None
    # Beam search width. Smaller is faster but slightly less accurate; live
    # preview passes use 3 (RealtimeSTT's realtime default), final passes
    # keep the whisper default of 5.
    beam_size: int | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: tuple[Segment, ...] = ()
    audio_duration_s: int | None = None
    summary: str | None = None
    chapters: tuple[Chapter, ...] = ()
    language_code: str | None = None
    confidence: float | None = None
