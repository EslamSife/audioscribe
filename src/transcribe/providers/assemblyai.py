from pathlib import Path

import assemblyai as aai

from transcribe.transcription.model import (
    Chapter,
    Segment,
    Transcript,
    TranscriptionError,
    TranscriptionOptions,
)
from transcribe.transcription.segmenting import group_words


class AssemblyAIProvider:
    def __init__(self, api_key: str) -> None:
        self._transcriber = aai.Transcriber(api_key=api_key)

    def transcribe(
        self,
        path: Path,
        *,
        options: TranscriptionOptions | None = None,
    ) -> Transcript:
        opts = options or TranscriptionOptions()
        config_kwargs: dict = {}

        if opts.speakers or opts.speakers_expected is not None:
            config_kwargs["speaker_labels"] = True
            if opts.speakers_expected is not None:
                config_kwargs["speakers_expected"] = opts.speakers_expected

        if opts.auto_detect_language:
            config_kwargs["language_detection"] = True
        elif opts.language:
            config_kwargs["language_code"] = opts.language

        if opts.summarize:
            config_kwargs["summarization"] = True
            config_kwargs["summary_model"] = aai.SummarizationModel.informative
            config_kwargs["summary_type"] = aai.SummarizationType.bullets

        if opts.chapters:
            config_kwargs["auto_chapters"] = True

        if opts.boost_words:
            config_kwargs["word_boost"] = list(opts.boost_words)

        config = aai.TranscriptionConfig(**config_kwargs) if config_kwargs else None
        result = self._transcriber.transcribe(str(path), config)
        if result.status == aai.TranscriptStatus.error:
            raise TranscriptionError(f"Transcription failed (job {result.id}): {result.error}")
        return to_transcript(result)


def to_transcript(result) -> Transcript:
    if getattr(result, "utterances", None):
        segments = tuple(
            Segment(
                text=u.text,
                start_ms=u.start,
                end_ms=u.end,
                speaker=f"Speaker {u.speaker}",
            )
            for u in result.utterances
        )
    elif getattr(result, "words", None):
        segments = group_words([(w.text, w.start, w.end) for w in result.words])
    else:
        segments = ()

    chapters = ()
    raw_chapters = getattr(result, "chapters", None)
    if raw_chapters:
        chapters = tuple(
            Chapter(
                headline=c.headline,
                summary=c.summary,
                start_ms=c.start,
                end_ms=c.end,
                gist=getattr(c, "gist", None),
            )
            for c in raw_chapters
        )

    return Transcript(
        text=getattr(result, "text", "") or "",
        segments=segments,
        audio_duration_s=getattr(result, "audio_duration", None),
        summary=getattr(result, "summary", None),
        chapters=chapters,
        language_code=getattr(result, "language_code", None),
        confidence=getattr(result, "confidence", None),
    )
