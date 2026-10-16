from pathlib import Path
from typing import Any

from transcribe.transcription.model import (
    Segment,
    Transcript,
    TranscriptionError,
    TranscriptionOptions,
)


class LocalWhisperProvider:
    """Local offline transcription provider using faster-whisper."""

    def __init__(self, model_size: str = "base", device: str = "auto") -> None:
        self.model_size = model_size
        self.device = device
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise TranscriptionError(
                "Local Whisper requires 'faster-whisper'.\n"
                "Install it with:\n"
                "  uv add faster-whisper"
            )

        dev = "cuda" if self.device == "cuda" else ("cpu" if self.device == "cpu" else "auto")
        compute_type = "float16" if dev == "cuda" else "int8"
        self._model = WhisperModel(self.model_size, device=dev, compute_type=compute_type)
        return self._model

    def warm_up(self) -> None:
        """Load the model now (first use downloads it), so live sessions don't stall mid-stream."""
        self._get_model()

    def transcribe(
        self,
        path: Path,
        *,
        options: TranscriptionOptions | None = None,
    ) -> Transcript:
        opts = options or TranscriptionOptions()
        if opts.speakers:
            from transcribe.diarization import check_ready

            # Before the transcription, so a missing setup doesn't waste a long run
            check_ready()

        transcript = self._transcribe(self._get_model(), path, opts)

        if opts.speakers and transcript.segments:
            from transcribe.diarization import diarize_transcript

            transcript = diarize_transcript(
                path, transcript, speakers_expected=opts.speakers_expected
            )
        return transcript

    def _transcribe(self, model: Any, path: Path, opts: TranscriptionOptions) -> Transcript:
        kwargs: dict[str, Any] = {}
        if opts.language and not opts.auto_detect_language:
            kwargs["language"] = opts.language
        if opts.context_prompt:
            kwargs["initial_prompt"] = opts.context_prompt
        if opts.boost_words:
            # Whisper's version of AssemblyAI word boost: hint words it should expect
            kwargs["hotwords"] = ", ".join(opts.boost_words)
        if opts.beam_size:
            kwargs["beam_size"] = opts.beam_size
        if opts.skip_silence:
            kwargs["vad_filter"] = True
            # Without word times, faster-whisper can stretch a line's end over
            # the silence it cut out; word times keep each line's end right
            kwargs["word_timestamps"] = True

        raw_segments, info = model.transcribe(str(path), **kwargs)
        segments: list[Segment] = []
        full_text_chunks: list[str] = []

        for seg in raw_segments:
            text = seg.text.strip()
            if text:
                segments.append(
                    Segment(
                        text=text,
                        start_ms=int(seg.start * 1000),
                        end_ms=int(seg.end * 1000),
                    )
                )
                full_text_chunks.append(text)

        full_text = " ".join(full_text_chunks)
        duration_s = int(info.duration) if getattr(info, "duration", None) is not None else None
        lang = getattr(info, "language", None)

        return Transcript(
            text=full_text,
            segments=tuple(segments),
            audio_duration_s=duration_s,
            language_code=lang,
        )
