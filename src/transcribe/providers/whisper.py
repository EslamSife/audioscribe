from pathlib import Path
from typing import Any

from transcribe.transcription.model import (
    Segment,
    Transcript,
    TranscriptionError,
    TranscriptionOptions,
)


class LocalWhisperProvider:
    """Local offline transcription provider using faster-whisper or openai-whisper."""

    def __init__(self, model_size: str = "base", device: str = "auto") -> None:
        self.model_size = model_size
        self.device = device
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        # 1. Try faster-whisper (recommended, fast CTranslate2 backend)
        try:
            from faster_whisper import WhisperModel

            dev = "cuda" if self.device == "cuda" else ("cpu" if self.device == "cpu" else "auto")
            compute_type = "float16" if dev == "cuda" else "int8"
            self._model = (
                "faster_whisper",
                WhisperModel(self.model_size, device=dev, compute_type=compute_type),
            )
            return self._model
        except ImportError:
            pass

        # 2. Try standard openai-whisper
        try:
            import whisper

            dev = "cuda" if self.device == "cuda" else ("cpu" if self.device == "cpu" else None)
            self._model = ("openai_whisper", whisper.load_model(self.model_size, device=dev))
            return self._model
        except ImportError:
            pass

        raise TranscriptionError(
            "Local Whisper requires 'faster-whisper' or 'openai-whisper'.\n"
            "To use local offline transcription, install faster-whisper:\n"
            "  uv add faster-whisper\n"
            "or:\n"
            "  pip install faster-whisper"
        )

    def warm_up(self) -> None:
        """Load the model now (first use downloads it), so live sessions don't stall mid-stream."""
        self._get_model()

    def transcribe(
        self,
        path: Path,
        *,
        options: TranscriptionOptions | None = None,
        speakers: bool = False,
    ) -> Transcript:
        opts = options or TranscriptionOptions(speakers=speakers)
        backend_type, model = self._get_model()

        if backend_type == "faster_whisper":
            transcript = self._transcribe_faster_whisper(model, path, opts)
        else:
            transcript = self._transcribe_openai_whisper(model, path, opts)

        if opts.speakers and transcript.segments:
            from transcribe.diarization import diarize_transcript

            transcript = diarize_transcript(
                path, transcript, speakers_expected=opts.speakers_expected
            )
        return transcript

    def _transcribe_faster_whisper(
        self, model: Any, path: Path, opts: TranscriptionOptions
    ) -> Transcript:
        kwargs: dict[str, Any] = {}
        if opts.language and not opts.auto_detect_language:
            kwargs["language"] = opts.language
        if opts.context_prompt:
            kwargs["initial_prompt"] = opts.context_prompt
        if opts.beam_size:
            kwargs["beam_size"] = opts.beam_size

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

    def _transcribe_openai_whisper(
        self, model: Any, path: Path, opts: TranscriptionOptions
    ) -> Transcript:
        kwargs: dict[str, Any] = {}
        if opts.language and not opts.auto_detect_language:
            kwargs["language"] = opts.language
        if opts.context_prompt:
            kwargs["initial_prompt"] = opts.context_prompt
        if opts.beam_size:
            kwargs["beam_size"] = opts.beam_size

        result = model.transcribe(str(path), **kwargs)
        segments: list[Segment] = []

        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                segments.append(
                    Segment(
                        text=text,
                        start_ms=int(seg.get("start", 0) * 1000),
                        end_ms=int(seg.get("end", 0) * 1000),
                    )
                )

        full_text = result.get("text", "").strip()
        lang = result.get("language")

        return Transcript(
            text=full_text,
            segments=tuple(segments),
            language_code=lang,
        )
