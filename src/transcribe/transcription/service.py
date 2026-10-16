from pathlib import Path

from transcribe.providers.base import TranscriptionProvider

from .model import Transcript, TranscriptionError, TranscriptionOptions

SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def transcribe_file(
    path: Path,
    provider: TranscriptionProvider,
    *,
    options: TranscriptionOptions | None = None,
    speakers: bool = False,
) -> Transcript:
    validate_audio_file(path)
    opts = options or TranscriptionOptions(speakers=speakers)
    try:
        return provider.transcribe(path, options=opts)
    except TypeError:
        return provider.transcribe(path, speakers=opts.speakers)


def validate_audio_file(path: Path) -> None:
    if not path.exists():
        raise TranscriptionError(f"File not found: {path}")
    if not path.is_file():
        raise TranscriptionError(f"Not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise TranscriptionError(f"Unsupported file type '{path.suffix}'. Supported: {supported}")
    if path.stat().st_size == 0:
        raise TranscriptionError(f"File is empty: {path}")
