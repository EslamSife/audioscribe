from pathlib import Path
from typing import Protocol

from transcribe.transcription.model import Transcript, TranscriptionOptions


class TranscriptionProvider(Protocol):
    def transcribe(
        self,
        path: Path,
        *,
        options: TranscriptionOptions | None = None,
        speakers: bool = False,
    ) -> Transcript: ...
