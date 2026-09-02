from pathlib import Path

import pytest

from transcribe.transcription.model import Transcript, TranscriptionError
from transcribe.transcription.service import transcribe_file


class FakeProvider:
    def __init__(self):
        self.received = None

    def transcribe(self, path: Path, *, speakers: bool) -> Transcript:
        self.received = (path, speakers)
        return Transcript(text="hello")


def make_audio(tmp_path: Path, name: str = "a.mp3") -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake audio")
    return path


def test_happy_path_passes_file_and_options_to_provider(tmp_path):
    provider = FakeProvider()
    path = make_audio(tmp_path)
    result = transcribe_file(path, provider, speakers=True)
    assert result.text == "hello"
    assert provider.received == (path, True)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(TranscriptionError, match="not found"):
        transcribe_file(tmp_path / "nope.mp3", FakeProvider())


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"x")
    with pytest.raises(TranscriptionError, match="Unsupported"):
        transcribe_file(path, FakeProvider())


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.mp3"
    path.touch()
    with pytest.raises(TranscriptionError, match="empty"):
        transcribe_file(path, FakeProvider())
