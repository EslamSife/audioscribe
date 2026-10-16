from pathlib import Path

import pytest

from transcribe.transcription.model import Transcript, TranscriptionError, TranscriptionOptions
from transcribe.transcription.service import transcribe_file


class FakeProvider:
    def __init__(self):
        self.received = None
        self.calls = 0

    def transcribe(self, path: Path, *, options: TranscriptionOptions) -> Transcript:
        self.calls += 1
        self.received = (path, options)
        return Transcript(text="hello")


def make_audio(tmp_path: Path, name: str = "a.mp3") -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake audio")
    return path


def test_happy_path_passes_file_and_options_to_provider(tmp_path):
    provider = FakeProvider()
    path = make_audio(tmp_path)
    options = TranscriptionOptions(speakers=True, language="es")
    result = transcribe_file(path, provider, options=options)
    assert result.text == "hello"
    assert provider.received == (path, options)


def test_a_type_error_inside_the_provider_is_not_retried(tmp_path):
    """A bug in a provider must surface once, not start a second (paid) transcription."""

    class BrokenProvider(FakeProvider):
        def transcribe(self, path, *, options):
            self.calls += 1
            raise TypeError("bug deep inside the provider")

    provider = BrokenProvider()
    with pytest.raises(TypeError):
        transcribe_file(make_audio(tmp_path), provider)
    assert provider.calls == 1


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
