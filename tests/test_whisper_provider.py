from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from transcribe.providers.factory import get_provider
from transcribe.providers.whisper import LocalWhisperProvider
from transcribe.transcription.model import TranscriptionError, TranscriptionOptions


def test_factory_returns_whisper_provider():
    provider = get_provider("whisper", model="tiny")
    assert isinstance(provider, LocalWhisperProvider)
    assert provider.model_size == "tiny"


def test_factory_unknown_provider_raises_error():
    with pytest.raises(TranscriptionError, match="Unknown provider"):
        get_provider("invalid_provider")


def test_whisper_missing_libraries_raises_helpful_error(tmp_path):
    provider = LocalWhisperProvider(model_size="base")
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"audio")

    with patch.dict("sys.modules", {"faster_whisper": None, "whisper": None}):
        with pytest.raises(TranscriptionError, match="faster-whisper"):
            provider.transcribe(audio)


def test_whisper_faster_whisper_transcription(tmp_path):
    provider = LocalWhisperProvider(model_size="base")
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"audio")

    mock_model = MagicMock()
    mock_segments = [
        SimpleNamespace(text="Hello world.", start=0.0, end=1.5),
        SimpleNamespace(text="Second sentence.", start=2.0, end=4.0),
    ]
    mock_info = SimpleNamespace(duration=4.5, language="en")
    mock_model.transcribe.return_value = (mock_segments, mock_info)

    provider._model = ("faster_whisper", mock_model)
    transcript = provider.transcribe(audio, options=TranscriptionOptions(language="en"))

    assert transcript.text == "Hello world. Second sentence."
    assert len(transcript.segments) == 2
    assert transcript.segments[0].start_ms == 0
    assert transcript.segments[0].end_ms == 1500
    assert transcript.segments[1].start_ms == 2000
    assert transcript.segments[1].end_ms == 4000
    assert transcript.audio_duration_s == 4
    assert transcript.language_code == "en"


def test_whisper_openai_whisper_transcription(tmp_path):
    provider = LocalWhisperProvider(model_size="tiny")
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"audio")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "OpenAI whisper output.",
        "language": "es",
        "segments": [
            {"text": "OpenAI whisper output.", "start": 0.5, "end": 3.2},
        ],
    }

    provider._model = ("openai_whisper", mock_model)
    transcript = provider.transcribe(audio)

    assert transcript.text == "OpenAI whisper output."
    assert len(transcript.segments) == 1
    assert transcript.segments[0].start_ms == 500
    assert transcript.segments[0].end_ms == 3200
    assert transcript.language_code == "es"
