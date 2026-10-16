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

    with patch.dict("sys.modules", {"faster_whisper": None}):
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

    provider._model = mock_model
    transcript = provider.transcribe(audio, options=TranscriptionOptions(language="en"))

    assert transcript.text == "Hello world. Second sentence."
    assert len(transcript.segments) == 2
    assert transcript.segments[0].start_ms == 0
    assert transcript.segments[0].end_ms == 1500
    assert transcript.segments[1].start_ms == 2000
    assert transcript.segments[1].end_ms == 4000
    assert transcript.audio_duration_s == 4
    assert transcript.language_code == "en"


def test_whisper_sends_boost_words_as_hotwords(tmp_path):
    provider = LocalWhisperProvider(model_size="base")
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"audio")
    provider._model = MagicMock()
    provider._model.transcribe.return_value = ([], SimpleNamespace(duration=1.0, language="en"))

    provider.transcribe(audio, options=TranscriptionOptions(boost_words=("Kubernetes", "Helm")))

    assert provider._model.transcribe.call_args.kwargs["hotwords"] == "Kubernetes, Helm"


def test_whisper_skips_silence_by_default(tmp_path):
    """Whisper invents text on silence, so files and the final pass filter it out."""
    provider = LocalWhisperProvider(model_size="base")
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"audio")
    provider._model = MagicMock()
    provider._model.transcribe.return_value = ([], SimpleNamespace(duration=1.0, language="en"))

    provider.transcribe(audio)

    kwargs = provider._model.transcribe.call_args.kwargs
    assert kwargs["vad_filter"] is True
    # Keeps line ends from stretching over the silence that was cut out
    assert kwargs["word_timestamps"] is True


def test_whisper_keeps_silence_when_asked(tmp_path):
    provider = LocalWhisperProvider(model_size="base")
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"audio")
    provider._model = MagicMock()
    provider._model.transcribe.return_value = ([], SimpleNamespace(duration=1.0, language="en"))

    provider.transcribe(audio, options=TranscriptionOptions(skip_silence=False))

    kwargs = provider._model.transcribe.call_args.kwargs
    assert "vad_filter" not in kwargs
    assert "word_timestamps" not in kwargs


def test_whisper_checks_speaker_setup_before_transcribing(tmp_path, monkeypatch):
    """A missing HF_TOKEN must fail before a long transcription, not after it."""
    pytest.importorskip("pyannote.audio")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    provider = LocalWhisperProvider(model_size="base")
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"audio")
    provider._model = MagicMock()

    with pytest.raises(TranscriptionError, match="HF_TOKEN"):
        provider.transcribe(audio, options=TranscriptionOptions(speakers=True))
    provider._model.transcribe.assert_not_called()


def test_hotwords_is_a_real_faster_whisper_parameter():
    """Contract check against the installed library, so an API change fails here first."""
    import inspect

    from faster_whisper import WhisperModel

    params = inspect.signature(WhisperModel.transcribe).parameters
    assert {
        "hotwords",
        "initial_prompt",
        "beam_size",
        "language",
        "vad_filter",
        "word_timestamps",
    } <= params.keys()
