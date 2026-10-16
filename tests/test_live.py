from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from transcribe.live import LiveTranscriptionSession
from transcribe.transcription.model import TranscriptionError


def test_live_session_missing_drivers_raises_error():
    session = LiveTranscriptionSession(api_key="mock-key")
    with patch.dict("sys.modules", {"sounddevice": None, "pyaudio": None}):
        with pytest.raises(TranscriptionError, match="sounddevice"):
            list(session.get_microphone_stream())


def test_live_session_streams_and_builds_transcript():
    session = LiveTranscriptionSession(api_key="mock-key", language="en")

    # Mock audio chunk stream
    def fake_audio_stream():
        yield b"chunk1"
        yield b"chunk2"

    # Mock StreamingClient
    mock_client = MagicMock()
    registered_handlers = {}

    def mock_on(event_type):
        def decorator(handler):
            registered_handlers[event_type] = handler
            return handler

        return decorator

    mock_client.on = mock_on

    with (
        patch("assemblyai.streaming.v3.StreamingClient", return_value=mock_client),
        patch("assemblyai.streaming.v3.RealTimeEvents") as mock_events,
    ):
        mock_events.Turn = "Turn"
        mock_events.Error = "Error"

        def simulate_turns(chunk):
            # When client.stream is called, trigger turn event
            turn_handler = registered_handlers.get("Turn")
            if turn_handler:
                turn_handler(
                    mock_client,
                    SimpleNamespace(text=f"Heard {chunk.decode()}", start=0, end=1000),
                )

        mock_client.stream.side_effect = simulate_turns

        turns_received = []
        transcript = session.run(
            audio_stream=fake_audio_stream(),
            on_turn=lambda text, s, e: turns_received.append(text),
        )

        assert mock_client.connect.called
        assert mock_client.disconnect.called
        assert len(turns_received) == 2
        assert "Heard chunk1" in transcript.text
        assert "Heard chunk2" in transcript.text
        assert len(transcript.segments) == 2


def test_list_audio_devices_mocked():
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = [
        {"name": "Built-in Microphone", "max_input_channels": 1, "default_samplerate": 48000.0},
        {"name": "Built-in Output", "max_input_channels": 0, "default_samplerate": 48000.0},
        {"name": "BlackHole 2ch", "max_input_channels": 2, "default_samplerate": 44100.0},
    ]
    mock_sd.default.device = [0, 1]

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        from transcribe.live import list_audio_devices, resolve_device_id

        devices = list_audio_devices()
        assert len(devices) == 2
        assert devices[0]["name"] == "Built-in Microphone"
        assert devices[0]["is_default"] is True
        assert devices[1]["name"] == "BlackHole 2ch"

        # Resolve device
        assert resolve_device_id(None) is None
        assert resolve_device_id(0) == 0
        assert resolve_device_id("2") == 2
        assert resolve_device_id("blackhole") == 2
        assert resolve_device_id("Built-in") == 0


def test_resolve_device_id_not_found():
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = [
        {"name": "Built-in Mic", "max_input_channels": 1, "default_samplerate": 44100.0}
    ]
    mock_sd.default.device = [0, 0]

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        from transcribe.live import resolve_device_id

        with pytest.raises(TranscriptionError, match="not found"):
            resolve_device_id("NonExistentDevice")
