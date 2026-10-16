from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from transcribe.live import LiveTranscriptionSession
from transcribe.transcription.model import TranscriptionError, TranscriptionOptions


def test_live_session_missing_drivers_raises_error():
    session = LiveTranscriptionSession(api_key="mock-key")
    with patch.dict("sys.modules", {"sounddevice": None}):
        with pytest.raises(TranscriptionError, match="sounddevice"):
            list(session.get_microphone_stream())


def fake_streaming_client():
    """A fake with the real StreamingClient's method signatures (autospec), so a
    call the real SDK would reject, like connect() without parameters, fails here too."""
    import assemblyai.streaming.v3 as v3

    client = create_autospec(v3.StreamingClient, instance=True)
    handlers = {}

    client.on.side_effect = lambda event, handler: handlers.__setitem__(event, handler)
    return client, handlers


def turn_event(text, start_ms=1200, end_ms=1900, speaker=None, end_of_turn=True):
    return SimpleNamespace(
        transcript=text,
        end_of_turn=end_of_turn,
        words=[
            SimpleNamespace(start=start_ms, end=start_ms + 100),
            SimpleNamespace(start=end_ms - 100, end=end_ms),
        ],
        speaker_label=speaker,
    )


def test_live_session_streams_and_builds_transcript():
    import assemblyai.streaming.v3 as v3

    session = LiveTranscriptionSession(api_key="mock-key", language="es")
    client, handlers = fake_streaming_client()

    def answer(chunk):
        handlers[v3.RealTimeEvents.Turn](client, turn_event(f"Heard {chunk.hex()}", speaker="A"))

    client.stream.side_effect = answer
    received = []
    with patch("assemblyai.streaming.v3.StreamingClient", return_value=client):
        transcript = session.run(
            audio_stream=iter([b"\x01\x00" * 1600, b"\x02\x00" * 1600]),
            on_turn=received.append,
            options=TranscriptionOptions(language="es", speakers=True, boost_words=("Helm",)),
        )

    params = client.connect.call_args.args[0]
    assert isinstance(params, v3.StreamingParameters)
    assert params.sample_rate == 16000
    assert params.encoding == v3.Encoding.pcm_s16le
    assert params.language_code == "es"
    assert params.speaker_labels is True
    assert params.keyterms_prompt == ["Helm"]
    assert client.disconnect.called

    assert len(received) == 2
    # Times come from the turn's words, not from fields a turn doesn't have
    assert (transcript.segments[0].start_ms, transcript.segments[0].end_ms) == (1200, 1900)
    assert transcript.segments[0].speaker == "Speaker A"
    assert transcript.audio_duration_s == 0  # 0.2s of audio
    assert transcript.language_code == "es"


def test_live_session_ignores_partial_turns():
    import assemblyai.streaming.v3 as v3

    session = LiveTranscriptionSession(api_key="mock-key")
    client, handlers = fake_streaming_client()
    client.stream.side_effect = lambda chunk: handlers[v3.RealTimeEvents.Turn](
        client, turn_event("still talk", end_of_turn=False)
    )
    with patch("assemblyai.streaming.v3.StreamingClient", return_value=client):
        transcript = session.run(audio_stream=iter([b"\x01\x00" * 1600]))
    assert transcript.segments == ()
    assert transcript.language_code is None  # unknown, not a guessed "en"


def test_live_session_keeps_what_was_heard_when_the_device_fails():
    import assemblyai.streaming.v3 as v3

    session = LiveTranscriptionSession(api_key="mock-key")
    client, handlers = fake_streaming_client()
    client.stream.side_effect = lambda chunk: handlers[v3.RealTimeEvents.Turn](
        client, turn_event("before the failure")
    )

    def failing_stream():
        yield b"\x01\x00" * 1600
        raise TranscriptionError("Audio device error: unplugged")

    with patch("assemblyai.streaming.v3.StreamingClient", return_value=client):
        transcript = session.run(audio_stream=failing_stream())
    assert transcript.text == "before the failure"


def test_live_session_fails_when_the_device_never_opens():
    session = LiveTranscriptionSession(api_key="mock-key")
    client, _ = fake_streaming_client()

    def failing_stream():
        raise TranscriptionError("Audio device error: invalid device")
        yield

    with patch("assemblyai.streaming.v3.StreamingClient", return_value=client):
        with pytest.raises(TranscriptionError, match="invalid device"):
            session.run(audio_stream=failing_stream())


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


def test_record_stream_saves_wav_and_passes_chunks_through(tmp_path):
    import wave

    from transcribe.live import record_stream

    wav_path = tmp_path / "sub" / "meeting.wav"
    chunks = [b"\x01\x00" * 1600, b"\x02\x00" * 1600]

    passed = list(record_stream(iter(chunks), wav_path, sample_rate=16000))

    assert passed == chunks
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.readframes(wf.getnframes()) == b"".join(chunks)


def test_record_stream_file_is_playable_before_it_is_closed(tmp_path):
    """A crash mid-meeting must still leave a readable WAV with the audio so far."""
    import wave

    from transcribe.live import record_stream

    wav_path = tmp_path / "meeting.wav"
    recorder = record_stream(iter([b"\x01\x00" * 1600, b"\x02\x00" * 1600]), wav_path, 16000)
    next(recorder)  # one chunk written, recorder still open

    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnframes() == 1600
    recorder.close()
