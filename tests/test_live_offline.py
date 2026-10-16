"""The local live engine end to end, with fake audio and fake Whisper models."""

import queue
import struct
import time

import pytest

from transcribe.live import LiveTranscriptionSession, mix_chunks, pair_chunks
from transcribe.transcription.model import Segment, Transcript, TranscriptionError

LOUD = struct.pack("<1600h", *([8000, -8000] * 800))
QUIET = b"\x00\x00" * 1600


class FakeEngine:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, path, *, options):
        self.calls += 1
        return Transcript(text=self.text, segments=(Segment(self.text, 0, 100),))


class LoudnessDetector:
    """Stands in for Silero VAD: anything loud is speech."""

    def __init__(self, silence_threshold_db: float = -45.0) -> None:
        pass

    def is_speech(self, chunk: bytes) -> bool:
        return chunk != QUIET


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr("transcribe.live.SpeechDetector", LoudnessDetector)
    monkeypatch.setattr("transcribe.live.sys.platform", "test")  # no caffeinate
    return LiveTranscriptionSession(require_api_key=False)


def test_spoken_phrase_becomes_one_timed_line(session):
    stream = [QUIET] * 3 + [LOUD] * 10 + [QUIET] * 6
    main, preview = FakeEngine("hello there"), FakeEngine("hel")
    lines = []

    transcript = session.run_offline(
        main, audio_stream=iter(stream), on_turn=lines.append, preview_provider=preview
    )

    assert [s.text for s in transcript.segments] == ["hello there"]
    assert lines == list(transcript.segments)
    assert lines[0].start_ms == 0  # includes the 300ms before the first word
    assert main.calls == 1
    assert transcript.audio_duration_s == 1  # 1.9s of audio heard
    assert transcript.language_code is None


def test_speech_still_going_when_the_stream_ends_is_kept(session):
    transcript = session.run_offline(FakeEngine("last words"), audio_stream=iter([LOUD] * 8))
    assert transcript.text == "last words"


def test_device_that_never_opens_is_an_error_not_an_empty_session(session):
    def broken():
        raise TranscriptionError("Audio device error: invalid device")
        yield

    with pytest.raises(TranscriptionError, match="invalid device"):
        session.run_offline(FakeEngine("x"), audio_stream=broken())


def test_device_failing_mid_session_keeps_what_was_heard(session):
    def failing():
        yield from [LOUD] * 6 + [QUIET] * 6
        raise TranscriptionError("Audio device error: unplugged")

    transcript = session.run_offline(FakeEngine("before the failure"), audio_stream=failing())
    assert transcript.text == "before the failure"


# --- meeting mode: pairing two devices ---


def filled(chunks):
    q = queue.Queue()
    for chunk in chunks:
        q.put(chunk)
    return q


def test_pairing_mixes_both_devices():
    a = filled([b"\x01\x00", b"\x02\x00", None])
    b = filled([b"\x10\x00", b"\x20\x00"])
    mixed = list(pair_chunks(a, b, b"\x00\x00", lambda: False))
    assert mixed == [mix_chunks(b"\x01\x00", b"\x10\x00"), mix_chunks(b"\x02\x00", b"\x20\x00")]


def test_a_dead_second_device_does_not_slow_down_the_main_one():
    a = filled([b"\x01\x00"] * 50 + [None])
    b = queue.Queue()  # never delivers anything
    started = time.monotonic()
    mixed = list(pair_chunks(a, b, b"\x00\x00", lambda: False, b_wait_s=0.05, b_stall_after=3))
    # Waiting 50ms for every chunk would take 2.5s; after 3 misses it stops waiting
    assert time.monotonic() - started < 1.0
    assert mixed == [b"\x01\x00"] * 50


def test_second_device_backlog_is_trimmed_to_stay_in_step():
    a = filled([b"\x01\x00", None])
    b = filled([b"\x02\x00"] * 10)
    list(pair_chunks(a, b, b"\x00\x00", lambda: False, max_b_backlog=2))
    assert b.qsize() == 2
