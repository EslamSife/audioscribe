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
        self.options = []

    def transcribe(self, path, *, options):
        self.calls += 1
        self.options.append(options)
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


def test_live_phrases_are_not_silence_filtered_again(session):
    """Phrases are already cut at pauses; a second filter could drop short words."""
    stream = [QUIET] * 3 + [LOUD] * 10 + [QUIET] * 6
    main, preview = FakeEngine("hello there"), FakeEngine("hel")

    session.run_offline(main, audio_stream=iter(stream), preview_provider=preview)

    assert main.options and all(not o.skip_silence for o in main.options + preview.options)


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


def test_pairing_keeps_both_devices_side_by_side():
    a = filled([b"\x01\x00", b"\x02\x00", None])
    b = filled([b"\x10\x00", b"\x20\x00"])
    pairs = list(pair_chunks(a, b, b"\x00\x00", lambda: False))
    assert pairs == [(b"\x01\x00", b"\x10\x00"), (b"\x02\x00", b"\x20\x00")]


def test_mixing_adds_the_two_devices():
    assert mix_chunks(b"\x01\x00", b"\x10\x00") == b"\x11\x00"


def test_a_dead_second_device_does_not_slow_down_the_main_one():
    a = filled([b"\x01\x00"] * 50 + [None])
    b = queue.Queue()  # never delivers anything
    started = time.monotonic()
    pairs = list(pair_chunks(a, b, b"\x00\x00", lambda: False, b_wait_s=0.05, b_stall_after=3))
    # Waiting 50ms for every chunk would take 2.5s; after 3 misses it stops waiting
    assert time.monotonic() - started < 1.0
    assert pairs == [(b"\x01\x00", b"\x00\x00")] * 50  # silence stands in for B


def test_second_device_backlog_is_trimmed_to_stay_in_step():
    a = filled([b"\x01\x00", None])
    b = filled([b"\x02\x00"] * 10)
    list(pair_chunks(a, b, b"\x00\x00", lambda: False, max_b_backlog=2))
    assert b.qsize() == 2


# --- sentence ends found by the preview model, and model errors ---


def paced(chunks):
    """Deliver chunks at a steady pace, like a microphone, so the preview has time to run."""
    for chunk in chunks:
        time.sleep(0.01)
        yield chunk


class ScriptedEngine:
    """Returns the given results in order, then repeats the last one."""

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls = 0

    def transcribe(self, path, *, options):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


def test_finished_sentence_is_committed_before_the_pause(session):
    """When the preview sees two sentences, the first one is sent to the main model at once."""
    two_sentences = Transcript(
        text="First one. Second",
        segments=(Segment("First one.", 0, 200), Segment("Second", 200, 300)),
    )
    still_talking = Transcript(text="Second part", segments=(Segment("Second part", 0, 500),))
    preview = ScriptedEngine(two_sentences, still_talking)
    main = ScriptedEngine(
        Transcript(text="First one.", segments=()), Transcript(text="Second part", segments=())
    )

    transcript = session.run_offline(
        main, audio_stream=paced([LOUD] * 8 + [QUIET] * 6), preview_provider=preview
    )

    assert [s.text for s in transcript.segments] == ["First one.", "Second part"]
    first, second = transcript.segments
    assert first.start_ms == 0
    assert first.end_ms == second.start_ms  # cut at the sentence end, no audio lost


def test_a_model_error_on_one_phrase_does_not_end_the_session(session):
    main = ScriptedEngine(RuntimeError("model crashed"), Transcript(text="second", segments=()))
    stream = ([LOUD] * 4 + [QUIET] * 6) * 2

    transcript = session.run_offline(
        main, audio_stream=iter(stream), preview_provider=FakeEngine("x")
    )

    assert transcript.text == "second"
    assert main.calls == 2


# --- the speech detector ---


def test_quiet_audio_is_never_speech():
    from transcribe.live import SpeechDetector

    assert SpeechDetector().is_speech(QUIET) is False


def test_speech_detector_falls_back_to_loudness_when_the_vad_model_fails():
    from transcribe.live import SpeechDetector

    detector = SpeechDetector()

    def broken_vad(audio, options):
        raise RuntimeError("onnxruntime error")

    detector._get_speech_timestamps = broken_vad
    assert detector.is_speech(LOUD) is True
    assert detector._vad_ready is False  # the broken model is not tried again


# --- meeting mode: two voices, labeled Me and Them ---

SOFT = struct.pack("<1600h", *([3000, -3000] * 800))  # the microphone, quieter than the call


class VoiceEngine:
    """Says which fake voice a phrase holds, from its loudness: LOUD is the call, SOFT is you."""

    def transcribe(self, path, *, options):
        import wave

        import numpy as np

        with wave.open(str(path), "rb") as wf:
            peak = int(np.abs(np.frombuffer(wf.readframes(wf.getnframes()), np.int16)).max())
        text = "Can you deploy it tomorrow?" if peak == 8000 else "Yes, in the morning."
        return Transcript(text=text, segments=(Segment(text, 0, 100),))


@pytest.fixture
def meeting(monkeypatch):
    monkeypatch.setattr("transcribe.live.SpeechDetector", LoudnessDetector)
    monkeypatch.setattr("transcribe.live.sys.platform", "test")
    return LiveTranscriptionSession(device=0, mix_device=1, require_api_key=False)


def test_meeting_lines_say_who_spoke(meeting):
    pause = [(QUIET, QUIET)] * 6
    stream = [(LOUD, QUIET)] * 6 + pause + [(QUIET, SOFT)] * 6 + pause

    transcript = meeting.run_offline(VoiceEngine(), audio_stream=iter(stream))

    assert [(s.speaker, s.text) for s in transcript.segments] == [
        ("Them", "Can you deploy it tomorrow?"),
        ("Me", "Yes, in the morning."),
    ]
    assert transcript.segments[1].start_ms > transcript.segments[0].start_ms


def test_two_people_talking_at_once_are_both_kept(meeting):
    stream = [(LOUD, SOFT)] * 6 + [(QUIET, QUIET)] * 6

    transcript = meeting.run_offline(VoiceEngine(), audio_stream=iter(stream))

    assert sorted(s.speaker for s in transcript.segments) == ["Me", "Them"]


def test_the_microphone_hearing_the_call_is_not_a_line(meeting):
    """Without headphones the mic picks up the call from the speakers."""
    stream = [(LOUD, SOFT)] * 6 + [(QUIET, QUIET)] * 6
    lines = []

    transcript = meeting.run_offline(
        FakeEngine("We ship on Monday."), audio_stream=iter(stream), on_turn=lines.append
    )

    assert [(s.speaker, s.text) for s in transcript.segments] == [("Them", "We ship on Monday.")]
    assert lines == list(transcript.segments)  # the echo was never shown either


def test_meeting_recording_keeps_each_voice_on_its_own_channel(meeting, tmp_path):
    import wave

    from transcribe import voices

    wav_path = tmp_path / "meeting.wav"
    meeting.run_offline(FakeEngine("x"), audio_stream=iter([(LOUD, SOFT)] * 3), record_to=wav_path)

    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 2
    with voices.split_channels(wav_path) as (them_path, me_path):
        with wave.open(str(them_path), "rb") as wf:
            assert wf.readframes(wf.getnframes()) == LOUD * 3
        with wave.open(str(me_path), "rb") as wf:
            assert wf.readframes(wf.getnframes()) == SOFT * 3
