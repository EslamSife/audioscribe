"""Meeting mode labels: Me (microphone) and Them (the call), echo lines, stereo recordings."""

import struct
import wave

import numpy as np

from transcribe import voices
from transcribe.transcription.model import Segment, Transcript, TranscriptionOptions
from transcribe.voices import ME, THEM


def line(text, start_ms, end_ms, speaker):
    return Segment(text, start_ms, end_ms, speaker)


def write_stereo(path, left, right, rate=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(voices.interleave(left, right))
    return path


def test_interleave_puts_the_call_left_and_the_mic_right():
    left = struct.pack("<2h", 1, 2)
    right = struct.pack("<2h", 10, 20)
    assert struct.unpack("<4h", voices.interleave(left, right)) == (1, 10, 2, 20)


def test_split_channels_gives_back_each_side(tmp_path):
    left, right = struct.pack("<3h", 1, 2, 3), struct.pack("<3h", 7, 8, 9)
    path = write_stereo(tmp_path / "meeting.wav", left, right)

    with voices.split_channels(path) as (them_path, me_path):
        for channel_path, expected in ((them_path, left), (me_path, right)):
            with wave.open(str(channel_path), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getframerate() == 16000
                assert wf.readframes(wf.getnframes()) == expected
    assert not them_path.exists()  # temporary files are removed


def test_is_stereo(tmp_path):
    stereo = write_stereo(tmp_path / "stereo.wav", b"\x00\x00", b"\x00\x00")
    mono = tmp_path / "mono.wav"
    with wave.open(str(mono), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00")
    not_wav = tmp_path / "talk.mp3"
    not_wav.write_bytes(b"ID3 not a wav")

    assert voices.is_stereo(stereo) is True
    assert voices.is_stereo(mono) is False
    assert voices.is_stereo(not_wav) is False


def test_the_microphone_hearing_the_call_is_an_echo():
    call = line("We ship the release on Monday.", 5000, 8000, THEM)
    assert voices.is_echo(line("we ship the release on monday", 5100, 8100, ME), call)
    # Speech recognition hears the quieter echo a bit differently
    assert voices.is_echo(line("We ship release on Monday", 5200, 7900, ME), call)


def test_a_real_answer_is_not_an_echo():
    call = line("Can you deploy it tomorrow?", 5000, 7000, THEM)
    assert not voices.is_echo(line("Yes, I will deploy it in the morning.", 7200, 9000, ME), call)
    # Same words, but far apart in time: you repeating something later is real speech
    assert not voices.is_echo(line("Can you deploy it tomorrow?", 20000, 22000, ME), call)


def test_drop_echoes_keeps_the_call_and_real_overlapping_speech():
    segments = [
        line("We ship on Monday.", 0, 2000, THEM),
        line("We ship on Monday.", 100, 2100, ME),  # echo
        line("Thank you, bye.", 2500, 3500, ME),  # real, while the call also talks
        line("See you tomorrow then.", 2400, 3600, THEM),
    ]
    kept = voices.drop_echoes(segments)
    assert [(s.speaker, s.text) for s in kept] == [
        (THEM, "We ship on Monday."),
        (ME, "Thank you, bye."),
        (THEM, "See you tomorrow then."),
    ]


def test_drop_echoes_never_drops_lines_without_two_voices():
    segments = [line("Hello.", 0, 1000, None), line("Hello.", 100, 1100, None)]
    assert voices.drop_echoes(segments) == tuple(segments)


def test_merge_labels_orders_and_keeps_names_from_speaker_labels():
    them = Transcript(
        text="",
        segments=(Segment("Did you finish?", 0, 2000, "Speaker 1"), Segment("Great.", 6000, 7000)),
        audio_duration_s=10,
        language_code="en",
    )
    me = Transcript(
        text="",
        segments=(Segment("Did you finish?", 100, 2100), Segment("Yes, it is done.", 3000, 5000)),
        audio_duration_s=10,
    )

    merged = voices.merge(them, me)

    assert [(s.speaker, s.text) for s in merged.segments] == [
        ("Speaker 1", "Did you finish?"),
        (ME, "Yes, it is done."),
        (THEM, "Great."),
    ]
    assert merged.text == "Did you finish? Yes, it is done. Great."
    assert (merged.audio_duration_s, merged.language_code) == (10, "en")


class ChannelEngine:
    """Answers with what each channel of the test recording 'says', from its loudness."""

    def __init__(self):
        self.options = []

    def transcribe(self, path, *, options):
        self.options.append(options)
        with wave.open(str(path), "rb") as wf:
            peak = int(np.abs(np.frombuffer(wf.readframes(wf.getnframes()), np.int16)).max())
        text = "The call talks." if peak == 8000 else "I answer."
        return Transcript(text=text, segments=(Segment(text, 0, 1000),))


def test_two_voice_recording_is_transcribed_per_channel(tmp_path):
    path = write_stereo(
        tmp_path / "meeting.wav", struct.pack("<2h", 8000, -8000), struct.pack("<2h", 3000, -3000)
    )
    engine = ChannelEngine()

    transcript = voices.transcribe_two_voices(path, engine, TranscriptionOptions(speakers=True))

    assert [(s.speaker, s.text) for s in transcript.segments] == [
        (THEM, "The call talks."),
        (ME, "I answer."),
    ]
    # Speaker labels can split the people on the call; only you use the microphone
    assert [o.speakers for o in engine.options] == [True, False]
