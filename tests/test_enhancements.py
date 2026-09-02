import struct
from pathlib import Path

from transcribe import output
from transcribe.diarization import assign_speakers
from transcribe.live import mix_chunks
from transcribe.notes import _timestamped_text
from transcribe.transcription.model import Segment, Transcript


def make_chunk(value: int, samples: int = 4) -> bytes:
    return struct.pack(f"<{samples}h", *([value] * samples))


class TestMixChunks:
    def test_sums_samples(self):
        mixed = mix_chunks(make_chunk(100), make_chunk(200))
        values = struct.unpack("<4h", mixed)
        assert values == (300, 300, 300, 300)

    def test_clips_instead_of_overflowing(self):
        mixed = mix_chunks(make_chunk(30000), make_chunk(30000))
        values = struct.unpack("<4h", mixed)
        assert values == (32767, 32767, 32767, 32767)

    def test_pads_unequal_lengths(self):
        mixed = mix_chunks(make_chunk(100, samples=2), make_chunk(50, samples=4))
        values = struct.unpack("<4h", mixed)
        assert values == (150, 150, 50, 50)


class TestAssignSpeakers:
    def test_assigns_by_max_overlap(self):
        segments = (
            Segment(text="hello", start_ms=0, end_ms=1000),
            Segment(text="hi there", start_ms=1000, end_ms=2000),
        )
        turns = [(0, 900, "SPEAKER_01"), (900, 2000, "SPEAKER_00")]
        labeled = assign_speakers(segments, turns)
        assert labeled[0].speaker == "Speaker 1"
        assert labeled[1].speaker == "Speaker 2"

    def test_same_speaker_keeps_same_name(self):
        segments = (
            Segment(text="a", start_ms=0, end_ms=1000),
            Segment(text="b", start_ms=1000, end_ms=2000),
        )
        turns = [(0, 2000, "SPEAKER_05")]
        labeled = assign_speakers(segments, turns)
        assert labeled[0].speaker == "Speaker 1"
        assert labeled[1].speaker == "Speaker 1"

    def test_no_turns_returns_segments_unchanged(self):
        segments = (Segment(text="a", start_ms=0, end_ms=1000),)
        assert assign_speakers(segments, []) == segments

    def test_no_overlap_leaves_speaker_empty(self):
        segments = (Segment(text="a", start_ms=0, end_ms=1000),)
        turns = [(5000, 6000, "SPEAKER_00")]
        labeled = assign_speakers(segments, turns)
        assert labeled[0].speaker is None


class TestUniquePath:
    def test_free_name_is_kept(self, tmp_path: Path):
        p = tmp_path / "live_session.txt"
        assert output.unique_path(p) == p

    def test_taken_name_gets_number(self, tmp_path: Path):
        p = tmp_path / "live_session.txt"
        p.write_text("x")
        assert output.unique_path(p).name == "live_session_2.txt"

    def test_numbers_keep_counting(self, tmp_path: Path):
        p = tmp_path / "live_session.txt"
        p.write_text("x")
        (tmp_path / "live_session_2.txt").write_text("x")
        assert output.unique_path(p).name == "live_session_3.txt"


class TestLiveIncrementalWrite:
    def test_lines_are_appended_during_session(self, tmp_path: Path):
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from transcribe.cli import app

        out_file = tmp_path / "live.md"
        mid_session_content = {}

        def fake_run(on_turn=None):
            on_turn("first sentence", 0, 1000)
            on_turn("second sentence", 61000, 62000)
            # Capture what is on disk before the final render
            mid_session_content["text"] = out_file.read_text()
            return Transcript(text="first sentence second sentence")

        with patch("transcribe.cli.LiveTranscriptionSession") as session_cls:
            session = MagicMock()
            session.run.side_effect = fake_run
            session_cls.return_value = session
            result = CliRunner().invoke(app, ["live", "--out", str(out_file), "--format", "md"])

        assert result.exit_code == 0
        # Lines were on disk before the session ended (crash protection)
        assert mid_session_content["text"] == "00:00 first sentence\n01:01 second sentence\n"
        # Final formatted render replaced the raw lines
        assert "# Transcript: live_recording" in out_file.read_text()


class TestNotesTimestampedText:
    def test_uses_segments_with_timestamps(self):
        t = Transcript(
            text="hello world",
            segments=(
                Segment(text="hello", start_ms=0, end_ms=1000),
                Segment(text="world", start_ms=61000, end_ms=62000, speaker="Speaker 1"),
            ),
        )
        text = _timestamped_text(t)
        assert text == "[00:00] hello\n[01:01] Speaker 1: world"

    def test_falls_back_to_plain_text(self):
        t = Transcript(text="just text")
        assert _timestamped_text(t) == "just text"
