import json

import pytest

from transcribe import output
from transcribe.transcription.model import Chapter, Segment, Transcript, TranscriptionError

PLAIN = Transcript(
    text="Hello world. Next sentence.",
    segments=(
        Segment("Hello world.", 0, 1500),
        Segment("Next sentence.", 2000, 3800),
    ),
    audio_duration_s=4,
)

WITH_SPEAKERS = Transcript(
    text="Hi there. Hello.",
    segments=(
        Segment("Hi there.", 0, 1200, speaker="Speaker A"),
        Segment("Hello.", 1500, 2400, speaker="Speaker B"),
    ),
    audio_duration_s=3,
)

WITH_METADATA_AND_CHAPTERS = Transcript(
    text="Full transcript text.",
    segments=(Segment("Hello everyone.", 0, 2000, speaker="Speaker A"),),
    audio_duration_s=120,
    summary="Key meeting highlights.",
    chapters=(
        Chapter(
            headline="Kick-off",
            summary="Introductions and goal alignment.",
            start_ms=0,
            end_ms=60000,
        ),
    ),
    language_code="en",
    confidence=0.97,
)


def test_txt_without_speakers_is_plain_text():
    assert output.txt.render(PLAIN, "demo") == "Hello world. Next sentence.\n"


def test_txt_with_speakers_has_one_line_per_speaker_turn():
    assert output.txt.render(WITH_SPEAKERS, "demo") == ("Speaker A: Hi there.\nSpeaker B: Hello.\n")


def test_txt_with_summary():
    assert output.txt.render(WITH_METADATA_AND_CHAPTERS, "demo") == (
        "Summary:\nKey meeting highlights.\n\nSpeaker A: Hello everyone.\n"
    )


def test_md_contains_title_timestamps_and_speakers():
    text = output.md.render(WITH_SPEAKERS, "meeting")
    assert "# Transcript: meeting" in text
    assert "**[00:00]** **Speaker A:** Hi there." in text
    assert "**[00:01]** **Speaker B:** Hello." in text


def test_md_contains_summary_chapters_and_metadata():
    text = output.md.render(WITH_METADATA_AND_CHAPTERS, "quarterly_review")
    assert "# Transcript: quarterly_review" in text
    assert "**Duration:** 02:00" in text
    assert "**Language:** en" in text
    assert "**Confidence:** 97.0%" in text
    assert "## Summary" in text
    assert "Key meeting highlights." in text
    assert "## Chapters" in text
    assert "### [00:00 - 01:00] Kick-off" in text
    assert "Introductions and goal alignment." in text
    assert "## Dialogue" in text
    assert "**[00:00]** **Speaker A:** Hello everyone." in text


def test_srt_renders_numbered_cues_with_timestamps():
    text = output.srt.render(PLAIN, "demo")
    assert text == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello world.\n"
        "\n"
        "2\n00:00:02,000 --> 00:00:03,800\nNext sentence.\n"
    )


def test_srt_without_segments_fails_clearly():
    with pytest.raises(TranscriptionError, match="timing"):
        output.srt.render(Transcript(text="no timing"), "demo")


def test_vtt_renders_valid_webvtt():
    text = output.vtt.render(PLAIN, "demo")
    assert text == (
        "WEBVTT\n"
        "\n"
        "1\n00:00:00.000 --> 00:00:01.500\nHello world.\n"
        "\n"
        "2\n00:00:02.000 --> 00:00:03.800\nNext sentence.\n"
    )


def test_vtt_without_segments_fails_clearly():
    with pytest.raises(TranscriptionError, match="timing"):
        output.vtt.render(Transcript(text="no timing"), "demo")


def test_json_renders_structured_data():
    raw = output.json.render(WITH_METADATA_AND_CHAPTERS, "json_demo")
    data = json.loads(raw)
    assert data["title"] == "json_demo"
    assert data["text"] == "Full transcript text."
    assert data["summary"] == "Key meeting highlights."
    assert data["language_code"] == "en"
    assert data["confidence"] == 0.97
    assert len(data["chapters"]) == 1
    assert data["chapters"][0]["headline"] == "Kick-off"
    assert len(data["segments"]) == 1
    assert data["segments"][0]["speaker"] == "Speaker A"


def test_render_helper_dispatches_correctly():
    assert "WEBVTT" in output.render(PLAIN, "vtt", "demo")
    with pytest.raises(ValueError, match="Unknown format"):
        output.render(PLAIN, "invalid_fmt", "demo")


def test_write_refuses_to_overwrite_without_force(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("precious")
    with pytest.raises(FileExistsError, match="--force"):
        output.write(PLAIN, "txt", target, title="demo")
    assert target.read_text() == "precious"


def test_write_overwrites_with_force(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old")
    output.write(PLAIN, "txt", target, title="demo", force=True)
    assert target.read_text() == "Hello world. Next sentence.\n"
