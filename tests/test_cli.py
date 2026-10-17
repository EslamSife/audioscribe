from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from transcribe import config
from transcribe.cli import app
from transcribe.transcription.model import Segment, Transcript

runner = CliRunner()

DUMMY_TRANSCRIPT = Transcript(
    text="Hello world.",
    segments=(Segment("Hello world.", 0, 1000),),
    audio_duration_s=5,
    summary="Short summary",
    language_code="en",
    confidence=0.99,
)


def make_test_audio(tmp_path: Path, name: str = "meeting.mp3") -> Path:
    f = tmp_path / name
    f.write_bytes(b"dummy audio content")
    return f


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Transcribe audio files" in result.output


def test_cli_no_args_shows_error():
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Please provide audio files, URLs, or 'live'" in result.output


def test_cli_invalid_format(tmp_path):
    audio = make_test_audio(tmp_path)
    result = runner.invoke(app, [str(audio), "--format", "docx"])
    assert result.exit_code == 2
    assert "unknown format 'docx'" in result.output


def test_cli_transcribe_single_file(tmp_path, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    audio = make_test_audio(tmp_path, "talk.mp3")

    with patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT):
        result = runner.invoke(app, [str(audio), "--format", "txt"])
        assert result.exit_code == 0
        assert "Transcript written to" in result.output
        out_file = tmp_path / "talk.txt"
        assert out_file.exists()
        assert "Hello world." in out_file.read_text()


def test_cli_stdout_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    audio = make_test_audio(tmp_path, "talk.mp3")

    with patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT):
        result = runner.invoke(app, [str(audio), "--format", "vtt", "--stdout"])
        assert result.exit_code == 0
        assert "WEBVTT" in result.output
        assert not (tmp_path / "talk.vtt").exists()


def test_cli_batch_files(tmp_path, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    f1 = make_test_audio(tmp_path, "a.mp3")
    f2 = make_test_audio(tmp_path, "b.wav")

    with patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT):
        result = runner.invoke(app, [str(f1), str(f2), "--format", "json"])
        assert result.exit_code == 0
        assert (tmp_path / "a.json").exists()
        assert (tmp_path / "b.json").exists()


def test_cli_options_forwarding(tmp_path, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    audio = make_test_audio(tmp_path, "lecture.m4a")

    with patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT) as mock_transcribe:
        result = runner.invoke(
            app,
            [
                str(audio),
                "--speakers",
                "--language",
                "es",
                "--summarize",
                "--chapters",
                "--boost",
                "Python,AssemblyAI",
            ],
        )
        assert result.exit_code == 0
        call_args, call_kwargs = mock_transcribe.call_args
        opts = call_kwargs["options"]
        assert opts.speakers is True
        assert opts.language == "es"
        assert opts.summarize is True
        assert opts.chapters is True
        assert opts.boost_words == ("Python", "AssemblyAI")


def test_cli_config_set_key_and_show(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(config, "get_config_path", lambda: cfg_file)
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)

    # Set key using subcommand
    res_set = runner.invoke(app, ["config", "set-key", "my-secret-key-12345"])
    assert res_set.exit_code == 0
    assert "successfully saved" in res_set.output
    assert cfg_file.exists()

    # Show config
    res_show = runner.invoke(app, ["config", "show"])
    assert res_show.exit_code == 0
    assert "configured" in res_show.output


def test_cli_whisper_provider(tmp_path):
    audio = make_test_audio(tmp_path, "local_test.mp3")

    with (
        patch("transcribe.cli.get_provider") as mock_get_provider,
        patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT),
    ):
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        result = runner.invoke(
            app,
            [str(audio), "--provider", "whisper", "--model", "tiny"],
        )
        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("whisper", model="tiny")


def test_cli_live_transcription_command(tmp_path):
    out_file = tmp_path / "live_out.md"
    with patch("transcribe.cli.LiveTranscriptionSession") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.run.return_value = DUMMY_TRANSCRIPT
        mock_session_cls.return_value = mock_session

        result = runner.invoke(
            app, ["live", "--out", str(out_file), "--format", "md", "--device", "1"]
        )
        assert result.exit_code == 0
        assert out_file.exists()
        assert "live_recording" in out_file.read_text()
        mock_session_cls.assert_called_once_with(
            language=None, device="1", mix_device=None, require_api_key=True
        )


def test_cli_list_devices():
    with patch("transcribe.cli.print_audio_devices_table") as mock_print_table:
        result = runner.invoke(app, ["--list-devices"])
        assert result.exit_code == 0
        mock_print_table.assert_called_once()


def test_cli_url_transcription(tmp_path, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    audio = make_test_audio(tmp_path, "downloaded.mp3")
    out_target = tmp_path / "yt_output.txt"

    with (
        patch("transcribe.cli.url_audio_source") as mock_url_source,
        patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT),
    ):
        # Context manager returns (audio, "YouTube_Video_Title")
        mock_url_source.return_value.__enter__.return_value = (audio, "YouTube_Video_Title")

        result = runner.invoke(
            app, ["https://youtube.com/watch?v=123", "--format", "txt", "--out", str(out_target)]
        )
        assert result.exit_code == 0
        assert "yt_output.txt" in result.output
        assert out_target.exists()


FINAL_TRANSCRIPT = Transcript(
    text="Hello world, said clearly.",
    segments=(Segment("Hello world, said clearly.", 0, 1000, speaker="Speaker 1"),),
    audio_duration_s=5,
    language_code="en",
)


def _live_session_that_records():
    """A fake live session that writes a small WAV wherever the CLI asks it to record."""
    import wave

    def fake_run(on_turn=None, record_to=None, **kwargs):
        with wave.open(str(record_to), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)
        return DUMMY_TRANSCRIPT

    session = MagicMock()
    session.run.side_effect = fake_run
    return session


def test_cli_live_saves_audio_next_to_transcript(tmp_path):
    out_file = tmp_path / "meeting.md"
    with patch(
        "transcribe.cli.LiveTranscriptionSession", return_value=_live_session_that_records()
    ):
        result = runner.invoke(app, ["live", "--out", str(out_file), "--format", "md"])

    assert result.exit_code == 0
    assert (tmp_path / "meeting.wav").exists()
    assert "Audio saved to" in result.output


def test_cli_live_final_pass_replaces_live_transcript(tmp_path):
    out_file = tmp_path / "meeting.md"
    with (
        patch("transcribe.cli.LiveTranscriptionSession", return_value=_live_session_that_records()),
        patch("transcribe.cli.get_provider") as mock_get_provider,
        patch("transcribe.cli.transcribe_file", return_value=FINAL_TRANSCRIPT) as mock_transcribe,
    ):
        result = runner.invoke(
            app,
            ["live", "--out", str(out_file), "--format", "md", "--final-model", "small", "-s"],
        )

    assert result.exit_code == 0
    mock_get_provider.assert_called_once_with("assemblyai", model="small")
    assert mock_transcribe.call_args.kwargs["options"].speakers is True
    text = out_file.read_text()
    assert "said clearly" in text
    assert "Speaker 1" in text


def test_cli_live_final_pass_failure_keeps_live_transcript(tmp_path):
    from transcribe.transcription.model import TranscriptionError

    out_file = tmp_path / "meeting.md"
    with (
        patch("transcribe.cli.LiveTranscriptionSession", return_value=_live_session_that_records()),
        patch("transcribe.cli.get_provider"),
        patch("transcribe.cli.transcribe_file", side_effect=TranscriptionError("model broke")),
    ):
        result = runner.invoke(
            app, ["live", "--out", str(out_file), "--format", "md", "--final-model", "small"]
        )

    assert result.exit_code == 0
    assert "keeping the live transcript" in result.output
    assert "Hello world." in out_file.read_text()


def test_cli_live_no_save_audio_skips_recording(tmp_path):
    out_file = tmp_path / "meeting.md"
    session = MagicMock()
    session.run.return_value = DUMMY_TRANSCRIPT
    with patch("transcribe.cli.LiveTranscriptionSession", return_value=session):
        result = runner.invoke(
            app, ["live", "--out", str(out_file), "--format", "md", "--no-save-audio"]
        )

    assert result.exit_code == 0
    assert session.run.call_args.kwargs["record_to"] is None


def test_cli_live_final_model_needs_saved_audio():
    result = runner.invoke(app, ["live", "--stdout", "--final-model", "small"])
    assert result.exit_code == 2
    assert "--final-model needs the saved audio" in result.output


def test_cli_notes_dir_writes_vault_style_note(tmp_path, monkeypatch):
    from transcribe.notes import Notes

    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    audio = make_test_audio(tmp_path)
    inbox = tmp_path / "00_Inbox"
    inbox.mkdir()
    (inbox / "release-planning-2026-10-16.md").write_text("an older note")

    with (
        patch("transcribe.cli.get_provider"),
        patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT),
        patch("transcribe.notes.get_notes_provider", return_value=MagicMock(label="Fake")),
        patch("transcribe.notes.generate_notes", return_value=Notes("# Release Planning\n")),
        patch("transcribe.cli.date") as mock_date,
    ):
        mock_date.today.return_value = __import__("datetime").date(2026, 10, 16)
        result = runner.invoke(app, [str(audio), "--notes", "--notes-dir", str(inbox)])

    assert result.exit_code == 0, result.output
    note = inbox / "release-planning-2026-10-16-2.md"  # never overwrites the older note
    assert note.read_text().startswith("---\ncreated: 2026-10-16\n")
    assert (inbox / "release-planning-2026-10-16.md").read_text() == "an older note"


def test_cli_missing_notes_dir_stops_before_recording(tmp_path):
    with patch("transcribe.cli.LiveTranscriptionSession") as session_cls:
        result = runner.invoke(app, ["live", "--notes", "--notes-dir", str(tmp_path / "missing")])

    assert result.exit_code == 2
    assert "notes folder not found" in result.output
    session_cls.assert_not_called()


def test_cli_config_set_key_asks_for_the_key_when_not_given(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(config, "get_config_path", lambda: cfg_file)

    result = runner.invoke(app, ["config", "set-key"], input="typed-secret-key\n")

    assert result.exit_code == 0
    assert "typed-secret-key" not in result.output  # hidden while typing
    assert "typed-secret-key" in cfg_file.read_text()


def test_cli_warns_about_options_whisper_cannot_use(tmp_path):
    audio = make_test_audio(tmp_path)
    with (
        patch("transcribe.cli.get_provider"),
        patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT),
    ):
        result = runner.invoke(app, [str(audio), "--provider", "whisper", "--summarize"])

    assert result.exit_code == 0
    assert "--summarize only works" in result.output


def test_cli_prints_transcript_warnings(tmp_path, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    audio = make_test_audio(tmp_path)
    warned = Transcript(text="hi", warnings=("Speaker labels skipped: no GPU",))
    with patch("transcribe.cli.transcribe_file", return_value=warned):
        result = runner.invoke(app, [str(audio)])

    assert result.exit_code == 0
    assert "Speaker labels skipped: no GPU" in result.output


def test_cli_bad_notes_setup_stops_before_recording(monkeypatch):
    monkeypatch.setenv("NOTES_PROVIDER", "ollama")
    monkeypatch.delenv("NOTES_MODEL", raising=False)
    monkeypatch.setattr("transcribe.notes.factory.load_dotenv", lambda: None)
    with patch("transcribe.cli.LiveTranscriptionSession") as session_cls:
        result = runner.invoke(app, ["live", "--notes"])

    assert result.exit_code == 2
    assert "NOTES_MODEL is not set" in result.output
    session_cls.assert_not_called()


def test_cli_template_reaches_the_notes_and_the_tags(tmp_path, monkeypatch):
    from transcribe.notes import Notes

    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    audio = make_test_audio(tmp_path)

    with (
        patch("transcribe.cli.get_provider"),
        patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT),
        patch("transcribe.notes.get_notes_provider", return_value=MagicMock(label="Fake")),
        patch("transcribe.notes.generate_notes", return_value=Notes("# Standup\n")) as notes,
    ):
        result = runner.invoke(app, [str(audio), "--notes", "--template", "Standup"])

    assert result.exit_code == 0, result.output
    assert notes.call_args.kwargs["template"] == "standup"
    note = tmp_path / "meeting_notes.md"
    assert "tags: [meeting, transcript, standup]" in note.read_text()


def test_cli_unknown_template_stops_before_recording(monkeypatch):
    with patch("transcribe.notes.get_notes_provider", return_value=MagicMock(label="Fake")):
        result = runner.invoke(app, ["live", "--notes", "--template", "retro"])
    assert result.exit_code == 2
    assert "Unknown note template 'retro'" in result.output


def test_cli_template_without_notes_says_it_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    audio = make_test_audio(tmp_path)
    with (
        patch("transcribe.cli.get_provider"),
        patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT),
    ):
        result = runner.invoke(app, [str(audio), "--template", "incident"])
    assert result.exit_code == 0
    assert "--template only applies with --notes" in result.output


def test_cli_live_final_pass_labels_a_meeting_recording_me_and_them(tmp_path):
    """Meeting mode saves the call (left) and your microphone (right) as one stereo WAV."""
    import struct
    import wave

    from transcribe import voices

    def record_meeting(engine, on_turn=None, record_to=None, **kwargs):
        with wave.open(str(record_to), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(voices.interleave(struct.pack("<h", 8000), struct.pack("<h", 3000)))
        return DUMMY_TRANSCRIPT

    class ChannelEngine:
        def transcribe(self, path, *, options):
            with wave.open(str(path), "rb") as wf:
                sample = struct.unpack("<h", wf.readframes(1))[0]
            text = "Can you deploy it?" if sample == 8000 else "Yes, tomorrow."
            return Transcript(text=text, segments=(Segment(text, 0, 1000),))

    session = MagicMock()
    session.run_offline.side_effect = record_meeting
    out_file = tmp_path / "meeting.md"
    with (
        patch("transcribe.cli.LiveTranscriptionSession", return_value=session),
        patch("transcribe.cli.get_provider", return_value=ChannelEngine()),
    ):
        result = runner.invoke(
            app,
            ["live", "-p", "whisper", "--out", str(out_file), "--format", "md"]
            + ["--device", "0", "--mix-device", "1", "--final-model", "small"],
        )

    assert result.exit_code == 0, result.output
    text = out_file.read_text()
    assert "**Them:** Can you deploy it?" in text
    assert "**Me:** Yes, tomorrow." in text


def test_cli_assemblyai_meeting_mode_says_lines_are_not_labeled(tmp_path):
    session = MagicMock()
    session.run.return_value = DUMMY_TRANSCRIPT
    with patch("transcribe.cli.LiveTranscriptionSession", return_value=session):
        result = runner.invoke(
            app,
            ["live", "--out", str(tmp_path / "m.md"), "--device", "0", "--mix-device", "1"],
        )

    assert result.exit_code == 0
    assert "--provider whisper" in " ".join(result.output.split())
