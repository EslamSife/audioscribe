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
                "--language", "es",
                "--summarize",
                "--chapters",
                "--boost", "Python,AssemblyAI",
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

    with patch("transcribe.cli.get_provider") as mock_get_provider, \
         patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT):
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

        result = runner.invoke(app, ["live", "--out", str(out_file), "--format", "md", "--device", "1"])
        assert result.exit_code == 0
        assert out_file.exists()
        assert "live_recording" in out_file.read_text()
        mock_session_cls.assert_called_once_with(language=None, device="1", mix_device=None, require_api_key=True)



def test_cli_list_devices():
    with patch("transcribe.cli.print_audio_devices_table") as mock_print_table:
        result = runner.invoke(app, ["--list-devices"])
        assert result.exit_code == 0
        mock_print_table.assert_called_once()


def test_cli_url_transcription(tmp_path, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "mock-key")
    audio = make_test_audio(tmp_path, "downloaded.mp3")
    out_target = tmp_path / "yt_output.txt"

    with patch("transcribe.cli.url_audio_source") as mock_url_source, \
         patch("transcribe.cli.transcribe_file", return_value=DUMMY_TRANSCRIPT):

        # Context manager returns (audio, "YouTube_Video_Title")
        mock_url_source.return_value.__enter__.return_value = (audio, "YouTube_Video_Title")

        result = runner.invoke(app, ["https://youtube.com/watch?v=123", "--format", "txt", "--out", str(out_target)])
        assert result.exit_code == 0
        assert "yt_output.txt" in result.output
        assert out_target.exists()



