import io
import json
import sys
import urllib.error
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from transcribe import doctor
from transcribe.cli import app
from transcribe.doctor import FAIL, OK, WARN, Check

runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    # No .env file and no real settings may leak into the checks
    monkeypatch.chdir(tmp_path)
    for name in (
        "NOTES_PROVIDER",
        "NOTES_MODEL",
        "NOTES_BASE_URL",
        "NOTES_API_KEY",
        "NOTES_DIR",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def fake_sounddevice(*names: str):
    devices = [{"name": n, "max_input_channels": 2} for n in names]
    devices.append({"name": "Speakers", "max_input_channels": 0})
    return SimpleNamespace(query_devices=lambda: devices)


def test_missing_ffmpeg_is_a_warning_with_the_fix(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    check = doctor.check_ffmpeg()
    assert check.status == WARN
    assert "ffmpeg" in check.fix


def test_loopback_device_is_found(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "sounddevice", fake_sounddevice("MacBook Mic", "BlackHole 2ch")
    )
    check = doctor.check_audio_devices()
    assert check.status == OK
    assert "BlackHole 2ch" in check.detail


def test_no_loopback_device_warns_that_only_the_mic_is_heard(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice("MacBook Mic"))
    check = doctor.check_audio_devices()
    assert check.status == WARN
    assert "only your mic" in check.detail


def test_missing_portaudio_is_a_warning_not_a_crash(monkeypatch):
    class Broken:
        def query_devices(self):
            raise OSError("PortAudio library not found")

    monkeypatch.setitem(sys.modules, "sounddevice", Broken())
    check = doctor.check_audio_devices()
    assert check.status == WARN
    assert "portaudio" in check.fix.lower()


def test_old_yt_dlp_warns(monkeypatch):
    monkeypatch.setattr("yt_dlp.version.__version__", "2026.01.02")
    assert doctor.check_yt_dlp(today=date(2026, 10, 17)).status == WARN
    assert doctor.check_yt_dlp(today=date(2026, 2, 1)).status == OK


def test_missing_notes_folder_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("NOTES_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    checks = doctor.check_notes()
    assert checks[0].status == FAIL


def test_claude_without_a_key_is_only_a_warning():
    [check] = doctor.check_notes()
    assert check.status == WARN
    assert "ANTHROPIC_API_KEY" in check.detail


def test_a_chosen_but_broken_notes_provider_fails(monkeypatch):
    monkeypatch.setenv("NOTES_PROVIDER", "nonsense")
    [check] = doctor.check_notes()
    assert check.status == FAIL


def ollama_tags(*models: str):
    body = json.dumps({"models": [{"name": m} for m in models]}).encode()
    return lambda url, timeout: io.BytesIO(body)


def test_ollama_with_the_model_is_ok():
    with patch.object(doctor.urllib.request, "urlopen", ollama_tags("qwen2.5:7b")):
        check = doctor.check_ollama("http://localhost:11434/v1", "qwen2.5:7b")
    assert check.status == OK
    assert "OLLAMA_CONTEXT_LENGTH" in check.detail


def test_ollama_model_name_without_tag_means_latest():
    with patch.object(doctor.urllib.request, "urlopen", ollama_tags("llama3:latest")):
        assert doctor.check_ollama("http://localhost:11434/v1", "llama3").status == OK
    with patch.object(doctor.urllib.request, "urlopen", ollama_tags("qwen2.5:7b")):
        check = doctor.check_ollama("http://localhost:11434/v1", "qwen2.5")
    assert check.status == FAIL
    assert check.fix == "Run: ollama pull qwen2.5"


def test_ollama_not_running_fails():
    def refuse(url, timeout):
        raise urllib.error.URLError("connection refused")

    with patch.object(doctor.urllib.request, "urlopen", refuse):
        check = doctor.check_ollama("http://localhost:11434/v1", "qwen2.5:7b")
    assert check.status == FAIL
    assert "localhost:11434/api/tags" in check.detail


def test_cli_doctor_exits_1_only_when_something_is_broken():
    warn_only = [Check("ffmpeg", WARN, "not found", "brew install ffmpeg")]
    with patch("transcribe.doctor.run_checks", return_value=warn_only):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "brew install ffmpeg" in result.output

    broken = [Check("Notes folder", FAIL, "missing folder")]
    with patch("transcribe.doctor.run_checks", return_value=broken):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1


def test_assemblyai_key_is_only_needed_for_the_cloud_engine(monkeypatch):
    def no_key():
        raise doctor.config.ConfigError("not set")

    monkeypatch.setattr(doctor.config, "load_api_key", no_key)
    check = doctor.check_assemblyai_key()
    assert check.status == WARN
    assert check.fix == "Run: transcribe config set-key"

    monkeypatch.setattr(doctor.config, "load_api_key", lambda: "key")
    assert doctor.check_assemblyai_key().status == OK


def test_speaker_labels_not_ready_shows_only_the_first_line(monkeypatch):
    def not_ready():
        raise doctor.TranscriptionError("HF_TOKEN is not set.\nGet a token at ...")

    monkeypatch.setattr("transcribe.diarization.check_ready", not_ready)
    check = doctor.check_speakers()
    assert check.status == WARN
    assert "HF_TOKEN is not set." in check.detail
    assert "Get a token" not in check.detail

    monkeypatch.setattr("transcribe.diarization.check_ready", lambda: None)
    assert doctor.check_speakers().status == OK


def test_yt_dlp_version_that_is_not_a_date_is_shown_as_it_is(monkeypatch):
    monkeypatch.setattr("yt_dlp.version.__version__", "nightly")
    assert doctor.check_yt_dlp() == Check("yt-dlp", OK, "nightly")


def test_openai_provider_with_a_key_is_ok(monkeypatch):
    monkeypatch.setenv("NOTES_PROVIDER", "openai")
    monkeypatch.setenv("NOTES_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    [check] = doctor.check_notes()
    assert check.status == OK
    assert "gpt-test" in check.detail


def test_openai_style_provider_without_the_openai_package_fails(monkeypatch):
    monkeypatch.setenv("NOTES_PROVIDER", "ollama")
    monkeypatch.setenv("NOTES_MODEL", "qwen2.5:7b")
    monkeypatch.setitem(sys.modules, "openai", None)  # import fails as if not installed
    [check] = doctor.check_notes()
    assert check.status == FAIL
    assert "openai" in check.fix


def test_ollama_provider_is_checked_at_the_configured_address(monkeypatch):
    monkeypatch.setenv("NOTES_PROVIDER", "ollama")
    monkeypatch.setenv("NOTES_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("NOTES_BASE_URL", "http://gpu-box:11434/v1")
    asked = []

    def tags(url, timeout):
        asked.append(url)
        return ollama_tags("qwen2.5:7b")(url, timeout)

    with patch.object(doctor.urllib.request, "urlopen", tags):
        [check] = doctor.check_notes()
    assert check.status == OK
    assert asked == ["http://gpu-box:11434/api/tags"]


@pytest.mark.parametrize(
    "platform, ffmpeg_fix, loopback_fix",
    [
        ("darwin", "brew install ffmpeg", "BlackHole"),
        ("win32", "winget install ffmpeg", "Stereo Mix"),
        ("linux", "sudo apt install ffmpeg", "pavucontrol"),
    ],
)
def test_fixes_match_the_operating_system(monkeypatch, platform, ffmpeg_fix, loopback_fix):
    monkeypatch.setattr(doctor.sys, "platform", platform)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice("Mic"))
    assert ffmpeg_fix in doctor.check_ffmpeg().fix
    assert loopback_fix in doctor.check_audio_devices().fix
