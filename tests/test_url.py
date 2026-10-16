from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe.audio.url import (
    fetch_audio_from_url,
    is_url,
    sanitize_title,
    url_audio_source,
)
from transcribe.transcription.model import TranscriptionError


def test_is_url():
    assert is_url("https://www.youtube.com/watch?v=123") is True
    assert is_url("http://example.com/audio.mp3") is True
    assert is_url("www.vimeo.com/456") is True
    assert is_url("meeting.mp3") is False
    assert is_url("/path/to/file.wav") is False


def test_sanitize_title():
    assert sanitize_title("My Great Video! (2026)") == "My_Great_Video___2026_"
    assert sanitize_title("") == "web_audio"


def test_fetch_audio_from_url_missing_yt_dlp():
    with patch.dict("sys.modules", {"yt_dlp": None}):
        with pytest.raises(TranscriptionError, match="yt-dlp"):
            fetch_audio_from_url("https://example.com/video")


def test_fetch_audio_from_url_success(tmp_path):
    mock_ydl_cls = MagicMock()
    mock_ydl = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

    def fake_extract_info(url, download=True):
        # Create a dummy file in the temp dir
        # Grab temp directory from outtmpl
        opts = mock_ydl_cls.call_args[0][0]
        outtmpl = Path(opts["outtmpl"])
        dummy_file = outtmpl.parent / "sample_video.mp3"
        dummy_file.write_bytes(b"downloaded audio")
        return {"title": "Sample Video Title"}

    mock_ydl.extract_info.side_effect = fake_extract_info

    with patch.dict("sys.modules", {"yt_dlp": MagicMock(YoutubeDL=mock_ydl_cls)}):
        audio_path, title = fetch_audio_from_url("https://youtube.com/watch?v=sample")
        assert audio_path.exists()
        assert title == "Sample_Video_Title"
        # A watch?v=...&list=... link must fetch the one video, not the whole playlist
        assert mock_ydl_cls.call_args[0][0]["noplaylist"] is True
        # Cleanup
        audio_path.parent.rmdir() if not any(audio_path.parent.iterdir()) else None


def test_url_audio_source_cleans_up_temp_dir(tmp_path):
    mock_ydl_cls = MagicMock()
    mock_ydl = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

    created_dirs: list[Path] = []

    def fake_extract_info(url, download=True):
        opts = mock_ydl_cls.call_args[0][0]
        outtmpl = Path(opts["outtmpl"])
        temp_dir = outtmpl.parent
        created_dirs.append(temp_dir)
        dummy_file = temp_dir / "clip.mp3"
        dummy_file.write_bytes(b"audio data")
        return {"title": "Short Clip"}

    mock_ydl.extract_info.side_effect = fake_extract_info

    with patch.dict("sys.modules", {"yt_dlp": MagicMock(YoutubeDL=mock_ydl_cls)}):
        with url_audio_source("https://youtube.com/watch?v=test") as (audio, title):
            assert audio.exists()
            assert title == "Short_Clip"

        # Once exited, temp dir is cleaned up
        assert not created_dirs[0].exists()
