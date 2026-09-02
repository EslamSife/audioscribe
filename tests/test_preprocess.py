from pathlib import Path
from unittest.mock import MagicMock, patch

from transcribe.audio.preprocess import (
    optimized_audio_source,
    prepare_audio,
    should_compress,
)


def test_should_compress_detects_video_and_large_files(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"dummy video")
    assert should_compress(video) is True

    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"small audio")
    assert should_compress(audio) is False


def test_prepare_audio_falls_back_when_no_ffmpeg(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"dummy video")

    with patch("transcribe.audio.preprocess.is_ffmpeg_available", return_value=False):
        processed, is_temp = prepare_audio(video, compress=True)
        assert processed == video
        assert is_temp is False


def test_prepare_audio_converts_with_ffmpeg(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video data")

    with patch("transcribe.audio.preprocess.is_ffmpeg_available", return_value=True), \
         patch("subprocess.run") as mock_run:

        def fake_ffmpeg(cmd, **kwargs):
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"extracted mp3 audio")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_ffmpeg
        processed, is_temp = prepare_audio(video, compress=True)

        assert is_temp is True
        assert processed.exists()
        assert processed.suffix == ".mp3"
        processed.unlink()


def test_optimized_audio_source_cleans_up_temp_file(tmp_path):
    video = tmp_path / "sample.mov"
    video.write_bytes(b"mov data")

    temp_created: list[Path] = []

    with patch("transcribe.audio.preprocess.is_ffmpeg_available", return_value=True), \
         patch("subprocess.run") as mock_run:

        def fake_ffmpeg(cmd, **kwargs):
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"extracted audio")
            temp_created.append(out_file)
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_ffmpeg

        with optimized_audio_source(video, compress=True) as src:
            assert src.exists()
            assert src != video

        # Once context manager exits, temp file should be deleted
        assert not temp_created[0].exists()
