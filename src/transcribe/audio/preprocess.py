import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v"}
LARGE_FILE_THRESHOLD_BYTES = 25 * 1024 * 1024  # 25 MB


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is accessible on the system PATH."""
    return shutil.which("ffmpeg") is not None


def should_compress(path: Path) -> bool:
    """Determine if an audio/video file benefits from preprocessing/downsampling."""
    if not path.exists():
        return False
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return True
    try:
        return path.stat().st_size > LARGE_FILE_THRESHOLD_BYTES
    except OSError:
        return False


def prepare_audio(path: Path, *, compress: bool = True) -> tuple[Path, bool]:
    """
    Extract or downsample audio for efficient cloud upload or faster local inference.
    Returns (path_to_transcribe, is_temporary_file).
    """
    if not compress or not is_ffmpeg_available() or not should_compress(path):
        return path, False

    temp_fd, temp_path_str = tempfile.mkstemp(suffix=".mp3", prefix="transcribe_")
    os.close(temp_fd)
    temp_path = Path(temp_path_str)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(temp_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
            return temp_path, True
    except Exception:
        pass

    # Fallback to original if ffmpeg encountered an issue
    if temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            pass
    return path, False


@contextmanager
def optimized_audio_source(path: Path, *, compress: bool = True) -> Generator[Path, None, None]:
    """Context manager yielding the optimal audio file path and ensuring temp cleanup."""
    processed_path, is_temp = prepare_audio(path, compress=compress)
    try:
        yield processed_path
    finally:
        if is_temp and processed_path.exists():
            try:
                processed_path.unlink()
            except OSError:
                pass
