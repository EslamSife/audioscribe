import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from transcribe.transcription.model import TranscriptionError


def is_url(target: str | Path) -> bool:
    """Check if the provided target path or string is a web URL."""
    s = str(target).strip()
    return s.startswith(("http://", "https://", "www."))


def sanitize_title(title: str) -> str:
    """Sanitize title for safe filesystem naming."""
    clean = "".join(c if c.isalnum() or c in ("-", "_", " ") else "_" for c in title)
    return clean.strip().replace(" ", "_")[:80] or "web_audio"


def fetch_audio_from_url(url: str) -> tuple[Path, str]:
    """
    Download audio track from a YouTube or web video URL using yt-dlp.
    Returns (downloaded_audio_path, video_title).
    """
    try:
        import yt_dlp
    except ImportError:
        raise TranscriptionError(
            "Transcribing web/YouTube URLs requires 'yt-dlp'.\n"
            "To enable URL transcription, install yt-dlp:\n"
            "  uv add yt-dlp\n"
            "or:\n"
            "  pip install yt-dlp"
        )

    temp_dir = tempfile.mkdtemp(prefix="transcribe_url_")
    outtmpl = str(Path(temp_dir) / "%(title).60s.%(ext)s")

    postprocessors = []
    if shutil.which("ffmpeg"):
        postprocessors.append(
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": postprocessors,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            raw_title = info.get("title") if info else "web_audio"
            title = sanitize_title(raw_title or "web_audio")

            files = [f for f in Path(temp_dir).iterdir() if f.is_file()]
            if not files:
                raise TranscriptionError(f"Failed to download audio from {url}")
            return files[0], title
    except Exception as e:
        # Cleanup temp directory if download fails
        shutil.rmtree(temp_dir, ignore_errors=True)
        if isinstance(e, TranscriptionError):
            raise
        raise TranscriptionError(f"Could not extract audio from URL '{url}': {e}") from e


@contextmanager
def url_audio_source(url: str) -> Generator[tuple[Path, str], None, None]:
    """Context manager for URL audio extraction with automatic temporary file cleanup."""
    audio_path, title = fetch_audio_from_url(url)
    parent_dir = audio_path.parent
    try:
        yield audio_path, title
    finally:
        shutil.rmtree(parent_dir, ignore_errors=True)
