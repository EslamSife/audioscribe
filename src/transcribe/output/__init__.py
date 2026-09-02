from pathlib import Path

from transcribe.transcription.model import Transcript

from . import json, md, srt, txt, vtt

FORMATS = {
    "txt": txt.render,
    "md": md.render,
    "srt": srt.render,
    "vtt": vtt.render,
    "json": json.render,
}


def render(transcript: Transcript, fmt: str, title: str) -> str:
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Choose from {', '.join(sorted(FORMATS))}")
    return FORMATS[fmt](transcript, title)


def write(
    transcript: Transcript,
    fmt: str,
    path: Path,
    *,
    title: str,
    force: bool = False,
) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists, use --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(transcript, fmt, title), encoding="utf-8")
    return path


def unique_path(path: Path) -> Path:
    """Return `path` if free, otherwise the first free numbered variant (name_2.ext, name_3.ext, ...)."""
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1

