"""`transcribe doctor`: check the setup before a meeting depends on it.

Each check is independent and returns a Check. A missing optional feature is a
warning with the fix; "fail" is kept for something the user set up that is
broken (e.g. NOTES_DIR points to a missing folder), so the exit code means
"your setup will break a run".
"""

import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from transcribe import config
from transcribe.transcription.model import TranscriptionError

OK, WARN, FAIL = "ok", "warn", "fail"

# Names of virtual devices that carry computer audio (BlackHole on macOS,
# Stereo Mix on Windows, PulseAudio/PipeWire monitors on Linux)
LOOPBACK_NAMES = ("blackhole", "soundflower", "loopback", "stereo mix", "monitor of")

# YouTube changes often and old yt-dlp versions stop working; its version is a date
YT_DLP_MAX_AGE_DAYS = 90


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def run_checks() -> list[Check]:
    load_dotenv()
    return [
        check_ffmpeg(),
        check_assemblyai_key(),
        check_audio_devices(),
        check_speakers(),
        check_yt_dlp(),
        *check_notes(),
    ]


def check_ffmpeg() -> Check:
    if shutil.which("ffmpeg"):
        return Check("ffmpeg", OK, "found")
    return Check(
        "ffmpeg",
        WARN,
        "not found: video and large audio files are used as they are, without shrinking",
        f"Install it: {_install_hint('ffmpeg')}",
    )


def check_assemblyai_key() -> Check:
    try:
        config.load_api_key()
    except config.ConfigError:
        return Check(
            "AssemblyAI key",
            WARN,
            "not set: needed for the default cloud engine, not for --provider whisper",
            "Run: transcribe config set-key",
        )
    return Check("AssemblyAI key", OK, "found")


def check_audio_devices() -> Check:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except (ImportError, OSError) as error:
        # On Linux, a missing PortAudio library shows up as an OSError on import
        return Check(
            "Audio devices",
            WARN,
            f"cannot list devices, live mode will not work ({error})",
            "Linux: sudo apt install libportaudio2",
        )

    inputs = [str(d.get("name", "")) for d in devices if d.get("max_input_channels", 0) > 0]
    if not inputs:
        return Check("Audio devices", WARN, "no input devices found", "Connect a microphone")
    loopbacks = [name for name in inputs if any(k in name.lower() for k in LOOPBACK_NAMES)]
    if loopbacks:
        return Check(
            "Audio devices",
            OK,
            f"{len(inputs)} inputs; computer audio: {', '.join(loopbacks)}",
        )
    return Check(
        "Audio devices",
        WARN,
        f"{len(inputs)} inputs, but none carries computer audio, so live mode hears only your mic",
        _loopback_hint(),
    )


def check_speakers() -> Check:
    from transcribe.diarization import check_ready

    try:
        check_ready()
    except TranscriptionError as error:
        first_line = str(error).splitlines()[0]
        return Check(
            "Local speaker labels",
            WARN,
            f"not ready: {first_line} (only needed for --speakers with Whisper)",
            "See 'Local speaker labels' in the README",
        )
    return Check("Local speaker labels", OK, "pyannote and HF_TOKEN found")


def check_yt_dlp(today: date | None = None) -> Check:
    try:
        from yt_dlp.version import __version__ as version
    except ImportError:
        return Check("yt-dlp", WARN, "not installed: video links will not work")
    try:
        released = date(*(int(part) for part in version.split(".")[:3]))
    except (TypeError, ValueError):
        return Check("yt-dlp", OK, version)
    age = ((today or date.today()) - released).days
    if age > YT_DLP_MAX_AGE_DAYS:
        return Check(
            "yt-dlp",
            WARN,
            f"{version} is {age} days old; YouTube links may fail",
            "Update it: uv tool upgrade audioscribe (or uv lock --upgrade-package yt-dlp)",
        )
    return Check("yt-dlp", OK, version)


def check_notes() -> list[Check]:
    from transcribe.notes.factory import OLLAMA_BASE_URL, get_notes_provider

    checks = []
    notes_dir = _env("NOTES_DIR")
    if notes_dir:
        folder = Path(notes_dir).expanduser()
        if folder.is_dir():
            checks.append(Check("Notes folder", OK, str(folder)))
        else:
            checks.append(
                Check(
                    "Notes folder",
                    FAIL,
                    f"NOTES_DIR points to a missing folder: {folder}",
                    "Create the folder or fix NOTES_DIR",
                )
            )

    name = (_env("NOTES_PROVIDER") or "claude").lower()
    try:
        provider = get_notes_provider()
    except TranscriptionError as error:
        # The user chose a provider, or --notes would use the default one, and it would fail
        status = FAIL if _env("NOTES_PROVIDER") else WARN
        checks.append(Check("Notes (--notes)", status, str(error).splitlines()[0]))
        return checks

    if name in ("claude", "anthropic"):
        if _env("NOTES_API_KEY") or _env("ANTHROPIC_API_KEY"):
            checks.append(Check("Notes (--notes)", OK, provider.label))
        else:
            checks.append(
                Check(
                    "Notes (--notes)",
                    WARN,
                    f"{provider.label}: no ANTHROPIC_API_KEY found (fine if you use an 'ant auth login' profile)",
                    "Set ANTHROPIC_API_KEY, or choose another NOTES_PROVIDER (see the README)",
                )
            )
        return checks

    try:
        import openai  # noqa: F401
    except ImportError:
        checks.append(
            Check(
                "Notes (--notes)",
                FAIL,
                f"NOTES_PROVIDER={name} needs the 'openai' package",
                "Install it: uv tool install 'audioscribe[openai]' (or uv sync --extra openai)",
            )
        )
        return checks

    if name == "ollama":
        checks.append(check_ollama(_env("NOTES_BASE_URL") or OLLAMA_BASE_URL, _env("NOTES_MODEL")))
    else:
        checks.append(Check("Notes (--notes)", OK, provider.label))
    return checks


def check_ollama(base_url: str, model: str | None) -> Check:
    # Ollama's own API lists the models it has; the OpenAI-style URL ends in /v1
    tags_url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=3) as response:
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return Check(
            "Notes (--notes)",
            FAIL,
            f"Ollama is not answering at {tags_url}",
            "Start it with a bigger context: OLLAMA_CONTEXT_LENGTH=32768 ollama serve",
        )

    names = {m.get("name", "") for m in data.get("models", [])}
    # 'qwen2.5:7b' and 'qwen2.5' (meaning ':latest') both name a pulled model
    if model and model not in names and f"{model}:latest" not in names:
        return Check(
            "Notes (--notes)",
            FAIL,
            f"Ollama has no model '{model}'",
            f"Run: ollama pull {model}",
        )
    # The server's context size can't be read from here, so always say how to raise it
    return Check(
        "Notes (--notes)",
        OK,
        f"Ollama ({model}). For meetings over ~15 min, start it with "
        "OLLAMA_CONTEXT_LENGTH=32768 ollama serve",
    )


def _env(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


def _install_hint(package: str) -> str:
    if sys.platform == "darwin":
        return f"brew install {package}"
    if sys.platform == "win32":
        return f"winget install {package}"
    return f"sudo apt install {package}"


def _loopback_hint() -> str:
    if sys.platform == "darwin":
        return "Install BlackHole (brew install blackhole-2ch), then use --device BlackHole"
    if sys.platform == "win32":
        return "Enable 'Stereo Mix' in Sound settings > Recording, then use --device 'Stereo Mix'"
    # PortAudio usually shows one 'pulse' device; pavucontrol picks what it records
    return "Use --device pulse, then pick 'Monitor of ...' for it in pavucontrol > Recording"
