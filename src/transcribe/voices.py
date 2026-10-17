"""Meeting mode with two devices: label each line "Me" (your microphone) or "Them" (the call).

The two devices are kept apart instead of mixed, so every line knows who said
it and two people talking at once are both heard. The recording is one stereo
WAV: the call on the left channel, your microphone on the right.

Without headphones the microphone also hears the call through the speakers.
Those echo lines repeat what "Them" said at the same moment, so they are dropped.
"""

import re
import tempfile
import wave
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator

from transcribe.transcription.model import Segment, Transcript, TranscriptionOptions
from transcribe.transcription.service import transcribe_file

ME, THEM = "Me", "Them"

# A "Me" line this close in time to a "Them" line may be the same words heard twice
ECHO_WINDOW_MS = 1500
# ...and it is an echo when this share of its words also appear in that line
ECHO_WORD_SHARE = 0.6


def interleave(left: bytes, right: bytes) -> bytes:
    """Join two 16-bit mono chunks into one stereo chunk (left, right, left, right, ...)."""
    import numpy as np

    a = np.frombuffer(left, dtype=np.int16)
    b = np.frombuffer(right, dtype=np.int16)
    size = max(len(a), len(b))
    stereo = np.zeros(size * 2, dtype=np.int16)
    stereo[0 : len(a) * 2 : 2] = a
    stereo[1 : len(b) * 2 : 2] = b
    return stereo.tobytes()


def is_stereo(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnchannels() == 2
    except (wave.Error, EOFError, OSError):
        return False


@contextmanager
def split_channels(path: Path) -> Iterator[tuple[Path, Path]]:
    """Write the left (Them) and right (Me) channels of a stereo WAV to two temporary mono WAVs."""
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    with tempfile.TemporaryDirectory(prefix="transcribe_") as folder:
        paths = (Path(folder) / "them.wav", Path(folder) / "me.wav")
        for channel, channel_path in enumerate(paths):
            with wave.open(str(channel_path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes(frames[channel::2].tobytes())
        yield paths


def is_echo(line: Segment, other: Segment) -> bool:
    """True when `line` repeats most of the words of `other` at about the same time."""
    if line.start_ms > other.end_ms + ECHO_WINDOW_MS:
        return False
    if line.end_ms < other.start_ms - ECHO_WINDOW_MS:
        return False
    words = _words(line.text)
    if not words:
        return False
    heard = set(_words(other.text))
    return sum(word in heard for word in words) / len(words) >= ECHO_WORD_SHARE


def drop_echoes(segments: Iterable[Segment]) -> tuple[Segment, ...]:
    """Drop "Me" lines that are only the microphone hearing the call."""
    segments = tuple(segments)
    others = [s for s in segments if s.speaker != ME]
    return tuple(s for s in segments if s.speaker != ME or not any(is_echo(s, o) for o in others))


def merge(them: Transcript, me: Transcript) -> Transcript:
    """One transcript in time order from the two channels, without echo lines.

    Lines from the call keep a speaker label they already have (e.g. "Speaker 1"
    from --speakers), so several people on the call stay apart.
    """
    lines = [replace(s, speaker=s.speaker or THEM) for s in them.segments]
    lines += [replace(s, speaker=ME) for s in me.segments]
    segments = drop_echoes(sorted(lines, key=lambda s: s.start_ms))
    durations = [d for d in (them.audio_duration_s, me.audio_duration_s) if d is not None]
    return Transcript(
        text=" ".join(s.text for s in segments),
        segments=segments,
        audio_duration_s=max(durations) if durations else None,
        language_code=them.language_code or me.language_code,
        warnings=them.warnings + me.warnings,
    )


def transcribe_two_voices(path: Path, engine: Any, options: TranscriptionOptions) -> Transcript:
    """Transcribe a stereo meeting recording channel by channel, as Them and Me lines."""
    with split_channels(path) as (them_path, me_path):
        them = transcribe_file(them_path, engine, options=options)
        # Only one person speaks into your microphone
        me = transcribe_file(me_path, engine, options=replace(options, speakers=False))
    return merge(them, me)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
