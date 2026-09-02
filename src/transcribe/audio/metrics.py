import math
import struct
from typing import Tuple


def calculate_rms_and_peak(pcm16_bytes: bytes) -> Tuple[float, int]:
    """
    Calculate the Root Mean Square (RMS) and Peak sample value for 16-bit signed PCM audio.
    """
    if not pcm16_bytes or len(pcm16_bytes) < 2:
        return 0.0, 0

    num_samples = len(pcm16_bytes) // 2
    # Unpack 16-bit signed integers (little-endian)
    samples = struct.unpack(f"<{num_samples}h", pcm16_bytes[: num_samples * 2])

    sum_sq = sum(s * s for s in samples)
    peak = max(abs(s) for s in samples)
    rms = math.sqrt(sum_sq / num_samples)
    return rms, peak


def calculate_dbfs(pcm16_bytes: bytes) -> float:
    """
    Calculate loudness in decibels relative to full scale (dBFS).
    Returns value between -96.0 dBFS (silence) and 0.0 dBFS (max loudness).
    """
    rms, _ = calculate_rms_and_peak(pcm16_bytes)
    if rms <= 0:
        return -96.0

    # 32767 is max amplitude for signed 16-bit PCM
    db = 20.0 * math.log10(rms / 32768.0)
    return max(-96.0, min(0.0, db))


def assess_audio_level(pcm16_bytes: bytes) -> Tuple[str, str, str]:
    """
    Assess audio quality status.
    Returns: (category, color, message)
      - category: "CLIPPING", "GOOD", "LOW", "SILENT"
      - color: rich style color string
      - message: human-readable diagnostics
    """
    rms, peak = calculate_rms_and_peak(pcm16_bytes)
    if rms <= 0:
        return "SILENT", "bold red", "Dead silence or muted input"

    db = calculate_dbfs(pcm16_bytes)

    if peak >= 32700:
        return "CLIPPING", "bold red", "Audio is clipping / distorted (too loud)"
    if db > -10.0:
        return "HOT", "bold yellow", "Very loud input"
    if db >= -30.0:
        return "GOOD", "bold green", "Optimal speech level"
    if db >= -48.0:
        return "LOW", "yellow", "Low volume (speak closer to mic)"
    return "SILENT", "bold red", "Extremely quiet or silent (check mic input volume)"


def render_vu_meter(dbfs: float, bar_length: int = 16) -> str:
    """
    Render a text-based VU meter bar from -60 dBFS to 0 dBFS.
    Example: '[████████░░░░░░░░] -18.5 dB'
    """
    # Map range [-60, 0] to [0, bar_length]
    clamped_db = max(-60.0, min(0.0, dbfs))
    ratio = (clamped_db + 60.0) / 60.0
    filled = int(round(ratio * bar_length))
    filled = max(0, min(bar_length, filled))
    empty = bar_length - filled

    bar = "█" * filled + "░" * empty
    return f"[{bar}] {dbfs:.1f} dB"
