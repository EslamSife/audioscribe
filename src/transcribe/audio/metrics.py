import math
import struct


def calculate_rms_and_peak(pcm16_bytes: bytes) -> tuple[float, int]:
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
