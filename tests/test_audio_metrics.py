import struct

from transcribe.audio.metrics import calculate_dbfs, calculate_rms_and_peak


def make_pcm16_samples(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}h", *values)


def test_silence_metrics():
    silence = make_pcm16_samples([0, 0, 0, 0])
    rms, peak = calculate_rms_and_peak(silence)
    assert rms == 0.0
    assert peak == 0
    assert calculate_dbfs(silence) == -96.0


def test_speech_level():
    # Moderate speech amplitude around ~6000 (roughly -15 dBFS)
    samples = make_pcm16_samples([5500, -6200, 5800, -6000])
    rms, peak = calculate_rms_and_peak(samples)
    assert rms > 5000
    assert peak == 6200
    assert -20.0 < calculate_dbfs(samples) < -10.0


def test_full_scale_peak():
    _, peak = calculate_rms_and_peak(make_pcm16_samples([32767, -32767, 32750]))
    assert peak == 32767
