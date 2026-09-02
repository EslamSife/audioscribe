import struct

from transcribe.audio.metrics import (
    assess_audio_level,
    calculate_dbfs,
    calculate_rms_and_peak,
    render_vu_meter,
)


def make_pcm16_samples(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}h", *values)


def test_silence_metrics():
    silence = make_pcm16_samples([0, 0, 0, 0])
    rms, peak = calculate_rms_and_peak(silence)
    assert rms == 0.0
    assert peak == 0
    assert calculate_dbfs(silence) == -96.0

    category, color, msg = assess_audio_level(silence)
    assert category == "SILENT"


def test_optimal_speech_level():
    # Moderate speech amplitude around ~6000 (roughly -15 dBFS)
    samples = make_pcm16_samples([5500, -6200, 5800, -6000])
    rms, peak = calculate_rms_and_peak(samples)
    assert rms > 5000
    assert peak == 6200

    db = calculate_dbfs(samples)
    assert -20.0 < db < -10.0

    category, color, msg = assess_audio_level(samples)
    assert category == "GOOD"


def test_clipping_detection():
    # Maxed out 16-bit PCM (>= 32700)
    clipping_samples = make_pcm16_samples([32767, -32767, 32750])
    rms, peak = calculate_rms_and_peak(clipping_samples)
    assert peak == 32767

    category, color, msg = assess_audio_level(clipping_samples)
    assert category == "CLIPPING"


def test_render_vu_meter():
    meter_silence = render_vu_meter(-60.0, bar_length=10)
    assert "░░░░░░░░░░" in meter_silence
    assert "-60.0 dB" in meter_silence

    meter_half = render_vu_meter(-30.0, bar_length=10)
    assert "█████░░░░░" in meter_half
    assert "-30.0 dB" in meter_half

    meter_max = render_vu_meter(0.0, bar_length=10)
    assert "██████████" in meter_max
    assert "0.0 dB" in meter_max
