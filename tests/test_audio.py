"""The question "does this file really carry two streams of speech" must be answered right.

A wrong answer in either direction breaks the run: treating fake stereo as real assigns every
sentence to both roles at the same timestamp; treating real stereo as fake throws away the most
reliable information in the file and goes guessing instead.
"""

from __future__ import annotations

import numpy as np

from monosplit.audio import channel_difference, one_stream_only, peak_levels, slice_pcm

SAMPLE_RATE = 16_000


def pcm(values: list[int]) -> bytes:
    return np.array(values, dtype=np.int16).tobytes()


def tone(freq: float, ms: int, amplitude: int = 8000, phase: float = 0.0) -> bytes:
    t = np.arange(int(SAMPLE_RATE * ms / 1000), dtype=np.float32) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t + phase) * amplitude).astype(np.int16).tobytes()


def test_mono_duplicated_into_stereo_is_one_stream():
    wave = tone(180, 500)
    assert one_stream_only(wave, wave) is True


def test_small_codec_drift_is_still_one_stream():
    """Lossy compression makes the two channels differ slightly — still one stream."""
    wave = np.frombuffer(tone(180, 500), dtype=np.int16).astype(np.int32)
    noisy = (wave + np.random.default_rng(7).integers(-20, 20, wave.size)).astype(np.int16)
    assert one_stream_only(wave.astype(np.int16).tobytes(), noisy.tobytes()) is True


def test_one_muted_channel_is_one_stream():
    wave = tone(180, 500)
    assert one_stream_only(wave, pcm([0] * (len(wave) // 2))) is True


def test_two_different_channels_are_two_streams():
    assert one_stream_only(tone(180, 500), tone(320, 500, phase=1.1)) is False


def test_both_channels_silent_is_not_a_role_split_case():
    """A silent file is a "no speech" error, not a "one stream" error."""
    silence = pcm([0] * 8000)
    assert one_stream_only(silence, silence) is False


def test_channel_difference_is_measured_for_the_report():
    wave = tone(180, 300)
    assert channel_difference(wave, wave) == 0.0
    assert channel_difference(wave, tone(320, 300, phase=1.1)) > 0.5


def test_slicing_follows_the_millisecond_marks():
    """32 bytes per millisecond: slicing by mark is slicing by byte, not off by one sample."""
    wave = tone(180, 1000)
    assert len(slice_pcm(wave, 100, 200)) == 100 * 32


def test_waveform_has_one_point_per_frame():
    assert len(peak_levels(tone(180, 200), frame_ms=20)) == 10
