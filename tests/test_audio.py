"""Câu hỏi «tệp này có hai luồng tiếng thật không» phải trả lời đúng.

Trả lời sai theo hướng nào cũng hỏng: tưởng stereo giả là thật thì mỗi câu bị
gán cho cả hai vai cùng mốc thời gian; tưởng stereo thật là giả thì vứt đi
thông tin chắc chắn nhất có trong tệp để đi đoán.
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


def test_mono_nhan_doi_thanh_stereo_la_mot_luong():
    wave = tone(180, 500)
    assert one_stream_only(wave, wave) is True


def test_lech_nho_do_nen_van_la_mot_luong():
    """Nén mất mát làm hai kênh lệch chút xíu — vẫn là một luồng."""
    wave = np.frombuffer(tone(180, 500), dtype=np.int16).astype(np.int32)
    noisy = (wave + np.random.default_rng(7).integers(-20, 20, wave.size)).astype(np.int16)
    assert one_stream_only(wave.astype(np.int16).tobytes(), noisy.tobytes()) is True


def test_mot_ben_cam_la_mot_luong():
    wave = tone(180, 500)
    assert one_stream_only(wave, pcm([0] * (len(wave) // 2))) is True


def test_hai_kenh_khac_nhau_la_hai_luong():
    assert one_stream_only(tone(180, 500), tone(320, 500, phase=1.1)) is False


def test_ca_hai_kenh_cam_khong_phai_chuyen_tach_vai():
    """Tệp câm là lỗi «không có tiếng nói», không phải lỗi «một luồng»."""
    silence = pcm([0] * 8000)
    assert one_stream_only(silence, silence) is False


def test_chenh_lech_do_duoc_de_in_bao_cao():
    wave = tone(180, 300)
    assert channel_difference(wave, wave) == 0.0
    assert channel_difference(wave, tone(320, 300, phase=1.1)) > 0.5


def test_cat_doan_theo_moc_thoi_gian():
    """32 byte mỗi mili giây: cắt theo mốc là cắt byte, không lệch một mẫu."""
    wave = tone(180, 1000)
    assert len(slice_pcm(wave, 100, 200)) == 100 * 32


def test_dang_song_co_mot_diem_moi_khung():
    assert len(peak_levels(tone(180, 200), frame_ms=20)) == 10
