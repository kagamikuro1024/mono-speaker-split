"""Chia chữ cho từng lượt — phần chạy được mà không cần Whisper.

Một từ đi vào hai lượt là lỗi im lặng nguy hiểm nhất của cả đường chạy: bản ghi
trông vẫn đúng, chỉ có điều lượt của bên này mang theo câu bên kia vừa nói.
"""

from __future__ import annotations

from monosplit.transcribe import Word, words_per_piece


def test_mot_tu_chi_thuoc_dung_mot_luot():
    """Từ ở vùng giáp ranh: thuộc lượt nó GIAO nhiều nhất, không thuộc cả hai."""
    words = [
        Word(100, 900, "gọi"),
        Word(4_900, 5_200, "nhé"),
        Word(5_400, 6_000, "số-0903"),
    ]
    pieces = [(0, 5_262, 0), (5_262, 8_160, 1)]

    assert words_per_piece(words, pieces) == ["gọi nhé", "số-0903"]


def test_tu_roi_vao_khoang_lang_ve_luot_gan_nhat():
    """Mốc ASR lệch vài trăm mili giây nên vẫn phải có đường cứu — nhưng cứu về
    ĐÚNG MỘT bên, bên gần hơn."""
    words = [Word(5_300, 5_500, "dạ")]

    assert words_per_piece(words, [(0, 5_000, 0), (6_000, 8_000, 1)]) == ["dạ", ""]


def test_tu_qua_xa_moi_luot_thi_khong_thuoc_ai():
    """Bỏ hẳn còn hơn gán bừa: cách mọi lượt hơn 400 ms thì đó là tiếng nền,
    tiếng gõ, hoặc mốc ASR hỏng — gán vào lượt nào cũng là bịa lời cho người."""
    words = [Word(20_000, 20_400, "rè")]

    assert words_per_piece(words, [(0, 5_000, 0), (6_000, 8_000, 1)]) == ["", ""]
