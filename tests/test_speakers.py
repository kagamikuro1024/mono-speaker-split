"""Hai quyết định KHÔNG cần mô hình: cắt lượt, và chọn cụm nào là agent.

Đây là phần dễ sai nhất và cũng là phần chạy được không cần ONNX, nên nó phải
có bài kiểm riêng — hỏng ở đây thì mọi bản gỡ băng đều đảo vai một cách im lặng.
"""

from __future__ import annotations

from monosplit.speakers import co_bang_chung_hai_vai, pick_agent_cluster, split_runs


def test_mot_quang_lien_mach_bi_cat_tai_cho_doi_nguoi():
    """Hai bên nói đè nhau: VAD không thấy khoảng lặng, cụm mới là chỗ cắt."""
    pieces = split_runs(
        [{"start_ms": 0, "end_ms": 4000}],
        [(0, 1500, 0), (1500, 4000, 1)],
    )
    assert pieces == [(0, 1500, 0), (1500, 4000, 1)]


def test_hai_quang_cung_cum_van_la_hai_luot():
    """Cách nhau khoảng lặng thì là hai lượt — nhập lại là nuốt mất một lượt."""
    pieces = split_runs(
        [{"start_ms": 0, "end_ms": 3000}, {"start_ms": 9000, "end_ms": 12000}],
        [(0, 12000, 0)],
    )
    assert pieces == [(0, 3000, 0), (9000, 12000, 0)]


def test_ben_doc_lai_gia_tri_la_agent():
    """Khách đưa số, agent nhắc lại để xác nhận — không bao giờ ngược lại."""
    cluster, reason = pick_agent_cluster(
        [(0, "gọi giúp tôi số 0903 456 789"), (1, "số 0903 456 789 đúng không ạ")],
        ["0903 456 789"],
    )
    assert cluster == 1
    assert "sau" in reason


def test_cau_cua_mieng_cua_khach_khong_bien_khach_thanh_agent():
    """Đếm một phía thì 'cảm ơn' của khách cũng thành bằng chứng buộc tội."""
    cluster, _reason = pick_agent_cluster(
        [(0, "kiểm tra giúp tôi, cảm ơn"), (1, "dạ vâng em kiểm tra ngay ạ")],
        [],
    )
    assert cluster == 1


def test_khong_co_can_cu_thi_noi_thang_la_suy_doan():
    cluster, reason = pick_agent_cluster([(0, "ừ"), (1, "à")], [])
    assert cluster == 1
    assert "yếu" in reason


def test_mot_nguoi_noi_khong_duoc_chia_thanh_hai_vai():
    """Một giọng + lời không cho thấy hai vai = từ chối, không bịa người thứ hai."""
    assert co_bang_chung_hai_vai([(0, "hôm nay trời đẹp"), (1, "tôi đi làm lúc bảy giờ")], []) is False


def test_hai_nhom_noi_nguoc_nhau_la_hai_vai():
    assert co_bang_chung_hai_vai([(0, "giúp tôi đặt lịch"), (1, "dạ em xác nhận ạ")], []) is True


def test_cung_mot_gia_tri_duoc_ca_hai_ben_noc_la_hai_vai():
    """Có người đưa tin và có người nhắc lại — đó là hội thoại, không phải độc thoại."""
    assert (
        co_bang_chung_hai_vai(
            [(0, "địa chỉ 18 Nguyễn Huệ"), (1, "18 Nguyễn Huệ phải không")],
            ["18 Nguyễn Huệ"],
        )
        is True
    )
