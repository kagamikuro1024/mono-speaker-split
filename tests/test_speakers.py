"""Two decisions that need NO model: cutting turns, and picking which cluster is the agent.

This is the easiest part to get wrong and also the part that runs without ONNX, so it needs its
own tests — broken here, every transcript swaps the roles silently.
"""

from __future__ import annotations

from monosplit.speakers import (
    co_bang_chung_hai_vai,
    overlap_spans,
    pick_agent_cluster,
    split_runs,
)


def test_continuous_run_is_cut_where_the_speaker_changes():
    """Both sides speak over each other: VAD sees no silence, the new cluster is the cut point."""
    pieces = split_runs(
        [{"start_ms": 0, "end_ms": 4000}],
        [(0, 1500, 0), (1500, 4000, 1)],
    )
    assert pieces == [(0, 1500, 0), (1500, 4000, 1)]


def test_two_runs_of_the_same_cluster_stay_two_turns():
    """Separated by silence means two turns — merging them swallows one turn."""
    pieces = split_runs(
        [{"start_ms": 0, "end_ms": 3000}, {"start_ms": 9000, "end_ms": 12000}],
        [(0, 12000, 0)],
    )
    assert pieces == [(0, 3000, 0), (9000, 12000, 0)]


def test_the_side_that_reads_a_value_back_is_the_agent():
    """The caller gives the number, the agent repeats it to confirm — never the other way round."""
    cluster, reason = pick_agent_cluster(
        [(0, "gọi giúp tôi số 0903 456 789"), (1, "số 0903 456 789 đúng không ạ")],
        ["0903 456 789"],
    )
    assert cluster == 1
    assert "after" in reason


def test_caller_stock_phrases_do_not_turn_the_caller_into_the_agent():
    """Counting one side only turns the caller's "thank you" into incriminating evidence."""
    cluster, _reason = pick_agent_cluster(
        [(0, "kiểm tra giúp tôi, cảm ơn"), (1, "dạ vâng em kiểm tra ngay ạ")],
        [],
    )
    assert cluster == 1


def test_no_evidence_is_reported_as_an_inference():
    cluster, reason = pick_agent_cluster([(0, "ừ"), (1, "à")], [])
    assert cluster == 1
    assert "weak" in reason


def test_a_single_speaker_is_not_split_into_two_roles():
    """One voice + wording that shows no second role = refuse, do not invent a second person."""
    assert co_bang_chung_hai_vai([(0, "hôm nay trời đẹp"), (1, "tôi đi làm lúc bảy giờ")], []) is False


def test_two_groups_with_opposite_stock_phrases_are_two_roles():
    assert co_bang_chung_hai_vai([(0, "giúp tôi đặt lịch"), (1, "dạ em xác nhận ạ")], []) is True


def test_the_same_value_spoken_by_both_sides_is_two_roles():
    """Someone gives the information and someone repeats it — that is a dialogue, not a monologue."""
    assert (
        co_bang_chung_hai_vai(
            [(0, "địa chỉ 18 Nguyễn Huệ"), (1, "18 Nguyễn Huệ phải không")],
            ["18 Nguyễn Huệ"],
        )
        is True
    )


def test_overlap_reports_both_the_mark_and_the_barge_in_direction():
    """``split_runs`` cuts clusters into adjacent pieces, which erases every trace of overlap.

    Reading straight from the clusters still shows it: cluster 1 comes in at 4.42 s while
    cluster 0 is still speaking until 5 s, and cluster 0 is the side that leaves the overlap
    first.
    """
    assert overlap_spans([(0, 5_000, 0), (4_420, 8_000, 1)]) == [
        {"start_ms": 4_420, "duration_ms": 580, "cum_chen": 1, "cum_nhuong": 0}
    ]


def test_two_runs_of_one_speaker_are_not_overlapping_speech():
    assert overlap_spans([(0, 5_000, 0), (4_000, 8_000, 0)]) == []


def test_overlap_shorter_than_the_boundary_blur_is_not_counted():
    """An 80 ms overlap is smaller than the model's boundary blur (receptive field 62 ms):
    counting it is counting your own error."""
    assert overlap_spans([(0, 5_000, 0), (4_920, 8_000, 1)]) == []
