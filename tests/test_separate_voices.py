"""Rules for recovering text out of an overlap - separator and ASR are faked.

These pin the three decisions that cost the most if they are wrong: an overlap
too short is NOT separated, recovered text never merges into the turns, and a
role stays empty when the voice embeddings are too close to tell apart (which
is exactly the synthetic-recording case, where guessing blames the wrong side).
"""

import numpy as np

from monosplit.separate_voices import MIN_SEPARATE_MS, assign_roles, recover_overlap_text


class FakeSeparator:
    """Returns prepared streams and counts how often it was asked to work."""

    rate = 16_000

    def __init__(self, streams: list[np.ndarray] | None = None) -> None:
        self.calls = 0
        self._streams = streams

    def streams(self, window: np.ndarray) -> list[np.ndarray]:
        self.calls += 1
        if self._streams is not None:
            return self._streams
        half = len(window) // 2 or 1
        return [np.full(half, 0.5, dtype="float32"), np.full(half, 0.4, dtype="float32")]


def mixture(seconds: float = 6.0) -> np.ndarray:
    return np.full(int(seconds * 16_000), 0.2, dtype="float32")


def test_overlap_below_the_threshold_never_reaches_the_separator() -> None:
    separator = FakeSeparator()

    out = recover_overlap_text(
        [{"start_ms": 2_000, "duration_ms": MIN_SEPARATE_MS - 1}],
        mixture(),
        separator=separator,
        transcribe=lambda track: "must never be called",
    )

    assert separator.calls == 0
    assert out[0]["recovered"] is None
    # Original keys survive: recovery adds information, it must not drop the
    # timestamp or the barge-in direction the earlier layers measured.
    assert out[0]["start_ms"] == 2_000


def test_long_enough_overlap_returns_text_from_both_streams() -> None:
    texts = iter(["Read the number back.", "Zero nine zero three."])

    out = recover_overlap_text(
        [{"start_ms": 2_000, "duration_ms": 900}],
        mixture(),
        separator=FakeSeparator(),
        transcribe=lambda track: next(texts),
    )

    assert [line["text"] for line in out[0]["recovered"]] == [
        "Read the number back.",
        "Zero nine zero three.",
    ]
    # No enrollment samples: the words still count, the labels do not.
    assert [line["role"] for line in out[0]["recovered"]] == [None, None]


def test_clear_embedding_margin_names_each_stream() -> None:
    caller = np.array([1.0, 0.0], dtype="float32")
    agent = np.array([0.0, 1.0], dtype="float32")
    streams = [np.full(8_000, 0.9, dtype="float32"), np.full(8_000, 0.3, dtype="float32")]

    out = recover_overlap_text(
        [{"start_ms": 1_000, "duration_ms": 800}],
        mixture(),
        separator=FakeSeparator(streams),
        transcribe=lambda track: "anything",
        profiles={"caller": caller, "agent": agent},
        embed=lambda track: caller if float(np.abs(track).max()) > 0.5 else agent,
    )

    assert [line["role"] for line in out[0]["recovered"]] == ["caller", "agent"]


def test_one_voice_reading_both_parts_leaves_the_roles_empty() -> None:
    same = np.array([1.0, 0.0], dtype="float32")

    out = recover_overlap_text(
        [{"start_ms": 1_000, "duration_ms": 800}],
        mixture(),
        separator=FakeSeparator(),
        transcribe=lambda track: "anything",
        profiles={"caller": same, "agent": same},
        embed=lambda track: same,
    )

    # Both pairings score identically, so neither may be picked.
    assert [line["role"] for line in out[0]["recovered"]] == [None, None]


def test_roles_are_assigned_as_a_pairing_not_per_stream() -> None:
    """Both streams may score highest against the same role - and must not both
    get it: the separator already guarantees they are two different people."""
    caller = np.array([1.0, 0.0], dtype="float32")
    agent = np.array([0.6, 0.8], dtype="float32")
    leaning_caller = np.array([0.99, 0.14], dtype="float32")
    vectors = iter([caller, leaning_caller])
    recovered = [{"role": None, "text": "a", "_track": 1}, {"role": None, "text": "b", "_track": 2}]

    assign_roles(recovered, {"caller": caller, "agent": agent}, lambda _track: next(vectors))

    assert [item["role"] for item in recovered] == ["caller", "agent"]


def test_silent_and_empty_streams_are_dropped() -> None:
    streams = [np.zeros(8_000, dtype="float32"), np.full(8_000, 0.4, dtype="float32")]

    out = recover_overlap_text(
        [{"start_ms": 1_000, "duration_ms": 800}],
        mixture(),
        separator=FakeSeparator(streams),
        transcribe=lambda track: "  Only one side spoke.  ",
    )

    assert out[0]["recovered"] == [{"role": None, "text": "Only one side spoke."}]


def test_a_broken_separator_costs_words_not_the_whole_call() -> None:
    class Broken(FakeSeparator):
        def streams(self, window: np.ndarray) -> list[np.ndarray]:
            raise RuntimeError("model is broken")

    out = recover_overlap_text(
        [{"start_ms": 1_000, "duration_ms": 800}],
        mixture(),
        separator=Broken(),
        transcribe=lambda track: "never reached",
    )

    assert out[0]["recovered"] is None
    assert out[0]["duration_ms"] == 800
