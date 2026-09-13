"""Rules for rebuilding one channel per role - separator and embeddings faked.

These pin the three decisions that cost the most if they are wrong: an overlap
too short is NOT separated, nothing is built when the two voices cannot be told
apart (swapping the channels swaps both sides' words), and the joint between
mixture and separated stream crosses over instead of cutting hard.
"""

import numpy as np

from monosplit.separate_voices import FADE_MS, MIN_SEPARATE_MS, build_role_tracks

CALLER = np.array([1.0, 0.0], dtype="float32")
AGENT = np.array([0.0, 1.0], dtype="float32")


class FakeSeparator:
    """Returns prepared streams and counts how often it was asked to work."""

    rate = 16_000

    def __init__(self, streams: list[np.ndarray] | None = None) -> None:
        self.calls = 0
        self._streams = streams

    def streams(self, window: np.ndarray) -> list[np.ndarray]:
        self.calls += 1
        if self._streams is not None:
            return [stream[: len(window)] for stream in self._streams]
        return [
            np.full(len(window), 0.5, dtype="float32"),
            _wobble(0.4, len(window)),
        ]


def mixture(seconds: float = 8.0) -> np.ndarray:
    return np.full(int(seconds * 16_000), 0.2, dtype="float32")


def _wobble(value: float, length: int) -> np.ndarray:
    """A stream that alternates sign - a different SHAPE from a flat one."""
    track = np.full(length, value, dtype="float32")
    track[1::2] *= -1.0
    return track


def by_shape(track: np.ndarray) -> np.ndarray:
    """Fake embedding: tells streams apart by shape, not by amplitude.

    The code matches a separated stream's level before embedding it, so a fake
    that keys on amplitude would see both streams as identical.
    """
    return AGENT if float(track.min()) < 0.0 else CALLER


def test_overlap_below_the_threshold_never_reaches_the_separator() -> None:
    separator = FakeSeparator()

    built = build_role_tracks(
        [{"start_ms": 2_000, "duration_ms": MIN_SEPARATE_MS - 1}],
        mixture(),
        separator=separator,
        profiles={"caller": CALLER, "agent": AGENT},
        embed=by_shape,
    )

    assert separator.calls == 0
    # Nothing separable means no channels at all: the transcript keeps the
    # mixture instead of passing through a second model for no gain.
    assert built is None


def test_long_enough_overlap_builds_one_channel_per_role() -> None:
    built = build_role_tracks(
        [{"start_ms": 3_000, "duration_ms": 800}],
        mixture(),
        separator=FakeSeparator(),
        profiles={"caller": CALLER, "agent": AGENT},
        embed=by_shape,
    )

    assert built is not None
    tracks, marked = built
    assert sorted(tracks) == ["agent", "caller"]
    assert marked[0]["separated"] is True
    # Each channel is as long as the original, so the step that follows can read
    # it like one channel of a stereo pair and turn timestamps still apply.
    assert all(len(track) == len(mixture()) for track in tracks.values())
    # Outside the overlap it is the untouched mixture.
    assert tracks["caller"][0] == np.float32(0.2)


def test_one_voice_reading_both_parts_builds_nothing() -> None:
    same = np.array([1.0, 0.0], dtype="float32")

    built = build_role_tracks(
        [{"start_ms": 3_000, "duration_ms": 800}],
        mixture(),
        separator=FakeSeparator(),
        profiles={"caller": same, "agent": same},
        embed=lambda _track: same,
    )

    # Both pairings score the same. Picking one anyway would swap both sides'
    # words across the whole overlap - worse than leaving the mixture alone.
    assert built is None


def test_the_joint_crosses_over_instead_of_cutting_hard() -> None:
    # An alternating stream still differs from the mixture after level
    # matching, so the joint is measurable. A flat stream would be scaled to
    # exactly the mixture's 0.2 and show nothing.
    patch = _wobble(1.0, 16_000 * 8)
    built = build_role_tracks(
        [{"start_ms": 3_000, "duration_ms": 800}],
        mixture(),
        separator=FakeSeparator([np.full(16_000 * 8, 0.9, dtype="float32"), patch]),
        profiles={"caller": CALLER, "agent": AGENT},
        embed=by_shape,
    )

    assert built is not None
    tracks, _marked = built
    track = tracks["agent"]  # the alternating stream is assigned to the agent
    # The window starts at 3000 - PAD_MS = 1500 ms. At the very edge the signal
    # must still be close to the mixture; a hard cut leaves a step, and ASR
    # reads that step as a stray syllable.
    edge = 1_500 * 16
    assert abs(float(track[edge]) - 0.2) < 0.05
    # Past the crossfade it is the separated stream: only that carries negatives.
    assert float(track[edge + FADE_MS * 16 + 101]) < 0.0
    # Before the window, the mixture is untouched.
    assert float(track[edge - 10]) == np.float32(0.2)


def test_a_broken_separator_costs_words_not_the_whole_call() -> None:
    class Broken(FakeSeparator):
        def streams(self, window: np.ndarray) -> list[np.ndarray]:
            raise RuntimeError("model is broken")

    built = build_role_tracks(
        [{"start_ms": 3_000, "duration_ms": 800}],
        mixture(),
        separator=Broken(),
        profiles={"caller": CALLER, "agent": AGENT},
        embed=by_shape,
    )

    assert built is None


def test_a_silent_stream_means_only_one_speaker_was_found() -> None:
    silent = np.zeros(16_000 * 8, dtype="float32")

    built = build_role_tracks(
        [{"start_ms": 3_000, "duration_ms": 800}],
        mixture(),
        separator=FakeSeparator([silent, _wobble(0.4, 16_000 * 8)]),
        profiles={"caller": CALLER, "agent": AGENT},
        embed=by_shape,
    )

    assert built is None
