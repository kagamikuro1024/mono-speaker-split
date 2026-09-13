"""Handing text out to each turn — the part that runs without Whisper.

One word landing in two turns is the most dangerous silent failure of the whole pipeline: the
transcript still looks correct, except one side's turn carries the sentence the other side just
said.
"""

from __future__ import annotations

from monosplit.transcribe import Word, words_per_piece


def test_a_word_belongs_to_exactly_one_turn():
    """A word on the boundary belongs to the turn it OVERLAPS most, not to both."""
    words = [
        Word(100, 900, "call"),
        Word(4_900, 5_200, "okay"),
        Word(5_400, 6_000, "num-0903"),
    ]
    pieces = [(0, 5_262, 0), (5_262, 8_160, 1)]

    assert words_per_piece(words, pieces) == ["call okay", "num-0903"]


def test_word_in_a_gap_goes_to_the_nearest_turn():
    """ASR marks drift by a few hundred milliseconds, so a rescue path is still required — but
    it rescues to EXACTLY ONE side, the closer one."""
    words = [Word(5_300, 5_500, "yes")]

    assert words_per_piece(words, [(0, 5_000, 0), (6_000, 8_000, 1)]) == ["yes", ""]


def test_word_too_far_from_every_turn_belongs_to_nobody():
    """Dropping it beats assigning it blindly: more than 400 ms away from every turn means
    background noise, a keystroke, or a broken ASR mark — assigning it to any turn puts words in
    someone's mouth."""
    words = [Word(20_000, 20_400, "buzz")]

    assert words_per_piece(words, [(0, 5_000, 0), (6_000, 8_000, 1)]) == ["", ""]
