"""Listening again to a turn the full read left empty.

Whisper reading the WHOLE recording folds short backchannels — "ừm", "à", "dạ" — into the
surrounding silence, and a turn with no text is dropped further down the pipeline. The agent then
reads as silent for the entire caller turn, and every turn-taking measure loses its evidence.
Measured on a 192 second call (2026-09-14): two agent "Ờm" at 106.9 s and 108.9 s produced no word
from the full read, and both came back once that stretch was cut out and read on its own.
"""

from __future__ import annotations

from pathlib import Path

from monosplit import audio
from monosplit.pipeline import Turn, _rescue_silent_turns


class _Transcriber:
    """Whisper stand-in: answers per clip file name, records what it was asked to read."""

    def __init__(self, replies: dict[str, tuple[str, float]]) -> None:
        self._replies = replies
        self.asked: list[str] = []

    def listen_again(self, path: Path, language: str = "vi") -> tuple[str, float]:
        self.asked.append(Path(path).name)
        return self._replies.get(Path(path).name, ("", 1.0))


def _pcm(ms: int) -> bytes:
    return b"\x01\x02" * (ms * audio.PCM_BYTES_PER_MS // 2)


def test_an_empty_turn_is_read_again_on_its_own(tmp_path: Path):
    turns = [
        Turn("caller", 100_000, 105_000, "already has text"),
        Turn("agent", 106_960, 108_330, ""),
    ]
    transcriber = _Transcriber({"again_106960_108330.wav": ("Ờm...", 0.33)})

    out = _rescue_silent_turns(turns, _pcm(110_000), transcriber, tmp_path)

    assert [turn.text for turn in out] == ["already has text", "Ờm..."]
    # A turn that already has text keeps the full read: that read had the context on both sides.
    assert transcriber.asked == ["again_106960_108330.wav"]


def test_a_sentence_too_long_for_the_clip_is_refused(tmp_path: Path):
    """Read a short stretch out of context and Whisper returns a line memorised from its training
    data — a 270 ms tail came back as a full "subscribe to the channel" sentence. Thirteen words do
    not fit in 270 ms."""
    turns = [Turn("agent", 190_480, 190_750, "")]
    transcriber = _Transcriber(
        {
            "again_190480_190750.wav": (
                "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn",
                0.2,
            )
        }
    )

    assert _rescue_silent_turns(turns, _pcm(191_000), transcriber, tmp_path)[0].text == ""


def test_a_clip_whisper_itself_calls_silence_stays_empty(tmp_path: Path):
    turns = [Turn("caller", 1_930, 2_410, "")]
    transcriber = _Transcriber({"again_1930_2410.wav": ("khịt", 0.92)})

    assert _rescue_silent_turns(turns, _pcm(3_000), transcriber, tmp_path)[0].text == ""
