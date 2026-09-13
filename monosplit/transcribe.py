"""Transcribe with faster-whisper, taking WORD-level timestamps.

Whisper's segment timestamps cannot be used to cut turns: Whisper cuts on
grammar, not on when people stop speaking. A recording where the agent speaks
twice 2.5 seconds apart still comes back as ONE segment — take the segment as
the turn and that turn swallows the whole silence, and the response latency of
the following turn disappears. With per-word timestamps the 2.4 second gap
shows up clearly.

The whole module is OPTIONAL: without faster-whisper installed the speaker
splitting path still runs, the turns just carry no text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TranscriberUnavailable(RuntimeError):
    """faster-whisper is not installed."""


@dataclass(frozen=True)
class Word:
    start_ms: int
    end_ms: int
    text: str


class Transcriber:
    """Wrapper around faster-whisper. Model loaded once, reused for every file."""

    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - package missing in the environment
            raise TranscriberUnavailable("faster-whisper is not installed") from exc
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def words(self, path: Path, language: str = "vi", vad_filter: bool = True) -> list[Word]:
        # ``vad_filter=False`` for already separated clips at an overlap: those
        # clips are short and known to contain speech, and VAD on a short clip
        # often cuts the first word off — measured on the 30-case suite, turning
        # VAD off for separated clips gives more accurate text.
        segments, _info = self._model.transcribe(
            str(path), language=language, word_timestamps=True, vad_filter=vad_filter
        )
        out: list[Word] = []
        for segment in segments:
            words: list[Any] = list(getattr(segment, "words", None) or [])
            if not words:
                # A segment with no word timestamps (Whisper drops them now and
                # then) still has to be kept: losing the text is worse than a
                # coarse timestamp.
                text = segment.text.strip()
                if text:
                    out.append(
                        Word(max(0, round(segment.start * 1000)), max(0, round(segment.end * 1000)), text)
                    )
                continue
            for word in words:
                text = word.word.strip()
                if text:
                    out.append(
                        Word(max(0, round(word.start * 1000)), max(0, round(word.end * 1000)), text)
                    )
        return out


def words_per_piece(
    words: list[Word], pieces: list[tuple[int, int, int]], pad_ms: int = 400
) -> list[str]:
    """Hand out the text per piece — every word belongs to EXACTLY ONE piece.

    The old approach (`text_in_range`) let each piece widen both ends by
    ``pad_ms`` and then scan independently, so a word near a boundary went into
    BOTH pieces. The consequence is not just duplicated text: one side's turn
    carries the sentence the other side just said, and whoever reads the
    transcript afterwards believes this person spoke the other one's line.

    Rule: a word belongs to the piece it OVERLAPS most. Only a word that
    overlaps no piece (ASR timestamp drift, or it falls into silence) goes to
    the nearest piece, and only while still within ``pad_ms`` — further away
    than that it is not the speech of any turn.
    """
    buckets: list[list[str]] = [[] for _ in pieces]
    for word in words:
        best, best_overlap = -1, 0
        for index, (start, end, _cluster) in enumerate(pieces):
            overlap = min(end, word.end_ms) - max(start, word.start_ms)
            if overlap > best_overlap:
                best, best_overlap = index, overlap
        if best < 0:
            gap, index = min(
                (max(start - word.end_ms, word.start_ms - end, 0), index)
                for index, (start, end, _c) in enumerate(pieces)
            )
            if gap > pad_ms:
                continue
            best = index
        buckets[best].append(word.text)
    return [" ".join(bucket).strip() for bucket in buckets]
