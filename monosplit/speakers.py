"""Label caller / agent on a SINGLE-channel recording.

A two-channel recording needs none of this: the user declares which channel is
who. A single-channel recording — a caller recording on a phone inside a car, a
contact centre exporting mono — has to be inferred, and inferring it wrong
means blaming the agent for the caller's words.

Three layers, in this order, each one fixing what the previous cannot do:

1. **Blind clustering** (``sherpa-onnx``: pyannote segmentation 3.0 + ERes2Net
   voice embeddings, both ONNX) splits the recording into two voice clusters.
   This layer knows *there are two people* and *when the speaker changes*, but
   not who is who, and it often swallows a short turn into the long turn next
   to it.
2. **Pick which cluster is the agent by WORDS, not by voice.** The agent voice
   changes with each user's configuration, so it cannot be pinned down; the
   role, though, shows up in the text: the agent is the side that reads values
   back for confirmation and says the contact centre lines. This is the only
   place where "who is who" is decided.
3. **Re-score every span by voice embedding** taken from those same two
   clusters (a reduced form of Target-Speaker VAD, Medennikov 2020): take the
   longest piece of each cluster as the enrollment sample, then compare cosine
   for every VAD span. This layer pulls back to the right side the short turns
   layer 1 swallowed.

What this CANNOT do and must not pretend to do: two people speaking at once on
one channel leave a single waveform, so overlapping speech / barge-in has to be
reported as "not measurable" rather than as 0.
"""

from __future__ import annotations

import logging
import re
from array import array
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Two turns of the same cluster less than this apart are one turn. Must be the
# same number as ``pipeline.TURN_MERGE_GAP_MS`` — two places splitting turns
# with different numbers turn one recording into two different timelines.
MERGE_GAP_MS = 400

# Minimum cosine margin required to trust the label from layer 3. Below it the
# two voices are too alike (or the span too short): keep the label from layer 1
# and lower the confidence.
MIN_COSINE_MARGIN = 0.10

# A piece shorter than this is not enough for a voice embedding: ERes2Net needs
# roughly one second of speech before the vector is stable.
MIN_ENROLL_MS = 1_000

# When a cluster has no piece reaching one second: still take its longest piece
# from this length up. A short sample gives a weaker vector, but the cosine
# threshold in layer 3 still blocks it — whereas with no sample at all layer 3
# switches off entirely, which loses every chance of fixing layer 1.
MIN_SAMPLE_MS = 400

# If the two furthest-apart pieces still score above this cosine, the recording
# holds only ONE person. Measured on the 30-case acceptance suite: same speaker
# 0.54 to 0.74, different speakers 0.06 to 0.27. 0.40 sits between the two
# ranges; re-measure once there are real in-car recordings.
MAX_SAME_VOICE_COSINE = 0.40

# A piece long enough TO BE COMPARED during the rescue step. Shorter than the
# normal enrollment length because what gets swallowed is exactly the "ừ", "dạ"
# backchannel turns — drop those and there is nothing left to rescue.
MIN_RESCUE_MS = 180

# Stock phrases of each side, in Vietnamese: these strings are recognition data
# for Vietnamese speech, not prose. Used when the scenario declares no value
# that has to be read back — that is, when there is no firmer evidence. Scored
# on the DIFFERENCE between the two sides, not by counting the agent side
# alone: a bare "cảm ơn" says nothing about who is who, "giúp tôi" does.
AGENT_CUES = (
    "tổng đài", "xin nghe", "em xin", "dạ em", "bên em", "quý khách",
    "em hỗ trợ", "em kiểm tra", "em xác nhận", "cảm ơn anh", "cảm ơn chị",
    "dạ", "vâng", " ạ",
)

# The caller is the side ASKING FOR something: their lines address someone who
# will carry it out. Vietnamese cue strings, kept as data.
CALLER_CUES = (
    "giúp tôi", "cho tôi", "tôi muốn", "tôi cần", "em ơi", "a lô",
    "gọi cho tôi", "của tôi",
)


class MonoSplitUnavailable(RuntimeError):
    """Missing model or library: importable but not runnable."""


@dataclass(frozen=True)
class Labelled:
    start_ms: int
    end_ms: int
    speaker: Literal["caller", "agent"]
    # Cosine distance between the two hypotheses. The smaller, the more suspect;
    # the UI reads this number to say "inferred label" instead of staying silent.
    margin: float


def _norm(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.casefold())


def speech_energy_runs(
    pcm: bytes, frame_ms: int, min_speech_ms: int, bytes_per_ms: int
) -> list[dict[str, int]]:
    """Speech runs, with the threshold derived from THIS recording.

    Take the 20th percentile of frame energy as the floor, then multiply by
    three. Pinning an absolute number makes a quietly recorded file come out
    entirely silent, while in a noisy file every silence becomes speech.
    """
    frame_bytes = frame_ms * bytes_per_ms
    frames = [pcm[at : at + frame_bytes] for at in range(0, len(pcm) - frame_bytes + 1, frame_bytes)]
    if not frames:
        return []
    levels = [max(abs(value) for value in array("h", frame)) for frame in frames]
    floor = sorted(levels)[len(levels) // 5]
    threshold = max(floor * 3, 200)
    runs: list[dict[str, int]] = []
    for index, level in enumerate(levels):
        if level < threshold:
            continue
        start, end = index * frame_ms, index * frame_ms + frame_ms
        if runs and start - runs[-1]["end_ms"] < MERGE_GAP_MS:
            runs[-1]["end_ms"] = end
            continue
        runs.append({"start_ms": start, "end_ms": end})
    return [run for run in runs if run["end_ms"] - run["start_ms"] >= min_speech_ms]


class MonoSpeakerSplitter:
    """The three layers from the module docstring. Models loaded once, reused for every case."""

    def __init__(self, segmentation_model: str, embedding_model: str) -> None:
        for path in (segmentation_model, embedding_model):
            if not Path(path).is_file():
                raise MonoSplitUnavailable(f"missing speaker splitting model: {path}")
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - package missing in the environment
            raise MonoSplitUnavailable("sherpa-onnx is not installed") from exc
        self._sherpa = sherpa_onnx
        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(
            sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=segmentation_model
                    ),
                    num_threads=4,
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=embedding_model, num_threads=4
                ),
                # Two sides: caller and agent. Letting the model guess the
                # number of speakers turns one cough from the person sitting
                # next to them into a third speaker.
                clustering=sherpa_onnx.FastClusteringConfig(num_clusters=2),
                min_duration_on=0.2,
                min_duration_off=0.3,
            )
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_model, num_threads=4)
        )

    # ── layer 1 ────────────────────────────────────────────────────────────
    def clusters(self, samples: Any) -> list[tuple[int, int, int]]:
        """(start_ms, end_ms, cluster) for the whole recording, sorted by time."""
        result = self._diarizer.process(samples).sort_by_start_time()
        return [
            (round(item.start * 1000), round(item.end * 1000), int(item.speaker))
            for item in result
        ]

    # ── layer 3 ────────────────────────────────────────────────────────────
    def embed(self, samples: Any) -> Any:
        import numpy as np

        stream = self._extractor.create_stream()
        stream.accept_waveform(16000, samples)
        stream.input_finished()
        vector = np.asarray(self._extractor.compute(stream), dtype="float32")
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    # ── layer 1b ───────────────────────────────────────────────────────────
    def split_by_voice(
        self, samples: Any, pieces: list[tuple[int, int, int]]
    ) -> list[tuple[int, int, int]] | None:
        """Rescue when blind clustering merged both people into one.

        Happens when one side speaks only a few very short turns, or when the
        two voices are close. Take the two pieces FURTHEST APART by voice
        embedding as anchors, then assign the rest to the nearer anchor.
        ``None`` = it really is one person, do not invent a second one.
        """
        import numpy as np

        usable = [piece for piece in pieces if piece[1] - piece[0] >= MIN_RESCUE_MS]
        if len(usable) < 2:
            return None
        vectors = {
            piece: self.embed(samples[piece[0] * 16 : piece[1] * 16]) for piece in usable
        }
        # The furthest pair must contain at least one long enough piece: two
        # tiny pieces disagreeing is ordinary voice-embedding noise, building a
        # second person out of that is fabrication.
        pairs = [
            (a, b)
            for index, a in enumerate(usable)
            for b in usable[index + 1 :]
            if max(a[1] - a[0], b[1] - b[0]) >= MIN_SAMPLE_MS
        ]
        if not pairs:
            return None
        far = min(pairs, key=lambda pair: float(np.dot(vectors[pair[0]], vectors[pair[1]])))
        if float(np.dot(vectors[far[0]], vectors[far[1]])) > MAX_SAME_VOICE_COSINE:
            return None
        seeds = [vectors[far[0]], vectors[far[1]]]
        out: list[tuple[int, int, int]] = []
        for piece in pieces:
            vector = vectors.get(piece)
            if vector is None:
                out.append((piece[0], piece[1], out[-1][2] if out else 0))
                continue
            scores = [float(np.dot(vector, seed)) for seed in seeds]
            out.append((piece[0], piece[1], 0 if scores[0] >= scores[1] else 1))
        return out



# A piece shorter than this after cutting on cluster boundaries is debris: merge
# it back into the piece next to it. A 120 ms "turn" is not a turn, it is a
# filler sound.
MIN_PIECE_MS = 300


def split_runs(
    runs: list[dict[str, int]], clusters: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """Cut speech runs where the SPEAKER CHANGES, returns (start, end, cluster).

    Without this step an unbroken caller-then-agent run (the two sides
    overlapping, VAD finding no silence to cut on) goes into a single label as
    one block: a whole turn is lost, and that turn's response latency goes with
    it.

    Merging only happens WITHIN a run. Two runs separated by silence are two
    turns even when they share a cluster — merging them swallows the turn in
    between, which the voice-embedding layer would otherwise still have had a
    chance to re-score.
    """
    pieces: list[tuple[int, int, int]] = []
    for run in runs:
        marks = {run["start_ms"], run["end_ms"]}
        for start, end, _cluster in clusters:
            for mark in (start, end):
                if run["start_ms"] < mark < run["end_ms"]:
                    marks.add(mark)
        edges = sorted(marks)
        cut: list[tuple[int, int, int]] = []
        for left, right in pairwise(edges):
            cluster, best = -1, 0
            for start, end, owner in clusters:
                overlap = min(end, right) - max(start, left)
                if overlap > best:
                    cluster, best = owner, overlap
            if cluster < 0:
                cluster = cut[-1][2] if cut else (pieces[-1][2] if pieces else 0)
            if cut and (right - left < MIN_PIECE_MS or cut[-1][2] == cluster):
                cut[-1] = (cut[-1][0], right, cut[-1][2])
                continue
            cut.append((left, right, cluster))
        while len(cut) > 1 and cut[0][1] - cut[0][0] < MIN_PIECE_MS:
            cut[1] = (cut[0][0], cut[1][1], cut[1][2])
            cut.pop(0)
        pieces.extend(cut)
    return pieces


# An overlap shorter than this is boundary debris, not one instance of
# overlapping speech: the segmentation model has a receptive field of 991
# samples (62 ms), so every boundary is blurred by about that much. Measured on
# the sample set: a real 0.59 s overlap was reported as 0.78 s.
MIN_OVERLAP_MS = 150


def overlap_spans(clusters: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    """Spans where TWO CLUSTERS are active at once — what ``split_runs`` erases.

    The segmentation model is a 7-class powerset (``num_classes=7``,
    ``powerset_max_classes=2``): the last three classes are two people speaking
    at once, so ``OfflineSpeakerDiarization`` returns segments that OVERLAP in
    time. ``split_runs`` cuts them into adjacent pieces — needed so that each
    turn carries exactly one label — and that is where the trace of "two people
    speaking at once" is lost. This function reads it before it is lost.

    ``cum_chen`` is the cluster that comes in later, ``cum_nhuong`` the cluster
    that leaves the overlap first; ``-1`` when the two marks are equal, because
    then neither one cut in on the other.

    This is INFERRED, not measured: overlaps under 200 ms are missed often (OSD
    F1 on telephone speech is around 0.60 — DIHARD III) and the blurred
    boundaries inflate the total duration by about 1.4x. Counting instances and
    taking their timestamps works; adding them up into a total number of seconds
    does not.
    """
    out: list[dict[str, int]] = []
    for index, (a_start, a_end, a_cluster) in enumerate(clusters):
        for b_start, b_end, b_cluster in clusters[index + 1 :]:
            if a_cluster == b_cluster:
                continue
            start, end = max(a_start, b_start), min(a_end, b_end)
            if end - start < MIN_OVERLAP_MS:
                continue
            late = b_cluster if b_start > a_start else (a_cluster if a_start > b_start else -1)
            early = a_cluster if a_end < b_end else (b_cluster if b_end < a_end else -1)
            out.append(
                {
                    "start_ms": start,
                    "duration_ms": end - start,
                    "cum_chen": late,
                    "cum_nhuong": early,
                }
            )
    return sorted(out, key=lambda span: span["start_ms"])


def co_bang_chung_hai_vai(spoken: list[tuple[int, str]], requirements: list[str]) -> bool:
    """Are there really two roles in the speech — asked when the VOICE cannot say.

    Synthesised recordings often have both caller and agent speaking in one
    voice, so the voice embedding is helpless and the labels have to alternate
    per turn. But "one voice" also fits a recording with only ONE person
    speaking — and alternating labels there invents a second person. Tell them
    apart by the words, not by the sound:

    - the two groups pull in opposite directions on the stock phrases (one side
      sounds like the contact centre, the other like someone asking for help),
      or
    - the same value from the requirements is spoken by BOTH groups — meaning
      one side gave the information and the other read it back to confirm.

    With neither sign present, return ``False``: better to refuse than to split
    one person's speech in two and then score the agent with the caller's own
    words.
    """
    normalized: dict[int, str] = {}
    for cluster, text in spoken:
        normalized[cluster] = f"{normalized.get(cluster, '')} {_norm(text)}".strip()
    if len(normalized) < 2:
        return False
    cues = [
        sum(cue in text for cue in AGENT_CUES) - sum(cue in text for cue in CALLER_CUES)
        for text in normalized.values()
    ]
    if max(cues) > 0 and min(cues) < 0:
        return True
    wanted = [value for value in (_norm(item).strip() for item in requirements) if value]
    return any(
        sum(value in text for text in normalized.values()) >= 2 for value in wanted
    )


def pick_agent_cluster(
    spoken: list[tuple[int, str]],
    requirements: list[str],
) -> tuple[int, str]:
    """Which cluster is the agent, and why. Decided by WORDS, not by voice.

    ``spoken`` is the speech pieces IN TIME ORDER: (cluster, recovered text).
    The order is evidence, not decoration — see ground 2.

    The agent voice is configured by the user, so it cannot be pinned in
    advance; the role shows up in the text. Four grounds, in decreasing
    certainty:

    1. The cluster that reads out **more of the values the user stated in the
       requirements** — the agent reads them back to confirm. A freshly uploaded
       recording usually has NO requirements yet; this ground then stays silent
       and the three below decide.
    2. Both sides read out the same value: the side that reads it **LATER** is
       the agent. The caller gives the information first, the agent repeats it
       to confirm — never the other way round.
    3. The **difference** in stock phrases: contact centre lines minus
       asking-for-help lines. Counting one side only turns the caller's "cảm ơn"
       into incriminating evidence too.
    4. The cluster that **speaks the last turn** — the agent is the side closing
       the call. Weakest, so the returned reason says outright that it is a
       guess, for the UI to lower the confidence.
    """
    order = [cluster for cluster, _text in spoken]
    keys = sorted(set(order))
    if len(keys) < 2:
        return keys[0] if keys else 0, "only one voice recognised"
    normalized: dict[int, str] = {}
    for cluster, text in spoken:
        normalized[cluster] = f"{normalized.get(cluster, '')} {_norm(text)}".strip()

    wanted = [_norm(value).strip() for value in requirements]
    wanted = [value for value in wanted if value]
    if wanted:
        hits = {
            cluster: sum(value in text for value in wanted) for cluster, text in normalized.items()
        }
        best, second = sorted(hits.values(), reverse=True)[:2]
        if best > second:
            cluster = max(hits, key=lambda key: hits[key])
            return cluster, f"reads back {hits[cluster]}/{len(wanted)} scripted values"
        if best > 0:
            said_at: dict[int, int] = {}
            for index, (cluster, text) in enumerate(spoken):
                normalized_text = _norm(text)
                if any(value in normalized_text for value in wanted):
                    said_at.setdefault(cluster, index)
            if len(said_at) == 2:
                cluster = max(said_at, key=lambda key: said_at[key])
                return cluster, "repeats the scripted value after the other side"

    cues = {
        cluster: sum(cue in text for cue in AGENT_CUES)
        - sum(cue in text for cue in CALLER_CUES)
        for cluster, text in normalized.items()
    }
    best, second = sorted(cues.values(), reverse=True)[:2]
    if best > second:
        cluster = max(cues, key=lambda key: cues[key])
        return cluster, f"{best - second} more contact centre stock phrases"

    return order[-1], "guessed from the last turn speaker — weak evidence, re-check the labels"
