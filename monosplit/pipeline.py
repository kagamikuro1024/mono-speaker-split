"""The whole run: one audio file in, a list of role-labelled turns out.

Five steps, and every one of them can say "not measurable" instead of guessing:

1. ``audio``  - read the file, decide whether it is one stream or two real ones.
2. ``speakers`` layer 1 - blind clustering, cut speech runs where the speaker
   changes.
3. ``transcribe`` - words with per-word timestamps (optional).
4. ``speakers`` layers 2 + 3 - words pick the role, voice embeddings re-score
   every run.
5. ``separate_voices`` - recover the words spoken while both people talked at
   once (optional; needs a separator model).

The rule that runs through all of it: **a role label on a single-channel
recording is INFERRED**, and the result has to say so. A transcript that looks
certain but has the roles swapped is worse than one that says "unsure, check".
"""

from __future__ import annotations

import tempfile
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from monosplit import audio
from monosplit.separate_voices import MIN_SEPARATE_MS, recover_overlap_text
from monosplit.speakers import (
    MIN_COSINE_MARGIN,
    MIN_ENROLL_MS,
    MIN_SAMPLE_MS,
    Labelled,
    MonoSpeakerSplitter,
    MonoSplitUnavailable,
    co_bang_chung_hai_vai,
    overlap_spans,
    pick_agent_cluster,
    speech_energy_runs,
    split_runs,
)
from monosplit.transcribe import Transcriber, words_per_piece

# VAD frame and turn thresholds - the same numbers as the two-channel path, so
# both paths produce the same timeline for the same recording.
VAD_FRAME_MS = 20
MIN_SPEECH_MS = 200
TURN_MERGE_GAP_MS = 400


class SeparationError(RuntimeError):
    """Cannot separate - carries an error code so callers show the right text."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


ERRORS: dict[str, str] = {
    "khong_co_tieng_noi": "This recording has almost no speech in it.",
    "khong_tach_nguoi_noi": "Only one voice was heard; the two roles cannot be separated.",
    "thieu_mo_hinh": "No speaker-separation model available (seg.onnx / emb.onnx).",
}


@dataclass
class Turn:
    """One turn: who spoke, from when to when, what they said, how sure we are."""

    speaker: Literal["caller", "agent"]
    start_ms: int
    end_ms: int
    text: str = ""
    #: Cosine margin between the two voice embeddings on this run. 0 = layer 3
    #: did not score it (no enrollment sample); below 0.10 = the voices are too
    #: close, so the label stays as layer 1 left it.
    margin: float = 0.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class Result:
    """One separation result, enough to draw the UI and to score the benchmark."""

    source: str
    duration_ms: int
    channels: int
    #: Why the single-channel path was taken: `mono`, `stereo_trung_nhau`,
    #: `stereo_mot_ben_cam`.
    mode: str
    turns: list[Turn] = field(default_factory=list)
    #: How layer 2 decided which cluster is the agent.
    role_reason: str = ""
    #: Both sides sound like one voice - labels alternate by turn order.
    same_voice: bool = False
    #: Warnings for whoever reads the result.
    warnings: list[str] = field(default_factory=list)
    #: Moments where both sides are SUSPECTED to have spoken at once, read from
    #: the segmentation model's powerset layer. Each item: `start_ms`,
    #: `duration_ms`, `who_cut_in`, `who_yielded` (role names, `None` when the
    #: direction cannot be inferred), and `recovered` - the words separated out
    #: of that overlap, or `None` when it was too short to separate.
    #:
    #: INFERRED, not measured - see `speakers.overlap_spans`. Use the count and
    #: the timestamps; do not add `duration_ms` up into a total overlap time.
    overlaps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "turns": [{**asdict(turn), "duration_ms": turn.duration_ms} for turn in self.turns],
        }


def separate(
    source: Path,
    splitter: MonoSpeakerSplitter,
    transcriber: Transcriber | None = None,
    requirements: list[str] | None = None,
    voice_separator: Any | None = None,
) -> Result:
    """Split one file into turns labelled caller / agent.

    ``voice_separator`` is optional: pass a ``separate_voices.VoiceSeparator``
    and every overlap at least ``MIN_SEPARATE_MS`` long also comes back with
    the words recovered from it. It needs ``transcriber`` too - there is no
    point separating audio nobody is going to read.
    """
    requirements = requirements or []
    info = audio.probe(source)

    with tempfile.TemporaryDirectory(prefix="monosplit-") as tmp:
        work = Path(tmp)
        mode = "mono"
        if info.channels >= 2:
            left, right = audio.decode_channels(source, work)
            if not audio.one_stream_only(left, right):
                raise SeparationError(
                    "hai_kenh_that",
                    "This file has two real per-role channels - use each channel "
                    "directly, there is nothing to infer.",
                )
            import numpy as np

            r_level = float(np.abs(np.frombuffer(right, dtype=np.int16)).mean())
            l_level = float(np.abs(np.frombuffer(left, dtype=np.int16)).mean())
            quiet = min(l_level, r_level)
            loud = max(l_level, r_level)
            mode = "stereo_mot_ben_cam" if loud and quiet <= 0.01 * loud else "stereo_trung_nhau"

        mono_path = work / "mono.wav"
        pcm = audio.decode_mono(source, mono_path)
        samples = audio.to_samples(pcm)

        runs = speech_energy_runs(pcm, VAD_FRAME_MS, MIN_SPEECH_MS, audio.PCM_BYTES_PER_MS)
        if not runs:
            raise SeparationError("khong_co_tieng_noi", ERRORS["khong_co_tieng_noi"])

        pieces, same_voice, spans = _pieces(splitter, samples, runs)
        if len(pieces) < 2:
            raise SeparationError("khong_tach_nguoi_noi", ERRORS["khong_tach_nguoi_noi"])

        words = transcriber.words(mono_path) if transcriber is not None else []
        texts = words_per_piece(words, pieces, TURN_MERGE_GAP_MS)

        labels, reason = _label(splitter, samples, pieces, texts, requirements)
        warnings: list[str] = []
        if same_voice:
            # Synthetic recordings often use ONE voice for both roles: voice
            # embeddings then tell nobody apart. The only real evidence left is
            # TURN ORDER - usable only when the words themselves show two roles,
            # because "one voice" also describes a recording of one person.
            spoken = [(cluster, text) for (_s, _e, cluster), text in zip(pieces, texts, strict=True)]
            if not co_bang_chung_hai_vai(spoken, requirements):
                raise SeparationError("khong_tach_nguoi_noi", ERRORS["khong_tach_nguoi_noi"])
            warnings.append("both sides sound like one voice - labels alternate by turn order")

        turns = [
            Turn(label.speaker, label.start_ms, label.end_ms, text, label.margin)
            for label, text in zip(labels, texts, strict=True)
        ]

        # Clusters are numbers, readers need role names: take the mapping from
        # layer 2's own labels - a cluster belongs to whichever role it
        # contributes the most speech time to.
        by_cluster: dict[tuple[int, str], int] = {}
        for (start, end, cluster), label in zip(pieces, labels, strict=True):
            key = (cluster, label.speaker)
            by_cluster[key] = by_cluster.get(key, 0) + (end - start)
        role_of: dict[int, str] = {}
        for cluster in {piece[2] for piece in pieces}:
            sides = [(ms, role) for (c, role), ms in by_cluster.items() if c == cluster]
            if sides:
                role_of[cluster] = max(sides)[1]
        overlaps = [
            {
                "start_ms": span["start_ms"],
                "duration_ms": span["duration_ms"],
                "who_cut_in": role_of.get(span["cum_chen"]),
                "who_yielded": role_of.get(span["cum_nhuong"]),
            }
            for span in spans
        ]
        if voice_separator is not None and transcriber is not None:
            # The mixture swallowed whoever was quieter. Separate a window
            # around each overlap and read the streams back - words only: the
            # timestamps still come from the original waveform, never from a
            # separated stream (its clock starts at the padded window edge).
            overlaps = recover_overlap_text(
                overlaps,
                samples,
                separator=voice_separator,
                transcribe=lambda track: _transcribe_track(transcriber, track, work),
                embed=splitter.embed,
            )
        if transcriber is not None:
            turns = [turn for turn in turns if turn.text] or turns

        return Result(
            source=source.name,
            duration_ms=info.duration_ms,
            channels=info.channels,
            mode=mode,
            turns=turns,
            role_reason=reason,
            same_voice=same_voice,
            warnings=warnings,
            overlaps=overlaps,
        )


def _pieces(
    splitter: MonoSpeakerSplitter, samples, runs
) -> tuple[list[tuple[int, int, int]], bool, list[dict[str, int]]]:
    """Layer 1: blind clustering, cut runs where the speaker changes, rescue if
    the clustering swallowed one side.

    The third element holds the spans where both clusters are active, read
    BEFORE ``split_runs`` cuts them into adjacent pieces.
    """
    clusters = splitter.clusters(samples)
    spans = overlap_spans(clusters)
    pieces = split_runs(runs, clusters)
    if len({piece[2] for piece in pieces}) >= 2:
        return pieces, False, spans
    # Blind clustering swallowed one side - common when one party only says
    # "yeah", "uh-huh". Retry with voice embeddings before concluding there is
    # only one voice.
    rescued = splitter.split_by_voice(samples, pieces)
    if rescued is not None and len({piece[2] for piece in rescued}) >= 2:
        return rescued, False, spans
    # Once labels alternate, cluster numbers mean nothing: keep the timestamps,
    # drop the direction - inferring "who cut in on whom" from a broken mapping
    # would be making it up.
    return (
        [(start, end, index % 2) for index, (start, end, _c) in enumerate(pieces)],
        True,
        [{**span, "cum_chen": -1, "cum_nhuong": -1} for span in spans],
    )


def _label(
    splitter: MonoSpeakerSplitter,
    samples,
    pieces: list[tuple[int, int, int]],
    texts: list[str],
    requirements: list[str],
) -> tuple[list[Labelled], str]:
    """Layer 2 (words pick the role) and layer 3 (embeddings re-score each run)."""
    import numpy as np

    agent_cluster, reason = pick_agent_cluster(
        [(cluster, text) for (_s, _e, cluster), text in zip(pieces, texts, strict=True)],
        requirements,
    )

    # Enrollment sample: the LONGEST piece of each cluster. A long piece dilutes
    # whatever leaked in from the other voice, so the vector still belongs to
    # the right person. A cluster with only short pieces still needs a sample:
    # missing one turns layer 3 off entirely, and that is exactly the case where
    # layer 1 went wrong and layer 3 is needed most.
    profiles: dict[int, object] = {}
    for cluster in {piece[2] for piece in pieces}:
        pool = [piece for piece in pieces if piece[2] == cluster]
        enrollable = [p for p in pool if p[1] - p[0] >= MIN_ENROLL_MS] or [
            p for p in pool if p[1] - p[0] >= MIN_SAMPLE_MS
        ]
        if not enrollable:
            continue
        longest = max(enrollable, key=lambda piece: piece[1] - piece[0])
        profiles[cluster] = splitter.embed(samples[longest[0] * 16 : longest[1] * 16])

    labels: list[Labelled] = []
    for start, end, cluster in pieces:
        speaker: Literal["caller", "agent"] = "agent" if cluster == agent_cluster else "caller"
        margin = 0.0
        if len(profiles) == 2:
            vector = splitter.embed(samples[start * 16 : end * 16])
            scores = {key: float(np.dot(vector, profile)) for key, profile in profiles.items()}
            winner = max(scores, key=lambda key: scores[key])
            margin = round(abs(scores[winner] - min(scores.values())), 4)
            # Embeddings may only flip a label when they are CERTAIN. When the
            # scores sit close together, keep the clustering layer's label:
            # flipping on a margin of 0.02 is a coin toss reported as a
            # measurement.
            if margin >= MIN_COSINE_MARGIN:
                speaker = "agent" if winner == agent_cluster else "caller"
        labels.append(Labelled(start, end, speaker, margin))
    return labels, reason


def _role_profiles(
    splitter: MonoSpeakerSplitter, samples, labels: list[Labelled]
) -> dict[str, object] | None:
    """One voice embedding per ROLE, taken from that role's longest turn.

    Used to name the separated streams. A long turn dilutes whatever leaked in
    from the other voice. Returns ``None`` when either role has no usable
    sample: naming two streams from one embedding is naming them by feel.
    """
    profiles: dict[str, object] = {}
    for role in ("caller", "agent"):
        mine = [label for label in labels if label.speaker == role]
        if not mine:
            return None
        longest = max(mine, key=lambda label: label.end_ms - label.start_ms)
        if longest.end_ms - longest.start_ms < MIN_SAMPLE_MS:
            return None
        profiles[role] = splitter.embed(samples[longest.start_ms * 16 : longest.end_ms * 16])
    return profiles


def _transcribe_track(transcriber: Transcriber, track, work: Path) -> str:
    """Read words off a separated stream - words only, never timestamps.

    A separated stream's clock starts at the padded window edge, so its
    timestamps do not belong on the call's timeline.

    VAD is off here: the clip is short and already known to contain speech,
    while VAD on a short clip tends to clip the first word.
    """
    import numpy as np

    clip = work / "overlap_clip.wav"
    with wave.open(str(clip), "wb") as dest:
        dest.setnchannels(1)
        dest.setsampwidth(2)
        dest.setframerate(audio.TARGET_SAMPLE_RATE)
        dest.writeframes((np.clip(track, -1.0, 1.0) * 32767).astype("int16").tobytes())
    return " ".join(word.text for word in transcriber.words(clip, vad_filter=False)).strip()


__all__ = [
    "ERRORS",
    "MIN_SEPARATE_MS",
    "MonoSplitUnavailable",
    "Result",
    "SeparationError",
    "Transcriber",
    "Turn",
    "separate",
]
