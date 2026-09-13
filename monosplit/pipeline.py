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
from monosplit.separate_voices import (
    MIN_SEPARATE_MS,
    build_role_tracks,
    continuous_role_tracks,
)
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

# Overlapped time divided by total speech time. Above this the recording counts
# as "densely overlapped": no stretch holds one voice alone to sample from, and
# the clustering layer has no silence left to cut turns on. Measured on a 30 s
# dense set (2026-09-13): at 0.56 the normal path collapsed 17 turns into 3,
# while separating the whole recording kept every sentence on the right side.
DENSE_OVERLAP_SHARE = 0.30


def _overlap_share(overlaps: list[dict], runs: list[dict[str, int]]) -> float:
    """How much of the speech time is overlapped."""
    speech = sum(run["end_ms"] - run["start_ms"] for run in runs)
    if speech <= 0:
        return 0.0
    return sum(int(span.get("duration_ms") or 0) for span in overlaps) / speech


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


def _cut_in_direction(cut_in: str | None, yielded: str | None) -> dict[str, str | None]:
    """The direction, only when the two sides are two DIFFERENT roles."""
    if cut_in is not None and cut_in == yielded:
        return {"who_cut_in": None, "who_yielded": None}
    return {"who_cut_in": cut_in, "who_yielded": yielded}


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
    #: direction cannot be inferred), and `separated` - whether the words of the
    #: turns covering that stretch were re-read off a separated stream instead
    #: of the mixture. `separated` marks where to listen before trusting the
    #: text.
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
    and every turn covering an overlap of at least ``MIN_SEPARATE_MS`` is
    re-read off that role's recovery channel, so its text holds only the words
    of the person who spoke it. It needs ``transcriber`` too - there is no point
    separating audio nobody is going to read.
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
        can_separate = voice_separator is not None and transcriber is not None
        if len(pieces) < 2 and not can_separate:
            raise SeparationError("khong_tach_nguoi_noi", ERRORS["khong_tach_nguoi_noi"])

        words = transcriber.words(mono_path) if transcriber is not None else []
        texts = words_per_piece(words, pieces, TURN_MERGE_GAP_MS)

        labels, reason = _label(splitter, samples, pieces, texts, requirements)
        warnings: list[str] = []
        # Layer 1 gave up: one cluster only, or two clusters that sound like one
        # voice with no evidence in the words either. On a densely overlapped
        # recording that is exactly what happens - every piece contains both
        # people, so every piece resembles every other. A separator turns this
        # from "refuse" into "separate first, label after", so try that before
        # giving up.
        layer_one_failed = len(pieces) < 2 or (
            same_voice
            and not co_bang_chung_hai_vai(
                [(cluster, text) for (_s, _e, cluster), text in zip(pieces, texts, strict=True)],
                requirements,
            )
        )
        if layer_one_failed and not can_separate:
            raise SeparationError("khong_tach_nguoi_noi", ERRORS["khong_tach_nguoi_noi"])
        if same_voice and not layer_one_failed:
            # Synthetic recordings often use ONE voice for both roles: voice
            # embeddings then tell nobody apart. The only real evidence left is
            # TURN ORDER - usable only when the words themselves show two roles,
            # because "one voice" also describes a recording of one person.
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
                # Both clusters mapping to one role makes the direction
                # meaningless: "caller cut in on caller" is not two people
                # talking over each other, it is a broken cluster->role mapping.
                # Keep the timestamp, drop the direction - reading further from
                # a mapping already known to be wrong is making it up.
                **_cut_in_direction(
                    role_of.get(span["cum_chen"]), role_of.get(span["cum_nhuong"])
                ),
            }
            for span in spans
        ]
        dense = layer_one_failed or _overlap_share(overlaps, runs) >= DENSE_OVERLAP_SHARE
        if can_separate and dense:
            # Overlapped so densely that no stretch holds one voice alone: there
            # is nothing to sample enrollment from, and the clustering layer has
            # no silence left to cut turns on either - it merges several turns
            # into one. Change tack: separate the WHOLE recording into two
            # channels and read each one like a real second channel.
            rebuilt = _transcript_from_continuous_split(
                samples, splitter, transcriber, voice_separator, requirements, work
            )
            if rebuilt is not None:
                turns, reason_suffix = rebuilt
                reason = f"{reason}; {reason_suffix}"
                overlaps = [
                    {**span, "separated": span["duration_ms"] >= MIN_SEPARATE_MS}
                    for span in overlaps
                ]
                dense = False
            elif layer_one_failed:
                # Even separating the whole recording did not produce two
                # people: that is a one-speaker recording, say so.
                raise SeparationError("khong_tach_nguoi_noi", ERRORS["khong_tach_nguoi_noi"])
        if (
            voice_separator is not None
            and transcriber is not None
            and not dense
            and not any(span.get("separated") for span in overlaps)
        ):
            # The mixture swallowed whoever was quieter, so every turn that
            # overlaps is missing words or carrying the other person's. Rebuild
            # one channel per role and re-read those turns off the channel of
            # the person who spoke them - the same thing the two-channel path
            # does, so the words land straight in the transcript.
            #
            # Timestamps still come from the original waveform: a separated
            # stream's clock starts at the padded window edge.
            turns, overlaps = _recover_overlapping_turns(
                turns, overlaps, samples, splitter, transcriber, voice_separator, work
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
    splitter: MonoSpeakerSplitter, samples, turns: list[Turn]
) -> dict[str, object] | None:
    """One voice embedding per ROLE, taken from that role's longest turn.

    Used to name the separated streams. A long turn dilutes whatever leaked in
    from the other voice. Returns ``None`` when either role has no usable
    sample: naming two streams from one embedding is naming them by feel.
    """
    profiles: dict[str, object] = {}
    for role in ("caller", "agent"):
        mine = [turn for turn in turns if turn.speaker == role]
        if not mine:
            return None
        longest = max(mine, key=lambda turn: turn.duration_ms)
        if longest.duration_ms < MIN_SAMPLE_MS:
            return None
        profiles[role] = splitter.embed(samples[longest.start_ms * 16 : longest.end_ms * 16])
    return profiles


def _recover_overlapping_turns(
    turns: list[Turn],
    overlaps: list[dict],
    samples,
    splitter: MonoSpeakerSplitter,
    transcriber: Transcriber,
    voice_separator,
    work: Path,
) -> tuple[list[Turn], list[dict]]:
    """Re-read the overlapping turns off each role's channel, and rebuild the
    turns the mixture swallowed whole.

    Three things, in order:

    1. **Words.** A turn covering an overlap is re-read off its own role's
       channel. Turns clear of every overlap keep their text: there the mixture
       holds one voice only, so it is already the cleanest signal, and sending
       it through a second model would only add a chance to be wrong.
    2. **Start time.** The turn of whoever CUT IN gets its start pulled back to
       the start of the overlap. Blind clustering cannot cut inside a stretch
       where both people speak, so it starts the interrupter's turn AFTER the
       overlap and their first words get counted into the other person's turn.
       Pulled back, the two turns genuinely overlap in time - which is what
       talking over each other is.
    3. **Missing turns.** A short turn the other person talks straight THROUGH
       leaves no silence for the VAD to cut on, so it disappears from the
       transcript entirely - not wrong words, a lost turn. Measured on a 30 s
       dense-overlap set: 2 of 11 turns vanished this way. The recovery channel
       has them separated, so the VAD is run again over exactly the separated
       windows: a stretch of speech no turn of that role covers becomes a new
       turn.
    """
    import numpy as np

    profiles = _role_profiles(splitter, samples, turns)
    if profiles is None:
        return turns, [{**span, "separated": False} for span in overlaps]

    built = build_role_tracks(
        overlaps, samples, separator=voice_separator, profiles=profiles, embed=splitter.embed
    )
    if built is None:
        return turns, [{**span, "separated": False} for span in overlaps]
    tracks, marked = built
    touched = [span for span in marked if span.get("separated")]

    moved = list(turns)
    for span in touched:
        role = span.get("who_cut_in")
        if role is None:
            continue
        end = span["start_ms"] + span["duration_ms"]
        for index, turn in enumerate(moved):
            if turn.speaker != role or turn.start_ms <= span["start_ms"]:
                continue
            if turn.start_ms > end + TURN_MERGE_GAP_MS:
                break
            # Pull back to the overlap's start, but never into the previous turn
            # of the SAME role: one person cannot speak twice at once, and two
            # overlapping same-role turns make the word split below hand the
            # same sentence to both.
            floor = max(
                (other.end_ms for other in moved[:index] if other.speaker == role), default=0
            )
            moved[index] = Turn(
                turn.speaker, max(span["start_ms"], floor), turn.end_ms, turn.text, turn.margin
            )
            break

    heard: dict[str, list] = {}
    for role, track in tracks.items():
        clip = work / f"recovered_{role}.wav"
        pcm = (np.clip(track, -1.0, 1.0) * 32767).astype("int16").tobytes()
        with wave.open(str(clip), "wb") as dest:
            dest.setnchannels(1)
            dest.setsampwidth(2)
            dest.setframerate(audio.TARGET_SAMPLE_RATE)
            dest.writeframes(pcm)
        heard[role] = transcriber.words(clip)
        moved += _turns_missed_on_channel(role, pcm, moved, touched)

    moved.sort(key=lambda turn: (turn.start_ms, turn.speaker))

    for role, words in heard.items():
        # Split the words between turns of the SAME role exactly the way the
        # mixture path does: every word goes to exactly ONE turn. The two
        # recovery channels are independent, so turns of DIFFERENT roles may
        # still overlap in time - but within one role they may not, and missing
        # that is three caller turns all carrying one identical sentence.
        mine = [(index, turn) for index, turn in enumerate(moved) if turn.speaker == role]
        role_texts = words_per_piece(
            words, [(turn.start_ms, turn.end_ms, 0) for _index, turn in mine], TURN_MERGE_GAP_MS
        )
        for (index, turn), spoken in zip(mine, role_texts, strict=True):
            if not any(
                turn.start_ms < span["start_ms"] + span["duration_ms"]
                and span["start_ms"] < turn.end_ms
                for span in touched
            ):
                continue
            # A turn inside an overlap: the recovery channel wins even when it
            # yields less text - the mixture there tends to glue both sides'
            # words into one sentence.
            if spoken.strip():
                moved[index] = Turn(
                    turn.speaker, turn.start_ms, turn.end_ms, spoken, turn.margin
                )

    # A turn we invented but read no words on is dropped: an empty line in the
    # transcript is a line the reviewer has to fill in by hand, which is worse
    # than no line at all.
    kept = [turn for turn in moved if turn.text.strip() or turn in turns]
    return kept, marked


def _transcript_from_continuous_split(
    samples,
    splitter: MonoSpeakerSplitter,
    transcriber: Transcriber,
    voice_separator,
    requirements: list[str],
    work: Path,
) -> tuple[list[Turn], str] | None:
    """Separate the whole recording into two channels and read each one.

    For densely overlapped recordings: there the clustering layer has no silence
    left to cut turns on, so it merges several turns into one and short turns
    disappear entirely. Two separated channels hold one voice each, and turn
    boundaries become measurable again.

    The two channels carry NO role names - the separator only guarantees "two
    different people". Which one is the caller is still decided by the WORDS
    (``pick_agent_cluster``), the one place allowed to decide that.
    """
    import numpy as np

    built = continuous_role_tracks(samples, separator=voice_separator, embed=splitter.embed)
    if built is None:
        return None
    tracks, stitch = built

    heard: list[tuple[int, int, int, list[str]]] = []
    for index, track in enumerate(tracks):
        clip = work / f"continuous_{index}.wav"
        with wave.open(str(clip), "wb") as dest:
            dest.setnchannels(1)
            dest.setsampwidth(2)
            dest.setframerate(audio.TARGET_SAMPLE_RATE)
            dest.writeframes((np.clip(track, -1.0, 1.0) * 32767).astype("int16").tobytes())
        # Turn boundaries come from WORD timestamps, not from a VAD on this
        # channel: a separated channel still leaks the other voice at a low
        # level, so a VAD sees speech almost continuously and merges ten seconds
        # into one turn. Word timestamps show the real pauses.
        for word in transcriber.words(clip):
            if heard and heard[-1][0] == index and word.start_ms - heard[-1][2] <= TURN_MERGE_GAP_MS:
                channel, start, _end, spoken = heard[-1]
                heard[-1] = (channel, start, word.end_ms, [*spoken, word.text])
            else:
                heard.append((index, word.start_ms, word.end_ms, [word.text]))
    if not heard:
        return None

    texts = [" ".join(words).strip() for _index, _start, _end, words in heard]
    agent_channel, reason = pick_agent_cluster(
        [(item[0], text) for item, text in zip(heard, texts, strict=True)], requirements
    )
    order = sorted(range(len(heard)), key=lambda i: heard[i][1])
    turns = [
        Turn(
            "agent" if heard[i][0] == agent_channel else "caller",
            heard[i][1],
            heard[i][2],
            texts[i],
            # Nothing was compared here: the label comes from the words, and
            # "two different people" is what the separator guarantees.
            0.0,
        )
        for i in order
        if texts[i]
    ]
    note = (
        f"densely overlapped, so the whole recording was separated into two channels "
        f"({stitch['windows']} windows, {stitch['weak_seams']} weak seams); {reason}"
    )
    if stitch["weak_seams"]:
        # A weak seam means the two channels may have swapped there - say so
        # rather than presenting it as certain.
        note += " - one join is uncertain, re-check both sides"
    return turns, note


def _turns_missed_on_channel(
    role: str, pcm: bytes, turns: list[Turn], touched: list[dict]
) -> list[Turn]:
    """Turns of ``role`` the mixture swallowed, found again on its channel.

    Only inside the separated overlaps: everywhere else the recovery channel IS
    the mixture, and the clustering layer already split that - accepting runs
    there would duplicate turns.

    A run the role already has a turn for is skipped. A run merely grazing an
    existing turn (VAD edges drift) is not a new turn: the bar is overlapping by
    more than ``MIN_SAMPLE_MS``.
    """
    mine = [turn for turn in turns if turn.speaker == role]
    out: list[Turn] = []
    for run in speech_energy_runs(pcm, VAD_FRAME_MS, MIN_SPEECH_MS, audio.PCM_BYTES_PER_MS):
        start, end = run["start_ms"], run["end_ms"]
        if end - start < MIN_SAMPLE_MS:
            continue
        if not any(
            start < span["start_ms"] + span["duration_ms"] and span["start_ms"] < end
            for span in touched
        ):
            continue
        covered = max(
            (min(end, turn.end_ms) - max(start, turn.start_ms) for turn in mine), default=0
        )
        if covered > MIN_SAMPLE_MS:
            continue
        # margin 0.0: this turn came from the recovery channel, with no two
        # embeddings to compare - do not invent a confidence for it.
        out.append(Turn(role, start, end, "", 0.0))
    return out


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
