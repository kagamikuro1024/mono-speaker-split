"""Rebuild one channel per role, so the words of both sides come back.

A single-channel recording carries one waveform, so the quieter speaker is
swallowed by the mix: ASR on the mixture returns only the louder side, and the
turn that overlaps carries either nothing or the other person's words.

What this module builds is a **recovery channel per role**: outside the overlap
it is the mixture itself (only one person is speaking there, so the mixture is
already the cleanest signal available), inside the overlap it is that role's
separated stream. The step that follows then reads those channels exactly the
way it reads the two channels of a stereo recording - so each turn ends up with
only the words of the person who spoke it, and the words land straight in the
transcript for the user to correct.

Three limits are wired into the code, not exposed as options:

1. **Only overlaps long enough to separate.** LibriCSS
   ([arXiv:2001.11482](https://arxiv.org/abs/2001.11482)) measured 0% overlap
   and found separation made WER *worse*: 11.8 -> 12.7. The 30-case acceptance
   suite in this repo agrees: at 200 ms of overlap, 2 of 6 cases came out worse
   than not separating at all. Below ``MIN_SEPARATE_MS`` nothing is separated.
2. **The words are still INFERRED, and the result says so.** The signature
   failure of cascaded separation -> ASR is *speaker leakage*: one person's
   words land in the other person's stream
   ([arXiv:2608.22196](https://arxiv.org/abs/2608.22196), Interspeech 2026,
   up to 71-77% WER on AMI). Whisper can also invent whole sentences on short
   clips ([arXiv:2402.08021](https://arxiv.org/abs/2402.08021)). Every overlap
   that was separated is marked ``separated: True`` so the reader knows which
   stretch to listen to before trusting it - the whole point of a transcript
   the user can edit.
3. **No channels at all rather than guessed roles.** The two streams leave the
   separator unnamed (the permutation problem). Enrollment samples of the two
   roles decide which is which, scored as a pairing; below
   ``MIN_ASSIGN_MARGIN`` nothing is built. One TTS voice reading both parts
   lands here by design: swapping the channels would swap both sides' words
   across the whole overlap, which is far worse than leaving the mixture alone.

Measured on the 30-case acceptance suite (2026-09-13, SepFormer WHAMR 16 kHz
ONNX + Whisper large-v3 on the separated streams):

| Overlap  | CER on caller words | CER on agent words |
|----------|---------------------|--------------------|
| mixture  | 0.65                | 0.78               |
| 200 ms   | 0.86 -> 0.37        | 0.63 -> 0.46       |
| 600 ms   | 0.74 -> 0.27        | 0.83 -> 0.28       |
| 1200 ms  | 0.59 -> 0.14        | 0.49 -> 0.16       |
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Overlaps shorter than this are left alone. See limit 1 in the module
# docstring: separating them makes the result worse than leaving it be.
MIN_SEPARATE_MS = 400

# Context kept on both sides of the overlap. The separator needs to hear each
# voice alone to tell them apart; cut flush to the overlap and it sees only the
# mixed part and folds both voices into one stream. 1.5 s scored best on the
# 30-case suite.
PAD_MS = 1_500

# How far one pairing must beat the other before a role is assigned. Same value
# as ``speakers.MIN_COSINE_MARGIN``: two places that assign roles with two
# different thresholds report two different answers for one recording.
MIN_ASSIGN_MARGIN = 0.10

# Streams below this peak carry no speech: the separator found only one person.
SILENT_PEAK = 1e-4


class SeparatorUnavailable(RuntimeError):
    """Model could not be loaded: text recovery is off, the rest still runs."""


class VoiceSeparator:
    """Two-speaker separator, ONNX, loaded once and reused for every call.

    Measured alternatives (30-case suite, same ASR, same windows):

    | Model                      | Size    | CER caller | CER agent |
    |----------------------------|---------|------------|-----------|
    | SepFormer WHAMR **16 kHz** | 106 MB  | **0.33**   | **0.33**  |
    | SepFormer wsj0-2mix 8 kHz  | 28.5 MB | 0.64       | 0.34      |
    | MossFormer2 16 kHz         | 639 MB  | 0.56       | 0.53      |

    The 8 kHz model throws away the upper band that ASR needs, and MossFormer2
    wins on SI-SDRi yet loses on words - which is the only thing measured here.
    """

    def __init__(self, model_path: str | Path, model_rate: int = 16_000) -> None:
        if not Path(model_path).is_file():
            raise SeparatorUnavailable(f"separator model not found: {model_path}")
        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SeparatorUnavailable("onnxruntime is not installed") from exc
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 4
        self._session = onnxruntime.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self.rate = model_rate

    def streams(self, window: Any) -> list[Any]:
        """One mixed channel at the model's rate -> one array per speaker."""
        import numpy as np

        outputs = self._session.run(None, {self._input: window[None, :].astype("float32")})
        if len(outputs) > 1:  # one output tensor per speaker
            return [np.asarray(item, dtype="float32").reshape(-1) for item in outputs]
        block = np.asarray(outputs[0], dtype="float32")
        if block.ndim == 3 and block.shape[-1] <= 4:  # [1, time, speaker]
            return [block[0, :, index] for index in range(block.shape[-1])]
        if block.ndim == 3:  # [1, speaker, time]
            return [block[0, index, :] for index in range(block.shape[1])]
        return [block[index].reshape(-1) for index in range(block.shape[0])]


def resample(samples: Any, source_rate: int, target_rate: int) -> Any:
    """Change sample rate through ffmpeg - the same filter the decoder uses."""
    import numpy as np

    if source_rate == target_rate:
        return samples
    raw = (np.clip(samples, -1.0, 1.0) * 32767).astype("int16").tobytes()
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(source_rate), "-ac", "1",
         "-i", "pipe:0", "-ar", str(target_rate), "-f", "s16le", "pipe:1"],
        input=raw,
        capture_output=True,
        timeout=120,
    )
    if done.returncode != 0:
        raise SeparatorUnavailable("ffmpeg could not resample the window")
    return np.frombuffer(done.stdout, dtype="int16").astype("float32") / 32768.0


# The joint between mixture and separated stream has to cross over gradually. A
# hard cut produces a click that ASR reads as a stray syllable, and it sits
# right at the edge of the overlap - the very place the words matter most.
FADE_MS = 20


# Window and hop of the CONTINUOUS mode. Consecutive windows share 1 s: that
# shared second is the only evidence for which stream of the later window
# continues which stream of the earlier one.
CONT_WINDOW_MS = 4_000
CONT_HOP_MS = 3_000

# A weak seam: both ways of joining score about the same. Weak seams are counted
# rather than silently guessed - the caller has to be able to say "this may be
# swapped in the middle" instead of presenting it as certain.
MIN_STITCH_MARGIN = 0.10


def continuous_role_tracks(
    samples: Any,
    *,
    separator: VoiceSeparator,
    embed: Callable[[Any], Any],
    sample_rate: int = 16_000,
) -> tuple[list[Any], dict[str, Any]] | None:
    """Separate the WHOLE recording into two channels, and name them later.

    For recordings so densely overlapped that no stretch holds one voice alone:
    ``build_role_tracks`` needs enrollment samples taken outside the overlaps,
    and here there are none. So the order is reversed - separate first, assign
    roles afterwards.

    The recording is cut into overlapping windows, each window separated into two
    streams, and the windows STITCHED: the later window's stream joins whichever
    earlier stream its voice embedding is closer to. The separator only
    guarantees "two different people" WITHIN a window; "who is who throughout"
    is what this stitching step establishes (continuous speech separation,
    LibriCSS - [arXiv:2001.11482](https://arxiv.org/abs/2001.11482)).

    Returns two channels as long as the original, with NO role names: they are
    only "two different people". Which one is the caller is for the text layer
    to decide, exactly as on a two-channel recording whose channels are unlabelled.

    The second element reports the stitching: window count and weak-seam count.
    Many weak seams means the two channels may have swapped somewhere in the
    middle.

    Measured on a 30 s set with 56% of its speech overlapped: the normal path
    collapsed 17 turns into 3, this one keeps every sentence on the right side.
    """
    import numpy as np

    total_ms = len(samples) * 1000 // sample_rate
    if total_ms < CONT_WINDOW_MS:
        return None

    per_ms = sample_rate // 1000
    tracks = [np.zeros(len(samples), dtype="float32") for _ in range(2)]
    weights = np.zeros(len(samples), dtype="float32")
    anchors: list[Any] = [None, None]
    windows = weak = 0

    start_ms = 0
    while start_ms < total_ms:
        end_ms = min(total_ms, start_ms + CONT_WINDOW_MS)
        window = samples[start_ms * per_ms : end_ms * per_ms]
        if len(window) < sample_rate // 2:
            break
        try:
            streams = separator.streams(resample(window, sample_rate, separator.rate))
        except Exception:  # pragma: no cover - a broken model drops the mode
            logger.warning("continuous separation failed at %d ms", start_ms, exc_info=True)
            return None
        streams = [
            resample(stream, separator.rate, sample_rate)[: len(window)] for stream in streams
        ]
        if len(streams) < 2:
            return None

        vectors = [
            embed(stream / (float(np.abs(stream).max()) or 1.0) * 0.9) for stream in streams[:2]
        ]
        if anchors[0] is None:
            order = (0, 1)
        else:
            straight = float(np.dot(vectors[0], anchors[0])) + float(np.dot(vectors[1], anchors[1]))
            crossed = float(np.dot(vectors[0], anchors[1])) + float(np.dot(vectors[1], anchors[0]))
            order = (0, 1) if straight >= crossed else (1, 0)
            if abs(straight - crossed) < MIN_STITCH_MARGIN:
                weak += 1
        windows += 1

        # Overlap-add with a trapezoid weight: the joint between two windows
        # crosses over, so it leaves no step for ASR to read as a stray syllable.
        reference = float(np.sqrt(np.mean(np.square(window)))) or 1.0
        ramp = np.minimum(
            np.minimum(
                np.linspace(0.0, 4.0, len(window), dtype="float32"),
                np.linspace(4.0, 0.0, len(window), dtype="float32"),
            ),
            1.0,
        )
        for track_index, stream_index in enumerate(order):
            stream = streams[stream_index]
            level = float(np.sqrt(np.mean(np.square(stream)))) or 1.0
            span = slice(start_ms * per_ms, start_ms * per_ms + len(stream))
            tracks[track_index][span] += stream * (reference / level) * ramp[: len(stream)]
            # The anchor updates gradually: one person's voice does not change
            # over a call, so a running mean is steadier than the last window.
            vector = vectors[stream_index]
            base = anchors[track_index]
            merged = vector if base is None else base * 0.7 + vector * 0.3
            anchors[track_index] = merged / (float(np.linalg.norm(merged)) or 1.0)
        weights[start_ms * per_ms : start_ms * per_ms + len(window)] += ramp
        start_ms += CONT_HOP_MS

    if windows < 2:
        return None
    weights[weights < 1e-4] = 1.0
    return [track / weights for track in tracks], {"windows": windows, "weak_seams": weak}


def build_role_tracks(
    overlaps: list[dict[str, Any]],
    samples: Any,
    *,
    separator: VoiceSeparator,
    profiles: dict[str, Any],
    embed: Callable[[Any], Any],
    sample_rate: int = 16_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Build one recovery channel per role out of a single-channel recording.

    Each channel is as long as the original: the mixture outside every overlap,
    that role's separated stream inside. The caller can then transcribe the two
    channels the way it transcribes a stereo pair.

    ``samples`` is the mixed waveform as float32 at ``sample_rate`` - the same
    array the labelling layers work on, so overlap timestamps index into it
    directly with no conversion. ``profiles`` holds one voice embedding per role
    (``caller`` / ``agent``) taken from speech OUTSIDE any overlap.

    Returns ``None`` when nothing was separable, or when the streams could not
    be assigned to roles - see limit 3 in the module docstring.

    The second element is the overlap list with ``separated`` set, so the reader
    knows which stretches were rebuilt rather than heard directly.
    """
    import numpy as np

    roles = [role for role in ("caller", "agent") if role in profiles]
    if len(roles) < 2:
        return None

    per_ms = sample_rate // 1000
    tracks = {role: np.array(samples, dtype="float32", copy=True) for role in roles}
    marked: list[dict[str, Any]] = []
    used = 0
    for span in overlaps:
        duration = int(span.get("duration_ms") or 0)
        if duration < MIN_SEPARATE_MS:
            marked.append({**span, "separated": False})
            continue

        start = max(0, int(span["start_ms"]) - PAD_MS)
        end = min(len(samples) // per_ms, int(span["start_ms"]) + duration + PAD_MS)
        window = samples[start * per_ms : end * per_ms]
        if len(window) < sample_rate // 2:
            marked.append({**span, "separated": False})
            continue

        try:
            streams = separator.streams(resample(window, sample_rate, separator.rate))
        except Exception:  # pragma: no cover - a broken model costs words, not the call
            logger.warning("separation failed at %d ms", span["start_ms"], exc_info=True)
            marked.append({**span, "separated": False})
            continue

        picked = _pick_streams(streams, window, roles, profiles, embed, separator, sample_rate)
        if picked is None:
            marked.append({**span, "separated": False})
            continue

        for role, patch in picked.items():
            _splice(tracks[role], patch, start * per_ms, per_ms)
        marked.append({**span, "separated": True})
        used += 1

    if not used:
        return None
    return tracks, marked


def _pick_streams(
    streams: list[Any],
    window: Any,
    roles: list[str],
    profiles: dict[str, Any],
    embed: Callable[[Any], Any],
    separator: VoiceSeparator,
    sample_rate: int,
) -> dict[str, Any] | None:
    """Assign the two streams to the two roles, and match their level.

    Scored as a PAIRING, not per stream: both streams can score highest against
    the same role, and assigning them independently would name one person twice
    - while the separator already guarantees they are two people.
    """
    import numpy as np

    usable = [resample(stream, separator.rate, sample_rate) for stream in streams]
    usable = [track for track in usable if float(np.abs(track).max()) > SILENT_PEAK]
    if len(usable) < 2:
        return None
    vectors = [embed(track / (float(np.abs(track).max()) or 1.0) * 0.9) for track in usable[:2]]
    scores = [[float(np.dot(vector, profiles[role])) for role in roles] for vector in vectors]
    straight = scores[0][0] + scores[1][1]
    crossed = scores[0][1] + scores[1][0]
    if abs(straight - crossed) < MIN_ASSIGN_MARGIN:
        return None
    order = (0, 1) if straight > crossed else (1, 0)

    # The separator returns whatever amplitude it likes. Pasted in unchanged,
    # the overlap ends up louder or quieter than its surroundings, and the VAD
    # downstream reads that step as a turn boundary. Match the window's RMS.
    reference = float(np.sqrt(np.mean(np.square(window)))) or 1.0
    picked: dict[str, Any] = {}
    for role, index in zip(roles, order, strict=True):
        track = usable[index]
        level = float(np.sqrt(np.mean(np.square(track)))) or 1.0
        picked[role] = np.clip(track * (reference / level), -1.0, 1.0)
    return picked


def _splice(target: Any, patch: Any, offset: int, per_ms: int) -> None:
    """Paste ``patch`` into ``target`` at ``offset``, crossfading both edges."""
    import numpy as np

    length = min(len(patch), len(target) - offset)
    if length <= 0:
        return
    fade = min(FADE_MS * per_ms, length // 2)
    ramp = np.ones(length, dtype="float32")
    if fade > 0:
        ramp[:fade] = np.linspace(0.0, 1.0, fade, dtype="float32")
        ramp[-fade:] = np.linspace(1.0, 0.0, fade, dtype="float32")
    window = target[offset : offset + length]
    target[offset : offset + length] = window * (1.0 - ramp) + patch[:length] * ramp


__all__ = [
    "FADE_MS",
    "MIN_ASSIGN_MARGIN",
    "MIN_SEPARATE_MS",
    "PAD_MS",
    "SeparatorUnavailable",
    "VoiceSeparator",
    "build_role_tracks",
    "resample",
]
