"""Recover the WORDS spoken while both people talked at once.

A single-channel recording carries one waveform, so the quieter speaker is
swallowed by the mix: running ASR on the mixture returns only the louder side.
This module separates a window around each overlap into two streams and
transcribes each one, which brings the missing words back.

Three limits are wired into the code, not exposed as options:

1. **Only overlaps long enough to separate.** LibriCSS
   ([arXiv:2001.11482](https://arxiv.org/abs/2001.11482)) measured 0% overlap
   and found separation made WER *worse*: 11.8 -> 12.7. The 30-case acceptance
   suite in this repo agrees: at 200 ms of overlap, 2 of 6 cases came out worse
   than not separating at all. Below ``MIN_SEPARATE_MS`` nothing is separated
   and the result says so instead of guessing.
2. **What comes out is RECOVERED text, not measured text.** The signature
   failure of cascaded separation -> ASR is *speaker leakage*: one person's
   words land in the other person's stream
   ([arXiv:2608.22196](https://arxiv.org/abs/2608.22196), Interspeech 2026,
   up to 71-77% WER on AMI). Whisper can also invent whole sentences on short
   clips ([arXiv:2402.08021](https://arxiv.org/abs/2402.08021)). So this text
   never merges into the transcript: it is returned separately, flagged, for a
   human to confirm by ear.
3. **Roles come from voice embeddings, or stay empty.** The two streams leave
   the separator unnamed (the permutation problem). Enrollment samples of the
   two roles decide which is which; when both sides sound like one voice -
   common in synthetic recordings where a single TTS voice reads both parts -
   ``role`` stays ``None`` rather than being guessed.

Measured on the 30-case acceptance suite (2026-09-13, SepFormer WHAMR 16 kHz
ONNX + Whisper large-v3 on the separated streams):

| Overlap  | CER on caller words | CER on agent words |
|----------|---------------------|--------------------|
| mixture  | 0.68                | 0.67               |
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


def recover_overlap_text(
    overlaps: list[dict[str, Any]],
    samples: Any,
    *,
    separator: VoiceSeparator,
    transcribe: Callable[[Any], str],
    profiles: dict[str, Any] | None = None,
    embed: Callable[[Any], Any] | None = None,
    sample_rate: int = 16_000,
) -> list[dict[str, Any]]:
    """Attach ``recovered`` to every overlap long enough to separate.

    ``samples`` is the mixed waveform as float32 at ``sample_rate`` - the same
    array the labelling layers work on, so overlap timestamps index into it
    directly with no conversion.

    ``profiles`` holds one voice embedding per role (``caller`` / ``agent``)
    taken from speech OUTSIDE any overlap. Without it, or when the two
    embeddings are too close to separate, the words still come back but
    ``role`` is left ``None``.

    Every returned overlap keeps its original keys. ``recovered is None`` means
    "could not separate" - a shorter overlap than ``MIN_SEPARATE_MS``, a
    separator failure, or no speech in either stream - which is not the same as
    "nobody said anything".
    """
    import numpy as np

    per_ms = sample_rate // 1000
    out: list[dict[str, Any]] = []
    for span in overlaps:
        duration = int(span.get("duration_ms") or 0)
        if duration < MIN_SEPARATE_MS:
            out.append({**span, "recovered": None})
            continue

        start = max(0, int(span["start_ms"]) - PAD_MS)
        end = int(span["start_ms"]) + duration + PAD_MS
        window = samples[start * per_ms : end * per_ms]
        if len(window) < sample_rate // 2:
            out.append({**span, "recovered": None})
            continue

        try:
            streams = separator.streams(resample(window, sample_rate, separator.rate))
        except Exception:  # pragma: no cover - a broken model costs words, not the call
            logger.warning("separation failed at %d ms", span["start_ms"], exc_info=True)
            out.append({**span, "recovered": None})
            continue

        recovered: list[dict[str, Any]] = []
        for stream in streams:
            track = resample(stream, separator.rate, sample_rate)
            peak = float(np.abs(track).max())
            if peak < SILENT_PEAK:
                continue
            text = transcribe(track / peak * 0.9).strip()
            if not text:
                continue
            recovered.append({"role": None, "text": text, "_track": track})

        if not recovered:
            out.append({**span, "recovered": None})
            continue

        assign_roles(recovered, profiles, embed)
        out.append(
            {**span, "recovered": [{"role": item["role"], "text": item["text"]} for item in recovered]}
        )
    return out


def assign_roles(
    recovered: list[dict[str, Any]],
    profiles: dict[str, Any] | None,
    embed: Callable[[Any], Any] | None,
) -> None:
    """Name each stream from voice embeddings, or leave both unnamed.

    Scored as a PAIRING, not per stream: two streams can both score highest
    against the same role, and assigning them independently would report one
    person twice - while the separator already guarantees they are two people.
    """
    import numpy as np

    if not profiles or embed is None or len(recovered) < 2:
        return
    roles = [role for role in ("caller", "agent") if role in profiles]
    if len(roles) < 2:
        return
    vectors = [embed(item["_track"]) for item in recovered[:2]]
    scores = [[float(np.dot(vector, profiles[role])) for role in roles] for vector in vectors]
    straight = scores[0][0] + scores[1][1]
    crossed = scores[0][1] + scores[1][0]
    if abs(straight - crossed) < MIN_ASSIGN_MARGIN:
        # One voice reading both parts, or a window too short to embed: the
        # words are worth keeping, the labels are not. Empty beats wrong -
        # a wrong label blames the wrong side.
        return
    order = (0, 1) if straight > crossed else (1, 0)
    for item, index in zip(recovered[:2], order, strict=True):
        item["role"] = roles[index]


__all__ = [
    "MIN_ASSIGN_MARGIN",
    "MIN_SEPARATE_MS",
    "PAD_MS",
    "SeparatorUnavailable",
    "VoiceSeparator",
    "assign_roles",
    "recover_overlap_text",
    "resample",
]
