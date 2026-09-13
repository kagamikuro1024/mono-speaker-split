"""Score the words recovered from overlaps, against per-side ground truth.

Answers one question: **when two people talk at once on a single-channel
recording, do the words of both sides come back?** Scored as CER against the
known text, not SI-SDR - this is about words, not about a clean waveform.

## The case set

Every case needs per-side ground truth, so build it from a TWO-channel source:
take one role per channel, shift the agent channel earlier by X ms to create
exactly X ms of overlap, then mix down to one channel. Shift the WHOLE channel
rather than splicing one turn - a splice leaves a seam, and the separator locks
onto the seam instead of the voice, which produces flattering numbers that do
not survive real audio.

``manifest.json`` next to the audio, one entry per case:

```json
{"cases": [{
  "file": "audio/a01_overlap200.wav",
  "group": "tts_overlap",
  "overlap_ms": 200,
  "overlap_start_ms": 3336,
  "agent_level_db": 0,
  "overlapping_turns": {
    "caller": {"text": "...", "start_ms": 0,    "end_ms": 3536},
    "agent":  {"text": "...", "start_ms": 3336, "end_ms": 7169}
  }
}]}
```

## Reading the numbers

Compare the `mixture -> separated` delta, not the absolute CER: the reference
is the whole turn while the scoring window only covers the overlap plus padding,
so text outside the window counts as missing on both sides equally.

Streams are paired to roles by whichever pairing scores better. That is
deliberate: the question here is "can the words be recovered", not "is the role
label right" - role assignment has its own threshold and its own tests.

Run:

    python -m benchmark.overlap ~/Downloads/benchmark-overlap-30 \\
        --separator ~/.cache/voice-models/sepformer16k.onnx --model large-v3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
import wave
from pathlib import Path

from monosplit.separate_voices import PAD_MS, VoiceSeparator, resample
from monosplit.transcribe import Transcriber

SAMPLE_RATE = 16_000


def load(path: Path):
    import numpy as np

    with wave.open(str(path)) as source:
        rate = source.getframerate()
        raw = np.frombuffer(source.readframes(source.getnframes()), dtype="int16")
    return raw.astype("float32") / 32768.0, rate


def write(path: Path, samples, rate: int) -> None:
    import numpy as np

    with wave.open(str(path), "wb") as dest:
        dest.setnchannels(1)
        dest.setsampwidth(2)
        dest.setframerate(rate)
        dest.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("int16").tobytes())


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return " ".join("".join(c for c in text if c.isalnum() or c.isspace()).split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate - Levenshtein over normalised characters."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.overlap",
        description="Score the words recovered from overlapping speech against ground truth.",
    )
    parser.add_argument("case_set", type=Path, help="directory holding manifest.json and audio/")
    parser.add_argument("--separator", type=Path, required=True, help="SepFormer ONNX model")
    parser.add_argument("--separator-rate", type=int, default=16_000, help="model sample rate")
    parser.add_argument("--model", default="small", help="Whisper model name (default: small)")
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = parser.parse_args(argv)

    import numpy as np

    manifest = json.loads((args.case_set / "manifest.json").read_text(encoding="utf-8"))
    separator = VoiceSeparator(args.separator, args.separator_rate)
    transcriber = Transcriber(args.model)
    work = args.case_set / ".work"
    work.mkdir(exist_ok=True)

    def text_of(samples) -> str:
        peak = float(np.abs(samples).max()) or 1.0
        clip = work / "clip.wav"
        write(clip, samples / peak * 0.9, SAMPLE_RATE)
        return " ".join(word.text for word in transcriber.words(clip, vad_filter=False)).strip()

    rows: list[dict] = []
    separation_seconds = audio_seconds = 0.0
    for case in manifest["cases"]:
        truth = case.get("overlapping_turns") or case.get("luot_chong")
        if not truth:
            continue  # control case with no ground-truth text
        caller_truth = (truth.get("caller") or truth["khach"])["text"]
        agent_truth = truth["agent"]["text"]
        overlap_ms = case["overlap_ms"]
        overlap_start = case.get("overlap_start_ms") or case.get("overlap_bat_dau_ms")

        samples, rate = load(args.case_set / case["file"])
        start = max(0, overlap_start - PAD_MS)
        end = overlap_start + max(0, overlap_ms) + PAD_MS
        window = samples[int(start / 1000 * rate) : int(end / 1000 * rate)]

        mixture_text = text_of(resample(window, rate, SAMPLE_RATE))

        began = time.perf_counter()
        streams = separator.streams(resample(window, rate, separator.rate))
        separation_seconds += time.perf_counter() - began
        audio_seconds += len(window) / rate
        texts = [text_of(resample(stream, separator.rate, SAMPLE_RATE)) for stream in streams]

        # Best pairing wins - see "Reading the numbers" in the module docstring.
        best = min(
            ((cer(caller_truth, texts[a]) + cer(agent_truth, texts[b]), a, b)
             for a in range(len(texts)) for b in range(len(texts)) if a != b),
            default=(0.0, 0, 0),
        )
        _score, caller_index, agent_index = best
        row = {
            "case": Path(case["file"]).stem,
            "group": case.get("group") or case.get("nhom"),
            "overlap_ms": overlap_ms,
            "agent_level_db": case.get("agent_level_db") or case.get("muc_agent_so_voi_khach_db") or 0,
            "cer_mixture_caller": cer(caller_truth, mixture_text),
            "cer_mixture_agent": cer(agent_truth, mixture_text),
            "cer_separated_caller": cer(caller_truth, texts[caller_index]),
            "cer_separated_agent": cer(agent_truth, texts[agent_index]),
            "separated_caller": texts[caller_index],
            "separated_agent": texts[agent_index],
            "mixture": mixture_text,
        }
        rows.append(row)
        print(
            f"{row['case']:22s} {overlap_ms:>5} ms | caller "
            f"{row['cer_mixture_caller']:.2f} -> {row['cer_separated_caller']:.2f} | agent "
            f"{row['cer_mixture_agent']:.2f} -> {row['cer_separated_agent']:.2f}",
            flush=True,
        )

    if not rows:
        print("no case carried ground-truth text")
        return 1

    def mean(key: str) -> float:
        return statistics.mean(row[key] for row in rows)

    print(
        f"\n{len(rows)} cases | caller CER {mean('cer_mixture_caller'):.3f} -> "
        f"{mean('cer_separated_caller'):.3f} | agent CER {mean('cer_mixture_agent'):.3f} -> "
        f"{mean('cer_separated_agent'):.3f} | separation RTF "
        f"{separation_seconds / audio_seconds:.2f}"
    )
    for group in sorted({row["group"] for row in rows}):
        subset = [row for row in rows if row["group"] == group]
        improved = sum(
            1
            for row in subset
            if row["cer_separated_caller"] + row["cer_separated_agent"]
            < row["cer_mixture_caller"] + row["cer_mixture_agent"] - 0.02
        )
        print(f"  {group:14s} n={len(subset):<3} improved {improved}/{len(subset)}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {"rows": rows, "separation_rtf": separation_seconds / audio_seconds},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"report: {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
