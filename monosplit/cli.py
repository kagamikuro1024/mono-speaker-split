"""Command line: one file in, a table of turns out.

Everything heavy (the ONNX models, Whisper) is loaded only when genuinely
needed, so `--help` and argument errors answer instantly instead of waiting
seconds for a model to load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from monosplit.models import ensure_models
from monosplit.pipeline import Result, SeparationError, separate
from monosplit.separate_voices import SeparatorUnavailable, VoiceSeparator
from monosplit.speakers import MonoSpeakerSplitter, MonoSplitUnavailable
from monosplit.transcribe import Transcriber, TranscriberUnavailable

VAI = {"caller": "Caller", "agent": "Agent"}

# Truncate the text in the table so rows do not wrap; use --json for the full text.
LOI_TOI_DA = 64


def _dong_ho(ms: int) -> str:
    return f"{ms // 60000:d}:{ms % 60000 / 1000:06.3f}"


def _bang(result: Result) -> str:
    """Plain-text table, columns sized to the actual content."""
    dau = ("#", "Role", "Start", "End", "Length", "Margin", "Text")
    dong = [
        (
            str(i),
            VAI.get(turn.speaker, turn.speaker),
            _dong_ho(turn.start_ms),
            _dong_ho(turn.end_ms),
            f"{turn.duration_ms / 1000:.2f}s",
            f"{turn.margin:.3f}",
            turn.text if len(turn.text) <= LOI_TOI_DA else turn.text[: LOI_TOI_DA - 1] + "…",
        )
        for i, turn in enumerate(result.turns, 1)
    ]
    # Measured with len() rather than display width: Vietnamese diacritics still
    # take one terminal cell, so len() is accurate enough here.
    rong = [max(len(hang[c]) for hang in (dau, *dong)) for c in range(len(dau))]
    ke = lambda hang: "  ".join(o.ljust(w) for o, w in zip(hang, rong, strict=True)).rstrip()  # noqa: E731
    return "\n".join([ke(dau), "  ".join("-" * w for w in rong), *(ke(hang) for hang in dong)])


def _in_ket_qua(result: Result) -> None:
    print(_bang(result))
    print()
    print(f"Source     : {result.source}  ({result.duration_ms / 1000:.1f}s, {result.channels} channels)")
    print(f"Route      : {result.mode}")
    print(f"Role choice: {result.role_reason or '—'}")
    if result.overlaps:
        # Count and timestamps, NOT a total in seconds: the model's boundaries are
        # fuzzy to ±62 ms, so summing them invents a number more precise than
        # anything that was measured.
        print(f"Suspected overlap: {len(result.overlaps)} time(s) (inferred, not measured)")
        for span in result.overlaps:
            ai = span["who_cut_in"]
            huong = f" — {VAI[ai]} cut in" if ai in VAI else ""
            # Mark where the words came off a separated stream rather than the
            # mixture. The separator can leak one voice into the other stream,
            # so this is where a human should listen before trusting the text.
            tach = " · voices separated to re-read the words" if span.get("separated") else ""
            print(f"             {span['start_ms'] / 1000:.1f}s{huong}{tach}")
    if result.same_voice:
        print("Note       : both sides sound like one voice")
    for canh_bao in result.warnings:
        print(f"Warning    : {canh_bao}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="monosplit",
        description="Split the caller / agent voices out of a single-channel call recording.",
    )
    parser.add_argument("audio", type=Path, help="audio file to split")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--no-asr", action="store_true", help="skip transcription for speed")
    parser.add_argument("--models", type=Path, default=None, help="directory holding seg.onnx / emb.onnx")
    parser.add_argument("--model", default="small", help="Whisper model name (default: small)")
    parser.add_argument(
        "--requirement",
        action="append",
        default=[],
        metavar="SENTENCE",
        help="describe what the agent must do, helps layer 2 pick the roles; repeatable",
    )
    parser.add_argument(
        "--separator",
        type=Path,
        default=None,
        metavar="MODEL.onnx",
        help="SepFormer ONNX model; recovers the words spoken during overlaps "
        "(measured: caller CER 0.68 -> 0.33 on the 30-case suite)",
    )
    parser.add_argument(
        "--separator-rate",
        type=int,
        default=16_000,
        help="sample rate the separator was trained at (default: 16000). A wrong "
        "value still produces audio, just pitch-shifted and unreadable by ASR",
    )
    args = parser.parse_args(argv)

    if not args.audio.is_file():
        print(f"File not found: {args.audio}", file=sys.stderr)
        return 1

    try:
        seg, emb = ensure_models(args.models)
        splitter = MonoSpeakerSplitter(str(seg), str(emb))
    except MonoSplitUnavailable as exc:
        print(f"Missing speaker diarization model: {exc}", file=sys.stderr)
        print("Install: uv pip install 'monosplit'  — models download to ~/.cache/monosplit", file=sys.stderr)
        print("Or point MONOSPLIT_MODELS / --models at a folder with seg.onnx and emb.onnx.", file=sys.stderr)
        return 1

    transcriber = None
    if not args.no_asr:
        try:
            transcriber = Transcriber(args.model)
        except TranscriberUnavailable:
            print("faster-whisper is not installed, so transcription is unavailable.", file=sys.stderr)
            print("Install: uv pip install 'monosplit[asr]'  — or rerun with --no-asr.", file=sys.stderr)
            return 1

    voice_separator = None
    if args.separator is not None:
        if transcriber is None:
            print("--separator needs transcription; drop --no-asr.", file=sys.stderr)
            return 1
        try:
            voice_separator = VoiceSeparator(args.separator, args.separator_rate)
        except SeparatorUnavailable as exc:
            print(f"Cannot load the separator: {exc}", file=sys.stderr)
            return 1

    try:
        result = separate(args.audio, splitter, transcriber, args.requirement, voice_separator)
    except SeparationError as exc:
        print(f"Could not split: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _in_ket_qua(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
