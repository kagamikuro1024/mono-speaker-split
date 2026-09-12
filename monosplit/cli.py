"""Dòng lệnh: một tệp vào, bảng lượt nói ra.

Mọi thứ nặng (mô hình ONNX, Whisper) chỉ nạp khi thật sự cần, để `--help` và
lỗi tham số trả lời tức thì thay vì đợi vài giây nạp mô hình.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from monosplit.models import ensure_models
from monosplit.pipeline import Result, SeparationError, separate
from monosplit.speakers import MonoSpeakerSplitter, MonoSplitUnavailable
from monosplit.transcribe import Transcriber, TranscriberUnavailable

VAI = {"caller": "Khách", "agent": "Agent"}

# Cắt lời trong bảng cho khỏi tràn dòng; ai cần đủ chữ thì dùng --json.
LOI_TOI_DA = 64


def _dong_ho(ms: int) -> str:
    return f"{ms // 60000:d}:{ms % 60000 / 1000:06.3f}"


def _bang(result: Result) -> str:
    """Bảng thuần văn bản, cột co theo nội dung thật."""
    dau = ("STT", "Vai", "Bắt đầu", "Kết thúc", "Độ dài", "Margin", "Lời")
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
    # Đo bằng len() chứ không bằng bề rộng hiển thị: chữ Việt có dấu vẫn là một
    # ô trên terminal, nên len() đủ đúng ở đây.
    rong = [max(len(hang[c]) for hang in (dau, *dong)) for c in range(len(dau))]
    ke = lambda hang: "  ".join(o.ljust(w) for o, w in zip(hang, rong, strict=True)).rstrip()  # noqa: E731
    return "\n".join([ke(dau), "  ".join("-" * w for w in rong), *(ke(hang) for hang in dong)])


def _in_ket_qua(result: Result) -> None:
    print(_bang(result))
    print()
    print(f"Nguồn      : {result.source}  ({result.duration_ms / 1000:.1f}s, {result.channels} kênh)")
    print(f"Đường chạy : {result.mode}")
    print(f"Chọn vai   : {result.role_reason or '—'}")
    if result.same_voice:
        print("Lưu ý      : hai bên nghe như một giọng")
    for canh_bao in result.warnings:
        print(f"Cảnh báo   : {canh_bao}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="monosplit",
        description="Tách giọng khách / agent khỏi bản ghi cuộc gọi một kênh.",
    )
    parser.add_argument("audio", type=Path, help="tệp âm thanh cần tách")
    parser.add_argument("--json", action="store_true", help="in JSON thay vì bảng")
    parser.add_argument("--no-asr", action="store_true", help="bỏ chép lời cho nhanh")
    parser.add_argument("--models", type=Path, default=None, help="thư mục chứa seg.onnx / emb.onnx")
    parser.add_argument("--model", default="small", help="tên mô hình Whisper (mặc định: small)")
    parser.add_argument(
        "--requirement",
        action="append",
        default=[],
        metavar="CÂU",
        help="mô tả việc agent phải làm, giúp lớp 2 chọn vai; lặp lại được",
    )
    args = parser.parse_args(argv)

    if not args.audio.is_file():
        print(f"Không thấy tệp: {args.audio}", file=sys.stderr)
        return 1

    try:
        seg, emb = ensure_models(args.models)
        splitter = MonoSpeakerSplitter(str(seg), str(emb))
    except MonoSplitUnavailable as exc:
        print(f"Thiếu mô hình tách người nói: {exc}", file=sys.stderr)
        print("Cài: uv pip install 'monosplit'  — mô hình tự tải về ~/.cache/monosplit", file=sys.stderr)
        print("Hoặc trỏ MONOSPLIT_MODELS / --models tới thư mục có seg.onnx và emb.onnx.", file=sys.stderr)
        return 1

    transcriber = None
    if not args.no_asr:
        try:
            transcriber = Transcriber(args.model)
        except TranscriberUnavailable:
            print("Chưa cài faster-whisper nên không chép lời được.", file=sys.stderr)
            print("Cài: uv pip install 'monosplit[asr]'  — hoặc chạy lại với --no-asr.", file=sys.stderr)
            return 1

    try:
        result = separate(args.audio, splitter, transcriber, args.requirement)
    except SeparationError as exc:
        print(f"Không tách được: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _in_ket_qua(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
