"""Chạy cả bộ mẫu qua `monosplit.separate` và chấm điểm từng ca.

Đây là benchmark CHẠY THẬT: nạp mô hình thật, giải mã tệp thật, đo đồng hồ
thật. Không có mẫu giả, không có mock — vì thứ cần biết ở đây là bản ghi có
thật thì máy tách được hay không, và tách hết một phút thoại mất bao lâu.

Cách chạy:

    MONOSPLIT_MODELS=$HOME/.cache/voice-diar \\
        uv run --with-editable '.[asr]' python -m benchmark.run
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from monosplit import MonoSpeakerSplitter, SeparationError, ensure_models, separate
from monosplit.transcribe import Transcriber, TranscriberUnavailable

BENCHMARK_DIR = Path(__file__).resolve().parent

DAT = "ĐẠT"
KHONG_DAT = "KHÔNG ĐẠT"
TU_CHOI_DUNG = "TỪ CHỐI ĐÚNG"


def cham_ca(ky_vong: dict, ket_qua: dict | None, loi: str | None) -> tuple[str, str]:
    """So kết quả một ca với kỳ vọng → (phán quyết, lý do)."""
    loi_mong_doi = ky_vong.get("loi")
    if loi_mong_doi:
        if loi == loi_mong_doi:
            return TU_CHOI_DUNG, f"từ chối đúng bằng mã `{loi}`"
        if loi:
            return KHONG_DAT, f"chờ mã `{loi_mong_doi}`, nhận `{loi}`"
        return KHONG_DAT, f"chờ bị từ chối bằng `{loi_mong_doi}` nhưng vẫn tách ra kết quả"

    if loi:
        return KHONG_DAT, f"ném lỗi `{loi}`"
    assert ket_qua is not None
    vai = {turn["speaker"] for turn in ket_qua["turns"]}
    if ky_vong.get("hai_vai") and len(vai) < 2:
        return KHONG_DAT, f"chỉ ra {len(vai)} vai, chờ 2"
    toi_thieu = ky_vong.get("so_luot_toi_thieu", 0)
    if len(ket_qua["turns"]) < toi_thieu:
        return KHONG_DAT, f"chỉ {len(ket_qua['turns'])} lượt, chờ tối thiểu {toi_thieu}"
    return DAT, ""


def _margin_trung_vi(turns: list[dict]) -> float:
    """Trung vị margin, bỏ các lượt lớp 3 không chấm (margin = 0)."""
    cham = [turn["margin"] for turn in turns if turn["margin"] > 0]
    return round(statistics.median(cham), 3) if cham else 0.0


def chay_mot_ca(ca: dict, splitter: MonoSpeakerSplitter, transcriber: Transcriber | None) -> dict:
    """Chạy một ca, trả về hàng báo cáo đã chấm điểm."""
    source = Path(ca["file"])
    hang: dict = {"file": source.name, "duong_dan": str(source), "mo_ta": ca["mo_ta"]}

    if not source.is_file():
        hang |= {"phan_quyet": KHONG_DAT, "ly_do": "không tìm thấy tệp", "giay": 0.0}
        return hang

    dong_ho = time.perf_counter()
    ket_qua: dict | None = None
    loi: str | None = None
    try:
        ket_qua = separate(source, splitter, transcriber).to_dict()
    except SeparationError as exc:
        loi = exc.code
        hang["thong_bao_loi"] = str(exc)
    hang["giay"] = round(time.perf_counter() - dong_ho, 2)

    if ket_qua is not None:
        turns = ket_qua["turns"]
        hang |= {
            "mode": ket_qua["mode"],
            "channels": ket_qua["channels"],
            "duration_ms": ket_qua["duration_ms"],
            "so_luot": len(turns),
            "luot_caller": sum(turn["speaker"] == "caller" for turn in turns),
            "luot_agent": sum(turn["speaker"] == "agent" for turn in turns),
            "margin_trung_vi": _margin_trung_vi(turns),
            "role_reason": ket_qua["role_reason"],
            "same_voice": ket_qua["same_voice"],
            "warnings": ket_qua["warnings"],
        }
    hang["loi"] = loi
    hang["phan_quyet"], ly_do = cham_ca(ca["ky_vong"], ket_qua, loi)
    hang["ly_do"] = ly_do
    return hang


COT = ["Ca", "Chế độ", "Giây", "Lượt", "Khách/Agent", "Margin", "Cùng giọng", "Căn cứ chọn vai", "Kết quả"]


def _o(hang: dict) -> list[str]:
    if hang.get("loi"):
        can_cu = f"lỗi `{hang['loi']}`"
    else:
        can_cu = (hang.get("role_reason") or "—").replace("|", "\\|").replace("\n", " ")
    return [
        hang["file"],
        hang.get("mode", "—"),
        f"{hang['giay']:.1f}",
        str(hang.get("so_luot", "—")),
        f"{hang['luot_caller']}/{hang['luot_agent']}" if "so_luot" in hang else "—",
        f"{hang['margin_trung_vi']:.3f}" if "margin_trung_vi" in hang else "—",
        "có" if hang.get("same_voice") else "không" if "so_luot" in hang else "—",
        can_cu,
        hang["phan_quyet"],
    ]


def bang_markdown(hangs: list[dict]) -> str:
    dong = [_o(hang) for hang in hangs]
    rong = [max(len(COT[i]), *(len(d[i]) for d in dong)) for i in range(len(COT))]
    def line(o: list[str]) -> str:
        return "| " + " | ".join(v.ljust(rong[i]) for i, v in enumerate(o)) + " |"
    return "\n".join(
        [line(COT), "|" + "|".join("-" * (r + 2) for r in rong) + "|", *(line(d) for d in dong)]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.run", description="Chấm điểm monosplit trên bộ mẫu thật."
    )
    parser.add_argument("--audio", type=Path, help="Thư mục chứa tệp mẫu, thay cho đường dẫn trong cases")
    parser.add_argument("--cases", type=Path, default=BENCHMARK_DIR / "cases.json")
    parser.add_argument("--no-asr", action="store_true", help="Bỏ chép lời — nhanh hơn, lớp 2 mất căn cứ")
    parser.add_argument("--models", type=Path, help="Thư mục chứa seg.onnx / emb.onnx")
    parser.add_argument("--out", type=Path, default=BENCHMARK_DIR / "report.json")
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.audio:
        # Người dùng đặt mẫu của họ ở chỗ khác: giữ nguyên tên tệp, đổi thư mục.
        for ca in cases:
            thay = args.audio / Path(ca["file"]).name
            if thay.is_file():
                ca["file"] = str(thay)

    seg, emb = ensure_models(args.models)
    splitter = MonoSpeakerSplitter(str(seg), str(emb))

    transcriber = None
    if not args.no_asr:
        try:
            transcriber = Transcriber()
        except TranscriberUnavailable as exc:
            print(f"cảnh báo: {exc} — chạy tiếp không có chữ")

    hangs = []
    for thu_tu, ca in enumerate(cases, 1):
        print(f"[{thu_tu}/{len(cases)}] {Path(ca['file']).name} …", flush=True)
        hangs.append(chay_mot_ca(ca, splitter, transcriber))

    dat = sum(hang["phan_quyet"] in (DAT, TU_CHOI_DUNG) for hang in hangs)
    tong_giay = sum(hang["giay"] for hang in hangs)
    tong_phut_audio = sum(hang.get("duration_ms", 0) for hang in hangs) / 60_000
    rtf = round(tong_giay / tong_phut_audio, 1) if tong_phut_audio else 0.0

    print()
    print(bang_markdown(hangs))
    print()
    print(f"**Đạt {dat}/{len(hangs)} ca** — {tong_giay:.1f} giây xử lý cho "
          f"{tong_phut_audio:.1f} phút audio tách được, tức {rtf} giây mỗi phút audio "
          f"(RTF {round(rtf / 60, 3)}).")
    for hang in hangs:
        if hang["phan_quyet"] == KHONG_DAT:
            print(f"- KHÔNG ĐẠT `{hang['file']}`: {hang['ly_do']}")

    tom_tat = {
        "so_ca": len(hangs),
        "so_ca_dat": dat,
        "tong_giay": round(tong_giay, 2),
        "tong_phut_audio": round(tong_phut_audio, 2),
        "giay_moi_phut_audio": rtf,
        "rtf": round(rtf / 60, 4),
        "co_asr": transcriber is not None,
    }
    args.out.write_text(
        json.dumps({"tom_tat": tom_tat, "cac_ca": hangs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nĐã ghi {args.out}")
    return 0 if dat == len(hangs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
