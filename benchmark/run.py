"""Run the whole sample set through `monosplit.separate` and score every case.

This benchmark RUNS FOR REAL: it loads the real models, decodes real files, and reads a real
clock. No fake samples, no mocks — because what has to be known here is whether a real
recording can be separated by the machine, and how long separating a minute of speech takes.

How to run:

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

DAT = "PASS"
KHONG_DAT = "FAIL"
TU_CHOI_DUNG = "CORRECTLY REJECTED"


def cham_ca(ky_vong: dict, ket_qua: dict | None, loi: str | None) -> tuple[str, str]:
    """Compare one case result against its expectation -> (verdict, reason)."""
    loi_mong_doi = ky_vong.get("loi")
    if loi_mong_doi:
        if loi == loi_mong_doi:
            return TU_CHOI_DUNG, f"correctly rejected with code `{loi}`"
        if loi:
            return KHONG_DAT, f"expected code `{loi_mong_doi}`, got `{loi}`"
        return KHONG_DAT, f"expected rejection with `{loi_mong_doi}` but a result was separated"

    if loi:
        return KHONG_DAT, f"raised error `{loi}`"
    assert ket_qua is not None
    vai = {turn["speaker"] for turn in ket_qua["turns"]}
    if ky_vong.get("hai_vai") and len(vai) < 2:
        return KHONG_DAT, f"found only {len(vai)} role(s), expected 2"
    toi_thieu = ky_vong.get("so_luot_toi_thieu", 0)
    if len(ket_qua["turns"]) < toi_thieu:
        return KHONG_DAT, f"only {len(ket_qua['turns'])} turns, expected at least {toi_thieu}"
    return DAT, ""


def _margin_trung_vi(turns: list[dict]) -> float:
    """Median margin, dropping the unscored layer-3 turns (margin = 0)."""
    cham = [turn["margin"] for turn in turns if turn["margin"] > 0]
    return round(statistics.median(cham), 3) if cham else 0.0


def chay_mot_ca(ca: dict, splitter: MonoSpeakerSplitter, transcriber: Transcriber | None) -> dict:
    """Run one case, return the scored report row."""
    source = Path(ca["file"])
    hang: dict = {"file": source.name, "duong_dan": str(source), "mo_ta": ca["mo_ta"]}

    if not source.is_file():
        hang |= {"phan_quyet": KHONG_DAT, "ly_do": "file not found", "giay": 0.0}
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


COT = ["Case", "Mode", "Seconds", "Turns", "Caller/Agent", "Margin", "Same voice", "Role evidence", "Result"]


def _o(hang: dict) -> list[str]:
    if hang.get("loi"):
        can_cu = f"error `{hang['loi']}`"
    else:
        can_cu = (hang.get("role_reason") or "—").replace("|", "\\|").replace("\n", " ")
    return [
        hang["file"],
        hang.get("mode", "—"),
        f"{hang['giay']:.1f}",
        str(hang.get("so_luot", "—")),
        f"{hang['luot_caller']}/{hang['luot_agent']}" if "so_luot" in hang else "—",
        f"{hang['margin_trung_vi']:.3f}" if "margin_trung_vi" in hang else "—",
        "yes" if hang.get("same_voice") else "no" if "so_luot" in hang else "—",
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
        prog="python -m benchmark.run", description="Score monosplit on the real sample set."
    )
    parser.add_argument(
        "--audio", type=Path, help="Directory holding the sample files, replacing the paths in cases"
    )
    parser.add_argument("--cases", type=Path, default=BENCHMARK_DIR / "cases.json")
    parser.add_argument(
        "--no-asr", action="store_true", help="Skip transcription - faster, layer 2 loses its evidence"
    )
    parser.add_argument("--models", type=Path, help="Directory holding seg.onnx / emb.onnx")
    parser.add_argument("--out", type=Path, default=BENCHMARK_DIR / "report.json")
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.audio:
        # The user keeps their samples elsewhere: keep the file names, swap the directory.
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
            print(f"warning: {exc} — continuing without text")

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
    print(f"**Passed {dat}/{len(hangs)} cases** — {tong_giay:.1f} seconds of processing for "
          f"{tong_phut_audio:.1f} minutes of separated audio, i.e. {rtf} seconds per minute of audio "
          f"(RTF {round(rtf / 60, 3)}).")
    for hang in hangs:
        if hang["phan_quyet"] == KHONG_DAT:
            print(f"- FAIL `{hang['file']}`: {hang['ly_do']}")

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
    print(f"\nWrote {args.out}")
    return 0 if dat == len(hangs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
