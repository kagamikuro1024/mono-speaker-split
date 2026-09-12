"""Đường chạy đầy đủ: một tệp âm thanh vào, danh sách lượt nói có nhãn vai ra.

Bốn bước, và bước nào cũng có thể nói "không làm được" thay vì đoán bừa:

1. ``audio``  — đọc tệp, quyết định nó là một luồng hay hai luồng thật.
2. ``speakers`` lớp 1 — phân cụm mù, cắt quãng tại chỗ đổi người.
3. ``transcribe`` — chép lời, mốc theo từ (tuỳ chọn).
4. ``speakers`` lớp 2 + 3 — chữ chọn vai, vân giọng chấm lại từng quãng.

Nguyên tắc xuyên suốt: **nhãn trên bản ghi một kênh là SUY ĐOÁN**, và kết quả
phải nói ra điều đó. Một bản gỡ băng trông chắc chắn mà sai vai còn tệ hơn một
bản ghi rõ "chưa chắc, soát lại".
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from monosplit import audio
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
from monosplit.transcribe import Transcriber, text_in_range

# Khung VAD và ngưỡng lượt — cùng bộ số với đường hai kênh để hai đường cho ra
# cùng một dòng thời gian trên cùng một bản ghi.
VAD_FRAME_MS = 20
MIN_SPEECH_MS = 200
TURN_MERGE_GAP_MS = 400


class SeparationError(RuntimeError):
    """Không tách được — kèm mã lỗi để phía gọi hiện đúng câu cho người dùng."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


ERRORS: dict[str, str] = {
    "khong_co_tieng_noi": "Bản ghi gần như không có tiếng nói nào.",
    "khong_tach_nguoi_noi": "Chỉ nghe ra một giọng, không tách được hai vai.",
    "thieu_mo_hinh": "Chưa có mô hình tách người nói (seg.onnx / emb.onnx).",
}


@dataclass
class Turn:
    """Một lượt nói: ai, từ đâu đến đâu, nói gì, và tin được bao nhiêu."""

    speaker: Literal["caller", "agent"]
    start_ms: int
    end_ms: int
    text: str = ""
    #: Khoảng cách cosine giữa hai vân giọng ở quãng này. 0 = lớp 3 không chấm
    #: (thiếu mẫu), dưới 0,10 = hai giọng quá sát, nhãn giữ theo lớp 1.
    margin: float = 0.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class Result:
    """Kết quả một lần tách, đủ để vẽ giao diện và để chấm điểm benchmark."""

    source: str
    duration_ms: int
    channels: int
    #: Vì sao đi đường một kênh: `mono`, `stereo_trung_nhau`, `stereo_mot_ben_cam`.
    mode: str
    turns: list[Turn] = field(default_factory=list)
    #: Câu giải thích lớp 2 đã chọn agent bằng căn cứ nào.
    role_reason: str = ""
    #: Hai bên nghe như một giọng — nhãn gán luân phiên theo lượt.
    same_voice: bool = False
    #: Cảnh báo cho người đọc kết quả.
    warnings: list[str] = field(default_factory=list)
    #: Các lần NGHI hai bên cùng nói, đọc từ lớp powerset của mô hình phân đoạn.
    #: Mỗi mục: `start_ms`, `duration_ms`, `who_cut_in`, `who_yielded` (tên vai,
    #: `None` khi không suy được hướng).
    #:
    #: PHỎNG ĐOÁN, không phải phép đo — xem `speakers.overlap_spans`. Dùng số
    #: lần và mốc; đừng cộng `duration_ms` thành tổng số giây nói chồng.
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
) -> Result:
    """Tách một tệp thành các lượt nói có nhãn khách / agent."""
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
                    "Tệp này có hai kênh tách vai thật — dùng thẳng từng kênh, "
                    "không cần đoán ai với ai.",
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
        texts = [text_in_range(words, start, end, TURN_MERGE_GAP_MS) for start, end, _c in pieces]

        labels, reason = _label(splitter, samples, pieces, texts, requirements)
        warnings: list[str] = []
        if same_voice:
            # Bản ghi tổng hợp hay dùng CÙNG MỘT giọng cho cả hai vai: vân giọng
            # không phân biệt được ai với ai. Dữ kiện thật còn lại là THỨ TỰ
            # LƯỢT — nhưng chỉ dùng được khi chính lời nói cho thấy có hai vai,
            # vì "một giọng" cũng đúng với bản ghi chỉ có một người nói.
            spoken = [(cluster, text) for (_s, _e, cluster), text in zip(pieces, texts, strict=True)]
            if not co_bang_chung_hai_vai(spoken, requirements):
                raise SeparationError("khong_tach_nguoi_noi", ERRORS["khong_tach_nguoi_noi"])
            warnings.append("hai bên nghe như một giọng — nhãn gán luân phiên theo lượt")

        turns = [
            Turn(label.speaker, label.start_ms, label.end_ms, text, label.margin)
            for label, text in zip(labels, texts, strict=True)
        ]

        # Cụm là số, người đọc cần tên vai: lấy mapping từ chính nhãn lớp 2 —
        # cụm nào đóng góp nhiều thời lượng nhất cho một vai thì thuộc vai đó.
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
    """Lớp 1: phân cụm mù, cắt quãng tại chỗ đổi người, cứu nếu gom hụt.

    Phần tử thứ ba là các khoảng hai cụm cùng hoạt động, đọc TRƯỚC khi
    ``split_runs`` cắt chúng thành mảnh kề nhau.
    """
    clusters = splitter.clusters(samples)
    spans = overlap_spans(clusters)
    pieces = split_runs(runs, clusters)
    if len({piece[2] for piece in pieces}) >= 2:
        return pieces, False, spans
    # Phân cụm mù nuốt mất một bên — hay gặp khi một bên chỉ "ừ", "dạ". Thử lại
    # bằng vân giọng trước khi kết luận chỉ có một giọng.
    rescued = splitter.split_by_voice(samples, pieces)
    if rescued is not None and len({piece[2] for piece in rescued}) >= 2:
        return rescued, False, spans
    # Gán luân phiên thì số cụm hết nghĩa: giữ mốc, bỏ hướng chen — suy "ai chen
    # ai" từ một mapping đã hỏng là bịa.
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
    """Lớp 2 (chữ chọn vai) và lớp 3 (vân giọng chấm lại từng quãng)."""
    import numpy as np

    agent_cluster, reason = pick_agent_cluster(
        [(cluster, text) for (_s, _e, cluster), text in zip(pieces, texts, strict=True)],
        requirements,
    )

    # Mẫu giọng: mảnh DÀI NHẤT của mỗi cụm. Mảnh dài thì phần lẫn giọng bên kia
    # bị pha loãng, nên vector ra vẫn là của đúng người. Cụm chỉ có mảnh ngắn
    # vẫn phải lấy mẫu: thiếu một mẫu là mất hẳn lớp 3, mà đó đúng là lúc lớp 1
    # đã lệch và cần lớp 3 kéo về nhất.
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
            # Vân giọng chỉ được lật nhãn khi nó CHẮC. Sát nhau thì giữ nhãn của
            # lớp phân cụm: đổi nhãn bằng một con số 0,02 là tung đồng xu rồi
            # gọi đó là kết quả đo.
            if margin >= MIN_COSINE_MARGIN:
                speaker = "agent" if winner == agent_cluster else "caller"
        labels.append(Labelled(start, end, speaker, margin))
    return labels, reason


__all__ = [
    "ERRORS",
    "MonoSplitUnavailable",
    "Result",
    "SeparationError",
    "Transcriber",
    "Turn",
    "separate",
]
