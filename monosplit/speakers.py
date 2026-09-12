"""Gán nhãn khách / agent cho bản ghi MỘT kênh.

Bản ghi hai kênh thì không cần gì cả: kênh nào là ai do người dùng khai. Bản ghi
một kênh — khách ghi bằng điện thoại trong xe, tổng đài xuất mono — thì phải suy
ra, và suy sai là chấm oan agent bằng lời của khách.

Ba lớp, đúng thứ tự, mỗi lớp sửa cái lớp trước không làm được:

1. **Phân cụm mù** (``sherpa-onnx``: pyannote segmentation 3.0 + vân giọng
   ERes2Net, đều ONNX) chia bản ghi thành hai cụm giọng. Lớp này biết *có hai
   người* và *đổi người lúc nào*, nhưng không biết ai là ai, và hay nuốt một
   lượt ngắn vào lượt dài bên cạnh.
2. **Chọn cụm nào là agent bằng LỜI, không bằng giọng.** Giọng agent đổi theo
   cấu hình của từng người dùng nên không ghim được; còn vai thì lộ ra trong
   chữ: agent là bên đọc lại giá trị cần xác nhận và nói những câu của tổng đài.
   Đây là chỗ duy nhất quyết "ai là ai".
3. **Chấm lại từng quãng bằng vân giọng** của chính hai cụm đó (lối rút gọn của
   Target-Speaker VAD, Medennikov 2020): lấy đoạn dài nhất mỗi cụm làm mẫu, rồi
   so cosine cho từng quãng VAD. Lớp này kéo về đúng bên những lượt ngắn mà lớp
   1 đã nuốt.

Cái KHÔNG làm được và không được giả vờ làm được: hai người cùng nói trên một
kênh thì chỉ còn một luồng sóng âm, nên nói chồng / cướp lời phải báo "chưa đo
được" chứ không báo 0.
"""

from __future__ import annotations

import logging
import re
from array import array
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Hai lượt cùng cụm cách nhau dưới mức này là một lượt. Phải cùng con số với
# ``pipeline.TURN_MERGE_GAP_MS`` — hai chỗ tách lượt mà lệch nhau thì cùng một
# bản ghi ra hai dòng thời gian khác nhau.
MERGE_GAP_MS = 400

# Khoảng cách cosine tối thiểu để tin nhãn của lớp 3. Dưới mức này là hai giọng
# quá giống nhau (hoặc quãng quá ngắn): giữ nhãn của lớp 1 và hạ độ tin cậy.
MIN_COSINE_MARGIN = 0.10

# Đoạn ngắn hơn mức này không đủ để lấy vân giọng: ERes2Net cần ít nhất chừng
# một giây tiếng nói mới ra vector ổn định.
MIN_ENROLL_MS = 1_000

# Khi một cụm không có mảnh nào đủ một giây: vẫn lấy mảnh dài nhất từ mức này
# trở lên. Mẫu ngắn thì vector kém chắc, nhưng ngưỡng cosine ở lớp 3 vẫn chặn —
# còn không có mẫu thì lớp 3 tắt hẳn, tức là mất luôn cơ hội sửa lớp 1.
MIN_SAMPLE_MS = 400

# Hai mảnh xa nhau nhất mà cosine vẫn trên mức này thì bản ghi chỉ có MỘT người.
# Đo trên bộ nghiệm thu 30 ca: cùng người 0,54 tới 0,74, khác người 0,06 tới 0,27.
# Lấy 0,40 cho vào giữa hai khoảng; đo lại khi có bản ghi thật trong xe.
MAX_SAME_VOICE_COSINE = 0.40

# Mảnh đủ để ĐEM RA SO trong bước cứu. Ngắn hơn mức lấy mẫu bình thường vì cái
# hay bị nuốt đúng là lượt "ừ", "dạ" — bỏ nó đi thì không còn gì để cứu.
MIN_RESCUE_MS = 180

# Câu cửa miệng của hai bên. Dùng khi kịch bản không khai giá trị nào phải đọc
# lại — tức là không có căn cứ chắc hơn. Chấm theo HIỆU hai bên chứ không chỉ
# đếm phía agent: "cảm ơn" một mình không nói lên ai là ai, "giúp tôi" thì có.
AGENT_CUES = (
    "tổng đài", "xin nghe", "em xin", "dạ em", "bên em", "quý khách",
    "em hỗ trợ", "em kiểm tra", "em xác nhận", "cảm ơn anh", "cảm ơn chị",
    "dạ", "vâng", " ạ",
)

# Người gọi là bên NHỜ VIỆC: câu của họ có người nhận lệnh ở cuối.
CALLER_CUES = (
    "giúp tôi", "cho tôi", "tôi muốn", "tôi cần", "em ơi", "a lô",
    "gọi cho tôi", "của tôi",
)


class MonoSplitUnavailable(RuntimeError):
    """Thiếu mô hình hoặc thư viện: gọi được nhưng không chạy được."""


@dataclass(frozen=True)
class Labelled:
    start_ms: int
    end_ms: int
    speaker: Literal["caller", "agent"]
    # Khoảng cách cosine giữa hai giả thuyết. Càng nhỏ càng đáng ngờ; giao diện
    # đọc con số này để nói "nhãn suy đoán" thay vì im lặng.
    margin: float


def _norm(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.casefold())


def speech_energy_runs(
    pcm: bytes, frame_ms: int, min_speech_ms: int, bytes_per_ms: int
) -> list[dict[str, int]]:
    """Quãng có tiếng nói, ngưỡng suy từ CHÍNH bản ghi.

    Lấy phân vị 20 của năng lượng khung làm nền rồi nhân ba. Ghim một con số
    tuyệt đối thì bản ghi thu nhỏ tiếng thành im hết, còn bản ghi ồn thì khoảng
    lặng nào cũng thành tiếng nói.
    """
    frame_bytes = frame_ms * bytes_per_ms
    frames = [pcm[at : at + frame_bytes] for at in range(0, len(pcm) - frame_bytes + 1, frame_bytes)]
    if not frames:
        return []
    levels = [max(abs(value) for value in array("h", frame)) for frame in frames]
    floor = sorted(levels)[len(levels) // 5]
    threshold = max(floor * 3, 200)
    runs: list[dict[str, int]] = []
    for index, level in enumerate(levels):
        if level < threshold:
            continue
        start, end = index * frame_ms, index * frame_ms + frame_ms
        if runs and start - runs[-1]["end_ms"] < MERGE_GAP_MS:
            runs[-1]["end_ms"] = end
            continue
        runs.append({"start_ms": start, "end_ms": end})
    return [run for run in runs if run["end_ms"] - run["start_ms"] >= min_speech_ms]


class MonoSpeakerSplitter:
    """Ba lớp ở docstring đầu tệp. Mô hình nạp một lần, dùng lại cho mọi ca."""

    def __init__(self, segmentation_model: str, embedding_model: str) -> None:
        for path in (segmentation_model, embedding_model):
            if not Path(path).is_file():
                raise MonoSplitUnavailable(f"thiếu mô hình tách người nói: {path}")
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - môi trường thiếu gói
            raise MonoSplitUnavailable("chưa cài sherpa-onnx") from exc
        self._sherpa = sherpa_onnx
        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(
            sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=segmentation_model
                    ),
                    num_threads=4,
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=embedding_model, num_threads=4
                ),
                # Hai bên: khách và agent. Để máy tự đoán số người thì một tiếng
                # ho của người ngồi cạnh cũng thành người thứ ba.
                clustering=sherpa_onnx.FastClusteringConfig(num_clusters=2),
                min_duration_on=0.2,
                min_duration_off=0.3,
            )
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_model, num_threads=4)
        )

    # ── lớp 1 ──────────────────────────────────────────────────────────────
    def clusters(self, samples: Any) -> list[tuple[int, int, int]]:
        """(start_ms, end_ms, cụm) của cả bản ghi, đã sắp theo thời gian."""
        result = self._diarizer.process(samples).sort_by_start_time()
        return [
            (round(item.start * 1000), round(item.end * 1000), int(item.speaker))
            for item in result
        ]

    # ── lớp 3 ──────────────────────────────────────────────────────────────
    def embed(self, samples: Any) -> Any:
        import numpy as np

        stream = self._extractor.create_stream()
        stream.accept_waveform(16000, samples)
        stream.input_finished()
        vector = np.asarray(self._extractor.compute(stream), dtype="float32")
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    # ── lớp 1b ─────────────────────────────────────────────────────────────
    def split_by_voice(
        self, samples: Any, pieces: list[tuple[int, int, int]]
    ) -> list[tuple[int, int, int]] | None:
        """Cứu khi phân cụm mù gộp cả hai người thành một.

        Xảy ra khi một bên chỉ nói vài lượt rất ngắn, hoặc hai giọng gần nhau.
        Lấy hai mảnh XA NHAU NHẤT theo vân giọng làm mốc rồi chia phần còn lại
        theo mốc gần hơn. ``None`` = đúng là một người, đừng bịa ra người thứ hai.
        """
        import numpy as np

        usable = [piece for piece in pieces if piece[1] - piece[0] >= MIN_RESCUE_MS]
        if len(usable) < 2:
            return None
        vectors = {
            piece: self.embed(samples[piece[0] * 16 : piece[1] * 16]) for piece in usable
        }
        # Cặp xa nhau nhất phải có ít nhất một mảnh đủ dài: hai mảnh vụn lệch
        # nhau là chuyện thường của vân giọng, dựng người thứ hai từ đó là bịa.
        pairs = [
            (a, b)
            for index, a in enumerate(usable)
            for b in usable[index + 1 :]
            if max(a[1] - a[0], b[1] - b[0]) >= MIN_SAMPLE_MS
        ]
        if not pairs:
            return None
        far = min(pairs, key=lambda pair: float(np.dot(vectors[pair[0]], vectors[pair[1]])))
        if float(np.dot(vectors[far[0]], vectors[far[1]])) > MAX_SAME_VOICE_COSINE:
            return None
        seeds = [vectors[far[0]], vectors[far[1]]]
        out: list[tuple[int, int, int]] = []
        for piece in pieces:
            vector = vectors.get(piece)
            if vector is None:
                out.append((piece[0], piece[1], out[-1][2] if out else 0))
                continue
            scores = [float(np.dot(vector, seed)) for seed in seeds]
            out.append((piece[0], piece[1], 0 if scores[0] >= scores[1] else 1))
        return out



# Mảnh ngắn hơn mức này sau khi cắt theo ranh giới cụm là vụn: nhập lại vào
# mảnh bên cạnh. Một "lượt" 120 ms không phải lượt nói, nó là tiếng đệm.
MIN_PIECE_MS = 300


def split_runs(
    runs: list[dict[str, int]], clusters: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """Cắt quãng có tiếng tại chỗ ĐỔI NGƯỜI, trả (start, end, cụm).

    Không có bước này thì một quãng khách-rồi-agent liền mạch (hai bên nói đè
    nhau, VAD không thấy khoảng lặng nào để cắt) đi nguyên khối vào một nhãn
    duy nhất: mất hẳn một lượt, và độ trễ đáp của lượt đó biến mất theo.

    Gộp chỉ xảy ra TRONG một quãng. Hai quãng cách nhau bởi khoảng lặng là hai
    lượt kể cả khi cùng cụm — nhập chúng lại là nuốt mất lượt nằm giữa mà lớp
    vân giọng lẽ ra còn cơ hội chấm lại.
    """
    pieces: list[tuple[int, int, int]] = []
    for run in runs:
        marks = {run["start_ms"], run["end_ms"]}
        for start, end, _cluster in clusters:
            for mark in (start, end):
                if run["start_ms"] < mark < run["end_ms"]:
                    marks.add(mark)
        edges = sorted(marks)
        cut: list[tuple[int, int, int]] = []
        for left, right in pairwise(edges):
            cluster, best = -1, 0
            for start, end, owner in clusters:
                overlap = min(end, right) - max(start, left)
                if overlap > best:
                    cluster, best = owner, overlap
            if cluster < 0:
                cluster = cut[-1][2] if cut else (pieces[-1][2] if pieces else 0)
            if cut and (right - left < MIN_PIECE_MS or cut[-1][2] == cluster):
                cut[-1] = (cut[-1][0], right, cut[-1][2])
                continue
            cut.append((left, right, cluster))
        while len(cut) > 1 and cut[0][1] - cut[0][0] < MIN_PIECE_MS:
            cut[1] = (cut[0][0], cut[1][1], cut[1][2])
            cut.pop(0)
        pieces.extend(cut)
    return pieces


def co_bang_chung_hai_vai(spoken: list[tuple[int, str]], requirements: list[str]) -> bool:
    """Có thật hai vai trong lời nói không — hỏi khi GIỌNG không trả lời được.

    Bản ghi tổng hợp hay để cả khách lẫn agent nói bằng một giọng, nên vân giọng
    bó tay và nhãn phải gán luân phiên theo lượt. Nhưng "một giọng" cũng đúng
    với bản ghi chỉ có MỘT người nói — và gán luân phiên ở đó là bịa ra người
    thứ hai. Phân biệt bằng chữ, không bằng tiếng:

    - hai nhóm nói ngược nhau về câu cửa miệng (một bên giọng tổng đài, một bên
      giọng người nhờ việc), hoặc
    - cùng một giá trị trong yêu cầu được CẢ HAI nhóm nói ra — tức có người đưa
      tin và có người nhắc lại để xác nhận.

    Không dấu hiệu nào thì trả ``False``: thà từ chối còn hơn chia đôi lời của
    một người rồi chấm agent bằng chính câu của khách.
    """
    normalized: dict[int, str] = {}
    for cluster, text in spoken:
        normalized[cluster] = f"{normalized.get(cluster, '')} {_norm(text)}".strip()
    if len(normalized) < 2:
        return False
    cues = [
        sum(cue in text for cue in AGENT_CUES) - sum(cue in text for cue in CALLER_CUES)
        for text in normalized.values()
    ]
    if max(cues) > 0 and min(cues) < 0:
        return True
    wanted = [value for value in (_norm(item).strip() for item in requirements) if value]
    return any(
        sum(value in text for text in normalized.values()) >= 2 for value in wanted
    )


def pick_agent_cluster(
    spoken: list[tuple[int, str]],
    requirements: list[str],
) -> tuple[int, str]:
    """Cụm nào là agent, và vì sao. Quyết bằng LỜI chứ không bằng giọng.

    ``spoken`` là các mảnh tiếng THEO THỨ TỰ THỜI GIAN: (cụm, chữ đọc được).
    Thứ tự là bằng chứng, không phải thứ trang trí — xem căn cứ 2.

    Giọng agent do người dùng cấu hình nên không ghim trước được; vai thì lộ ra
    trong chữ. Bốn căn cứ, chắc chắn giảm dần:

    1. Cụm đọc ra **nhiều giá trị người dùng nêu trong yêu cầu** hơn — agent đọc
       lại để xác nhận. Bản ghi vừa tải lên thường CHƯA có yêu cầu nào; khi đó
       căn cứ này im và ba căn cứ dưới quyết.
    2. Hai bên cùng đọc giá trị đó: bên đọc **SAU** là agent. Khách đưa thông
       tin trước, agent nhắc lại để xác nhận — không bao giờ ngược lại.
    3. **Hiệu** câu cửa miệng: câu của tổng đài trừ câu của người nhờ việc.
       Đếm một phía thì "cảm ơn" của khách cũng thành bằng chứng buộc tội.
    4. Cụm **nói lượt cuối** — agent là bên chốt cuộc gọi. Yếu nhất, nên lý do
       trả về nói thẳng là suy đoán để giao diện hạ độ tin cậy.
    """
    order = [cluster for cluster, _text in spoken]
    keys = sorted(set(order))
    if len(keys) < 2:
        return keys[0] if keys else 0, "chỉ nhận ra một giọng"
    normalized: dict[int, str] = {}
    for cluster, text in spoken:
        normalized[cluster] = f"{normalized.get(cluster, '')} {_norm(text)}".strip()

    wanted = [_norm(value).strip() for value in requirements]
    wanted = [value for value in wanted if value]
    if wanted:
        hits = {
            cluster: sum(value in text for value in wanted) for cluster, text in normalized.items()
        }
        best, second = sorted(hits.values(), reverse=True)[:2]
        if best > second:
            cluster = max(hits, key=lambda key: hits[key])
            return cluster, f"đọc lại {hits[cluster]}/{len(wanted)} giá trị kịch bản"
        if best > 0:
            said_at: dict[int, int] = {}
            for index, (cluster, text) in enumerate(spoken):
                normalized_text = _norm(text)
                if any(value in normalized_text for value in wanted):
                    said_at.setdefault(cluster, index)
            if len(said_at) == 2:
                cluster = max(said_at, key=lambda key: said_at[key])
                return cluster, "nhắc lại giá trị kịch bản sau bên kia"

    cues = {
        cluster: sum(cue in text for cue in AGENT_CUES)
        - sum(cue in text for cue in CALLER_CUES)
        for cluster, text in normalized.items()
    }
    best, second = sorted(cues.values(), reverse=True)[:2]
    if best > second:
        cluster = max(cues, key=lambda key: cues[key])
        return cluster, f"hơn {best - second} câu cửa miệng tổng đài"

    return order[-1], "đoán theo bên nói lượt cuối — căn cứ yếu, nên soát lại nhãn"
