"""Chép lời bằng faster-whisper, lấy mốc THEO TỪ.

Mốc theo đoạn của Whisper không dùng được để cắt lượt: Whisper cắt theo ngữ
pháp, không theo lúc người ta ngừng nói. Một bản ghi có agent nói hai lần cách
nhau 2,5 giây vẫn về đúng MỘT đoạn — lấy đoạn làm lượt thì lượt đó nuốt trọn
khoảng lặng và độ trễ đáp của lượt sau biến mất. Mốc từng từ thì khoảng cách
2,4 giây hiện ra rõ ràng.

Cả module là TUỲ CHỌN: không cài faster-whisper thì đường tách giọng vẫn chạy,
chỉ là các lượt không có chữ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TranscriberUnavailable(RuntimeError):
    """Chưa cài faster-whisper."""


@dataclass(frozen=True)
class Word:
    start_ms: int
    end_ms: int
    text: str


class Transcriber:
    """Bọc faster-whisper. Mô hình nạp một lần, dùng lại cho mọi tệp."""

    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - môi trường thiếu gói
            raise TranscriberUnavailable("chưa cài faster-whisper") from exc
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def words(self, path: Path, language: str = "vi") -> list[Word]:
        segments, _info = self._model.transcribe(
            str(path), language=language, word_timestamps=True, vad_filter=True
        )
        out: list[Word] = []
        for segment in segments:
            words: list[Any] = list(getattr(segment, "words", None) or [])
            if not words:
                # Đoạn không có mốc từ (Whisper thỉnh thoảng bỏ) vẫn phải giữ:
                # mất lời còn tệ hơn mốc thô.
                text = segment.text.strip()
                if text:
                    out.append(
                        Word(max(0, round(segment.start * 1000)), max(0, round(segment.end * 1000)), text)
                    )
                continue
            for word in words:
                text = word.word.strip()
                if text:
                    out.append(
                        Word(max(0, round(word.start * 1000)), max(0, round(word.end * 1000)), text)
                    )
        return out


def words_per_piece(
    words: list[Word], pieces: list[tuple[int, int, int]], pad_ms: int = 400
) -> list[str]:
    """Chia chữ cho từng mảnh — mỗi từ thuộc ĐÚNG MỘT mảnh.

    Cách cũ (`text_in_range`) để mỗi mảnh tự nới hai đầu ``pad_ms`` rồi quét
    độc lập, nên một từ ở vùng giáp ranh đi vào CẢ HAI mảnh. Hậu quả không chỉ
    là chữ lặp: lượt của một bên mang theo câu bên kia vừa nói, và ai đọc bản
    ghi sau đó tin rằng người này đã nói câu của người kia.

    Luật: từ thuộc mảnh mà nó GIAO nhiều nhất. Từ không giao mảnh nào (mốc ASR
    lệch, hoặc rơi vào khoảng lặng) mới về mảnh gần nhất, và chỉ khi còn trong
    ``pad_ms`` — xa hơn thì nó không phải lời của lượt nào cả.
    """
    buckets: list[list[str]] = [[] for _ in pieces]
    for word in words:
        best, best_overlap = -1, 0
        for index, (start, end, _cluster) in enumerate(pieces):
            overlap = min(end, word.end_ms) - max(start, word.start_ms)
            if overlap > best_overlap:
                best, best_overlap = index, overlap
        if best < 0:
            gap, index = min(
                (max(start - word.end_ms, word.start_ms - end, 0), index)
                for index, (start, end, _c) in enumerate(pieces)
            )
            if gap > pad_ms:
                continue
            best = index
        buckets[best].append(word.text)
    return [" ".join(bucket).strip() for bucket in buckets]
