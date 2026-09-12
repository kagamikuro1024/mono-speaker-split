"""Đọc tệp âm thanh thành PCM, và trả lời một câu hỏi hay bị bỏ qua:
tệp này có THẬT là hai kênh tách vai không.

Số kênh trong metadata không nói lên điều đó. Rất nhiều bản ghi "stereo" là
mono nhân đôi, hoặc thu một bên còn bên kia câm. Đi đường hai kênh với những
tệp đó thì mỗi câu bị phiên âm hai lần và gán cho cả hai vai với CÙNG mốc thời
gian — một bản gỡ băng nhìn thì đầy đủ mà vô nghĩa.
"""

from __future__ import annotations

import json
import subprocess
from array import array
from dataclasses import dataclass
from pathlib import Path

# Whisper ăn 16 kHz; PCM s16le mono ở mức này là 32 byte mỗi mili giây, nên cắt
# một đoạn theo mốc thời gian chỉ là cắt byte.
TARGET_SAMPLE_RATE = 16_000
PCM_BYTES_PER_MS = TARGET_SAMPLE_RATE * 2 // 1000

# Hai kênh "giống nhau đến mức này" thì coi là một luồng. Đo trên bộ mẫu: mono
# nhân đôi lệch 0,000–0,007% biên độ, bản ghi hai kênh thật lệch 180–196%.
# 2% nằm giữa hai khoảng và chịu được sai số nén.
SAME_STREAM_RATIO = 0.02

# Một kênh câm: nhỏ hơn 1% kênh kia thì nó không mang lời của ai.
SILENT_CHANNEL_RATIO = 0.01


class AudioError(RuntimeError):
    """Không đọc được tệp, hoặc tệp không dùng được cho việc tách giọng."""


@dataclass(frozen=True)
class Probe:
    """Những gì biết được trước khi tốn công giải mã cả tệp."""

    channels: int
    duration_ms: int


def probe(source: Path) -> Probe:
    """Số kênh và độ dài, đọc từ header."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels:format=duration", "-of", "json", str(source)],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise AudioError(f"ffprobe không đọc được tệp: {source}")
    try:
        payload = json.loads(result.stdout)
        return Probe(
            channels=int(payload["streams"][0]["channels"]),
            duration_ms=round(float(payload["format"]["duration"]) * 1000),
        )
    except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AudioError(f"tệp không có luồng âm thanh đọc được: {source}") from exc


def decode_mono(source: Path, dest: Path) -> bytes:
    """Trộn mọi kênh xuống một kênh 16 kHz PCM s16le và trả về byte."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ac", "1",
         "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest)],
        capture_output=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise AudioError(result.stderr.decode("utf-8", "replace")[:500] or "ffmpeg lỗi")
    return _read_pcm(dest)


def decode_channels(source: Path, work: Path) -> tuple[bytes, bytes]:
    """Tách kênh trái / phải thành hai luồng PCM riêng."""
    left, right = work / "left.wav", work / "right.wav"
    rate = str(TARGET_SAMPLE_RATE)
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source),
         "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[l][r]",
         "-map", "[l]", "-ac", "1", "-ar", rate, "-c:a", "pcm_s16le", str(left),
         "-map", "[r]", "-ac", "1", "-ar", rate, "-c:a", "pcm_s16le", str(right)],
        capture_output=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise AudioError(result.stderr.decode("utf-8", "replace")[:500] or "ffmpeg lỗi")
    return _read_pcm(left), _read_pcm(right)


def one_stream_only(left: bytes, right: bytes) -> bool:
    """Hai kênh có thực chất chỉ là MỘT luồng tiếng hay không.

    Trả ``True`` khi hai kênh trùng nhau hoặc một kênh câm — cả hai trường hợp
    đều phải đi đường một kênh. Cả hai kênh đều câm trả ``False``: đó là tệp
    không có tiếng nói, một lỗi khác hẳn, và gọi tên đúng lỗi mới sửa được.
    """
    import numpy as np

    l_pcm = np.frombuffer(left, dtype=np.int16).astype(np.float32)
    r_pcm = np.frombuffer(right, dtype=np.int16).astype(np.float32)
    size = min(l_pcm.size, r_pcm.size)
    l_pcm, r_pcm = l_pcm[:size], r_pcm[:size]
    l_level, r_level = float(np.abs(l_pcm).mean()), float(np.abs(r_pcm).mean())
    loud = max(l_level, r_level) if size else 0.0
    if loud == 0:
        return False
    if min(l_level, r_level) <= SILENT_CHANNEL_RATIO * loud:
        return True
    return float(np.abs(l_pcm - r_pcm).mean()) <= SAME_STREAM_RATIO * loud


def channel_difference(left: bytes, right: bytes) -> float:
    """Chênh lệch trung bình giữa hai kênh, theo tỉ lệ biên độ. Để in ra báo cáo."""
    import numpy as np

    l_pcm = np.frombuffer(left, dtype=np.int16).astype(np.float32)
    r_pcm = np.frombuffer(right, dtype=np.int16).astype(np.float32)
    size = min(l_pcm.size, r_pcm.size)
    if not size:
        return 0.0
    l_pcm, r_pcm = l_pcm[:size], r_pcm[:size]
    loud = max(float(np.abs(l_pcm).mean()), float(np.abs(r_pcm).mean()))
    return float(np.abs(l_pcm - r_pcm).mean() / loud) if loud else 0.0


def slice_pcm(pcm: bytes, start_ms: int, end_ms: int) -> bytes:
    return pcm[max(0, start_ms) * PCM_BYTES_PER_MS : max(0, end_ms) * PCM_BYTES_PER_MS]


def _read_pcm(path: Path) -> bytes:
    """Bỏ 44 byte header WAV, giữ mẫu thô."""
    raw = path.read_bytes()
    return raw[44:] if raw[:4] == b"RIFF" else raw


def to_samples(pcm: bytes):
    """PCM s16le → mảng float32 trong [-1, 1] cho các mô hình ONNX."""
    import numpy as np

    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def peak_levels(pcm: bytes, frame_ms: int = 20) -> list[int]:
    """Biên độ đỉnh mỗi khung — dùng vẽ dạng sóng ở giao diện."""
    frame_bytes = frame_ms * PCM_BYTES_PER_MS
    return [
        max((abs(value) for value in array("h", pcm[at : at + frame_bytes])), default=0)
        for at in range(0, max(0, len(pcm) - frame_bytes + 1), frame_bytes)
    ]
