"""Tải hai mô hình ONNX về máy, một lần.

Cả hai đều nhỏ và chạy trên CPU: segmentation 6 MB, vân giọng 38 MB. Không có
PyTorch trong đường chạy — đó là lý do dự án này cài được trong một phút và
chạy được trên máy không GPU.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

SEGMENTATION_URL = (
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx"
)
EMBEDDING_URL = (
    "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)


def models_dir() -> Path:
    """Nơi để mô hình. Đổi được bằng ``MONOSPLIT_MODELS``."""
    return Path(os.environ.get("MONOSPLIT_MODELS", Path.home() / ".cache" / "monosplit"))


def ensure_models(directory: Path | None = None) -> tuple[Path, Path]:
    """Trả về (segmentation, embedding); tải về nếu chưa có."""
    target = directory or models_dir()
    target.mkdir(parents=True, exist_ok=True)
    seg, emb = target / "seg.onnx", target / "emb.onnx"
    for path, url in ((seg, SEGMENTATION_URL), (emb, EMBEDDING_URL)):
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"tải {path.name} …")
        # Tải ra tệp tạm rồi mới đổi tên: một lần Ctrl-C giữa chừng không để
        # lại tệp cụt mà lần chạy sau tưởng là tải xong.
        partial = path.with_suffix(".part")
        urllib.request.urlopen  # noqa: B018 - giữ tên cho dễ đọc stack trace
        with urllib.request.urlopen(url) as response, partial.open("wb") as out:  # noqa: S310
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(path)
    return seg, emb
