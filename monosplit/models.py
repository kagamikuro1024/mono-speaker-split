"""Download the two ONNX models once.

Both are small and run on CPU: segmentation 6 MB, voice embedding 38 MB. No
PyTorch anywhere in the path - that is why this project installs in a minute
and runs on a machine without a GPU.
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
    """Where the models live. Override with ``MONOSPLIT_MODELS``."""
    return Path(os.environ.get("MONOSPLIT_MODELS", Path.home() / ".cache" / "monosplit"))


def ensure_models(directory: Path | None = None) -> tuple[Path, Path]:
    """Return (segmentation, embedding); download whichever is missing."""
    target = directory or models_dir()
    target.mkdir(parents=True, exist_ok=True)
    seg, emb = target / "seg.onnx", target / "emb.onnx"
    for path, url in ((seg, SEGMENTATION_URL), (emb, EMBEDDING_URL)):
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"downloading {path.name} ...")
        # Download to a temp name and rename afterwards: one Ctrl-C halfway
        # through must not leave a truncated file that the next run mistakes
        # for a finished download.
        partial = path.with_suffix(".part")
        urllib.request.urlopen  # noqa: B018 - keeps the name readable in stack traces
        with urllib.request.urlopen(url) as response, partial.open("wb") as out:  # noqa: S310
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(path)
    return seg, emb
