"""Web interface: drop in one file, see two lanes of turns.

The models are loaded ONCE and then kept in module-level variables. Reloading
them on every request burns a few seconds of CPU on work already finished by
the previous request.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from monosplit import audio
from monosplit.models import ensure_models
from monosplit.pipeline import SeparationError, separate
from monosplit.speakers import MonoSpeakerSplitter, MonoSplitUnavailable
from monosplit.transcribe import Transcriber, TranscriberUnavailable

STATIC = Path(__file__).parent / "static"

# Waveform frame: one column per 20 ms, fine enough to see the rhythm of speech
# while staying light (a 5-minute call yields 15,000 numbers).
WAVEFORM_FRAME_MS = 20

app = FastAPI(title="monosplit", description="Split caller / agent out of a single-channel recording")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_splitter: MonoSpeakerSplitter | None = None
_transcriber: Transcriber | None = None
_transcriber_loaded = False


def _get_splitter() -> MonoSpeakerSplitter:
    global _splitter
    if _splitter is None:
        try:
            seg, emb = ensure_models()
            _splitter = MonoSpeakerSplitter(str(seg), str(emb))
        except MonoSplitUnavailable as exc:
            raise HTTPException(503, {"code": "thieu_mo_hinh", "message": str(exc)}) from exc
    return _splitter


def _get_transcriber() -> Transcriber | None:
    """Without faster-whisper it still runs, the turns just carry no text."""
    global _transcriber, _transcriber_loaded
    if not _transcriber_loaded:
        _transcriber_loaded = True
        try:
            _transcriber = Transcriber()
        except TranscriberUnavailable:
            _transcriber = None
    return _transcriber


@app.get("/")
def trang_chu() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/separate")
async def api_separate(
    file: Annotated[UploadFile, File()],
    requirements: Annotated[str, Form()] = "",
) -> dict:
    splitter = _get_splitter()
    yeu_cau = [dong.strip() for dong in requirements.splitlines() if dong.strip()]

    with tempfile.TemporaryDirectory(prefix="monosplit-web-") as tmp:
        # Keep the file extension: ffmpeg guesses the format more easily with .m4a/.wav.
        nguon = Path(tmp) / (Path(file.filename or "upload").name or "upload")
        with nguon.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        try:
            result = separate(nguon, splitter, _get_transcriber(), yeu_cau)
        except SeparationError as exc:
            raise HTTPException(422, {"code": exc.code, "message": str(exc)}) from exc

        pcm = audio.decode_mono(nguon, Path(tmp) / "mono.wav")
        return {
            **result.to_dict(),
            "waveform": audio.peak_levels(pcm, WAVEFORM_FRAME_MS),
            "waveform_frame_ms": WAVEFORM_FRAME_MS,
        }
