"""Decode audio files into PCM, and answer a question that often gets skipped:
is this file REALLY two channels carrying one role each.

The channel count in the metadata does not answer that. Plenty of "stereo"
recordings are duplicated mono, or have one side recorded while the other is
silent. Sending those down the two-channel path transcribes every sentence
twice and assigns it to both roles with the SAME timestamps — a transcript that
looks complete and means nothing.
"""

from __future__ import annotations

import json
import subprocess
from array import array
from dataclasses import dataclass
from pathlib import Path

# Whisper takes 16 kHz; mono PCM s16le at that rate is 32 bytes per millisecond,
# so slicing a span by timestamp is just slicing bytes.
TARGET_SAMPLE_RATE = 16_000
PCM_BYTES_PER_MS = TARGET_SAMPLE_RATE * 2 // 1000

# Two channels "this similar" count as a single stream. Measured on the sample
# set: duplicated mono differs by 0.000-0.007% of amplitude, a real two-channel
# recording differs by 180-196%. 2% sits between the two ranges and absorbs
# compression error.
SAME_STREAM_RATIO = 0.02

# A silent channel: below 1% of the other one it carries nobody's speech.
SILENT_CHANNEL_RATIO = 0.01


class AudioError(RuntimeError):
    """The file cannot be read, or is unusable for speaker splitting."""


@dataclass(frozen=True)
class Probe:
    """What can be known before spending effort decoding the whole file."""

    channels: int
    duration_ms: int


def probe(source: Path) -> Probe:
    """Channel count and duration, read from the header."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels:format=duration", "-of", "json", str(source)],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise AudioError(f"ffprobe could not read the file: {source}")
    try:
        payload = json.loads(result.stdout)
        return Probe(
            channels=int(payload["streams"][0]["channels"]),
            duration_ms=round(float(payload["format"]["duration"]) * 1000),
        )
    except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AudioError(f"file has no readable audio stream: {source}") from exc


def decode_mono(source: Path, dest: Path) -> bytes:
    """Mix every channel down to one 16 kHz PCM s16le channel and return the bytes."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ac", "1",
         "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest)],
        capture_output=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise AudioError(result.stderr.decode("utf-8", "replace")[:500] or "ffmpeg error")
    return _read_pcm(dest)


def decode_channels(source: Path, work: Path) -> tuple[bytes, bytes]:
    """Split the left / right channels into two separate PCM streams."""
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
        raise AudioError(result.stderr.decode("utf-8", "replace")[:500] or "ffmpeg error")
    return _read_pcm(left), _read_pcm(right)


def one_stream_only(left: bytes, right: bytes) -> bool:
    """Whether the two channels are really just ONE speech stream.

    Returns ``True`` when the two channels are identical or one is silent — both
    cases must take the single-channel path. Two silent channels return
    ``False``: that is a file with no speech at all, an entirely different
    failure, and only naming the failure correctly makes it fixable.
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
    """Mean difference between the two channels, as a fraction of amplitude. For the report."""
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
    """Drop the 44-byte WAV header, keep the raw samples."""
    raw = path.read_bytes()
    return raw[44:] if raw[:4] == b"RIFF" else raw


def to_samples(pcm: bytes):
    """PCM s16le → float32 array in [-1, 1] for the ONNX models."""
    import numpy as np

    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def peak_levels(pcm: bytes, frame_ms: int = 20) -> list[int]:
    """Peak amplitude per frame — used to draw the waveform in the UI."""
    frame_bytes = frame_ms * PCM_BYTES_PER_MS
    return [
        max((abs(value) for value in array("h", pcm[at : at + frame_bytes])), default=0)
        for at in range(0, max(0, len(pcm) - frame_bytes + 1), frame_bytes)
    ]
