"""monosplit — split caller / agent apart from a SINGLE-channel call recording."""

from monosplit.models import ensure_models
from monosplit.pipeline import Result, SeparationError, Transcriber, Turn, separate
from monosplit.speakers import MonoSpeakerSplitter, MonoSplitUnavailable

__version__ = "0.1.0"

__all__ = [
    "MonoSpeakerSplitter",
    "MonoSplitUnavailable",
    "Result",
    "SeparationError",
    "Transcriber",
    "Turn",
    "ensure_models",
    "separate",
]
