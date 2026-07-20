"""SgChords: synchronized, editable chord transcription for play-along practice."""

from .analyzer import AnalyzerConfig, analyze_audio
from .models import AnalysisResult, BeatPoint, ChordSegment, KeyRegion, ProgressionPattern
from .service import analyze_request

__all__ = [
    "AnalysisResult",
    "AnalyzerConfig",
    "BeatPoint",
    "ChordSegment",
    "KeyRegion",
    "ProgressionPattern",
    "analyze_audio",
    "analyze_request",
]
__version__ = "0.1.0"
