from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ChordSegment:
    """One time-aligned chord estimate."""

    start: float
    end: float
    chord: str
    confidence: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_row(self) -> list[Any]:
        return [round(self.start, 3), round(self.end, 3), self.chord, round(self.confidence, 3)]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChordSegment:
        return cls(
            start=float(value["start"]),
            end=float(value["end"]),
            chord=str(value["chord"]),
            confidence=float(value.get("confidence", 0.0)),
        )


@dataclass(slots=True)
class AnalysisResult:
    """Serializable result returned by the analysis pipeline."""

    title: str
    source: str
    audio_path: str
    duration: float
    tempo_bpm: float | None
    key: str
    key_confidence: float
    tuning_cents: float
    segments: list[ChordSegment]
    warnings: list[str] = field(default_factory=list)
    used_beat_tracking: bool = True

    def to_state(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["segments"] = [asdict(segment) for segment in self.segments]
        return payload

    @classmethod
    def from_state(cls, value: dict[str, Any]) -> AnalysisResult:
        return cls(
            title=str(value["title"]),
            source=str(value["source"]),
            audio_path=str(value["audio_path"]),
            duration=float(value["duration"]),
            tempo_bpm=(None if value.get("tempo_bpm") is None else float(value["tempo_bpm"])),
            key=str(value.get("key", "Unknown")),
            key_confidence=float(value.get("key_confidence", 0.0)),
            tuning_cents=float(value.get("tuning_cents", 0.0)),
            segments=[ChordSegment.from_dict(item) for item in value.get("segments", [])],
            warnings=[str(item) for item in value.get("warnings", [])],
            used_beat_tracking=bool(value.get("used_beat_tracking", True)),
        )
