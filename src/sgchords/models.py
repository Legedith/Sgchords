from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BeatPoint:
    time: float
    index: int
    bar: int
    beat: float
    strength: float = 0.0
    downbeat: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BeatPoint:
        return cls(
            time=float(value["time"]),
            index=int(value.get("index", 0)),
            bar=int(value.get("bar", 1)),
            beat=float(value.get("beat", 1.0)),
            strength=float(value.get("strength", 0.0)),
            downbeat=bool(value.get("downbeat", False)),
        )


@dataclass(slots=True)
class KeyRegion:
    start: float
    end: float
    key: str
    confidence: float
    root: int
    mode: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KeyRegion:
        return cls(
            start=float(value["start"]),
            end=float(value["end"]),
            key=str(value["key"]),
            confidence=float(value.get("confidence", 0.0)),
            root=int(value.get("root", 0)),
            mode=str(value.get("mode", "major")),
        )


@dataclass(slots=True)
class ProgressionPattern:
    chords: list[str]
    length_bars: int
    occurrences: int
    coverage: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProgressionPattern:
        return cls(
            chords=[str(item) for item in value.get("chords", [])],
            length_bars=int(value.get("length_bars", 1)),
            occurrences=int(value.get("occurrences", 1)),
            coverage=float(value.get("coverage", 0.0)),
        )


@dataclass(slots=True)
class ChordSegment:
    start: float
    end: float
    chord: str
    confidence: float
    local_key: str | None = None
    bar: int | None = None
    beat: float | None = None
    bass: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_row(self) -> list[Any]:
        return [
            round(self.start, 3),
            round(self.end, 3),
            self.chord,
            round(self.confidence, 3),
            self.local_key or "",
            self.bar or "",
            "" if self.beat is None else round(self.beat, 2),
        ]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChordSegment:
        return cls(
            start=float(value["start"]),
            end=float(value["end"]),
            chord=str(value["chord"]),
            confidence=float(value.get("confidence", 0.0)),
            local_key=(str(value["local_key"]) if value.get("local_key") else None),
            bar=(int(value["bar"]) if value.get("bar") not in {None, ""} else None),
            beat=(float(value["beat"]) if value.get("beat") not in {None, ""} else None),
            bass=(str(value["bass"]) if value.get("bass") else None),
        )


@dataclass(slots=True)
class AnalysisResult:
    title: str
    source: str
    audio_path: str
    duration: float
    tempo_bpm: float | None
    key: str
    key_confidence: float
    tuning_cents: float
    segments: list[ChordSegment]
    tempo_confidence: float = 0.0
    meter: int = 4
    meter_confidence: float = 0.0
    beats: list[BeatPoint] = field(default_factory=list)
    key_regions: list[KeyRegion] = field(default_factory=list)
    patterns: list[ProgressionPattern] = field(default_factory=list)
    drone: str | None = None
    drone_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    used_beat_tracking: bool = True
    analysis_version: str = "0.2"

    def to_state(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state(cls, value: dict[str, Any]) -> AnalysisResult:
        return cls(
            title=str(value["title"]),
            source=str(value["source"]),
            audio_path=str(value["audio_path"]),
            duration=float(value["duration"]),
            tempo_bpm=(None if value.get("tempo_bpm") is None else float(value["tempo_bpm"])),
            tempo_confidence=float(value.get("tempo_confidence", 0.0)),
            meter=int(value.get("meter", 4)),
            meter_confidence=float(value.get("meter_confidence", 0.0)),
            key=str(value.get("key", "Unknown")),
            key_confidence=float(value.get("key_confidence", 0.0)),
            tuning_cents=float(value.get("tuning_cents", 0.0)),
            segments=[ChordSegment.from_dict(item) for item in value.get("segments", [])],
            beats=[BeatPoint.from_dict(item) for item in value.get("beats", [])],
            key_regions=[KeyRegion.from_dict(item) for item in value.get("key_regions", [])],
            patterns=[ProgressionPattern.from_dict(item) for item in value.get("patterns", [])],
            drone=(str(value["drone"]) if value.get("drone") else None),
            drone_confidence=float(value.get("drone_confidence", 0.0)),
            warnings=[str(item) for item in value.get("warnings", [])],
            used_beat_tracking=bool(value.get("used_beat_tracking", True)),
            analysis_version=str(value.get("analysis_version", "0.1")),
        )
