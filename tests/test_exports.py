from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sgchords.exports import rows_to_segments, write_exports
from sgchords.models import AnalysisResult, ChordSegment


def make_result(tmp_path: Path) -> AnalysisResult:
    return AnalysisResult(
        title="Test Song",
        source="test.wav",
        audio_path=str(tmp_path / "source.wav"),
        duration=8.0,
        tempo_bpm=100.0,
        key="C major",
        key_confidence=0.8,
        tuning_cents=-3.0,
        segments=[
            ChordSegment(0.0, 4.0, "C", 0.9),
            ChordSegment(4.0, 8.0, "G", 0.8),
        ],
    )


def test_rows_to_segments_accepts_dataframe() -> None:
    frame = pd.DataFrame(
        [[0, 2, "Am", 0.7], [2, 4, "F", 0.6]],
        columns=["Start (s)", "End (s)", "Chord", "Confidence"],
    )
    segments = rows_to_segments(frame)
    assert [segment.chord for segment in segments] == ["Am", "F"]


def test_write_all_export_formats(tmp_path: Path) -> None:
    result = make_result(tmp_path)
    files = write_exports(result, tmp_path / "exports", transpose=2, instrument="ukulele")
    assert len(files) == 5
    for file in files:
        assert Path(file).is_file()

    payload = json.loads((tmp_path / "exports" / "test-song-analysis.json").read_text())
    assert payload["key"] == "D major"
    assert [item["chord"] for item in payload["segments"]] == ["D", "A"]
    assert (tmp_path / "exports" / "test-song.cho").read_text().startswith("{title: Test Song}")
