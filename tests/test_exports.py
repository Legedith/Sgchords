from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sgchords.chords import render_chord_diagrams_html
from sgchords.exports import quantize_segments, render_bar_chart, rows_to_segments, write_exports
from sgchords.models import AnalysisResult, BeatPoint, ChordSegment, KeyRegion, ProgressionPattern


def make_result(tmp_path: Path) -> AnalysisResult:
    beats = [
        BeatPoint(float(index), index, index // 4 + 1, index % 4 + 1, 0.8, index % 4 == 0)
        for index in range(9)
    ]
    return AnalysisResult(
        title="Known Progression",
        source="test.wav",
        audio_path=str(tmp_path / "source.wav"),
        duration=8,
        tempo_bpm=60,
        tempo_confidence=0.9,
        meter=4,
        meter_confidence=0.8,
        key="C major",
        key_confidence=0.75,
        tuning_cents=2,
        segments=[
            ChordSegment(0, 2, "C", 0.9, "C major", 1, 1),
            ChordSegment(2, 4, "G/B", 0.8, "C major", 1, 3),
            ChordSegment(4, 6, "Am", 0.85, "C major", 2, 1),
            ChordSegment(6, 8, "F", 0.8, "C major", 2, 3),
        ],
        beats=beats,
        key_regions=[KeyRegion(0, 8, "C major", 0.75, 0, "major")],
        patterns=[ProgressionPattern(["C", "G/B", "Am", "F"], 2, 1, 1)],
    )


def test_rows_quantize_and_bar_chart(tmp_path: Path) -> None:
    result = make_result(tmp_path)
    frame = pd.DataFrame(
        [[0.08, 2.12, "C", 0.8, "C major", 1, 1], [2.12, 4.04, "G", 0.7, "C major", 1, 3]],
        columns=["Start (s)", "End (s)", "Chord", "Confidence", "Local key", "Bar", "Beat"],
    )
    snapped = quantize_segments(rows_to_segments(frame), result.beats, duration=4)
    assert (snapped[0].start, snapped[0].end, snapped[-1].end) == (0, 2, 4)
    chart = render_bar_chart(result, result.segments, capo=2)
    assert "Capo 2" in chart and "Bb" in chart
    assert "vi" in render_bar_chart(result, result.segments, notation="roman")
    assert "6m" in render_bar_chart(result, result.segments, notation="nashville")


def test_diagrams_and_exports(tmp_path: Path) -> None:
    assert render_chord_diagrams_html(["C", "Am", "G7"], "guitar").count("<svg") == 3
    result = make_result(tmp_path)
    files = write_exports(
        result, tmp_path / "exports", transpose=2, capo=2, notation="roman", instrument="ukulele"
    )
    assert len(files) == 7 and all(Path(path).is_file() for path in files)
    payload = json.loads((tmp_path / "exports" / "known-progression-analysis.json").read_text())
    assert payload["key"] == "D major" and payload["capo"] == 2 and payload["beats"]
    assert (tmp_path / "exports" / "known-progression.srt").read_text().startswith("1\n")
