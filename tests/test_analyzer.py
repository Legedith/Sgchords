from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sgchords.analyzer import AnalysisError, AnalyzerConfig, analyze_audio
from sgchords.chords import simplify_chord


def synth_chord(
    root_midi: int, intervals: tuple[int, ...], seconds: float, sample_rate: int
) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    values = np.zeros_like(time)
    for interval in intervals:
        frequency = 440 * 2 ** ((root_midi + interval - 69) / 12)
        for harmonic in range(1, 5):
            values += np.sin(2 * np.pi * frequency * harmonic * time) / harmonic
    bass = 440 * 2 ** ((root_midi - 12 - 69) / 12)
    values += 1.1 * np.sin(2 * np.pi * bass * time)
    values /= max(float(np.max(np.abs(values))), 1e-9)
    fade = min(int(0.025 * sample_rate), len(values) // 4)
    envelope = np.ones_like(values)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return 0.34 * values * envelope


def make_progression(sample_rate: int = 22_050) -> np.ndarray:
    blocks = []
    for root, intervals in [(48, (0, 4, 7)), (55, (0, 4, 7)), (57, (0, 3, 7)), (53, (0, 4, 7))]:
        block = synth_chord(root, intervals, 2, sample_rate)
        for offset in (0, 0.5, 1, 1.5):
            start = int(offset * sample_rate)
            block[start : start + 96] += np.hanning(96) * 0.08
        blocks.append(block)
    return np.concatenate(blocks)


def chord_at(segments, timestamp: float) -> str:
    for segment in segments:
        if segment.start <= timestamp < segment.end:
            return simplify_chord(segment.chord)
    return "N"


def test_analyze_synthetic_progression_and_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "known.wav"
    sf.write(path, make_progression(), 22_050)
    output = analyze_audio(
        path,
        detail="standard",
        smoothing=0.5,
        config=AnalyzerConfig(local_key_window_seconds=6, local_key_min_region_seconds=3),
    )
    assert [chord_at(output.segments, time) for time in (1, 3, 5, 7)] == ["C", "G", "Am", "F"]
    boundaries = [segment.start for segment in output.segments[1:]]
    for expected in (2, 4, 6):
        assert min(abs(actual - expected) for actual in boundaries) < 0.5
    assert output.beats
    assert output.key_regions


def test_silence_rejected(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(22_050 * 3), 22_050)
    with pytest.raises(AnalysisError, match="silent"):
        analyze_audio(path)
