from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sgchords.analyzer import AnalysisError, analyze_audio


def synth_chord(
    root_midi: int, intervals: tuple[int, ...], seconds: float, sample_rate: int
) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    signal = np.zeros_like(time)
    for interval in intervals:
        frequency = 440.0 * 2 ** ((root_midi + interval - 69) / 12)
        for harmonic in range(1, 5):
            signal += np.sin(2 * np.pi * frequency * harmonic * time) / harmonic
    signal /= max(float(np.max(np.abs(signal))), 1e-9)
    fade = min(int(0.03 * sample_rate), len(signal) // 4)
    envelope = np.ones_like(signal)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return 0.35 * signal * envelope


def test_analyze_synthetic_c_major(tmp_path: Path) -> None:
    sample_rate = 22_050
    audio = synth_chord(48, (0, 4, 7), 4.0, sample_rate)
    # Add quiet clicks to give the beat tracker useful transients without masking harmony.
    for second in np.arange(0.5, 4.0, 0.5):
        start = int(second * sample_rate)
        audio[start : start + 80] += np.hanning(80) * 0.08
    path = tmp_path / "c-major.wav"
    sf.write(path, audio, sample_rate)

    result = analyze_audio(path, detail="simple", smoothing=0.5)
    labels = [segment.chord for segment in result.segments]
    assert result.duration == pytest.approx(4.0, abs=0.05)
    assert "C" in labels
    assert result.segments


def test_silent_audio_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(22_050 * 3), 22_050)
    with pytest.raises(AnalysisError, match="silent"):
        analyze_audio(path)
