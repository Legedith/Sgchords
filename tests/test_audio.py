from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sgchords.audio import is_youtube_url, normalize_audio, probe_duration


def test_youtube_url_validation() -> None:
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert not is_youtube_url("https://example.com/watch?v=abc")
    assert not is_youtube_url("file:///etc/passwd")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_ffmpeg_normalization(tmp_path: Path) -> None:
    sample_rate = 44_100
    time = np.arange(sample_rate * 2) / sample_rate
    stereo = np.stack(
        [0.2 * np.sin(2 * np.pi * 220 * time), 0.2 * np.sin(2 * np.pi * 330 * time)],
        axis=1,
    )
    source = tmp_path / "source.flac"
    output = tmp_path / "output.wav"
    sf.write(source, stereo, sample_rate)

    normalize_audio(source, output)
    audio, output_rate = sf.read(output)
    assert output_rate == 22_050
    assert audio.ndim == 1
    assert probe_duration(output) == pytest.approx(2.0, abs=0.03)
