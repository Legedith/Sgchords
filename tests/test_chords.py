from __future__ import annotations

import numpy as np

from sgchords.analyzer import (
    _diatonic_duration_ratio,
    classify_chroma_segments,
)
from sgchords.chords import (
    estimate_key,
    key_prefers_flats,
    simplify_chord,
    suggest_capo,
    transpose_chord,
    transpose_key,
)
from sgchords.models import ChordSegment


def chord_vector(root: int, intervals: tuple[int, ...]) -> np.ndarray:
    vector = np.zeros(12, dtype=float)
    for interval, weight in zip(intervals, (1.0, 0.85, 0.7, 0.55), strict=False):
        vector[(root + interval) % 12] = weight
    return vector


def test_transpose_and_simplify() -> None:
    assert transpose_chord("C#m7", -1) == "Cm7"
    assert transpose_chord("Bb7", 2, prefer_flats=True) == "C7"
    assert transpose_chord("N", 7) == "N"
    assert simplify_chord("Cmaj7") == "C"
    assert simplify_chord("F#m7") == "F#m"
    assert transpose_key("C minor", 1) == "C# minor"
    assert key_prefers_flats("F minor")
    assert not key_prefers_flats("F# minor")


def test_key_estimation_prefers_c_major_profile() -> None:
    profile = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    key, confidence, root, mode = estimate_key(profile)
    assert key == "C major"
    assert root == 0
    assert mode == "major"
    assert confidence > 0


def test_classifier_decodes_clean_progression() -> None:
    vectors = np.stack(
        [
            chord_vector(0, (0, 4, 7)),
            chord_vector(9, (0, 3, 7)),
            chord_vector(5, (0, 4, 7)),
            chord_vector(7, (0, 4, 7)),
        ]
    )
    labels, confidence, _ = classify_chroma_segments(
        vectors,
        np.ones(4),
        detail="simple",
        smoothing=0.2,
        key_root=0,
        key_mode="major",
        key_confidence=1.0,
    )
    assert labels == ["C", "Am", "F", "G"]
    assert min(confidence) > 0.5


def test_capo_finds_easy_shapes() -> None:
    segments = [
        ChordSegment(0, 2, "B", 0.9),
        ChordSegment(2, 4, "E", 0.9),
        ChordSegment(4, 6, "F#", 0.9),
    ]
    capo, shapes, ease = suggest_capo(segments, "guitar")
    assert capo == 2
    assert shapes == ["A", "D", "E"]
    assert ease > 0.5



def test_classifier_uses_flat_spelling_for_g_minor() -> None:
    vectors = np.stack(
        [
            chord_vector(3, (0, 4, 7)),
            chord_vector(8, (0, 4, 7)),
            chord_vector(10, (0, 4, 7)),
        ]
    )
    labels, _, _ = classify_chroma_segments(
        vectors,
        np.ones(3),
        detail="simple",
        smoothing=0.2,
        key_root=7,
        key_mode="minor",
        key_confidence=1.0,
    )
    assert labels == ["Eb", "Ab", "Bb"]


def test_diatonic_ratio_detects_mixed_key_blocks() -> None:
    segments = [
        ChordSegment(0, 1, "Eb", 0.9),
        ChordSegment(1, 2, "Bb", 0.9),
        ChordSegment(2, 3, "Cm", 0.9),
        ChordSegment(3, 4, "E", 0.9),
        ChordSegment(4, 5, "A", 0.9),
        ChordSegment(5, 6, "B", 0.9),
    ]
    ratio = _diatonic_duration_ratio(
        segments,
        detail="simple",
        key_root=7,
        key_mode="minor",
    )
    assert ratio == 0.5
