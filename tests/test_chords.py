from __future__ import annotations

import numpy as np

from sgchords.analyzer import classify_chroma_segments, detect_drone, estimate_local_keys
from sgchords.chords import (
    estimate_key,
    nashville_number,
    roman_numeral,
    simplify_chord,
    transpose_chord,
    transpose_key,
)


def chord_vector(root: int, intervals: tuple[int, ...], *, drone: int | None = None) -> np.ndarray:
    vector = np.full(12, 0.005)
    for interval, weight in zip(intervals, (1.0, 0.82, 0.68, 0.54), strict=False):
        vector[(root + interval) % 12] += weight
    if drone is not None:
        vector[drone] += 0.70
    return vector


def test_transpose_and_number_notation_preserve_slash_bass() -> None:
    assert transpose_chord("C#m7/G#", 2) == "D#m7/A#"
    assert transpose_chord("Bb/F", 2, prefer_flats=True) == "C/G"
    assert simplify_chord("F#maj7/C#", keep_bass=True) == "F#/C#"
    assert transpose_key("D dorian", 2) == "E dorian"
    assert roman_numeral("Dm/F", "C major") == "ii/IV"
    assert nashville_number("G7/B", "C major") == "57/7"


def test_modal_key_estimation_can_recognize_d_dorian() -> None:
    profile = (
        chord_vector(2, (0, 3, 7)) * 4 + chord_vector(7, (0, 4, 7)) * 3 + chord_vector(0, (0, 4, 7))
    )
    key, confidence, root, mode = estimate_key(profile)
    assert (root, mode, key) == (2, "dorian", "D dorian")
    assert confidence > 0


def test_local_key_tracker_finds_modulation() -> None:
    c = [
        chord_vector(0, (0, 4, 7)),
        chord_vector(5, (0, 4, 7)),
        chord_vector(7, (0, 4, 7)),
        chord_vector(9, (0, 3, 7)),
    ]
    d = [
        chord_vector(2, (0, 4, 7)),
        chord_vector(7, (0, 4, 7)),
        chord_vector(9, (0, 4, 7)),
        chord_vector(11, (0, 3, 7)),
    ]
    vectors = np.asarray(c * 4 + d * 4)
    boundaries = np.arange(len(vectors) + 1, dtype=float) * 2
    local, regions, dominant, confidence = estimate_local_keys(
        vectors, np.ones(len(vectors)), boundaries, window_seconds=8, min_region_seconds=5
    )
    labels = [item[3] for item in local]
    assert "C major" in labels[: len(labels) // 2]
    assert "D major" in labels[len(labels) // 2 :]
    assert len(regions) >= 2
    assert dominant in {"C major", "D major"}
    assert confidence >= 0


def test_drone_detection_reduces_constant_pitch_bias() -> None:
    vectors = np.asarray(
        [
            chord_vector(2, (0, 3, 7), drone=2),
            chord_vector(7, (0, 4, 7), drone=2),
            chord_vector(0, (0, 4, 7), drone=2),
            chord_vector(9, (0, 3, 7), drone=2),
        ]
        * 4
    )
    root, confidence, corrected = detect_drone(vectors, np.ones(len(vectors)))
    assert root == 2
    assert confidence >= 0.38
    assert np.median(corrected[:, 2]) < np.median(vectors[:, 2])


def test_classifier_decodes_progression_and_inversion() -> None:
    vectors = np.asarray(
        [
            chord_vector(0, (0, 4, 7)),
            chord_vector(7, (0, 4, 7)),
            chord_vector(9, (0, 3, 7)),
            chord_vector(5, (0, 4, 7)),
        ]
    )
    bass = np.zeros_like(vectors)
    bass[0, 0] = 1
    bass[1, 11] = 1
    bass[2, 9] = 1
    bass[3, 5] = 1
    labels, confidence, _tonality, _states = classify_chroma_segments(
        vectors,
        np.ones(4),
        bass_vectors=bass,
        detail="standard",
        smoothing=0.2,
        local_keys=[(0, "major", 0.9, "C major")] * 4,
    )
    assert labels == ["C", "G/B", "Am", "F"]
    assert min(confidence) > 0.35



def test_common_chord_tones_are_not_mistaken_for_a_drone() -> None:
    vectors = np.asarray(
        [
            chord_vector(0, (0, 4, 7)),
            chord_vector(7, (0, 4, 7)),
            chord_vector(9, (0, 3, 7)),
            chord_vector(5, (0, 4, 7)),
        ]
        * 4
    )
    root, confidence, corrected = detect_drone(vectors, np.ones(len(vectors)))
    assert root is None
    assert confidence < 0.45
    assert np.allclose(corrected, vectors)


def test_clean_triad_is_not_promoted_to_major_seventh() -> None:
    vector = np.asarray([chord_vector(0, (0, 4, 7))])
    bass = np.zeros_like(vector)
    bass[0, 0] = 1
    labels, _, _, _ = classify_chroma_segments(
        vector,
        np.ones(1),
        bass_vectors=bass,
        detail="standard",
        smoothing=0.2,
        local_keys=[(0, "major", 1.0, "C major")],
    )
    assert labels == ["C"]



def test_persistent_chroma_floor_reduces_recording_wide_artifact() -> None:
    from sgchords.analyzer import suppress_persistent_chroma_floor

    vectors = np.asarray(
        [
            chord_vector(0, (0, 4, 7)),
            chord_vector(5, (0, 4, 7)),
            chord_vector(7, (0, 4, 7)),
            chord_vector(0, (0, 4, 7)),
        ]
        * 3
    )
    vectors[:, 8] += 0.45
    corrected, floor = suppress_persistent_chroma_floor(vectors, np.ones(len(vectors)))
    assert floor[8] > 0.05
    assert np.median(corrected[:, 8]) < np.median(vectors[:, 8])
    assert np.argmax(corrected[0]) == 0
