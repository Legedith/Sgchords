from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Remove a small recording-wide chroma floor. This targets persistent hum, compression artifacts,
# ringing strings and leakage that occur in nearly every segment; it does not remove a tone merely
# because it occurs in several legitimate chords.
replace_once(
    "src/sgchords/analyzer.py",
    '''def detect_drone(vectors: np.ndarray, energies: np.ndarray) -> tuple[int | None, float, np.ndarray]:
''',
    '''def suppress_persistent_chroma_floor(
    vectors: np.ndarray,
    energies: np.ndarray,
    *,
    percentile: float = 10.0,
    strength: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.maximum(np.asarray(vectors, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] < 4:
        return matrix, np.zeros(12, dtype=float)
    energy = np.asarray(energies, dtype=float)
    active = energy >= max(float(np.median(energy)) * 0.16, 1e-8)
    if np.count_nonzero(active) < 4:
        active = np.ones(matrix.shape[0], dtype=bool)
    totals = np.sum(matrix, axis=1, keepdims=True)
    mass = matrix / np.maximum(totals, 1e-12)
    floor = np.percentile(mass[active], percentile, axis=0)
    corrected = np.maximum(mass - strength * floor[None, :], 0.0)
    corrected /= np.maximum(np.sum(corrected, axis=1, keepdims=True), 1e-12)
    return corrected * totals, floor


def detect_drone(vectors: np.ndarray, energies: np.ndarray) -> tuple[int | None, float, np.ndarray]:
''',
)
replace_once(
    "src/sgchords/analyzer.py",
    '''    drone_root, drone_confidence, corrected = detect_drone(vectors, energies)
    local_keys, key_regions, dominant_key, key_confidence = estimate_local_keys(
''',
    '''    drone_root, drone_confidence, drone_corrected = detect_drone(vectors, energies)
    corrected, _persistent_floor = suppress_persistent_chroma_floor(drone_corrected, energies)
    local_keys, key_regions, dominant_key, key_confidence = estimate_local_keys(
''',
)

# Key context should resolve close inversion/quality ties, but remain proportional to local-key
# confidence so strong borrowed chords can still win on acoustic evidence.
replace_once(
    "src/sgchords/analyzer.py",
    '''                score += (0.07 if is_diatonic(spec, root, mode) else -0.018) * confidence
''',
    '''                score += (0.10 if is_diatonic(spec, root, mode) else -0.16) * confidence
''',
)

# The previous 0.68 default was too stable for two-beat and arpeggiated changes. 0.50 retains
# temporal continuity while allowing two adjacent evidence windows to establish a new chord.
for path in (
    "src/sgchords/analyzer.py",
    "src/sgchords/service.py",
    "src/sgchords/cli.py",
    "src/sgchords/ui.py",
):
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    text = text.replace("0.68", "0.50")
    target.write_text(text, encoding="utf-8")
replace_once(
    "scripts/benchmark_known_chords.py",
    '''    smoothing: float = 0.68
''',
    '''    smoothing: float = 0.50
''',
)

# Report both root-progression performance and exact-quality performance. A detected G instead of
# published G7 is a quality miss, but not a root-progression miss for a guitar play-along chart.
replace_once(
    "scripts/benchmark_known_chords.py",
    '''def collapse_sequence(segments: Sequence[ChordSegment]) -> list[str]:
''',
    '''def root_sequence(sequence: Sequence[str]) -> list[str]:
    roots: list[str] = []
    for chord in sequence:
        parsed = parse_chord(chord)
        if parsed is None:
            continue
        root = pitch_name(parsed.root, "b" in chord[:2])
        if not roots or roots[-1] != root:
            roots.append(root)
    return roots


def collapse_sequence(segments: Sequence[ChordSegment]) -> list[str]:
''',
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''    score = score_sequence(
        song.expected_sequence,
        detected,
        allow_global_transposition=song.allow_global_transposition,
        allow_repetition=song.allow_repetition,
    )
''',
    '''    score = score_sequence(
        song.expected_sequence,
        detected,
        allow_global_transposition=song.allow_global_transposition,
        allow_repetition=song.allow_repetition,
    )
    root_score = score_sequence(
        root_sequence(song.expected_sequence),
        root_sequence(detected),
        allow_global_transposition=song.allow_global_transposition,
        allow_repetition=song.allow_repetition,
    )
''',
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''        "score": score,
''',
    '''        "score": score,
        "root_score": root_score,
''',
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''        "| Recording | Key | Tempo | Mean confidence | Sequence recall | Sequence F1 | Vocabulary recall |",
        "|---|---|---:|---:|---:|---:|---:|",
''',
    '''        "| Recording | Key | Tempo | Mean confidence | Root recall | Root F1 | Quality vocabulary |",
        "|---|---|---:|---:|---:|---:|---:|",
''',
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''        score = result["score"]
        tempo = "—" if result["tempo_bpm"] is None else f"{result['tempo_bpm']:.1f}"
        lines.append(
            f"| {result['song']['title']} | {result['dominant_key']} | {tempo} | "
            f"{result['mean_chord_confidence']:.0%} | {score['sequence_recall']:.0%} | "
            f"{score['sequence_f1']:.0%} | {score['vocabulary_recall']:.0%} |"
        )
''',
    '''        score = result["score"]
        root_score = result["root_score"]
        tempo = "—" if result["tempo_bpm"] is None else f"{result['tempo_bpm']:.1f}"
        lines.append(
            f"| {result['song']['title']} | {result['dominant_key']} | {tempo} | "
            f"{result['mean_chord_confidence']:.0%} | {root_score['sequence_recall']:.0%} | "
            f"{root_score['sequence_f1']:.0%} | {score['vocabulary_recall']:.0%} |"
        )
''',
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''        score = result["score"]
        lines.extend(
''',
    '''        score = result["score"]
        root_score = result["root_score"]
        lines.extend(
''',
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''                    f"Best reference shift: {score['shift_semitones']:+d} semitones; "
                    f"sequence recall {score['sequence_recall']:.0%}; sequence precision "
                    f"{score['sequence_precision']:.0%}; F1 {score['sequence_f1']:.0%}; "
                    f"vocabulary recall {score['vocabulary_recall']:.0%}."
''',
    '''                    f"Best reference shift: {root_score['shift_semitones']:+d} semitones; "
                    f"root-sequence recall {root_score['sequence_recall']:.0%}; root precision "
                    f"{root_score['sequence_precision']:.0%}; root F1 {root_score['sequence_f1']:.0%}; "
                    f"quality-aware sequence F1 {score['sequence_f1']:.0%}; "
                    f"quality vocabulary recall {score['vocabulary_recall']:.0%}."
''',
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''    aggregate_recall = sum(item["score"]["sequence_recall"] for item in exact) / max(len(exact), 1)
    aggregate_vocab = sum(item["score"]["vocabulary_recall"] for item in exact) / max(len(exact), 1)
''',
    '''    aggregate_recall = sum(item["root_score"]["sequence_recall"] for item in exact) / max(len(exact), 1)
    aggregate_f1 = sum(item["root_score"]["sequence_f1"] for item in exact) / max(len(exact), 1)
    aggregate_vocab = sum(item["score"]["vocabulary_recall"] for item in exact) / max(len(exact), 1)
''',
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''            f"Mean literal-pitch sequence recall across the four exact-pitch clips: {aggregate_recall:.0%}.",
            "",
            f"Mean literal-pitch chord-vocabulary recall across those clips: {aggregate_vocab:.0%}.",
''',
    '''            f"Mean literal-pitch root-sequence recall across the four exact-pitch clips: {aggregate_recall:.0%}.",
            "",
            f"Mean literal-pitch root-sequence F1 across those clips: {aggregate_f1:.0%}.",
            "",
            f"Mean exact-quality chord-vocabulary recall across those clips: {aggregate_vocab:.0%}.",
''',
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''    aggregate = sum(item["score"]["sequence_recall"] for item in exact) / max(len(exact), 1)
''',
    '''    aggregate = sum(item["root_score"]["sequence_recall"] for item in exact) / max(len(exact), 1)
''',
)

with Path("tests/test_chords.py").open("a", encoding="utf-8") as handle:
    handle.write(
        '''


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
'''
    )
