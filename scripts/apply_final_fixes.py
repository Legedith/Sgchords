from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/sgchords/analyzer.py",
    """    estimate_key,
    is_diatonic,
    transpose_chord,
)""",
    """    estimate_key,
    is_diatonic,
    key_name,
    key_prefers_flats,
    transpose_chord,
)""",
)
replace_once(
    "src/sgchords/analyzer.py",
    "    prefer_flats = key_root in {1, 3, 5, 8, 10}\n",
    "    prefer_flats = key_prefers_flats(key_name(key_root, key_mode))\n",
)

helper = '''def _diatonic_duration_ratio(
    segments: Sequence[ChordSegment],
    *,
    detail: str,
    key_root: int,
    key_mode: str,
) -> float | None:
    specs = chord_vocabulary(detail)
    prefer_flats = key_prefers_flats(key_name(key_root, key_mode))
    by_label = {
        transpose_chord(spec.label, 0, prefer_flats=prefer_flats): spec
        for spec in specs
    }
    total_duration = 0.0
    diatonic_duration = 0.0
    for segment in segments:
        if segment.chord == "N" or segment.duration <= 0:
            continue
        total_duration += segment.duration
        spec = by_label.get(segment.chord)
        if spec is not None and is_diatonic(spec, key_root, key_mode):
            diatonic_duration += segment.duration
    if total_duration <= 0:
        return None
    return diatonic_duration / total_duration


'''
replace_once(
    "src/sgchords/analyzer.py",
    "\ndef analyze_audio(\n",
    "\n" + helper + "def analyze_audio(\n",
)
replace_once(
    "src/sgchords/analyzer.py",
    """    mean_tonality = float(np.mean(tonalities)) if tonalities else 0.0
    if not chord_segments:
""",
    """    mean_tonality = float(np.mean(tonalities)) if tonalities else 0.0
    diatonic_ratio = _diatonic_duration_ratio(
        chord_segments,
        detail=detail,
        key_root=key_root,
        key_mode=key_mode,
    )
    if not chord_segments:
""",
)
replace_once(
    "src/sgchords/analyzer.py",
    """    if key_confidence < 0.18:
""",
    """    if diatonic_ratio is not None and diatonic_ratio < 0.70:
        warnings.append(
            f"Only about {diatonic_ratio:.0%} of detected chord duration fits one {key} "
            "harmony. The song may modulate, be modal, or use borrowed chords; treat the "
            "global key label cautiously."
        )
    if key_confidence < 0.18:
""",
)

replace_once(
    "tests/test_chords.py",
    "from sgchords.analyzer import classify_chroma_segments\n",
    """from sgchords.analyzer import (
    _diatonic_duration_ratio,
    classify_chroma_segments,
)
""",
)
with Path("tests/test_chords.py").open("a", encoding="utf-8") as handle:
    handle.write(
        '''


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
'''
    )

Path("tests/test_benchmark.py").write_text(
    '''from scripts.benchmark_public_domain import align_reference_chords


def test_reference_alignment_handles_capo_or_global_transposition() -> None:
    result = align_reference_chords(
        ["C", "G", "Am", "F"],
        ["D", "A", "Bm", "G"],
    )
    assert result["literal_coverage"] == 0.25
    assert result["best_shift"] == 2
    assert result["aligned_expected"] == ["D", "A", "Bm", "G"]
    assert result["aligned_coverage"] == 1.0
    assert result["missing"] == []
    assert result["extra"] == []
''',
    encoding="utf-8",
)

replace_once(
    "README.md",
    """The benchmark covers a guitar-and-vocal recording with a published chord progression, an instrumental hymn arrangement, and a modal folk-song stress test. See `benchmarks/README.md` and the generated `benchmark-results/results.md` for interpretation.
""",
    """The benchmark covers a guitar-and-vocal recording with a published chord progression, an instrumental hymn arrangement, and a modal folk-song stress test. See the [benchmark design](benchmarks/README.md) and [latest checked results](benchmarks/results.md).
""",
)

Path(".github/workflows/ci.yml").write_text(
    '''name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - name: Install FFmpeg
        run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - name: Install uv and Python
        uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b # v8.1.0
        with:
          version: "0.11.29"
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      - name: Sync locked environment
        run: uv sync --locked
      - name: Lint
        run: uv run --frozen ruff check .
      - name: Test
        run: uv run --frozen pytest
      - name: Build
        run: uv build

  public-domain-benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install FFmpeg
        run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - name: Install uv and Python
        uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b # v8.1.0
        with:
          version: "0.11.29"
          python-version: "3.12"
          enable-cache: true
      - name: Sync locked environment
        run: uv sync --locked
      - name: Analyze public-domain songs
        run: uv run --frozen python scripts/benchmark_public_domain.py --output benchmark-results
      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: public-domain-song-results
          path: benchmark-results
          if-no-files-found: error
''',
    encoding="utf-8",
)
