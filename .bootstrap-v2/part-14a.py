from pathlib import Path

Path("tests/test_benchmark.py").write_text(
    '''from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_known_chords.py"
SPEC = importlib.util.spec_from_file_location("sgchords_known_chord_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

score_sequence = MODULE.score_sequence


def test_sequence_score_rewards_exact_progression() -> None:
    result = score_sequence(
        ["C", "F", "G", "C"],
        ["C", "F", "G", "C"],
        allow_global_transposition=False,
    )
    assert result["shift_semitones"] == 0
    assert result["sequence_recall"] == 1.0
    assert result["sequence_precision"] == 1.0
    assert result["sequence_f1"] == 1.0
    assert result["vocabulary_recall"] == 1.0


def test_sequence_score_handles_capo_relative_reference() -> None:
    result = score_sequence(
        ["C", "G", "Am", "F"],
        ["D", "A", "Bm", "G"],
        allow_global_transposition=True,
    )
    assert result["shift_semitones"] == 2
    assert result["aligned_expected"] == ["D", "A", "Bm", "G"]
    assert result["sequence_recall"] == 1.0
    assert result["sequence_f1"] == 1.0
    assert result["missing_vocabulary"] == []
    assert result["extra_vocabulary"] == []
''',
    encoding="utf-8",
)
