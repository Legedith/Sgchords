from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_public_domain.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sgchords_benchmark_script",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

align_reference_chords = MODULE.align_reference_chords


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
