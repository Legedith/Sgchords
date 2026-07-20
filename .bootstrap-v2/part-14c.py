from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/sgchords/analyzer.py",
    '''    if prevalence < 0.84 or dominance < 1.20 or confidence < 0.45 or top < 0.10:
        return None, confidence, matrix
''',
    '''    if prevalence < 0.84 or dominance < 1.20 or confidence < 0.45 or top < 0.10:
        # Confidence describes an accepted drone hypothesis. A rejected common-tone candidate
        # must not leak a high value into the UI or JSON result.
        return None, min(confidence, 0.44), matrix
''',
)

with Path("tests/test_benchmark.py").open("a", encoding="utf-8") as handle:
    handle.write(
        '''


def test_sequence_score_does_not_penalize_a_repeated_published_motif() -> None:
    result = score_sequence(
        ["C", "F", "G7"],
        ["C", "F", "G7", "C", "F", "G7"],
        allow_global_transposition=False,
        allow_repetition=True,
    )
    assert result["sequence_recall"] == 1.0
    assert result["sequence_precision"] == 1.0
    assert result["sequence_f1"] == 1.0
'''
    )
