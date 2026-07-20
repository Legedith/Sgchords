# Known-chord benchmark

This benchmark runs the production analyzer on five openly licensed recordings whose chord content is published by the source.

| Recording | Published reference | Evaluation |
|---|---|---|
| C major I–IV–V–I | `C → F → G → C` | Literal-pitch sequence and vocabulary |
| C major I–IV–V–I–I–V–I | `C → F → G → C → C → G → C` | Literal-pitch sequence and vocabulary |
| I–V–vi–IV resolving to I | `C → G → Am → F → C` | Literal-pitch sequence and vocabulary; inversions reduced to roots |
| La Bamba guitar clip | `C → F → G7` | Literal-pitch order and vocabulary |
| Hotaru no Hikari / Auld Lang Syne | Published four-line progression | Sequence and vocabulary with one permitted global transposition, because guitar chord names can be capo-relative |

Run it with uv:

```bash
uv sync --locked
uv run --frozen python scripts/benchmark_known_chords.py --output benchmark-results
```

The audio files are downloaded into a temporary directory and never committed. Generated output contains:

- `results.md`: readable report
- `results.json`: structured metrics and analyzer metadata
- `timelines/*.csv`: every detected segment with confidence, local key, bar, and beat

## What the score means

Adjacent duplicate labels are collapsed, then the detected sequence is compared with the reference using longest-common-subsequence matching. Sequence recall asks how much of the expected order appeared. Sequence precision penalizes excessive extra changes. Vocabulary recall ignores timing and asks whether the expected chord types appeared at all.

This is deliberately stricter than reporting only chord-set coverage, but it still does not validate exact boundary timestamps because the source recordings do not publish frame-accurate annotations.
