# Public-domain song benchmark

This benchmark is deliberately small and reproducible. It is not a leaderboard and it does not prove that the analyzer will work on every folk tradition.

The audio is downloaded temporarily during the run and is never committed to this repository.

| Recording | Arrangement | Rights | Reference used |
|---|---|---|---|
| Hotaru no Hikari (Auld Lang Syne in Japan) | Guitar and vocal | Public-domain dedication by the performer | The Wikimedia Commons description publishes `C → G → Am → F`, `C → G → F → G → C`, `C → G → C/G → F`, `C → G → F → G → C`. The benchmark compares the detected simple-chord vocabulary with `C`, `G`, `Am`, and `F`. It also reports the best single global transposition because published guitar labels can be capo-relative. Slash-bass recognition is not implemented. |
| Amazing Grace, instrumental arrangement | Synthesized instrumental | Public-domain dedication by the creator | No arrangement-level chord annotation is published. Results are reported descriptively rather than scored as correct or incorrect. |
| Scarborough Fair | Vocal/instrumental traditional performance | GFDL/CC BY-SA licensed recording of a traditional song | This is a modal stress test. A forced major/minor key label and triadic chord stream may be musically incomplete even when pitch detection is functioning. |

Run it with:

```bash
uv run --frozen python scripts/benchmark_public_domain.py --output benchmark-results
```

Generated files:

- `results.md`: readable summary
- `results.json`: full benchmark metadata
- one `*-timeline.csv` file per recording

Interpret the output using duration-weighted chord confidence, the warnings, and the recording type. The Hotaru coverage score checks whether the expected chord vocabulary appears after one global pitch shift; it does not validate exact change times. A high score on one arrangement does not imply that every segment is correct, and low confidence on a modal or drone-based recording does not automatically mean the pitch analysis failed.
