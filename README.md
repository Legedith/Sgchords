# SgChords

SgChords turns a YouTube link or an uploaded recording into a synchronized, editable guitar or ukulele play-along chart. It runs locally, uses `uv`, and does not require a paid API or a large model download.

The analyzer is intentionally honest: automatic chords are a draft. The app exposes confidence, local tonal regions, detected drones, tuning offset, beat and meter confidence, and editable corrections instead of pretending every label is certain.

## Highlights

### Play along in sync

- Current chord, next chord, and countdown update with audio playback.
- Click any chord to seek to it.
- Double-click a chord and enable looping for focused practice.
- Jump backward or forward five seconds.
- Practice at 0.75×, 1×, or 1.25× speed without changing pitch.
- The chord strip scrolls automatically as the recording plays.

### Guitar and ukulele tools

- Concert-pitch transposition from −11 to +11 semitones.
- Capo shapes without changing the sounding key.
- Automatic capo suggestion based on duration-weighted playability.
- Standard chord names, Roman numerals, or Nashville numbers.
- Inline guitar and ukulele chord diagrams.
- Slash-chord and bass-note estimates when the bass is clear.

### Better analysis for folk and local music

- Beat-synchronous harmonic chroma with tuning correction.
- Harmonic-change refinement around beat boundaries.
- Local key and mode tracking instead of one global-key assumption.
- Major, minor, Dorian, Mixolydian, and Phrygian candidates.
- Sustained-drone detection and partial drone suppression before chord scoring.
- Meter and downbeat estimates with manual 3-, 4-, or 6-beat override.
- Repeated progression detection.
- Simple, standard, and detailed chord vocabularies.

### Edit and export

- Edit chord names, timings, local keys, bars, and beats in a table.
- Quantize boundaries to detected beats.
- Merge adjacent repeated chords.
- Export text, CSV, JSON, ChordPro, chord-shape notes, SRT subtitles, and Audacity/MIREX-style LAB labels.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) 0.11.29
- FFmpeg and FFprobe on `PATH`
- A supported JavaScript runtime such as Node.js or Deno for the most reliable current YouTube extraction

Only download or process recordings you are allowed to use. YouTube can require authentication, cookies, or regional access, and platform changes can temporarily break automated extraction. Uploading a local file is the most reliable input path.

## Install and run with uv

```bash
git clone https://github.com/Legedith/Sgchords.git
cd Sgchords
uv sync --locked
uv run --frozen sgchords-web
```

Open `http://127.0.0.1:7860`.

`uv sync` creates and manages `.venv`; no manual activation or `pip install` step is required.

## Command line

```bash
uv run --frozen sgchords ./recording.mp3
uv run --frozen sgchords ./recording.m4a --instrument ukulele
uv run --frozen sgchords ./song.wav --detail detailed --meter 3
uv run --frozen sgchords "https://www.youtube.com/watch?v=VIDEO_ID"
```

Useful options:

```text
--detail simple|standard|detailed
--smoothing 0..1
--meter auto|3|4|6
--transpose -11..11
--capo 0..7
--notation standard|roman|nashville
--instrument guitar|ukulele
```

Keep `--transpose 0` when playing with the original recording. Transposition changes the concert key. A capo changes the displayed shapes while preserving the concert pitch.

## Docker

```bash
docker build -t sgchords .
docker run --rm -p 7860:7860 sgchords
```

The image installs from the committed `uv.lock` and includes FFmpeg and Node.js.

## Known-chord benchmark

The repository includes a reproducible benchmark over five openly licensed recordings whose chord content is stated by the source: two exact C-major progressions, I–V–vi–IV resolving to I, a C–F–G7 guitar clip, and Hotaru no Hikari/Auld Lang Syne.

```bash
uv run --frozen python scripts/benchmark_known_chords.py --output benchmark-results
```

See [`benchmarks/README.md`](benchmarks/README.md) for methodology and [`benchmarks/results.md`](benchmarks/results.md) for the latest checked results. Audio is downloaded temporarily and is never committed.

## Development

```bash
uv sync --locked
uv run --frozen ruff check .
uv run --frozen pytest
uv build
```

When dependencies intentionally change:

```bash
uv lock
uv sync --locked
```

## How the analyzer works

1. FFmpeg normalizes the source to mono 22.05 kHz WAV.
2. Harmonic audio is separated from percussion.
3. A tuning-aware Constant-Q chromagram represents pitch-class energy.
4. A low-frequency chromagram estimates bass notes and inversions.
5. Beat tracking creates the practice grid; a clearly marked fixed grid is used when tracking fails.
6. Harmonic novelty nudges candidate boundaries toward chord changes.
7. Sustained drone energy is estimated and partly removed before template scoring.
8. A Viterbi tracker estimates local tonal regions across major, minor, Dorian, Mixolydian, and Phrygian modes.
9. Chord templates are scored with local-key priors, bass evidence, silence handling, and transition smoothing.
10. Consecutive matches are merged, mapped to bars and beats, and searched for repeated patterns.

## Limits that still matter

No algorithm can guarantee the exact chords of every recording. Dense orchestration, melody notes that dominate the mix, poor field recordings, non-equal temperament, ambiguous power chords, rapid substitutions, rubato, and traditions organized around drones rather than functional harmony remain difficult.

The local-key and modal tracker improves interpretation, but its mode labels are hypotheses, not musicological proof. A low-confidence editable result is more useful than a confident wrong chart; check difficult passages by ear.
