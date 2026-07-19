# SgChords

SgChords estimates time-aligned chords from a YouTube URL or an uploaded audio/video file. It is designed for guitar and ukulele players who want a usable first chord sheet for songs that have never been transcribed online.

It produces an editable timeline, a compact play-along sheet, CSV, JSON, ChordPro, common chord shapes, and a capo suggestion.

## Important limitation

Automatic chord recognition is not exact. The result is a draft to check by ear, not a claim that every chord is correct. Dense arrangements, lead vocals, drums, drones, modal folk songs, microtonal instruments, poor live recordings, and instruments tuned away from A=440 can all reduce accuracy. SgChords shows confidence, tuning offset, and warnings so uncertainty is visible.

## How it works

1. A local file is decoded, or `yt-dlp` obtains the audio from a YouTube URL.
2. FFmpeg converts it to mono 22.05 kHz WAV.
3. Harmonic audio is separated from percussion.
4. A Constant-Q chromagram captures pitch-class energy; automatic tuning correction handles recordings slightly sharp or flat.
5. Beat-synchronous windows are compared with chord templates.
6. A key-aware Viterbi decoder suppresses implausible one-frame chord flicker.
7. Adjacent identical estimates are merged into a timeline that can be edited before export.

The built-in engine runs locally and does not require a paid API or a large model download.

## Requirements

- `uv` 0.11.29
- FFmpeg and FFprobe on `PATH`
- For reliable YouTube access, a supported JavaScript runtime such as Node.js or Deno

Install `uv` with the official standalone installer:

```bash
# Linux and macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Only download or process recordings you are allowed to use. Some YouTube videos require authentication, cookies, or regional access and will not be available to the tool.

## Install and run the web app with uv

```bash
git clone https://github.com/Legedith/Sgchords.git
cd Sgchords
uv sync --locked
uv run --frozen sgchords-web
```

`uv sync` creates and manages `.venv` automatically. No manual virtual-environment activation or `pip install` step is needed.

Open `http://127.0.0.1:7860`.

## Command line

```bash
uv run --frozen sgchords "https://www.youtube.com/watch?v=VIDEO_ID"
uv run --frozen sgchords ./recording.m4a --instrument ukulele
uv run --frozen sgchords ./song.mp3 --detail detailed --output-dir ./my-chords
```

Useful options:

```text
--detail simple|detailed   Major/minor only, or include 7/maj7/m7/sus/dim
--smoothing 0..1           Higher values reduce short chord flicker
--transpose -11..11        Transpose exported chord names
--instrument guitar|ukulele
```

Keep `--transpose 0` when playing with the original recording. A capo suggestion changes the shapes while preserving the sounding key; ordinary transposition changes the concert key.

## Docker

The image also installs and runs the project with the committed uv lockfile:

```bash
docker build -t sgchords .
docker run --rm -p 7860:7860 sgchords
```

Then open `http://127.0.0.1:7860`.

## Output files

Each analysis can export:

- `*-chords.txt`: compact time-stamped chord sheet
- `*-timeline.csv`: editable tabular timeline
- `*-analysis.json`: machine-readable metadata and confidence
- `*.cho`: ChordPro document
- `*-guitar-shapes.txt` or `*-ukulele-shapes.txt`: common voicings

## Public-domain song benchmark

The repository contains a reproducible benchmark for three openly usable folk recordings. It downloads the recordings temporarily, analyzes them, and writes Markdown, JSON, and CSV results without committing the audio:

```bash
uv run --frozen python scripts/benchmark_public_domain.py --output benchmark-results
```

The benchmark covers a guitar-and-vocal recording with a published chord progression, an instrumental hymn arrangement, and a modal folk-song stress test. See the [benchmark design](benchmarks/README.md) and [latest checked results](benchmarks/results.md).

## Configuration

Environment variables:

```text
SGCHORDS_MAX_DURATION_SECONDS=1200   Maximum song length; default 20 minutes
SGCHORDS_WORKSPACE=/path/to/cache    Generated job/output directory
SGCHORDS_JOB_TTL_SECONDS=86400       Cleanup age for generated jobs
SGCHORDS_COOKIES_FILE=/path/cookies.txt  Optional Netscape cookie file for permitted access
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
```

## Development with uv

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

The tests include deterministic chroma classification, synthetic audio analysis, FFmpeg normalization, exports, and a Gradio UI smoke test.

## Roadmap

The highest-value next improvement is an optional learned chord-recognition backend for difficult full-band recordings, while retaining this lightweight offline engine as a fallback. Other useful additions are lyric alignment, a tap-to-correct mobile view, and support for non-equal-temperament or explicitly modal analysis.
