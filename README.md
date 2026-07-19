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

- Python 3.10 or newer
- FFmpeg and FFprobe on `PATH`
- For reliable YouTube access, a current `yt-dlp` installation and a supported JavaScript runtime such as Node.js or Deno

Only download or process recordings you are allowed to use. Some YouTube videos require authentication, cookies, or regional access and will not be available to the tool.

## Install and run the web app

```bash
git clone https://github.com/Legedith/Sgchords.git
cd Sgchords
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and launch:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
sgchords-web
```

Open `http://127.0.0.1:7860`.

## Command line

```bash
sgchords "https://www.youtube.com/watch?v=VIDEO_ID"
sgchords ./recording.m4a --instrument ukulele
sgchords ./song.mp3 --detail detailed --output-dir ./my-chords
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

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

The tests include deterministic chroma classification, synthetic audio analysis, FFmpeg normalization, exports, and a Gradio UI smoke test.

## Roadmap

The highest-value next improvement is an optional learned chord-recognition backend for difficult full-band recordings, while retaining this lightweight offline engine as a fallback. Other useful additions are lyric alignment, a tap-to-correct mobile view, and support for non-equal-temperament or explicitly modal analysis.
