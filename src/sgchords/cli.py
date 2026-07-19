from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .audio import is_youtube_url
from .exports import render_summary, write_exports
from .service import analyze_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sgchords",
        description="Estimate time-aligned song chords from a YouTube URL or local media file.",
    )
    parser.add_argument("source", help="YouTube URL or local audio/video path")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("sgchords-output"),
        help="Directory for exported files (default: ./sgchords-output)",
    )
    parser.add_argument(
        "--detail",
        choices=("simple", "detailed"),
        default="simple",
        help="Chord vocabulary to use",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.65,
        help="Sequence smoothing from 0 to 1 (default: 0.65)",
    )
    parser.add_argument(
        "--transpose",
        type=int,
        default=0,
        choices=range(-11, 12),
        metavar="[-11..11]",
        help="Transpose exported chord names in semitones",
    )
    parser.add_argument(
        "--instrument",
        choices=("guitar", "ukulele"),
        default="guitar",
        help="Instrument used for shape and capo suggestions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.smoothing <= 1.0:
        parser.error("--smoothing must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workspace = args.output_dir / ".work"
    try:
        result, job_dir = analyze_request(
            youtube_url=args.source if is_youtube_url(args.source) else None,
            upload_path=None if is_youtube_url(args.source) else args.source,
            detail=args.detail,
            smoothing=args.smoothing,
            workspace_root=workspace,
        )
        files = write_exports(
            result,
            args.output_dir,
            transpose=args.transpose,
            instrument=args.instrument,
        )
        print(render_summary(result, result.segments, instrument=args.instrument))
        print("\nCreated:")
        for file in files:
            print(f"  {file}")
        shutil.rmtree(job_dir, ignore_errors=True)
        return 0
    except Exception as exc:
        print(f"sgchords: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
