from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .audio import is_youtube_url
from .chords import suggest_capo
from .exports import render_bar_chart, render_summary, write_exports
from .service import analyze_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate synchronized chords, bars, local keys, capo shapes and play-along charts.",
    )
    parser.add_argument("source", help="YouTube URL or local audio/video path")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("sgchords-output"))
    parser.add_argument("--detail", choices=("simple", "standard", "detailed"), default="standard")
    parser.add_argument("--smoothing", type=float, default=0.68)
    parser.add_argument("--meter", choices=("auto", "3", "4", "6"), default="auto")
    parser.add_argument(
        "--transpose", type=int, choices=range(-11, 12), default=0, metavar="[-11..11]"
    )
    parser.add_argument(
        "--capo", choices=("auto", "0", "1", "2", "3", "4", "5", "6", "7"), default="auto"
    )
    parser.add_argument(
        "--notation", choices=("standard", "roman", "nashville"), default="standard"
    )
    parser.add_argument("--instrument", choices=("guitar", "ukulele"), default="guitar")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.smoothing <= 1:
        parser.error("--smoothing must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meter = None if args.meter == "auto" else int(args.meter)
    try:
        result, job_dir = analyze_request(
            youtube_url=args.source if is_youtube_url(args.source) else None,
            upload_path=None if is_youtube_url(args.source) else args.source,
            detail=args.detail,
            smoothing=args.smoothing,
            meter_override=meter,
            workspace_root=args.output_dir / ".work",
        )
        suggested, _shapes, _ease = suggest_capo(result.segments, args.instrument)
        capo = suggested if args.capo == "auto" else int(args.capo)
        files = write_exports(
            result,
            args.output_dir,
            transpose=args.transpose,
            instrument=args.instrument,
            capo=capo,
            notation=args.notation,
            meter_override=meter,
        )
        print(
            render_summary(
                result,
                result.segments,
                transpose=args.transpose,
                instrument=args.instrument,
                capo=capo,
            )
        )
        print()
        print(
            render_bar_chart(
                result,
                result.segments,
                transpose=args.transpose,
                capo=capo,
                notation=args.notation,
                meter_override=meter,
            )
        )
        print("\nCreated:")
        for path in files:
            print(f"  {path}")
        shutil.rmtree(job_dir, ignore_errors=True)
        return 0
    except Exception as exc:
        print(f"sgchords: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
