from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sgchords.analyzer import analyze_audio
from sgchords.audio import prepare_local_audio
from sgchords.chords import simplify_chord

USER_AGENT = "SgChords public-domain benchmark/0.1 (+https://github.com/Legedith/Sgchords)"


@dataclass(frozen=True, slots=True)
class BenchmarkSong:
    slug: str
    title: str
    file_url: str
    source_page: str
    license_note: str
    arrangement_note: str
    expected_key: str | None = None
    expected_chords: tuple[str, ...] = ()


SONGS = (
    BenchmarkSong(
        slug="hotaru-no-hikari",
        title="Hotaru no Hikari (Auld Lang Syne in Japan)",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/f/f4/"
            "Hotaru_no_Hikari%28Auld_lang_syne_in_Japan%29.ogg"
        ),
        source_page=(
            "https://commons.wikimedia.org/wiki/"
            "File:Hotaru_no_Hikari(Auld_lang_syne_in_Japan).ogg"
        ),
        license_note="Public-domain dedication by the performer.",
        arrangement_note="Guitar and vocal; the source publishes a C/G/Am/F progression.",
        expected_key="C major",
        expected_chords=("C", "G", "Am", "F"),
    ),
    BenchmarkSong(
        slug="amazing-grace",
        title="Amazing Grace (instrumental arrangement)",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/4/4f/"
            "Anonimo_-_Amazing_Grace.ogg"
        ),
        source_page="https://commons.wikimedia.org/wiki/File:Anonimo_-_Amazing_Grace.ogg",
        license_note="Public-domain dedication by the creator.",
        arrangement_note="Synthesized instrumental; no arrangement-level chord reference.",
    ),
    BenchmarkSong(
        slug="scarborough-fair",
        title="Scarborough Fair",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/a/a4/Scarborough_Fair.ogg"
        ),
        source_page="https://commons.wikimedia.org/wiki/File:Scarborough_Fair.ogg",
        license_note="GFDL/CC BY-SA recording of a traditional song.",
        arrangement_note="Modal folk-song stress test; major/minor labels may be incomplete.",
    ),
)


def download(url: str, destination: Path, *, attempts: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, destination.open(
                "wb"
            ) as output:
                shutil.copyfileobj(response, output)
            if destination.stat().st_size < 1_000:
                raise RuntimeError("downloaded file is unexpectedly small")
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
            if attempt < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def duration_weighted_confidence(segments: list[Any]) -> float:
    sounding = [segment for segment in segments if segment.chord != "N"]
    total = sum(segment.duration for segment in sounding)
    if total <= 0:
        return 0.0
    return sum(segment.confidence * segment.duration for segment in sounding) / total


def analyze_song(song: BenchmarkSong, download_dir: Path, work_dir: Path, output: Path) -> dict[str, Any]:
    media_path = download_dir / f"{song.slug}.ogg"
    download(song.file_url, media_path)

    started = time.perf_counter()
    prepared = prepare_local_audio(media_path, work_dir, max_duration_seconds=600)
    result = analyze_audio(prepared.path, detail="simple", smoothing=0.65)
    elapsed = time.perf_counter() - started

    sequence = [segment.chord for segment in result.segments if segment.chord != "N"]
    simple_sequence = [simplify_chord(chord) for chord in sequence]
    unique_chords = ordered_unique(simple_sequence)
    durations: dict[str, float] = defaultdict(float)
    for segment in result.segments:
        durations[simplify_chord(segment.chord)] += segment.duration
    top_chords = [
        {"chord": chord, "seconds": round(seconds, 2), "share": round(seconds / result.duration, 4)}
        for chord, seconds in sorted(durations.items(), key=lambda item: item[1], reverse=True)
        if chord != "N"
    ][:10]

    expected = set(song.expected_chords)
    detected = set(unique_chords)
    coverage = None if not expected else len(expected & detected) / len(expected)
    extra = [] if not expected else sorted(detected - expected)
    missing = [] if not expected else sorted(expected - detected)

    timeline_path = output / f"{song.slug}-timeline.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_seconds", "end_seconds", "chord", "confidence"])
        for segment in result.segments:
            writer.writerow(
                [
                    f"{segment.start:.3f}",
                    f"{segment.end:.3f}",
                    segment.chord,
                    f"{segment.confidence:.3f}",
                ]
            )

    return {
        "song": asdict(song),
        "duration_seconds": round(result.duration, 3),
        "analysis_seconds": round(elapsed, 3),
        "tempo_bpm": None if result.tempo_bpm is None else round(result.tempo_bpm, 2),
        "key": result.key,
        "key_confidence": round(result.key_confidence, 4),
        "expected_key_match": None if song.expected_key is None else result.key == song.expected_key,
        "tuning_cents": round(result.tuning_cents, 2),
        "segment_count": len(result.segments),
        "mean_chord_confidence": round(duration_weighted_confidence(result.segments), 4),
        "unique_chords_in_order": unique_chords,
        "first_changes": simple_sequence[:24],
        "top_chords_by_duration": top_chords,
        "reference_chord_coverage": None if coverage is None else round(coverage, 4),
        "missing_reference_chords": missing,
        "extra_detected_chords": extra,
        "warnings": result.warnings,
        "timeline_csv": timeline_path.name,
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# SgChords public-domain benchmark results",
        "",
        "Generated by `scripts/benchmark_public_domain.py` using simple chords and smoothing 0.65.",
        "The recordings are not committed to the repository.",
        "",
        "| Recording | Duration | Key | Tempo | Mean chord confidence | Reference coverage |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for item in results:
        song = item["song"]
        tempo = "—" if item["tempo_bpm"] is None else f"{item['tempo_bpm']:.1f}"
        coverage = (
            "not scored"
            if item["reference_chord_coverage"] is None
            else f"{item['reference_chord_coverage']:.0%}"
        )
        lines.append(
            f"| {song['title']} | {item['duration_seconds']:.1f}s | {item['key']} "
            f"({item['key_confidence']:.0%}) | {tempo} | "
            f"{item['mean_chord_confidence']:.0%} | {coverage} |"
        )

    for item in results:
        song = item["song"]
        lines.extend(
            [
                "",
                f"## {song['title']}",
                "",
                f"Source: {song['source_page']}",
                "",
                f"Rights: {song['license_note']}",
                "",
                f"Test role: {song['arrangement_note']}",
                "",
                f"Analysis time: {item['analysis_seconds']:.1f}s; "
                f"segments: {item['segment_count']}; tuning: {item['tuning_cents']:+.0f} cents.",
                "",
                "Detected chords in first-occurrence order: "
                + ", ".join(f"`{value}`" for value in item["unique_chords_in_order"]),
                "",
                "First changes: " + " → ".join(item["first_changes"]),
            ]
        )
        if item["reference_chord_coverage"] is not None:
            lines.extend(
                [
                    "",
                    f"Published-chord set coverage: {item['reference_chord_coverage']:.0%}.",
                    "Missing: "
                    + (", ".join(item["missing_reference_chords"]) or "none")
                    + "; extra simple labels: "
                    + (", ".join(item["extra_detected_chords"]) or "none")
                    + ".",
                ]
            )
        if item["warnings"]:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in item["warnings"])

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The Hotaru recording is the only scored arrangement because its source publishes the "
            "actual chord progression. Coverage checks chord vocabulary, not exact timing. The other "
            "two recordings show real-world behaviour but do not have arrangement-specific ground truth.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sgchords-benchmark-") as temporary:
        root = Path(temporary)
        results = [
            analyze_song(song, root / "downloads", root / "work" / song.slug, args.output)
            for song in SONGS
        ]

    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(results)
    (args.output / "results.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
