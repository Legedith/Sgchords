from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sgchords.analyzer import analyze_audio
from sgchords.audio import prepare_local_audio
from sgchords.chords import parse_chord, pitch_name, transpose_chord
from sgchords.models import ChordSegment

USER_AGENT = "SgChords known-chord benchmark/0.2 (+https://github.com/Legedith/Sgchords)"


@dataclass(frozen=True, slots=True)
class BenchmarkSong:
    slug: str
    title: str
    file_url: str
    source_page: str
    license_note: str
    expected_sequence: tuple[str, ...]
    evaluation_note: str
    allow_global_transposition: bool = False
    allow_repetition: bool = False
    detail: str = "standard"
    smoothing: float = 0.68


SONGS = (
    BenchmarkSong(
        slug="c-major-i-iv-v-i",
        title="C major I–IV–V–I",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/5/5a/"
            "C_major%2C_Middle_C%2C_I-IV-V-I_chord_progression.ogg"
        ),
        source_page=(
            "https://commons.wikimedia.org/wiki/"
            "File:C_major,_Middle_C,_I-IV-V-I_chord_progression.ogg"
        ),
        license_note="CC0 / public-domain dedication.",
        expected_sequence=("C", "F", "G", "C"),
        evaluation_note="Exact piano progression named by the source; literal pitch is scored.",
    ),
    BenchmarkSong(
        slug="c-major-i-iv-v-i-i-v-i",
        title="C major I–IV–V–I–I–V–I",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/b/bc/"
            "C_major%2C_Middle_C%2C_I-IV-V-I-I-V-I_chord_progression.ogg"
        ),
        source_page=(
            "https://commons.wikimedia.org/wiki/"
            "File:C_major,_Middle_C,_I-IV-V-I-I-V-I_chord_progression.ogg"
        ),
        license_note="CC0 / public-domain dedication.",
        expected_sequence=("C", "F", "G", "C", "C", "G", "C"),
        evaluation_note="Exact piano progression named by the source; literal pitch is scored.",
    ),
    BenchmarkSong(
        slug="i-v-vi-iv-in-c",
        title="I–V–vi–IV resolving to I in C",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/f/f7/"
            "I-V-vi-IV_chord_progression_in_C.oga"
        ),
        source_page=(
            "https://commons.wikimedia.org/wiki/File:I-V-vi-IV_chord_progression_in_C.oga"
        ),
        license_note="CC BY-SA 3.0 / GFDL.",
        expected_sequence=("C", "G", "Am", "F", "C"),
        evaluation_note=(
            "The source states I–V–vi–IV resolving to I; inversions are reduced to chord roots."
        ),
        allow_repetition=True,
    ),
    BenchmarkSong(
        slug="la-bamba-c-f-g7",
        title="La Bamba guitar chords C–F–G7",
        file_url=("https://upload.wikimedia.org/wikipedia/commons/6/63/La_bamba_chords_cfg7.ogg"),
        source_page=("https://commons.wikimedia.org/wiki/File:La_bamba_chords_cfg7.ogg"),
        license_note="CC BY-SA 4.0.",
        expected_sequence=("C", "F", "G7"),
        evaluation_note=(
            "The source states the played guitar chords C–F–G7; order and vocabulary are scored."
        ),
        allow_repetition=True,
    ),
    BenchmarkSong(
        slug="hotaru-no-hikari",
        title="Hotaru no Hikari / Auld Lang Syne",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/f/f4/"
            "Hotaru_no_Hikari%28Auld_lang_syne_in_Japan%29.ogg"
        ),
        source_page=(
            "https://commons.wikimedia.org/wiki/File:Hotaru_no_Hikari(Auld_lang_syne_in_Japan).ogg"
        ),
        license_note="Public-domain dedication by the performer.",
        expected_sequence=(
            "C",
            "G",
            "Am",
            "F",
            "C",
            "G",
            "F",
            "G",
            "C",
            "C",
            "G",
            "C/G",
            "F",
            "C",
            "G",
            "F",
            "G",
            "C",
        ),
        evaluation_note=(
            "Guitar and vocal. One global transposition is allowed because published guitar "
            "shapes can be capo-relative; timing is not source-annotated."
        ),
        allow_global_transposition=True,
    ),
)


def download(url: str, destination: Path, *, attempts: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                destination.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            if destination.stat().st_size < 1_000:
                raise RuntimeError("downloaded file is unexpectedly small")
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def _quality_family(chord: str) -> str:
    parsed = parse_chord(chord)
    if parsed is None:
        return "N" if chord.strip().upper() in {"N", "NC", "N.C."} else chord
    root, suffix = parsed.root, parsed.suffix
    lower = suffix.lower().split("/", 1)[0]
    if lower.startswith("maj"):
        quality = ""
    elif lower.startswith("m"):
        quality = "m"
    elif lower.startswith("7"):
        quality = "7"
    elif lower.startswith("dim"):
        quality = "dim"
    else:
        quality = ""
    return f"{pitch_name(root, 'b' in chord[:2])}{quality}"


def collapse_sequence(segments: Sequence[ChordSegment]) -> list[str]:
    sequence: list[str] = []
    for segment in segments:
        chord = _quality_family(segment.chord)
        if chord == "N":
            continue
        if not sequence or sequence[-1] != chord:
            sequence.append(chord)
    return sequence


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _signed_shift(value: int) -> int:
    value %= 12
    return value if value <= 6 else value - 12


def _collapse_adjacent(sequence: Sequence[str]) -> list[str]:
    collapsed: list[str] = []
    for item in sequence:
        if not collapsed or collapsed[-1] != item:
            collapsed.append(item)
    return collapsed


def _transpose_reference(sequence: Sequence[str], semitones: int) -> list[str]:
    return _collapse_adjacent(
        [_quality_family(transpose_chord(chord, semitones)) for chord in sequence]
    )


def score_sequence(
    expected: Sequence[str],
    detected: Sequence[str],
    *,
    allow_global_transposition: bool,
    allow_repetition: bool = False,
) -> dict[str, Any]:
    shifts = range(12) if allow_global_transposition else (0,)
    candidates: list[tuple[float, float, float, int, list[str]]] = []
    detected_list = _collapse_adjacent(list(detected))
    for raw_shift in shifts:
        motif = _transpose_reference(expected, raw_shift)
        repeat_counts = range(1, max(2, len(detected_list) // max(len(motif), 1) + 2)) if allow_repetition else (1,)
        for repeats in repeat_counts:
            shifted = _collapse_adjacent(motif * repeats)
            match = lcs_length(shifted, detected_list)
            recall = match / max(len(shifted), 1)
            precision = match / max(len(detected_list), 1)
            f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
            candidates.append((f1, recall, precision, raw_shift, shifted))
    best = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2], -abs(_signed_shift(item[3]))),
    )
    shifted = best[4]
    expected_vocab = set(shifted)
    detected_vocab = set(detected_list)
    overlap = expected_vocab & detected_vocab
    vocabulary_recall = len(overlap) / max(len(expected_vocab), 1)
    vocabulary_precision = len(overlap) / max(len(detected_vocab), 1)
    return {
        "shift_semitones": _signed_shift(best[3]),
        "aligned_expected": shifted,
        "lcs_matches": lcs_length(shifted, detected_list),
        "sequence_recall": round(best[1], 4),
        "sequence_precision": round(best[2], 4),
        "sequence_f1": round(best[0], 4),
        "vocabulary_recall": round(vocabulary_recall, 4),
        "vocabulary_precision": round(vocabulary_precision, 4),
        "missing_vocabulary": sorted(expected_vocab - detected_vocab),
        "extra_vocabulary": sorted(detected_vocab - expected_vocab),
    }


def duration_weighted_confidence(segments: Sequence[ChordSegment]) -> float:
    sounding = [segment for segment in segments if segment.chord != "N"]
    total = sum(segment.duration for segment in sounding)
    if total <= 0:
        return 0.0
    return sum(segment.confidence * segment.duration for segment in sounding) / total


def write_timeline(path: Path, segments: Sequence[ChordSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "start_seconds",
                "end_seconds",
                "chord",
                "confidence",
                "local_key",
                "bar",
                "beat",
            ]
        )
        for segment in segments:
            writer.writerow(
                [
                    f"{segment.start:.3f}",
                    f"{segment.end:.3f}",
                    segment.chord,
                    f"{segment.confidence:.3f}",
                    segment.local_key or "",
                    segment.bar or "",
                    "" if segment.beat is None else f"{segment.beat:.2f}",
                ]
            )


def analyze_song(
    song: BenchmarkSong,
    downloads: Path,
    work: Path,
    output: Path,
) -> dict[str, Any]:
    media = downloads / f"{song.slug}.ogg"
    download(song.file_url, media)
    started = time.perf_counter()
    prepared = prepare_local_audio(media, work, max_duration_seconds=600)
    analyzed = analyze_audio(
        prepared.path,
        detail=song.detail,
        smoothing=song.smoothing,
    )
    elapsed = time.perf_counter() - started
    detected = collapse_sequence(analyzed.segments)
    score = score_sequence(
        song.expected_sequence,
        detected,
        allow_global_transposition=song.allow_global_transposition,
        allow_repetition=song.allow_repetition,
    )
    timeline_path = output / "timelines" / f"{song.slug}.csv"
    write_timeline(timeline_path, analyzed.segments)
    return {
        "song": asdict(song),
        "duration_seconds": round(analyzed.duration, 3),
        "analysis_seconds": round(elapsed, 3),
        "tempo_bpm": None if analyzed.tempo_bpm is None else round(analyzed.tempo_bpm, 2),
        "tempo_confidence": round(analyzed.tempo_confidence, 4),
        "meter": analyzed.meter,
        "meter_confidence": round(analyzed.meter_confidence, 4),
        "dominant_key": analyzed.key,
        "key_confidence": round(analyzed.key_confidence, 4),
        "local_key_regions": [asdict(region) for region in analyzed.key_regions],
        "tuning_cents": round(analyzed.tuning_cents, 2),
        "drone": analyzed.drone,
        "drone_confidence": round(analyzed.drone_confidence, 4),
        "segment_count": len(analyzed.segments),
        "mean_chord_confidence": round(duration_weighted_confidence(analyzed.segments), 4),
        "detected_sequence": detected,
        "first_detected_changes": detected[:32],
        "score": score,
        "warnings": analyzed.warnings,
        "timeline_csv": str(timeline_path.relative_to(output)),
    }


def render_markdown(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Known-chord benchmark results",
        "",
        (
            "Generated by `scripts/benchmark_known_chords.py` with the same analyzer used by "
            "the web app. Audio is downloaded temporarily and is not committed."
        ),
        "",
        (
            "Sequence scores use longest-common-subsequence matching after collapsing adjacent "
            "duplicate labels. They evaluate order, not exact boundary timestamps."
        ),
        "",
        "| Recording | Key | Tempo | Mean confidence | Sequence recall | Sequence F1 | Vocabulary recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        score = result["score"]
        tempo = "—" if result["tempo_bpm"] is None else f"{result['tempo_bpm']:.1f}"
        lines.append(
            f"| {result['song']['title']} | {result['dominant_key']} | {tempo} | "
            f"{result['mean_chord_confidence']:.0%} | {score['sequence_recall']:.0%} | "
            f"{score['sequence_f1']:.0%} | {score['vocabulary_recall']:.0%} |"
        )

    for result in results:
        song = result["song"]
        score = result["score"]
        lines.extend(
            [
                "",
                f"## {song['title']}",
                "",
                f"Source: {song['source_page']}",
                "",
                f"Rights: {song['license_note']}",
                "",
                f"Evaluation: {song['evaluation_note']}",
                "",
                "Expected: " + " → ".join(song["expected_sequence"]),
                "",
                "Detected: " + " → ".join(result["detected_sequence"]),
                "",
                (
                    f"Best reference shift: {score['shift_semitones']:+d} semitones; "
                    f"sequence recall {score['sequence_recall']:.0%}; sequence precision "
                    f"{score['sequence_precision']:.0%}; F1 {score['sequence_f1']:.0%}; "
                    f"vocabulary recall {score['vocabulary_recall']:.0%}."
                ),
                "",
                (
                    f"Duration {result['duration_seconds']:.1f}s; analysis "
                    f"{result['analysis_seconds']:.1f}s; key {result['dominant_key']} "
                    f"({result['key_confidence']:.0%}); tempo "
                    + (
                        "not stable"
                        if result["tempo_bpm"] is None
                        else f"{result['tempo_bpm']:.1f} BPM ({result['tempo_confidence']:.0%})"
                    )
                    + f"; meter {result['meter']} ({result['meter_confidence']:.0%})."
                ),
            ]
        )
        if result["local_key_regions"]:
            key_map = ", ".join(
                f"{region['start']:.1f}–{region['end']:.1f}s {region['key']}"
                for region in result["local_key_regions"]
            )
            lines.extend(["", f"Local key map: {key_map}."])
        if score["missing_vocabulary"] or score["extra_vocabulary"]:
            lines.extend(
                [
                    "",
                    "Missing vocabulary: "
                    + (", ".join(score["missing_vocabulary"]) or "none")
                    + "; extra labels: "
                    + (", ".join(score["extra_vocabulary"]) or "none")
                    + ".",
                ]
            )
        if result["warnings"]:
            lines.extend(["", "Analyzer warnings:"])
            lines.extend(f"- {warning}" for warning in result["warnings"])

    exact = [result for result in results if not result["song"]["allow_global_transposition"]]
    aggregate_recall = sum(item["score"]["sequence_recall"] for item in exact) / max(len(exact), 1)
    aggregate_vocab = sum(item["score"]["vocabulary_recall"] for item in exact) / max(len(exact), 1)
    lines.extend(
        [
            "",
            "## Aggregate interpretation",
            "",
            f"Mean literal-pitch sequence recall across the four exact-pitch clips: {aggregate_recall:.0%}.",
            "",
            f"Mean literal-pitch chord-vocabulary recall across those clips: {aggregate_vocab:.0%}.",
            "",
            (
                "These numbers do not prove arbitrary-song accuracy. The recordings are small, "
                "Western equal-temperament examples. Modal music, drones, noisy field recordings, "
                "and changing tempo remain harder; the app exposes local keys, drone warnings, "
                "meter confidence, and editable corrections for those cases."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    parser.add_argument(
        "--minimum-exact-sequence-recall",
        type=float,
        default=0.0,
        help="Exit non-zero when the mean exact-pitch sequence recall is below this value.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sgchords-known-chords-") as temporary:
        root = Path(temporary)
        results = [
            analyze_song(
                song,
                root / "downloads",
                root / "work" / song.slug,
                args.output,
            )
            for song in SONGS
        ]
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(results)
    (args.output / "results.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    exact = [result for result in results if not result["song"]["allow_global_transposition"]]
    aggregate = sum(item["score"]["sequence_recall"] for item in exact) / max(len(exact), 1)
    return 1 if aggregate < args.minimum_exact_sequence_recall else 0


if __name__ == "__main__":
    raise SystemExit(main())
