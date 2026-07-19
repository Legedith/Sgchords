from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chords import (
    key_prefers_flats,
    render_chord_shapes,
    suggest_capo,
    transpose_chord,
    transpose_key,
)
from .models import AnalysisResult, ChordSegment


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return (cleaned[:80] or "song").lower()


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:05.2f}"


def transpose_segments(
    segments: Sequence[ChordSegment], semitones: int, *, prefer_flats: bool = False
) -> list[ChordSegment]:
    return [
        ChordSegment(
            start=segment.start,
            end=segment.end,
            chord=transpose_chord(segment.chord, semitones, prefer_flats=prefer_flats),
            confidence=segment.confidence,
        )
        for segment in segments
    ]


def rows_to_segments(value: Any) -> list[ChordSegment]:
    """Convert a Gradio/pandas/native table into validated chord segments."""

    if value is None:
        return []
    if hasattr(value, "to_dict"):
        records = value.to_dict(orient="records")
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        records = value
    elif isinstance(value, list):
        records = [
            {
                "Start (s)": row[0],
                "End (s)": row[1],
                "Chord": row[2],
                "Confidence": row[3] if len(row) > 3 else 0.0,
            }
            for row in value
            if row and len(row) >= 3
        ]
    else:
        raise ValueError("The timeline table has an unsupported format.")

    def pick(record: dict[str, Any], *names: str) -> Any:
        lowered = {str(key).strip().lower(): item for key, item in record.items()}
        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]
        return None

    segments: list[ChordSegment] = []
    for index, record in enumerate(records, start=1):
        try:
            start = float(pick(record, "Start (s)", "start", "start_s"))
            end = float(pick(record, "End (s)", "end", "end_s"))
            chord = str(pick(record, "Chord", "chord") or "N").strip()
            confidence_raw = pick(record, "Confidence", "confidence")
            confidence = 0.0 if confidence_raw in {None, ""} else float(confidence_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Timeline row {index} contains an invalid number.") from exc
        if start < 0 or end <= start:
            raise ValueError(f"Timeline row {index} must have 0 <= start < end.")
        if not chord:
            raise ValueError(f"Timeline row {index} has an empty chord name.")
        segments.append(ChordSegment(start, end, chord, min(max(confidence, 0.0), 1.0)))
    segments.sort(key=lambda item: (item.start, item.end))
    return segments


def render_sheet(
    title: str,
    segments: Sequence[ChordSegment],
    *,
    key: str,
    tempo_bpm: float | None,
    transpose: int = 0,
) -> str:
    tempo = "unknown" if tempo_bpm is None else f"{tempo_bpm:.0f} BPM"
    lines = [
        title,
        f"Key: {transpose_key(key, transpose)} | Tempo: {tempo} | Transpose: {transpose:+d}",
        "",
    ]
    if not segments:
        return "\n".join(lines + ["No chord segments."])

    entries = [f"[{format_timestamp(segment.start)}] {segment.chord}" for segment in segments]
    for index in range(0, len(entries), 4):
        lines.append("    ".join(entries[index : index + 4]))
    return "\n".join(lines)


def render_summary(
    result: AnalysisResult,
    segments: Sequence[ChordSegment],
    *,
    transpose: int = 0,
    instrument: str = "guitar",
) -> str:
    tempo = "not stable" if result.tempo_bpm is None else f"{result.tempo_bpm:.0f} BPM"
    transposed_key = transpose_key(result.key, transpose)
    chord_segments = [item for item in segments if item.chord != "N"]
    capo, shapes, ease = suggest_capo(chord_segments, instrument)
    capo_text = "no capo" if capo == 0 else f"capo {capo}"
    shapes_text = ", ".join(shapes) if shapes else "none"
    confidence = (
        sum(item.confidence * item.duration for item in chord_segments)
        / max(sum(item.duration for item in chord_segments), 1e-9)
        if chord_segments
        else 0.0
    )
    lines = [
        f"### {result.title}",
        f"**Estimated key:** {transposed_key} ({result.key_confidence:.0%} key confidence)  ",
        f"**Tempo:** {tempo}  ",
        f"**Recording tuning:** {result.tuning_cents:+.0f} cents from A=440  ",
        f"**Mean chord confidence:** {confidence:.0%}  ",
        f"**Easy {instrument.lower()} option:** {capo_text}; play {shapes_text} "
        f"(ease score {ease:.0%})",
    ]
    if result.warnings:
        lines.extend(["", "**Check by ear:**"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def render_shapes_for_segments(segments: Sequence[ChordSegment], instrument: str) -> str:
    return render_chord_shapes((segment.chord for segment in segments), instrument)


def _write_csv(path: Path, segments: Sequence[ChordSegment]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_seconds", "end_seconds", "chord", "confidence"])
        for segment in segments:
            writer.writerow(
                [
                    f"{segment.start:.3f}",
                    f"{segment.end:.3f}",
                    segment.chord,
                    f"{segment.confidence:.3f}",
                ]
            )


def _write_chordpro(
    path: Path,
    result: AnalysisResult,
    segments: Sequence[ChordSegment],
    transpose: int,
) -> None:
    tempo = "" if result.tempo_bpm is None else f"{{tempo: {result.tempo_bpm:.0f}}}\n"
    lines = [
        f"{{title: {result.title}}}",
        f"{{key: {transpose_key(result.key, transpose)}}}",
        tempo.rstrip(),
        "{comment: Automatic estimate; verify low-confidence changes by ear.}",
        "",
    ]
    for segment in segments:
        lines.append(
            f"{{comment: {format_timestamp(segment.start)}–{format_timestamp(segment.end)}}} "
            f"[{segment.chord}]"
        )
    path.write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")


def write_exports(
    result: AnalysisResult,
    output_dir: str | Path,
    *,
    segments: Sequence[ChordSegment] | None = None,
    transpose: int = 0,
    instrument: str = "guitar",
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_segments = list(segments if segments is not None else result.segments)
    target_key = transpose_key(result.key, transpose)
    prefer_flats = key_prefers_flats(target_key)
    target_segments = transpose_segments(source_segments, transpose, prefer_flats=prefer_flats)
    base = safe_filename(result.title)

    text_path = output_dir / f"{base}-chords.txt"
    csv_path = output_dir / f"{base}-timeline.csv"
    json_path = output_dir / f"{base}-analysis.json"
    chordpro_path = output_dir / f"{base}.cho"
    shapes_path = output_dir / f"{base}-{instrument.lower()}-shapes.txt"

    text_path.write_text(
        render_sheet(
            result.title,
            target_segments,
            key=result.key,
            tempo_bpm=result.tempo_bpm,
            transpose=transpose,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, target_segments)
    json_payload = {
        "title": result.title,
        "source": result.source,
        "duration": result.duration,
        "tempo_bpm": result.tempo_bpm,
        "key": target_key,
        "key_confidence": result.key_confidence,
        "tuning_cents": result.tuning_cents,
        "transpose": transpose,
        "instrument": instrument,
        "warnings": result.warnings,
        "segments": [asdict(segment) for segment in target_segments],
    }
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_chordpro(chordpro_path, result, target_segments, transpose)
    shapes_path.write_text(
        re.sub(r"[*`]", "", render_shapes_for_segments(target_segments, instrument)) + "\n",
        encoding="utf-8",
    )
    return [str(path) for path in (text_path, csv_path, json_path, chordpro_path, shapes_path)]
