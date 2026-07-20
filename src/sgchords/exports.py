from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .chords import (
    format_chord_notation,
    key_prefers_flats,
    render_chord_shapes,
    suggest_capo,
    transpose_chord,
    transpose_key,
)
from .models import AnalysisResult, BeatPoint, ChordSegment


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return (cleaned[:80] or "song").lower()


def format_timestamp(seconds: float, *, milliseconds: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    remaining = seconds % 60
    if milliseconds:
        millis = int(round((remaining - int(remaining)) * 1000))
        if millis == 1000:
            remaining = int(remaining) + 1
            millis = 0
        return f"{hours:02d}:{minutes:02d}:{int(remaining):02d},{millis:03d}"
    return f"{minutes + hours * 60:02d}:{remaining:05.2f}"


def transpose_segments(
    segments: Sequence[ChordSegment], semitones: int, *, prefer_flats: bool = False
) -> list[ChordSegment]:
    return [
        ChordSegment(
            segment.start,
            segment.end,
            transpose_chord(segment.chord, semitones, prefer_flats),
            segment.confidence,
            transpose_key(segment.local_key, semitones) if segment.local_key else None,
            segment.bar,
            segment.beat,
            transpose_chord(segment.bass, semitones, prefer_flats) if segment.bass else None,
        )
        for segment in segments
    ]


def capo_shape_segments(
    segments: Sequence[ChordSegment], capo: int, *, prefer_flats: bool = False
) -> list[ChordSegment]:
    return [
        ChordSegment(
            segment.start,
            segment.end,
            transpose_chord(segment.chord, -capo, prefer_flats),
            segment.confidence,
            segment.local_key,
            segment.bar,
            segment.beat,
            transpose_chord(segment.bass, -capo, prefer_flats) if segment.bass else None,
        )
        for segment in segments
    ]


def rows_to_segments(value: Any) -> list[ChordSegment]:
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
                "Confidence": row[3] if len(row) > 3 else 0,
                "Local key": row[4] if len(row) > 4 else "",
                "Bar": row[5] if len(row) > 5 else "",
                "Beat": row[6] if len(row) > 6 else "",
            }
            for row in value
            if row and len(row) >= 3
        ]
    else:
        raise ValueError("The timeline table has an unsupported format.")

    def pick(record: dict[str, Any], *names: str) -> Any:
        lower = {str(key).strip().lower(): item for key, item in record.items()}
        return next((lower[name.lower()] for name in names if name.lower() in lower), None)

    segments: list[ChordSegment] = []
    for index, record in enumerate(records, start=1):
        try:
            start = float(pick(record, "Start (s)", "start", "start_s"))
            end = float(pick(record, "End (s)", "end", "end_s"))
            chord = str(pick(record, "Chord", "chord") or "N").strip()
            raw_confidence = pick(record, "Confidence", "confidence")
            confidence = 0.0 if raw_confidence in {None, ""} else float(raw_confidence)
            local_key = str(pick(record, "Local key", "local_key") or "").strip() or None
            raw_bar = pick(record, "Bar", "bar")
            raw_beat = pick(record, "Beat", "beat")
            bar = (
                None
                if raw_bar in {None, ""} or (isinstance(raw_bar, float) and np.isnan(raw_bar))
                else int(float(raw_bar))
            )
            beat = (
                None
                if raw_beat in {None, ""} or (isinstance(raw_beat, float) and np.isnan(raw_beat))
                else float(raw_beat)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Timeline row {index} contains invalid values.") from exc
        if start < 0 or end <= start:
            raise ValueError(f"Timeline row {index} must have 0 <= start < end.")
        if not chord:
            raise ValueError(f"Timeline row {index} has an empty chord.")
        segments.append(
            ChordSegment(
                start,
                end,
                chord,
                float(np.clip(confidence, 0, 1)),
                local_key,
                bar,
                beat,
                chord.split("/", 1)[1] if "/" in chord else None,
            )
        )
    return sorted(segments, key=lambda item: (item.start, item.end))


def merge_adjacent_segments(segments: Sequence[ChordSegment]) -> list[ChordSegment]:
    result: list[ChordSegment] = []
    for segment in sorted(segments, key=lambda item: item.start):
        if (
            result
            and result[-1].chord == segment.chord
            and result[-1].local_key == segment.local_key
            and segment.start <= result[-1].end + 0.08
        ):
            previous = result[-1]
            total = previous.duration + segment.duration
            previous.confidence = (
                previous.confidence * previous.duration + segment.confidence * segment.duration
            ) / max(total, 1e-9)
            previous.end = max(previous.end, segment.end)
        else:
            result.append(ChordSegment.from_dict(asdict(segment)))
    return result


def quantize_segments(
    segments: Sequence[ChordSegment], beats: Sequence[BeatPoint], *, duration: float
) -> list[ChordSegment]:
    if not segments or not beats:
        return list(segments)
    beat_times = np.asarray([0.0, *[beat.time for beat in beats], duration])
    beat_times = np.unique(np.clip(beat_times, 0, duration))
    snapped: list[ChordSegment] = []
    for segment in segments:
        start = float(beat_times[int(np.argmin(np.abs(beat_times - segment.start)))])
        end = float(beat_times[int(np.argmin(np.abs(beat_times - segment.end)))])
        if end <= start:
            after = beat_times[beat_times > start]
            end = float(after[0]) if after.size else duration
        nearest = min(beats, key=lambda beat: abs(beat.time - start))
        snapped.append(
            ChordSegment(
                start,
                end,
                segment.chord,
                segment.confidence,
                segment.local_key,
                nearest.bar,
                nearest.beat,
                segment.bass,
            )
        )
    snapped[0].start = 0.0
    snapped[-1].end = duration
    for index in range(1, len(snapped)):
        midpoint = (snapped[index - 1].end + snapped[index].start) / 2
        snapped[index - 1].end = snapped[index].start = midpoint
    return merge_adjacent_segments(snapped)


def _chord_at_time(segments: Sequence[ChordSegment], timestamp: float) -> ChordSegment | None:
    return next((segment for segment in segments if segment.start <= timestamp < segment.end), None)


def render_bar_chart(
    result: AnalysisResult,
    segments: Sequence[ChordSegment],
    *,
    transpose: int = 0,
    capo: int = 0,
    notation: str = "standard",
    meter_override: int | None = None,
) -> str:
    target_key = transpose_key(result.key, transpose)
    flats = key_prefers_flats(target_key)
    sounding = transpose_segments(segments, transpose, prefer_flats=flats)
    shape_key = transpose_key(target_key, -capo) if capo else target_key
    shapes = capo_shape_segments(sounding, capo, prefer_flats=key_prefers_flats(shape_key))
    meter = meter_override if meter_override in {3, 4, 6} else result.meter
    header = (
        f"{result.title}\nKey: {target_key} | Tempo: "
        + ("unknown" if result.tempo_bpm is None else f"{result.tempo_bpm:.1f} BPM")
        + f" | Meter: {meter}/4 | Transpose: {transpose:+d}"
        + (f" | Capo {capo} (shape names)" if capo else "")
        + "\n"
    )
    if not shapes:
        return header + "\nNo chord segments."
    if len(result.beats) < 2:
        entries = []
        for sound, shape in zip(sounding, shapes, strict=True):
            label = (
                shape.chord
                if notation == "standard"
                else format_chord_notation(sound.chord, target_key, notation)
            )
            entries.append(f"[{format_timestamp(sound.start)}] {label}")
        return (
            header
            + "\n"
            + "\n".join(
                "    ".join(entries[index : index + 4]) for index in range(0, len(entries), 4)
            )
        )

    beats = result.beats
    rows: dict[int, list[str]] = defaultdict(list)
    for beat in beats:
        if beat.time >= sounding[-1].end:
            continue
        sound = _chord_at_time(sounding, beat.time)
        shape = _chord_at_time(shapes, beat.time)
        if sound is None or shape is None:
            label = "N"
        elif notation == "standard":
            label = shape.chord
        else:
            local_key = transpose_key(sound.local_key, transpose) if sound.local_key else target_key
            label = format_chord_notation(sound.chord, local_key, notation)
        rows[beat.bar].append(label)
    lines = [header.rstrip(), ""]
    previous: str | None = None
    for bar in sorted(rows):
        cells = rows[bar][:meter]
        cells.extend([cells[-1] if cells else "N"] * (meter - len(cells)))
        display: list[str] = []
        for chord in cells:
            display.append("·" if chord == previous else chord)
            previous = chord
        lines.append(f"{bar:>3} | " + "  ".join(f"{cell:<8}" for cell in display).rstrip())
    return "\n".join(lines)


def render_summary(
    result: AnalysisResult,
    segments: Sequence[ChordSegment],
    *,
    transpose: int = 0,
    instrument: str = "guitar",
    capo: int = 0,
) -> str:
    target_key = transpose_key(result.key, transpose)
    sounding = [segment for segment in segments if segment.chord != "N"]
    confidence = (
        sum(segment.confidence * segment.duration for segment in sounding)
        / max(sum(segment.duration for segment in sounding), 1e-9)
        if sounding
        else 0.0
    )
    suggested, suggested_shapes, ease = suggest_capo(sounding, instrument)
    capo_message = f"capo {capo}" if capo else "no capo"
    if capo == 0 and suggested:
        capo_message += f"; automatic suggestion: capo {suggested} ({', '.join(suggested_shapes)})"
    regions = (
        ", ".join(
            f"{format_timestamp(region.start)}–{format_timestamp(region.end)} {transpose_key(region.key, transpose)} ({region.confidence:.0%})"
            for region in result.key_regions[:6]
        )
        or "no stable tonal regions"
    )
    tempo = (
        "not stable"
        if result.tempo_bpm is None
        else f"{result.tempo_bpm:.1f} BPM ({result.tempo_confidence:.0%})"
    )
    lines = [
        f"### {result.title}",
        f"**Dominant tonal region:** {target_key} ({result.key_confidence:.0%})  ",
        f"**Tempo / meter:** {tempo}; {result.meter}/4 ({result.meter_confidence:.0%} meter confidence)  ",
        f"**Tuning:** {result.tuning_cents:+.0f} cents from A=440  ",
        f"**Mean chord confidence:** {confidence:.0%}  ",
        f"**{instrument.title()} setup:** {capo_message}; ease score {ease:.0%}  ",
        f"**Local tonal map:** {regions}",
    ]
    if result.drone:
        lines.append(
            f"  \n**Sustained tonal centre:** {result.drone} ({result.drone_confidence:.0%})"
        )
    if result.patterns:
        pattern = result.patterns[0]
        lines.append(
            f"  \n**Repeated pattern:** {' → '.join(pattern.chords)} ({pattern.occurrences} occurrences)"
        )
    if result.warnings:
        lines.extend(["", "**Check by ear:**", *[f"- {warning}" for warning in result.warnings]])
    return "\n".join(lines)


def render_shapes_for_segments(segments: Sequence[ChordSegment], instrument: str) -> str:
    return render_chord_shapes((segment.chord for segment in segments), instrument)


def _write_csv(path: Path, segments: Sequence[ChordSegment]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["start_seconds", "end_seconds", "chord", "confidence", "local_key", "bar", "beat"]
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


def _write_chordpro(
    path: Path, result: AnalysisResult, segments: Sequence[ChordSegment], key: str, capo: int
) -> None:
    lines = [f"{{title: {result.title}}}", f"{{key: {key}}}"]
    if result.tempo_bpm is not None:
        lines.append(f"{{tempo: {result.tempo_bpm:.0f}}}")
    if capo:
        lines.append(f"{{capo: {capo}}}")
    lines.extend(["{comment: Automatic estimate; verify low-confidence changes by ear.}", ""])
    for segment in segments:
        lines.append(
            f"{{comment: {format_timestamp(segment.start)}–{format_timestamp(segment.end)}}} [{segment.chord}]"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_srt(path: Path, segments: Sequence[ChordSegment]) -> None:
    blocks = [
        f"{index}\n{format_timestamp(segment.start, milliseconds=True)} --> {format_timestamp(segment.end, milliseconds=True)}\n{segment.chord}\n"
        for index, segment in enumerate(segments, start=1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")


def _write_lab(path: Path, segments: Sequence[ChordSegment]) -> None:
    path.write_text(
        "".join(
            f"{segment.start:.3f}\t{segment.end:.3f}\t{segment.chord}\n" for segment in segments
        ),
        encoding="utf-8",
    )


def write_exports(
    result: AnalysisResult,
    output_dir: str | Path,
    *,
    segments: Sequence[ChordSegment] | None = None,
    transpose: int = 0,
    instrument: str = "guitar",
    capo: int = 0,
    notation: str = "standard",
    meter_override: int | None = None,
) -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = list(segments if segments is not None else result.segments)
    key = transpose_key(result.key, transpose)
    flats = key_prefers_flats(key)
    concert = transpose_segments(source, transpose, prefer_flats=flats)
    shape_key = transpose_key(key, -capo) if capo else key
    displayed = capo_shape_segments(concert, capo, prefer_flats=key_prefers_flats(shape_key))
    base = safe_filename(result.title)
    paths = {
        "text": output / f"{base}-chords.txt",
        "csv": output / f"{base}-timeline.csv",
        "json": output / f"{base}-analysis.json",
        "chordpro": output / f"{base}.cho",
        "shapes": output / f"{base}-{instrument}-shapes.txt",
        "srt": output / f"{base}.srt",
        "lab": output / f"{base}.lab",
    }
    paths["text"].write_text(
        render_bar_chart(
            result,
            source,
            transpose=transpose,
            capo=capo,
            notation=notation,
            meter_override=meter_override,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(paths["csv"], concert)
    payload = result.to_state()
    payload.update(
        {
            "key": key,
            "transpose": transpose,
            "capo": capo,
            "instrument": instrument,
            "notation": notation,
            "displayed_segments": [asdict(item) for item in displayed],
            "segments": [asdict(item) for item in concert],
        }
    )
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_chordpro(paths["chordpro"], result, displayed, key, capo)
    paths["shapes"].write_text(
        render_shapes_for_segments(displayed, instrument) + "\n", encoding="utf-8"
    )
    _write_srt(paths["srt"], concert)
    _write_lab(paths["lab"], concert)
    return [str(path) for path in paths.values()]
