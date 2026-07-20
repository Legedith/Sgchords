from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .models import ChordSegment

SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
PITCH_CLASS = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

MODE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
}

MODE_PROFILES: dict[str, np.ndarray] = {
    "major": np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "minor": np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    "dorian": np.asarray([6.15, 2.75, 3.70, 5.10, 2.20, 3.75, 2.35, 4.95, 2.35, 4.15, 3.70, 2.20]),
    "mixolydian": np.asarray(
        [6.20, 2.15, 3.55, 2.20, 4.15, 4.05, 2.25, 5.25, 2.20, 3.65, 4.05, 2.15]
    ),
    "phrygian": np.asarray(
        [6.15, 4.35, 2.25, 4.60, 2.00, 3.65, 2.10, 4.95, 3.90, 2.00, 3.65, 2.05]
    ),
}

_CHORD_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)([^/]*?)(?:/([A-Ga-g])([#b]?))?\s*$")
_KEY_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)\s+(major|minor|dorian|mixolydian|phrygian)\s*$", re.I)


@dataclass(frozen=True, slots=True)
class ParsedChord:
    root: int
    suffix: str
    bass: int | None = None


@dataclass(frozen=True, slots=True)
class ChordSpec:
    label: str
    root: int
    quality: str
    intervals: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TonalityCandidate:
    root: int
    mode: str
    label: str


QUALITY_DEFINITIONS: dict[str, tuple[tuple[int, ...], tuple[float, ...], str]] = {
    "maj": ((0, 4, 7), (1.00, 0.86, 0.72), ""),
    "min": ((0, 3, 7), (1.00, 0.86, 0.72), "m"),
    "7": ((0, 4, 7, 10), (1.00, 0.82, 0.68, 0.58), "7"),
    "maj7": ((0, 4, 7, 11), (1.00, 0.82, 0.68, 0.55), "maj7"),
    "min7": ((0, 3, 7, 10), (1.00, 0.82, 0.68, 0.56), "m7"),
    "sus2": ((0, 2, 7), (1.00, 0.78, 0.72), "sus2"),
    "sus4": ((0, 5, 7), (1.00, 0.78, 0.72), "sus4"),
    "dim": ((0, 3, 6), (1.00, 0.82, 0.68), "dim"),
    "aug": ((0, 4, 8), (1.00, 0.82, 0.68), "aug"),
    "6": ((0, 4, 7, 9), (1.00, 0.82, 0.68, 0.52), "6"),
    "min6": ((0, 3, 7, 9), (1.00, 0.82, 0.68, 0.52), "m6"),
    "add9": ((0, 2, 4, 7), (1.00, 0.48, 0.82, 0.68), "add9"),
    "power": ((0, 7), (1.00, 0.76), "5"),
}
SIMPLE_QUALITIES = ("maj", "min")
STANDARD_QUALITIES = ("maj", "min", "7", "maj7", "min7", "sus2", "sus4", "dim")
DETAILED_QUALITIES = STANDARD_QUALITIES + ("aug", "6", "min6", "add9", "power")

# Common, playable voicings. Guitar is EADGBE; ukulele is GCEA.
GUITAR_SHAPES: dict[str, str] = {
    "C": "x32010",
    "C7": "x32310",
    "Cmaj7": "x32000",
    "Cm": "x35543",
    "Csus2": "x30013",
    "Csus4": "x33011",
    "D": "xx0232",
    "Dm": "xx0231",
    "D7": "xx0212",
    "Dmaj7": "xx0222",
    "Dsus2": "xx0230",
    "Dsus4": "xx0233",
    "E": "022100",
    "Em": "022000",
    "E7": "020100",
    "Emaj7": "021100",
    "Esus4": "022200",
    "F": "133211",
    "Fm": "133111",
    "Fmaj7": "xx3210",
    "G": "320003",
    "Gm": "355333",
    "G7": "320001",
    "Gmaj7": "320002",
    "Gsus4": "330013",
    "A": "x02220",
    "Am": "x02210",
    "A7": "x02020",
    "Amaj7": "x02120",
    "Asus2": "x02200",
    "Asus4": "x02230",
    "Bb": "x13331",
    "Bbm": "x13321",
    "Bb7": "x13131",
    "B": "x24442",
    "Bm": "x24432",
    "B7": "x21202",
    "Bmaj7": "x24342",
    "F#": "244322",
    "F#m": "244222",
    "F#7": "242322",
    "Db": "x46664",
    "Dbm": "x46654",
    "Eb": "x68886",
    "Ebm": "x68876",
    "Ab": "466544",
    "Abm": "466444",
}
UKULELE_SHAPES: dict[str, str] = {
    "C": "0003",
    "Cm": "0333",
    "C7": "0001",
    "Cmaj7": "0002",
    "Csus2": "0233",
    "Csus4": "0013",
    "D": "2220",
    "Dm": "2210",
    "D7": "2223",
    "Dmaj7": "2224",
    "Dsus2": "2200",
    "Dsus4": "0230",
    "E": "1402",
    "Em": "0432",
    "E7": "1202",
    "Emaj7": "1302",
    "F": "2010",
    "Fm": "1013",
    "F7": "2310",
    "Fmaj7": "2410",
    "G": "0232",
    "Gm": "0231",
    "G7": "0212",
    "Gmaj7": "0222",
    "Gsus4": "0233",
    "A": "2100",
    "Am": "2000",
    "A7": "0100",
    "Amaj7": "1100",
    "Asus2": "2452",
    "Asus4": "2200",
    "Bb": "3211",
    "Bbm": "3111",
    "Bb7": "1211",
    "B": "4322",
    "Bm": "4222",
    "B7": "2322",
    "F#": "3121",
    "F#m": "2120",
    "Db": "1114",
    "Dbm": "1104",
    "Eb": "0331",
    "Ebm": "3321",
    "Ab": "5343",
    "Abm": "4342",
}

GUITAR_EASY = {
    "C",
    "C7",
    "Cmaj7",
    "D",
    "Dm",
    "D7",
    "E",
    "Em",
    "E7",
    "Fmaj7",
    "G",
    "G7",
    "A",
    "Am",
    "A7",
    "B7",
}
UKULELE_EASY = {
    "C",
    "C7",
    "Cmaj7",
    "D",
    "Dm",
    "D7",
    "E7",
    "Em",
    "F",
    "Fmaj7",
    "G",
    "G7",
    "A",
    "Am",
    "A7",
    "Bb",
    "B7",
}


def pitch_name(pitch_class: int, prefer_flats: bool = False) -> str:
    return (FLAT_NAMES if prefer_flats else SHARP_NAMES)[pitch_class % 12]


def _mode_prefers_flats(root: int, mode: str) -> bool:
    flat_roots = {
        "major": {1, 3, 5, 6, 8, 10},
        "minor": {0, 2, 3, 5, 7, 10},
        "dorian": {0, 2, 5, 7, 10},
        "mixolydian": {0, 2, 5, 7, 10},
        "phrygian": {0, 3, 5, 7, 10},
    }
    return root in flat_roots.get(mode, set())


def key_name(pitch_class: int, mode: str) -> str:
    return f"{pitch_name(pitch_class, _mode_prefers_flats(pitch_class % 12, mode))} {mode}"


def parse_key(key: str) -> tuple[int, str] | None:
    match = _KEY_RE.match(key)
    if match is None:
        return None
    return PITCH_CLASS[match.group(1).upper() + match.group(2)], match.group(3).lower()


def key_prefers_flats(key: str) -> bool:
    parsed = parse_key(key)
    return bool(parsed and _mode_prefers_flats(*parsed))


def parse_chord(chord: str) -> ParsedChord | None:
    if not chord or chord.strip().upper() in {"N", "NC", "N.C.", "NO CHORD"}:
        return None
    match = _CHORD_RE.match(chord)
    if match is None:
        return None
    root = PITCH_CLASS.get(match.group(1).upper() + match.group(2))
    if root is None:
        return None
    bass_name = (match.group(4) or "").upper() + (match.group(5) or "")
    return ParsedChord(
        root, match.group(3).strip(), PITCH_CLASS.get(bass_name) if bass_name else None
    )


def transpose_chord(chord: str, semitones: int, prefer_flats: bool = False) -> str:
    parsed = parse_chord(chord)
    if parsed is None:
        return "N" if chord.strip().upper() in {"N", "NC", "N.C.", "NO CHORD"} else chord
    label = f"{pitch_name(parsed.root + semitones, prefer_flats)}{parsed.suffix}"
    if parsed.bass is not None:
        label += f"/{pitch_name(parsed.bass + semitones, prefer_flats)}"
    return label


def _suffix_quality(suffix: str) -> str:
    lower = suffix.lower().strip()
    if lower in {"", "maj"}:
        return "maj"
    if lower.startswith("maj7"):
        return "maj7"
    if lower.startswith(("m7", "min7")):
        return "min7"
    if lower.startswith(("m6", "min6")):
        return "min6"
    if lower.startswith(("m", "min")):
        return "min"
    if lower.startswith("dim") or "°" in lower:
        return "dim"
    if lower.startswith("aug") or "+" in lower:
        return "aug"
    if lower.startswith("sus2"):
        return "sus2"
    if lower.startswith("sus"):
        return "sus4"
    if lower.startswith("add9"):
        return "add9"
    if lower == "5":
        return "power"
    if lower.startswith("7"):
        return "7"
    if lower.startswith("6"):
        return "6"
    return "maj"


def simplify_chord(chord: str, *, keep_bass: bool = False) -> str:
    parsed = parse_chord(chord)
    if parsed is None:
        return "N" if chord.strip().upper() in {"N", "NC", "N.C.", "NO CHORD"} else chord
    suffix = "m" if _suffix_quality(parsed.suffix) in {"min", "min7", "min6"} else ""
    flats = "b" in chord[:2]
    label = f"{pitch_name(parsed.root, flats)}{suffix}"
    if keep_bass and parsed.bass is not None:
        label += f"/{pitch_name(parsed.bass, flats)}"
    return label


def transpose_key(key: str, semitones: int) -> str:
    parsed = parse_key(key)
    return key if parsed is None else key_name(parsed[0] + semitones, parsed[1])


@lru_cache(maxsize=6)
def chord_vocabulary(detail: str = "standard") -> tuple[ChordSpec, ...]:
    qualities = {
        "simple": SIMPLE_QUALITIES,
        "standard": STANDARD_QUALITIES,
        "detailed": DETAILED_QUALITIES,
    }.get(detail.lower())
    if qualities is None:
        raise ValueError(f"Unknown chord vocabulary: {detail}")
    result: list[ChordSpec] = []
    for root in range(12):
        for quality in qualities:
            intervals, weights, suffix = QUALITY_DEFINITIONS[quality]
            result.append(
                ChordSpec(f"{SHARP_NAMES[root]}{suffix}", root, quality, intervals, weights)
            )
    return tuple(result)


def chord_template(spec: ChordSpec) -> np.ndarray:
    template = np.zeros(12, dtype=float)
    for interval, weight in zip(spec.intervals, spec.weights, strict=True):
        template[(spec.root + interval) % 12] = weight
    return template / (float(np.linalg.norm(template)) + 1e-12)


@lru_cache(maxsize=2)
def tonality_candidates(include_phrygian: bool = True) -> tuple[TonalityCandidate, ...]:
    modes = (
        ("major", "minor", "dorian", "mixolydian", "phrygian")
        if include_phrygian
        else ("major", "minor", "dorian", "mixolydian")
    )
    return tuple(
        TonalityCandidate(root, mode, key_name(root, mode)) for mode in modes for root in range(12)
    )


def tonality_scores(
    chroma: np.ndarray, *, include_phrygian: bool = True
) -> tuple[tuple[TonalityCandidate, ...], np.ndarray]:
    values = np.nan_to_num(np.asarray(chroma, dtype=float).reshape(12))
    if float(np.sum(np.maximum(values, 0))) <= 1e-10:
        return tonality_candidates(include_phrygian), np.zeros(
            len(tonality_candidates(include_phrygian))
        )
    normalized = (values - np.mean(values)) / (np.std(values) + 1e-9)
    candidates = tonality_candidates(include_phrygian)
    scores: list[float] = []
    for candidate in candidates:
        profile = np.roll(MODE_PROFILES[candidate.mode], candidate.root)
        profile = (profile - np.mean(profile)) / (np.std(profile) + 1e-9)
        score = float(np.dot(normalized, profile) / 12)
        # Avoid labeling ordinary major/minor material modal unless the characteristic pitch is present.
        if candidate.mode in {"dorian", "mixolydian", "phrygian"}:
            characteristic = {"dorian": 9, "mixolydian": 10, "phrygian": 1}[candidate.mode]
            mass = np.maximum(values, 0) / (float(np.sum(np.maximum(values, 0))) + 1e-12)
            characteristic_mass = float(mass[(candidate.root + characteristic) % 12])
            modal_bonus = 0.14 if candidate.mode == "phrygian" else 0.11
            modal_penalty = 0.07 if candidate.mode == "phrygian" else 0.045
            score += modal_bonus * characteristic_mass - modal_penalty
        scores.append(score)
    return candidates, np.asarray(scores)


def estimate_key(global_chroma: np.ndarray) -> tuple[str, float, int, str]:
    candidates, scores = tonality_scores(global_chroma)
    if not np.any(scores):
        return "Unknown", 0.0, 0, "major"
    order = np.argsort(scores)[::-1]
    best = candidates[int(order[0])]
    margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
    return best.label, float(np.clip(margin / 0.20, 0.0, 1.0)), best.root, best.mode


def _triad_quality(intervals: tuple[int, int, int]) -> str:
    shape = ((intervals[1] - intervals[0]) % 12, (intervals[2] - intervals[0]) % 12)
    if shape == (4, 7):
        return "maj"
    if shape == (3, 7):
        return "min"
    if shape == (3, 6):
        return "dim"
    return "other"


@lru_cache(maxsize=8)
def mode_diatonic_triads(mode: str) -> tuple[tuple[int, str], ...]:
    scale = MODE_INTERVALS.get(mode, MODE_INTERVALS["major"])
    result: list[tuple[int, str]] = []
    for degree in range(7):
        notes = tuple(
            scale[index % 7] + (12 if index >= 7 else 0)
            for index in (degree, degree + 2, degree + 4)
        )
        root = notes[0] % 12
        normalized = (0, (notes[1] - notes[0]) % 12, (notes[2] - notes[0]) % 12)
        result.append((root, _triad_quality(normalized)))
    return tuple(result)


def is_diatonic(spec: ChordSpec, key_root: int, key_mode: str) -> bool:
    relative = (spec.root - key_root) % 12
    triad = dict(mode_diatonic_triads(key_mode)).get(relative)
    if triad is None:
        return False
    if spec.quality == triad:
        return True
    if spec.quality == "7" and relative == 7:
        return True
    if spec.quality == "min7" and triad == "min":
        return True
    if spec.quality == "maj7" and triad == "maj":
        return True
    return spec.quality in {"sus2", "sus4", "power"}


def _degree_and_accidental(pitch: int, key_root: int, mode: str) -> tuple[int, str]:
    scale = MODE_INTERVALS.get(mode, MODE_INTERVALS["major"])
    relative = (pitch - key_root) % 12
    if relative in scale:
        return scale.index(relative) + 1, ""
    distances: list[tuple[int, int, str]] = []
    for index, note in enumerate(scale):
        if (note - 1) % 12 == relative:
            distances.append((0, index + 1, "♭"))
        if (note + 1) % 12 == relative:
            distances.append((0, index + 1, "♯"))
    if distances:
        _distance, degree, accidental = distances[0]
        return degree, accidental
    nearest = min(
        range(7),
        key=lambda index: min((relative - scale[index]) % 12, (scale[index] - relative) % 12),
    )
    return nearest + 1, ""


def roman_numeral(chord: str, key: str) -> str:
    parsed = parse_chord(chord)
    parsed_key = parse_key(key)
    if parsed is None or parsed_key is None:
        return chord
    degree, accidental = _degree_and_accidental(parsed.root, *parsed_key)
    symbol = ("I", "II", "III", "IV", "V", "VI", "VII")[degree - 1]
    quality = _suffix_quality(parsed.suffix)
    if quality in {"min", "min7", "min6"}:
        symbol = symbol.lower()
    elif quality == "dim":
        symbol = symbol.lower() + "°"
    elif quality == "aug":
        symbol += "+"
    if quality in {"7", "maj7", "min7"}:
        symbol += "7"
    result = accidental + symbol
    if parsed.bass is not None:
        bass_degree, bass_accidental = _degree_and_accidental(parsed.bass, *parsed_key)
        result += f"/{bass_accidental}{('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')[bass_degree - 1]}"
    return result


def nashville_number(chord: str, key: str) -> str:
    parsed = parse_chord(chord)
    parsed_key = parse_key(key)
    if parsed is None or parsed_key is None:
        return chord
    degree, accidental = _degree_and_accidental(parsed.root, *parsed_key)
    accidental = accidental.replace("♭", "b").replace("♯", "#")
    quality = _suffix_quality(parsed.suffix)
    suffix = {
        "min": "m",
        "min7": "m7",
        "min6": "m6",
        "dim": "dim",
        "aug": "+",
        "7": "7",
        "maj7": "maj7",
        "sus2": "sus2",
        "sus4": "sus4",
    }.get(quality, "")
    result = f"{accidental}{degree}{suffix}"
    if parsed.bass is not None:
        bass_degree, bass_accidental = _degree_and_accidental(parsed.bass, *parsed_key)
        bass_accidental = bass_accidental.replace("♭", "b").replace("♯", "#")
        result += f"/{bass_accidental}{bass_degree}"
    return result


def format_chord_notation(chord: str, key: str, notation: str) -> str:
    return (
        roman_numeral(chord, key)
        if notation == "roman"
        else nashville_number(chord, key)
        if notation == "nashville"
        else chord
    )


def shape_lookup(chord: str, instrument: str) -> tuple[str, str] | None:
    shapes = GUITAR_SHAPES if instrument.lower() == "guitar" else UKULELE_SHAPES
    parsed = parse_chord(chord)
    if parsed is None:
        return None
    candidates = [
        chord.split("/")[0],
        f"{pitch_name(parsed.root, True)}{parsed.suffix}",
        f"{pitch_name(parsed.root, False)}{parsed.suffix}",
    ]
    # Fall back to the triad when a detailed voicing is absent.
    simplified = simplify_chord(chord)
    candidates.extend(
        [simplified, transpose_chord(simplified, 0, True), transpose_chord(simplified, 0, False)]
    )
    for candidate in dict.fromkeys(candidates):
        if candidate in shapes:
            return candidate, shapes[candidate]
    return None


def render_chord_shapes(chords: Iterable[str], instrument: str) -> str:
    unique = list(dict.fromkeys(chord for chord in chords if chord != "N"))
    tuning = "E A D G B E (low to high)" if instrument == "guitar" else "G C E A"
    lines = [f"{instrument.title()} tuning: {tuning}", ""]
    for chord in unique:
        shape = shape_lookup(chord, instrument)
        lines.append(f"{chord}: {shape[1] if shape else 'no built-in voicing'}")
    return "\n".join(lines) if unique else "No playable chord shapes were detected."


def _shape_svg(label: str, frets: str, instrument: str) -> str:
    strings = 6 if instrument == "guitar" else 4
    values = list(frets)
    if len(values) != strings:
        return ""
    numeric = [int(value) for value in values if value.isdigit() and int(value) > 0]
    base_fret = max(1, min(numeric) if numeric and max(numeric) > 4 else 1)
    width, height = 118, 150
    left, top, grid_w, grid_h = 20, 32, 78, 92
    string_gap = grid_w / (strings - 1)
    fret_gap = grid_h / 5
    parts = [
        f'<div class="sg-diagram-card"><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)} chord diagram">',
        f'<text class="sg-diagram-title" x="59" y="17" text-anchor="middle">{html.escape(label)}</text>',
    ]
    for string in range(strings):
        x = left + string * string_gap
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + grid_h}"/>')
    for fret in range(6):
        y = top + fret * fret_gap
        stroke = 3 if fret == 0 and base_fret == 1 else 1
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + grid_w}" y2="{y:.1f}" stroke-width="{stroke}"/>'
        )
    if base_fret > 1:
        parts.append(
            f'<text class="sg-fret-label" x="4" y="{top + fret_gap:.1f}">{base_fret}fr</text>'
        )
    for string, value in enumerate(values):
        x = left + string * string_gap
        if value.lower() == "x":
            parts.append(
                f'<text class="sg-marker" x="{x:.1f}" y="27" text-anchor="middle">×</text>'
            )
        elif value == "0":
            parts.append(f'<circle class="sg-open" cx="{x:.1f}" cy="24" r="4"/>')
        elif value.isdigit():
            displayed = int(value) - base_fret + 1
            if 1 <= displayed <= 5:
                y = top + (displayed - 0.5) * fret_gap
                parts.append(f'<circle class="sg-finger" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
    parts.append("</svg></div>")
    return "".join(parts)


def render_chord_diagrams_html(chords: Iterable[str], instrument: str) -> str:
    unique = list(dict.fromkeys(chord for chord in chords if chord != "N"))
    diagrams: list[str] = []
    missing: list[str] = []
    for chord in unique:
        shape = shape_lookup(chord, instrument)
        if shape is None:
            missing.append(chord)
        else:
            diagrams.append(_shape_svg(chord, shape[1], instrument))
    if missing:
        diagrams.append(
            f'<div class="sg-missing-shapes">No built-in diagram: {html.escape(", ".join(missing))}</div>'
        )
    return (
        '<div class="sg-diagram-grid">' + "".join(diagrams) + "</div>"
        if diagrams
        else "<p>No chord diagrams.</p>"
    )


def suggest_capo(
    segments: Sequence[ChordSegment], instrument: str = "guitar", max_capo: int = 7
) -> tuple[int, list[str], float]:
    weighted = [
        (simplify_chord(segment.chord), max(segment.duration, 0.1))
        for segment in segments
        if segment.chord != "N"
    ]
    if not weighted:
        return 0, [], 0.0
    total = sum(duration for _chord, duration in weighted)
    easy = GUITAR_EASY if instrument == "guitar" else UKULELE_EASY
    best: tuple[float, int, list[str]] | None = None
    for capo in range(max_capo + 1):
        shapes: list[str] = []
        cost = 0.0
        for sounding, duration in weighted:
            played = transpose_chord(sounding, -capo, prefer_flats=True)
            lookup = shape_lookup(played, instrument)
            canonical = lookup[0] if lookup else played
            shapes.append(canonical)
            cost += (0.10 if canonical in easy else 0.60 if lookup else 1.35) * duration
        cost += capo * total * 0.028
        candidate = (cost, capo, list(dict.fromkeys(shapes)))
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], float(np.clip(1.0 - best[0] / (1.35 * total), 0.0, 1.0))
