from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .models import ChordSegment

SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
MAJOR_KEY_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINOR_KEY_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")
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

_CHORD_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)(.*)\s*$")


@dataclass(frozen=True, slots=True)
class ChordSpec:
    label: str
    root: int
    quality: str
    intervals: tuple[int, ...]
    weights: tuple[float, ...]


QUALITY_DEFINITIONS: dict[str, tuple[tuple[int, ...], tuple[float, ...], str]] = {
    "maj": ((0, 4, 7), (1.0, 0.86, 0.72), ""),
    "min": ((0, 3, 7), (1.0, 0.86, 0.72), "m"),
    "7": ((0, 4, 7, 10), (1.0, 0.82, 0.68, 0.58), "7"),
    "maj7": ((0, 4, 7, 11), (1.0, 0.82, 0.68, 0.55), "maj7"),
    "min7": ((0, 3, 7, 10), (1.0, 0.82, 0.68, 0.56), "m7"),
    "sus2": ((0, 2, 7), (1.0, 0.78, 0.72), "sus2"),
    "sus4": ((0, 5, 7), (1.0, 0.78, 0.72), "sus4"),
    "dim": ((0, 3, 6), (1.0, 0.82, 0.68), "dim"),
}

SIMPLE_QUALITIES = ("maj", "min")
DETAILED_QUALITIES = ("maj", "min", "7", "maj7", "min7", "sus2", "sus4", "dim")

_MAJOR_PROFILE = np.asarray(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
_MINOR_PROFILE = np.asarray(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)

# One common voicing. Strings are low-to-high for guitar (EADGBE) and GCEA for ukulele.
GUITAR_SHAPES: dict[str, str] = {
    "C": "x32010",
    "C7": "x32310",
    "Cmaj7": "x32000",
    "Cm": "x35543",
    "D": "xx0232",
    "Dm": "xx0231",
    "D7": "xx0212",
    "E": "022100",
    "Em": "022000",
    "E7": "020100",
    "F": "133211",
    "Fm": "133111",
    "Fmaj7": "xx3210",
    "G": "320003",
    "Gm": "355333",
    "G7": "320001",
    "A": "x02220",
    "Am": "x02210",
    "A7": "x02020",
    "Bb": "x13331",
    "Bbm": "x13321",
    "B": "x24442",
    "Bm": "x24432",
    "B7": "x21202",
}

UKULELE_SHAPES: dict[str, str] = {
    "C": "0003",
    "Cm": "0333",
    "C7": "0001",
    "Cmaj7": "0002",
    "D": "2220",
    "Dm": "2210",
    "D7": "2223",
    "E": "1402",
    "Em": "0432",
    "E7": "1202",
    "F": "2010",
    "Fm": "1013",
    "Fmaj7": "2410",
    "G": "0232",
    "Gm": "0231",
    "G7": "0212",
    "A": "2100",
    "Am": "2000",
    "A7": "0100",
    "Bb": "3211",
    "Bbm": "3111",
    "B": "4322",
    "Bm": "4222",
    "B7": "2322",
}

GUITAR_OPEN_SHAPES = {
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
UKULELE_EASY_SHAPES = {
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
    names = FLAT_NAMES if prefer_flats else SHARP_NAMES
    return names[pitch_class % 12]


def key_name(pitch_class: int, mode: str) -> str:
    names = MINOR_KEY_NAMES if mode.lower() == "minor" else MAJOR_KEY_NAMES
    return f"{names[pitch_class % 12]} {mode.lower()}"


def key_prefers_flats(key: str) -> bool:
    match = re.match(r"^\s*([A-G])([#b]?)\s+(major|minor)\s*$", key, re.IGNORECASE)
    if match is None:
        return False
    root = PITCH_CLASS[match.group(1).upper() + match.group(2)]
    mode = match.group(3).lower()
    flat_major_roots = {1, 3, 5, 6, 8, 10}
    flat_minor_roots = {0, 2, 3, 5, 7, 10}
    return root in (flat_minor_roots if mode == "minor" else flat_major_roots)


def parse_chord(chord: str) -> tuple[int, str] | None:
    if not chord or chord.strip().upper() in {"N", "NC", "N.C.", "NO CHORD"}:
        return None
    match = _CHORD_RE.match(chord)
    if match is None:
        return None
    root_name = match.group(1).upper() + match.group(2)
    root = PITCH_CLASS.get(root_name)
    if root is None:
        return None
    suffix = match.group(3).strip()
    return root, suffix


def transpose_chord(chord: str, semitones: int, prefer_flats: bool = False) -> str:
    parsed = parse_chord(chord)
    if parsed is None:
        return "N" if chord.strip().upper() in {"N", "NC", "N.C.", "NO CHORD"} else chord
    root, suffix = parsed
    return f"{pitch_name(root + semitones, prefer_flats)}{suffix}"


def simplify_chord(chord: str) -> str:
    parsed = parse_chord(chord)
    if parsed is None:
        return "N" if chord.strip().upper() in {"N", "NC", "N.C.", "NO CHORD"} else chord
    root, suffix = parsed
    lower = suffix.lower()
    quality = "m" if lower.startswith("m") and not lower.startswith("maj") else ""
    return f"{pitch_name(root, 'b' in chord[:2])}{quality}"


def transpose_key(key: str, semitones: int) -> str:
    match = re.match(r"^\s*([A-G])([#b]?)\s+(major|minor)\s*$", key, re.IGNORECASE)
    if match is None:
        return key
    root_name = match.group(1).upper() + match.group(2)
    root = PITCH_CLASS[root_name]
    mode = match.group(3).lower()
    target = (root + semitones) % 12
    return key_name(target, mode)


@lru_cache(maxsize=4)
def chord_vocabulary(detail: str = "simple") -> tuple[ChordSpec, ...]:
    qualities = DETAILED_QUALITIES if detail.lower() == "detailed" else SIMPLE_QUALITIES
    specs: list[ChordSpec] = []
    for root in range(12):
        for quality in qualities:
            intervals, weights, suffix = QUALITY_DEFINITIONS[quality]
            specs.append(
                ChordSpec(
                    label=f"{SHARP_NAMES[root]}{suffix}",
                    root=root,
                    quality=quality,
                    intervals=intervals,
                    weights=weights,
                )
            )
    return tuple(specs)


def chord_template(spec: ChordSpec) -> np.ndarray:
    template = np.zeros(12, dtype=float)
    for interval, weight in zip(spec.intervals, spec.weights, strict=True):
        template[(spec.root + interval) % 12] = weight
    norm = float(np.linalg.norm(template))
    return template / norm if norm else template


def estimate_key(global_chroma: np.ndarray) -> tuple[str, float, int, str]:
    chroma = np.asarray(global_chroma, dtype=float).reshape(12)
    if not np.any(np.isfinite(chroma)) or float(np.sum(chroma)) <= 1e-10:
        return "Unknown", 0.0, 0, "major"
    chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
    chroma = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-9)

    candidates: list[tuple[float, int, str]] = []
    for root in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rolled = np.roll(profile, root)
            rolled = (rolled - np.mean(rolled)) / (np.std(rolled) + 1e-9)
            score = float(np.dot(chroma, rolled) / len(chroma))
            candidates.append((score, root, mode))
    candidates.sort(reverse=True, key=lambda item: item[0])
    best, second = candidates[0], candidates[1]
    confidence = float(np.clip((best[0] - second[0]) / 0.22, 0.0, 1.0))
    return key_name(best[1], best[2]), confidence, best[1], best[2]


def is_diatonic(spec: ChordSpec, key_root: int, key_mode: str) -> bool:
    rel = (spec.root - key_root) % 12
    if key_mode == "major":
        allowed = {
            (0, "maj"),
            (2, "min"),
            (4, "min"),
            (5, "maj"),
            (7, "maj"),
            (7, "7"),
            (9, "min"),
            (11, "dim"),
        }
    else:
        allowed = {
            (0, "min"),
            (2, "dim"),
            (3, "maj"),
            (5, "min"),
            (7, "min"),
            (7, "maj"),
            (7, "7"),
            (8, "maj"),
            (10, "maj"),
        }
    return (rel, spec.quality) in allowed


def _shape_lookup(chord: str, instrument: str) -> tuple[str, str] | None:
    shapes = GUITAR_SHAPES if instrument.lower() == "guitar" else UKULELE_SHAPES
    candidates = [chord]
    parsed = parse_chord(chord)
    if parsed is not None:
        root, suffix = parsed
        candidates.extend(
            [
                f"{pitch_name(root, True)}{suffix}",
                f"{pitch_name(root, False)}{suffix}",
            ]
        )
    for candidate in candidates:
        if candidate in shapes:
            return candidate, shapes[candidate]
    return None


def render_chord_shapes(chords: Iterable[str], instrument: str) -> str:
    unique: list[str] = []
    for chord in chords:
        if chord != "N" and chord not in unique:
            unique.append(chord)
    if not unique:
        return "No playable chord shapes were detected."

    tuning = "E A D G B E; frets are low-to-high" if instrument.lower() == "guitar" else "G C E A"
    lines = [f"**{instrument.title()} shapes** — tuning: {tuning}", ""]
    for chord in unique:
        shape = _shape_lookup(chord, instrument)
        if shape is None:
            lines.append(f"- `{chord}` — no built-in simple voicing")
        else:
            display, frets = shape
            lines.append(f"- `{display}` — `{frets}`")
    return "\n".join(lines)


def suggest_capo(
    segments: Sequence[ChordSegment], instrument: str = "guitar", max_capo: int = 7
) -> tuple[int, list[str], float]:
    """Find a low capo position that turns the most-duration chords into known shapes."""

    if not segments:
        return 0, [], 0.0

    weighted: list[tuple[str, float]] = [
        (simplify_chord(segment.chord), max(segment.duration, 0.1))
        for segment in segments
        if segment.chord != "N"
    ]
    if not weighted:
        return 0, [], 0.0

    best: tuple[float, int, list[str]] | None = None
    total_duration = sum(duration for _, duration in weighted)
    for capo in range(max_capo + 1):
        shapes: list[str] = []
        cost = 0.0
        for sounding, duration in weighted:
            played = transpose_chord(sounding, -capo, prefer_flats=True)
            lookup = _shape_lookup(played, instrument)
            if lookup is None:
                cost += 1.35 * duration
                shapes.append(played)
            else:
                canonical, _frets = lookup
                easy_shapes = (
                    GUITAR_OPEN_SHAPES if instrument.lower() == "guitar" else UKULELE_EASY_SHAPES
                )
                cost += (0.12 if canonical in easy_shapes else 0.67) * duration
                shapes.append(canonical)
        cost += capo * total_duration * 0.025
        unique = list(dict.fromkeys(shapes))
        candidate = (cost, capo, unique)
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    ease = float(np.clip(1.0 - best[0] / (1.35 * total_duration), 0.0, 1.0))
    return best[1], best[2], ease
