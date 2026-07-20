from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .chords import (
    ChordSpec,
    chord_template,
    chord_vocabulary,
    is_diatonic,
    key_prefers_flats,
    pitch_name,
    simplify_chord,
    tonality_candidates,
    tonality_scores,
    transpose_chord,
)
from .models import BeatPoint, ChordSegment, KeyRegion, ProgressionPattern


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    sample_rate: int = 22_050
    hop_length: int = 512
    min_audio_seconds: float = 2.0
    fallback_window_seconds: float = 0.75
    max_segment_seconds: float = 1.5
    min_segment_seconds: float = 0.14
    local_key_window_seconds: float = 20.0
    local_key_min_region_seconds: float = 8.0
    boundary_search_ratio: float = 0.23


@dataclass(slots=True)
class AnalyzerOutput:
    duration: float
    tempo_bpm: float | None
    tempo_confidence: float
    meter: int
    meter_confidence: float
    key: str
    key_confidence: float
    tuning_cents: float
    segments: list[ChordSegment]
    beats: list[BeatPoint]
    key_regions: list[KeyRegion]
    patterns: list[ProgressionPattern]
    drone: str | None
    drone_confidence: float
    warnings: list[str]
    used_beat_tracking: bool


def _load_audio(path: str | Path, target_rate: int) -> tuple[np.ndarray, int]:
    try:
        y, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AnalysisError(f"Could not decode the normalized audio: {exc}") from exc
    if y.ndim > 1:
        y = np.mean(y, axis=1, dtype=np.float32)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))
    if sample_rate != target_rate and y.size:
        divisor = gcd(int(sample_rate), int(target_rate))
        y = signal.resample_poly(y, target_rate // divisor, sample_rate // divisor).astype(
            np.float32
        )
        sample_rate = target_rate
    return y, int(sample_rate)


def _estimate_beats(
    onset: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> tuple[float | None, np.ndarray, float]:
    envelope = np.nan_to_num(np.asarray(onset, dtype=float).reshape(-1))
    if envelope.size < 8 or float(np.max(envelope)) <= 1e-8:
        return None, np.asarray([], dtype=int), 0.0
    try:
        tempo_raw, beat_frames = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=sample_rate,
            hop_length=hop_length,
            trim=False,
            units="frames",
            sparse=True,
        )
        tempo = float(np.atleast_1d(tempo_raw)[0])
        beats = np.asarray(beat_frames, dtype=int)
    except Exception:
        tempo, beats = 0.0, np.asarray([], dtype=int)
    beats = beats[(beats >= 0) & (beats < envelope.size)]
    if beats.size < 3 or not 35 <= tempo <= 260:
        # Autocorrelation fallback: less expressive than the dynamic-programming tracker,
        # but it preserves a usable practice grid for sparse folk recordings and click-light audio.
        centered = np.maximum(envelope - np.median(envelope), 0.0)
        centered = np.convolve(centered, np.asarray([0.2, 0.6, 0.2]), mode="same")
        if float(np.max(centered)) <= 1e-8:
            return None, np.asarray([], dtype=int), 0.0
        centered /= float(np.max(centered)) + 1e-12
        min_lag = max(1, int(round(60 * sample_rate / (hop_length * 220))))
        max_lag = min(len(centered) - 1, int(round(60 * sample_rate / (hop_length * 45))))
        autocorrelation = signal.fftconvolve(centered, centered[::-1], mode="full")[
            len(centered) - 1 :
        ]
        lags = np.arange(min_lag, max_lag + 1)
        bpms = 60 * sample_rate / (hop_length * lags)
        preference = np.exp(-0.5 * (np.log2(bpms / 100.0) / 0.85) ** 2)
        scores = autocorrelation[lags] * preference
        if scores.size == 0 or float(np.max(scores)) <= 1e-8:
            return None, np.asarray([], dtype=int), 0.0
        lag = int(lags[int(np.argmax(scores))])
        phase_scores = np.asarray([np.sum(centered[phase::lag]) for phase in range(lag)])
        phase = int(np.argmax(phase_scores))
        nominal = np.arange(phase, len(centered), lag, dtype=int)
        radius = max(1, int(round(lag * 0.22)))
        refined = []
        for frame in nominal:
            left = max(0, frame - radius)
            right = min(len(centered), frame + radius + 1)
            refined.append(left + int(np.argmax(centered[left:right])))
        beats = np.asarray(sorted(set(refined)), dtype=int)
        tempo = float(60 * sample_rate / (hop_length * lag))
        if beats.size < 3:
            return None, np.asarray([], dtype=int), 0.0
        peak_ratio = float(np.max(scores) / (autocorrelation[0] + 1e-12))
        fallback_confidence = float(np.clip(0.20 + 0.80 * peak_ratio, 0.0, 0.62))
    else:
        fallback_confidence = 0.0

    gaps = np.diff(beats)
    median_gap = float(np.median(gaps)) if gaps.size else 0.0
    regularity = (
        float(np.mean(np.abs(gaps - median_gap) <= max(2.0, median_gap * 0.25)))
        if median_gap
        else 0.0
    )
    normalized = envelope / (float(np.max(envelope)) + 1e-12)
    beat_salience = float(np.mean(normalized[beats]))
    confidence = float(np.clip(0.55 * regularity + 0.45 * beat_salience, 0.0, 1.0))
    if fallback_confidence:
        confidence = min(confidence, fallback_confidence)
    return tempo, beats, confidence


def _estimate_meter(
    strengths: np.ndarray, meter_override: int | None = None
) -> tuple[int, int, float]:
    if meter_override in {3, 4, 6}:
        return int(meter_override), 0, 1.0
    values = np.nan_to_num(np.asarray(strengths, dtype=float).reshape(-1))
    if values.size < 8:
        return 4, 0, 0.0
    values = values / (float(np.max(values)) + 1e-12)
    candidates: list[tuple[float, int, int]] = []
    for meter in (3, 4, 6):
        if values.size < meter * 2:
            continue
        for phase in range(meter):
            down = values[phase::meter]
            other_mask = np.ones(values.size, dtype=bool)
            other_mask[phase::meter] = False
            others = values[other_mask]
            accent = float(np.mean(down) - np.mean(others))
            consistency = 1.0 - float(np.std(down))
            score = 0.72 * accent + 0.28 * consistency
            # Slight 4/4 prior prevents arbitrary 6/8 labels when accents are flat.
            if meter == 4:
                score += 0.025
            candidates.append((score, meter, phase))
    candidates.sort(reverse=True)
    if not candidates:
        return 4, 0, 0.0
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else (best[0], 4, 0)
    confidence = float(np.clip((best[0] - second[0]) / 0.16 + 0.18, 0.0, 1.0))
    return best[1], best[2], confidence


def _harmonic_novelty(chroma: np.ndarray, onset: np.ndarray) -> np.ndarray:
    normalized = chroma / np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-12)
    change = np.zeros(normalized.shape[1], dtype=float)
    if normalized.shape[1] > 1:
        change[1:] = 1.0 - np.sum(normalized[:, 1:] * normalized[:, :-1], axis=0)
    onset_values = np.asarray(onset, dtype=float)[: len(change)]
    onset_values = onset_values / (float(np.max(onset_values)) + 1e-12)
    change = change / (float(np.max(change)) + 1e-12)
    return np.nan_to_num(0.72 * change + 0.28 * onset_values)


def _refine_beat_frames(
    beats: np.ndarray,
    novelty: np.ndarray,
    ratio: float,
) -> np.ndarray:
    if beats.size < 3:
        return beats
    median_gap = max(2, int(round(float(np.median(np.diff(beats))))))
    radius = max(1, int(round(median_gap * ratio)))
    refined: list[int] = []
    for beat in beats:
        left = max(0, int(beat) - radius)
        right = min(len(novelty), int(beat) + radius + 1)
        refined.append(left + int(np.argmax(novelty[left:right])) if right > left else int(beat))
    return np.asarray(sorted(set(refined)), dtype=int)


def _fill_large_gaps(boundaries: list[float], max_gap: float) -> list[float]:
    result = [boundaries[0]]
    for end in boundaries[1:]:
        start = result[-1]
        gap = end - start
        if gap > max_gap:
            pieces = int(np.ceil(gap / max_gap))
            result.extend(start + gap * index / pieces for index in range(1, pieces))
        result.append(end)
    return result


def _segment_boundaries(
    duration: float,
    beat_times: np.ndarray,
    config: AnalyzerConfig,
) -> tuple[np.ndarray, bool]:
    valid = sorted(
        {
            float(value)
            for value in beat_times
            if config.min_segment_seconds < value < duration - config.min_segment_seconds
        }
    )
    used_beats = len(valid) >= 3
    if used_beats:
        raw = [0.0, *valid, duration]
    else:
        raw = list(np.arange(0.0, duration, config.fallback_window_seconds)) + [duration]
    filled = _fill_large_gaps(raw, config.max_segment_seconds)
    filtered = [filled[0]]
    for value in filled[1:]:
        if value == duration or value - filtered[-1] >= config.min_segment_seconds:
            filtered.append(value)
    if filtered[-1] != duration:
        filtered.append(duration)
    return np.asarray(filtered), used_beats


def _lowpass(y: np.ndarray, sample_rate: int, cutoff_hz: float = 330.0) -> np.ndarray:
    sos = signal.butter(4, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, y).astype(np.float32, copy=False)


def _aggregate_features(
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
    rms: np.ndarray,
    boundaries: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(chroma.shape[1], bass_chroma.shape[1], rms.shape[0])
    chroma, bass_chroma, rms = chroma[:, :count], bass_chroma[:, :count], rms[:count]
    times = librosa.frames_to_time(np.arange(count), sr=sample_rate, hop_length=hop_length)
    vectors: list[np.ndarray] = []
    bass_vectors: list[np.ndarray] = []
    energies: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        left = min(int(np.searchsorted(times, start, side="left")), count - 1)
        right = min(max(int(np.searchsorted(times, end, side="right")), left + 1), count)
        local_rms = rms[left:right]
        weights = local_rms + max(float(np.max(local_rms)), 1e-8) * 0.06
        harmonic_window = chroma[:, left:right]
        bass_window = bass_chroma[:, left:right]
        vector = 0.58 * np.median(harmonic_window, axis=1) + 0.42 * np.average(
            harmonic_window, axis=1, weights=weights
        )
        bass = 0.48 * np.median(bass_window, axis=1) + 0.52 * np.average(
            bass_window, axis=1, weights=weights
        )
        vectors.append(np.nan_to_num(vector))
        bass_vectors.append(np.nan_to_num(bass))
        energies.append(float(np.median(local_rms)))
    return np.asarray(vectors), np.asarray(bass_vectors), np.asarray(energies)


def _entropy(vector: np.ndarray) -> float:
    values = np.maximum(np.asarray(vector, dtype=float), 0.0)
    probabilities = values / (float(np.sum(values)) + 1e-12)
    nonzero = probabilities[probabilities > 1e-12]
    return float(-np.sum(nonzero * np.log(nonzero)) / np.log(12.0)) if nonzero.size else 1.0


def detect_drone(vectors: np.ndarray, energies: np.ndarray) -> tuple[int | None, float, np.ndarray]:
    matrix = np.maximum(np.asarray(vectors, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] < 6:
        return None, 0.0, matrix
    energy = np.asarray(energies, dtype=float)
    active = energy >= max(float(np.median(energy)) * 0.16, 1e-8)
    if np.count_nonzero(active) < 5:
        active = np.ones(matrix.shape[0], dtype=bool)
    mass = matrix / np.maximum(np.sum(matrix, axis=1, keepdims=True), 1e-12)
    profile = np.percentile(mass[active], 30, axis=0)
    order = np.argsort(profile)[::-1]
    root = int(order[0])
    top, second = float(profile[order[0]]), float(profile[order[1]]) + 1e-12
    root_mass = mass[active, root]
    prevalence = float(np.mean(root_mass >= max(0.11, top * 0.78)))
    dominance = top / second
    stability = float(
        np.clip(1.0 - np.std(root_mass) / (np.mean(root_mass) + 1e-12), 0.0, 1.0)
    )
    confidence = float(
        np.clip(
            0.50 * (prevalence - 0.82) / 0.18
            + 0.30 * (dominance - 1.20) / 0.90
            + 0.20 * (stability - 0.35) / 0.65,
            0.0,
            1.0,
        )
    )
    if prevalence < 0.84 or dominance < 1.20 or confidence < 0.45 or top < 0.10:
        # Confidence describes an accepted drone hypothesis. A rejected common-tone candidate
        # must not leak a high value into the UI or JSON result.
        return None, min(confidence, 0.44), matrix
    corrected = mass.copy()
    baseline = min(float(np.percentile(mass[active, root], 30)), 0.24)
    corrected[:, root] = np.maximum(0.0, corrected[:, root] - 0.58 * baseline)
    corrected /= np.maximum(np.sum(corrected, axis=1, keepdims=True), 1e-12)
    return root, confidence, corrected * np.sum(matrix, axis=1, keepdims=True)


def _key_transition_matrix() -> tuple[tuple, np.ndarray]:
    candidates = tonality_candidates()
    count = len(candidates)
    transition = np.full((count, count), -0.48)
    for left_index, left in enumerate(candidates):
        for right_index, right in enumerate(candidates):
            if left_index == right_index:
                transition[left_index, right_index] = 0.08
                continue
            distance = (right.root - left.root) % 12
            penalty = (
                -0.18
                if left.root == right.root
                else -0.26
                if distance in {5, 7}
                else -0.30
                if distance in {3, 4, 8, 9}
                else -0.42
            )
            if left.mode != right.mode:
                penalty -= 0.04
            transition[left_index, right_index] = penalty
    return candidates, transition


def _repair_short_runs(
    path: np.ndarray, confidence: np.ndarray, boundaries: np.ndarray, minimum: float
) -> np.ndarray:
    repaired = path.copy()
    for _pass in range(8):
        runs: list[tuple[int, int, int]] = []
        start = 0
        for index in range(1, len(repaired) + 1):
            if index == len(repaired) or repaired[index] != repaired[start]:
                runs.append((start, index, int(repaired[start])))
                start = index
        changed = False
        for run_index, (left, right, _state) in enumerate(runs):
            if (
                boundaries[right] - boundaries[left] >= minimum
                or float(np.mean(confidence[left:right])) >= 0.62
            ):
                continue
            previous = runs[run_index - 1] if run_index else None
            following = runs[run_index + 1] if run_index + 1 < len(runs) else None
            replacement = None
            if previous and following and previous[2] == following[2]:
                replacement = previous[2]
            elif previous and following:
                replacement = (
                    previous[2]
                    if previous[1] - previous[0] >= following[1] - following[0]
                    else following[2]
                )
            elif previous:
                replacement = previous[2]
            elif following:
                replacement = following[2]
            if replacement is not None:
                repaired[left:right] = replacement
                changed = True
                break
        if not changed:
            break
    return repaired


def estimate_local_keys(
    vectors: np.ndarray,
    energies: np.ndarray,
    boundaries: np.ndarray,
    *,
    window_seconds: float = 12.0,
    min_region_seconds: float = 4.0,
) -> tuple[list[tuple[int, str, float, str]], list[KeyRegion], str, float]:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.shape[0] == 0:
        return [], [], "Unknown", 0.0
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    durations = np.maximum(boundaries[1:] - boundaries[:-1], 1e-6)
    candidates, transition = _key_transition_matrix()
    emissions = np.zeros((len(matrix), len(candidates)))
    raw_confidence = np.zeros(len(matrix))
    median_energy = float(np.median(energies))
    for index, center in enumerate(centers):
        mask = np.abs(centers - center) <= window_seconds / 2
        if np.count_nonzero(mask) < min(4, len(matrix)):
            nearest = np.argsort(np.abs(centers - center))[: min(4, len(matrix))]
            mask = np.zeros(len(matrix), dtype=bool)
            mask[nearest] = True
        weights = durations[mask] * np.maximum(energies[mask], median_energy * 0.15 + 1e-8)
        aggregate = np.average(matrix[mask], axis=0, weights=weights)
        _items, scores = tonality_scores(aggregate)
        emissions[index] = scores
        order = np.argsort(scores)[::-1]
        raw_confidence[index] = np.clip((scores[order[0]] - scores[order[1]]) / 0.18, 0, 1)
    dp = np.full_like(emissions, -np.inf)
    back = np.zeros_like(emissions, dtype=np.int32)
    dp[0] = emissions[0]
    for index in range(1, len(matrix)):
        options = dp[index - 1][:, None] + transition
        back[index] = np.argmax(options, axis=0)
        dp[index] = emissions[index] + np.max(options, axis=0)
    path = np.zeros(len(matrix), dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for index in range(len(matrix) - 1, 0, -1):
        path[index - 1] = back[index, path[index]]
    path = _repair_short_runs(path, raw_confidence, boundaries, min_region_seconds)
    local = [
        (
            candidates[int(state)].root,
            candidates[int(state)].mode,
            float(raw_confidence[index]),
            candidates[int(state)].label,
        )
        for index, state in enumerate(path)
    ]
    regions: list[KeyRegion] = []
    start = 0
    for index in range(1, len(path) + 1):
        if index < len(path) and path[index] == path[start]:
            continue
        item = candidates[int(path[start])]
        confidence = float(np.average(raw_confidence[start:index], weights=durations[start:index]))
        regions.append(
            KeyRegion(
                float(boundaries[start]),
                float(boundaries[index]),
                item.label,
                confidence,
                item.root,
                item.mode,
            )
        )
        start = index
    totals: dict[str, float] = defaultdict(float)
    weighted: dict[str, float] = defaultdict(float)
    for region in regions:
        totals[region.key] += region.duration
        weighted[region.key] += region.duration * region.confidence
    dominant = max(totals, key=totals.get)
    total_duration = max(float(boundaries[-1] - boundaries[0]), 1e-9)
    confidence = float(
        np.clip(
            (totals[dominant] / total_duration) * weighted[dominant] / max(totals[dominant], 1e-9),
            0,
            1,
        )
    )
    return local, regions, dominant, confidence


def _chord_score(vector: np.ndarray, spec: ChordSpec) -> float:
    values = np.maximum(np.asarray(vector, dtype=float), 0.0)
    l1 = values / (float(np.sum(values)) + 1e-12)
    l2 = values / (float(np.linalg.norm(values)) + 1e-12)
    tones = [(spec.root + interval) % 12 for interval in spec.intervals]
    coverage = float(np.sum(l1[tones]))
    score = (
        0.59 * float(np.dot(l2, chord_template(spec)))
        + 0.50 * coverage
        + 0.10 * float(l1[spec.root])
        - 0.22 * (1 - coverage)
    )
    if len(spec.intervals) >= 4:
        extension = float(l1[tones[-1]])
        penalty = 0.10 if spec.quality == "maj7" else 0.075
        score += 0.38 * extension - penalty
    if spec.quality in {"aug", "dim", "sus2", "sus4", "add9", "power"}:
        score -= 0.018
    return score


def _chord_transition_matrix(specs: Sequence[ChordSpec]) -> np.ndarray:
    count = len(specs) + 1
    transition = np.full((count, count), -0.43)
    transition[0, 0] = 0.04
    transition[0, 1:] = transition[1:, 0] = -0.16
    for left_index, left in enumerate(specs, start=1):
        for right_index, right in enumerate(specs, start=1):
            if left_index == right_index:
                score = 0.055
            elif left.root == right.root:
                score = -0.11
            else:
                distance = (right.root - left.root) % 12
                score = (
                    -0.16
                    if distance in {5, 7}
                    else -0.22
                    if distance in {3, 4, 8, 9}
                    else -0.28
                    if distance in {2, 10}
                    else -0.37
                )
            transition[left_index, right_index] = score
    return transition


def classify_chroma_segments(
    vectors: np.ndarray,
    energies: np.ndarray,
    *,
    bass_vectors: np.ndarray | None = None,
    detail: str = "standard",
    smoothing: float = 0.68,
    local_keys: Sequence[tuple[int, str, float, str]] | None = None,
) -> tuple[list[str], list[float], list[float], np.ndarray]:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 12:
        raise ValueError("vectors must have shape (segments, 12)")
    if len(matrix) == 0:
        return [], [], [], np.asarray([], dtype=int)
    energy = np.asarray(energies, dtype=float).reshape(-1)
    bass = np.zeros_like(matrix) if bass_vectors is None else np.asarray(bass_vectors, dtype=float)
    if len(energy) != len(matrix) or bass.shape != matrix.shape:
        raise ValueError("feature arrays must have one row per segment")
    specs = chord_vocabulary(detail)
    emissions = np.zeros((len(matrix), len(specs) + 1))
    tonalities: list[float] = []
    positive = energy[energy > 1e-10]
    median_energy = float(np.median(positive)) if positive.size else 1.0
    for index, vector in enumerate(matrix):
        entropy = _entropy(vector)
        tonality = float(np.clip(1 - entropy, 0, 1))
        tonalities.append(tonality)
        relative_energy = float(energy[index] / (median_energy + 1e-12))
        silence = float(np.clip((0.16 - relative_energy) / 0.16, 0, 1))
        flatness = float(np.clip((entropy - 0.84) / 0.16, 0, 1))
        emissions[index, 0] = 0.22 + 0.94 * silence + 0.34 * flatness
        bass_mass = np.maximum(bass[index], 0)
        bass_mass /= float(np.sum(bass_mass)) + 1e-12
        local = local_keys[index] if local_keys and index < len(local_keys) else None
        for state, spec in enumerate(specs, start=1):
            tones = {(spec.root + interval) % 12 for interval in spec.intervals}
            score = (
                _chord_score(vector, spec)
                + 0.18 * float(bass_mass[spec.root])
                + 0.07 * float(np.sum(bass_mass[list(tones)]))
            )
            if local:
                root, mode, confidence, _label = local
                score += (0.07 if is_diatonic(spec, root, mode) else -0.018) * confidence
            emissions[index, state] = score
    transition = _chord_transition_matrix(specs) * (0.22 + 0.78 * float(np.clip(smoothing, 0, 1)))
    dp = np.full_like(emissions, -np.inf)
    back = np.zeros_like(emissions, dtype=np.int32)
    dp[0] = emissions[0]
    for index in range(1, len(matrix)):
        options = dp[index - 1][:, None] + transition
        back[index] = np.argmax(options, axis=0)
        dp[index] = emissions[index] + np.max(options, axis=0)
    states = np.zeros(len(matrix), dtype=np.int32)
    states[-1] = int(np.argmax(dp[-1]))
    for index in range(len(matrix) - 1, 0, -1):
        states[index - 1] = back[index, states[index]]
    labels: list[str] = []
    confidences: list[float] = []
    for index, state in enumerate(states):
        local = local_keys[index] if local_keys else None
        flats = key_prefers_flats(local[3]) if local else False
        if state == 0:
            label = "N"
        else:
            spec = specs[state - 1]
            chord_mass = np.maximum(matrix[index], 0)
            chord_mass /= float(np.sum(chord_mass)) + 1e-12
            extension_is_weak = False
            if spec.quality in {"7", "maj7", "min7"}:
                extension_pitch = (spec.root + spec.intervals[-1]) % 12
                threshold = 0.095 if spec.quality == "7" else 0.16 if spec.quality == "maj7" else 0.12
                extension_is_weak = float(chord_mass[extension_pitch]) < threshold
            if extension_is_weak:
                suffix = "m" if spec.quality == "min7" else ""
                label = f"{pitch_name(spec.root, flats)}{suffix}"
            else:
                label = transpose_chord(spec.label, 0, flats)
            bass_mass = np.maximum(bass[index], 0)
            bass_mass /= float(np.sum(bass_mass)) + 1e-12
            order = np.argsort(bass_mass)[::-1]
            bass_pitch = int(order[0])
            margin = float(bass_mass[order[0]] - bass_mass[order[1]])
            tones = {(spec.root + interval) % 12 for interval in spec.intervals}
            if (
                bass_pitch != spec.root
                and bass_pitch in tones
                and bass_mass[bass_pitch] >= 0.27
                and margin >= 0.05
            ):
                label += f"/{pitch_name(bass_pitch, flats)}"
        labels.append(label)
        row = emissions[index]
        runner_up = float(np.max(np.delete(row, int(state)))) if len(row) > 1 else float(row[state])
        margin = float(row[state] - runner_up)
        energy_factor = float(np.clip(energy[index] / (median_energy + 1e-12), 0, 1))
        confidences.append(
            float(
                np.clip(
                    0.34 + 1.75 * margin + 0.18 * tonalities[index] + 0.08 * energy_factor,
                    0.05,
                    0.99,
                )
            )
        )
    return labels, confidences, tonalities, states


def _repair_chord_blips(
    labels: list[str], confidences: list[float], boundaries: np.ndarray
) -> list[str]:
    repaired = labels.copy()
    for index in range(1, len(labels) - 1):
        duration = boundaries[index + 1] - boundaries[index]
        if labels[index - 1] == labels[index + 1] != labels[index] and (
            duration < 0.9 or confidences[index] < 0.48
        ):
            repaired[index] = labels[index - 1]
    return repaired


def _position_at(timestamp: float, beats: Sequence[BeatPoint]) -> tuple[int | None, float | None]:
    if not beats:
        return None, None
    index = int(np.argmin([abs(beat.time - timestamp) for beat in beats]))
    return beats[index].bar, beats[index].beat


def _merge_segments(segments: Sequence[ChordSegment]) -> list[ChordSegment]:
    merged: list[ChordSegment] = []
    for segment in segments:
        if segment.end <= segment.start:
            continue
        if (
            merged
            and merged[-1].chord == segment.chord
            and merged[-1].local_key == segment.local_key
        ):
            previous = merged[-1]
            total = previous.duration + segment.duration
            previous.confidence = (
                previous.confidence * previous.duration + segment.confidence * segment.duration
            ) / max(total, 1e-9)
            previous.end = segment.end
        else:
            merged.append(segment)
    return merged


def _chord_at_time(segments: Sequence[ChordSegment], timestamp: float) -> str:
    for segment in segments:
        if segment.start <= timestamp < segment.end:
            return simplify_chord(segment.chord)
    return "N"


def detect_progression_patterns(
    segments: Sequence[ChordSegment], beats: Sequence[BeatPoint], meter: int
) -> list[ProgressionPattern]:
    if not segments or len(beats) < meter * 4:
        return []
    bar_starts = [beat.time for beat in beats if beat.downbeat]
    if len(bar_starts) < 4:
        return []
    bars: list[tuple[str, ...]] = []
    for start_index, start in enumerate(bar_starts):
        end = bar_starts[start_index + 1] if start_index + 1 < len(bar_starts) else segments[-1].end
        samples = np.linspace(start, max(start, end - 1e-4), meter)
        bar = tuple(_chord_at_time(segments, float(value)) for value in samples)
        # Compress repeated beat labels within each bar.
        bars.append(
            tuple(chord for index, chord in enumerate(bar) if index == 0 or chord != bar[index - 1])
        )
    patterns: list[ProgressionPattern] = []
    for length in (1, 2, 4):
        if len(bars) < length * 2:
            continue
        windows = [
            tuple(chord for bar in bars[index : index + length] for chord in bar)
            for index in range(0, len(bars) - length + 1, length)
        ]
        counts = Counter(windows)
        sequence, occurrences = counts.most_common(1)[0]
        if occurrences >= 2 and sequence:
            patterns.append(
                ProgressionPattern(
                    list(sequence), length, occurrences, min(1.0, occurrences * length / len(bars))
                )
            )
    patterns.sort(
        key=lambda item: (item.coverage, item.occurrences, item.length_bars), reverse=True
    )
    return patterns[:3]


def _diatonic_duration_ratio(
    segments: Sequence[ChordSegment], key_root: int, key_mode: str, detail: str
) -> float | None:
    specs = chord_vocabulary(detail)
    by_simple: dict[str, ChordSpec] = {}
    for spec in specs:
        by_simple.setdefault(simplify_chord(spec.label), spec)
    total = diatonic = 0.0
    for segment in segments:
        if segment.chord == "N":
            continue
        total += segment.duration
        spec = by_simple.get(simplify_chord(segment.chord))
        if spec and is_diatonic(spec, key_root, key_mode):
            diatonic += segment.duration
    return None if total <= 0 else diatonic / total


def analyze_audio(
    audio_path: str | Path,
    *,
    detail: str = "standard",
    smoothing: float = 0.68,
    meter_override: int | None = None,
    config: AnalyzerConfig | None = None,
) -> AnalyzerOutput:
    config = config or AnalyzerConfig()
    y, sample_rate = _load_audio(audio_path, config.sample_rate)
    duration = float(len(y) / sample_rate) if sample_rate else 0.0
    if duration < config.min_audio_seconds:
        raise AnalysisError(
            f"The song must be at least {config.min_audio_seconds:.0f} seconds long."
        )
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-5:
        raise AnalysisError("The audio is silent or too quiet to analyze.")
    y = y / peak
    warnings: list[str] = []
    try:
        harmonic = librosa.effects.harmonic(y, margin=2.5)
    except Exception:
        harmonic = y
        warnings.append("Harmonic/percussive separation failed; the raw mix was analyzed.")
    try:
        tuning = float(
            librosa.estimate_tuning(y=harmonic, sr=sample_rate, bins_per_octave=12, resolution=0.01)
        )
        if not np.isfinite(tuning):
            tuning = 0.0
    except Exception:
        tuning = 0.0
    tuning_cents = float(np.clip(tuning * 100, -50, 50))
    try:
        chroma = librosa.feature.chroma_cqt(
            y=harmonic,
            sr=sample_rate,
            hop_length=config.hop_length,
            bins_per_octave=36,
            n_octaves=6,
            tuning=tuning * 3,
            norm=2,
            cqt_mode="hybrid",
        )
    except Exception:
        chroma = librosa.feature.chroma_stft(
            y=harmonic,
            sr=sample_rate,
            hop_length=config.hop_length,
            n_fft=4096,
            tuning=tuning,
            norm=2,
        )
        warnings.append("Constant-Q analysis failed; an STFT chromagram fallback was used.")
    try:
        bass_audio = _lowpass(harmonic, sample_rate)
        bass_chroma = librosa.feature.chroma_cqt(
            y=bass_audio,
            sr=sample_rate,
            hop_length=config.hop_length,
            fmin=librosa.note_to_hz("C1"),
            n_octaves=4,
            bins_per_octave=36,
            tuning=tuning * 3,
            norm=2,
            cqt_mode="hybrid",
        )
    except Exception:
        bass_chroma = chroma
        warnings.append("Bass-note isolation failed; slash-chord labels are less reliable.")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=config.hop_length)[0]
    onset = librosa.onset.onset_strength(
        y=y, sr=sample_rate, hop_length=config.hop_length, aggregate=np.median
    )
    tempo, beat_frames, tempo_confidence = _estimate_beats(
        onset, sample_rate=sample_rate, hop_length=config.hop_length
    )
    tracked_beats = beat_frames.size >= 3
    if tracked_beats:
        novelty = _harmonic_novelty(chroma, onset)
        beat_frames = _refine_beat_frames(beat_frames, novelty, config.boundary_search_ratio)
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sample_rate, hop_length=config.hop_length
        )
        beat_strengths = onset[np.clip(beat_frames, 0, max(0, len(onset) - 1))]
        meter, phase, meter_confidence = _estimate_meter(beat_strengths, meter_override)
    else:
        # Keep charts and the synchronized editor usable even when a beat tracker cannot lock.
        # This is explicitly marked as a low-confidence fixed grid rather than a detected tempo.
        beat_times = np.arange(0.0, duration, config.fallback_window_seconds)
        beat_strengths = np.zeros(len(beat_times), dtype=float)
        meter = int(meter_override) if meter_override in {3, 4, 6} else 4
        phase = 0
        meter_confidence = 1.0 if meter_override in {3, 4, 6} else 0.0
    beats = [
        BeatPoint(
            float(time),
            index,
            (index - phase) // meter + 1 if index >= phase else 1,
            float((index - phase) % meter + 1),
            float(beat_strengths[index] / (float(np.max(beat_strengths)) + 1e-12))
            if beat_strengths.size
            else 0.0,
            (index - phase) % meter == 0,
        )
        for index, time in enumerate(beat_times)
    ]
    boundaries, _grid_available = _segment_boundaries(duration, beat_times, config)
    used_beats = tracked_beats
    if not used_beats:
        warnings.append(
            "A stable beat grid was not found; a fixed practice grid was used for charting."
        )
    vectors, bass_vectors, energies = _aggregate_features(
        chroma, bass_chroma, rms, boundaries, sample_rate=sample_rate, hop_length=config.hop_length
    )
    drone_root, drone_confidence, corrected = detect_drone(vectors, energies)
    local_keys, key_regions, dominant_key, key_confidence = estimate_local_keys(
        corrected,
        energies,
        boundaries,
        window_seconds=config.local_key_window_seconds,
        min_region_seconds=config.local_key_min_region_seconds,
    )
    labels, confidences, tonalities, _states = classify_chroma_segments(
        corrected,
        energies,
        bass_vectors=bass_vectors,
        detail=detail,
        smoothing=smoothing,
        local_keys=local_keys,
    )
    labels = _repair_chord_blips(labels, confidences, boundaries)
    raw_segments: list[ChordSegment] = []
    for index, label in enumerate(labels):
        bar, beat = _position_at(float(boundaries[index]), beats)
        parsed_bass = label.split("/", 1)[1] if "/" in label else None
        raw_segments.append(
            ChordSegment(
                float(boundaries[index]),
                float(boundaries[index + 1]),
                label,
                confidences[index],
                local_keys[index][3] if local_keys else None,
                bar,
                beat,
                parsed_bass,
            )
        )
    segments = _merge_segments(raw_segments)
    patterns = detect_progression_patterns(segments, beats, meter)
    sounding = [segment for segment in segments if segment.chord != "N"]
    mean_confidence = (
        float(
            np.average(
                [item.confidence for item in sounding], weights=[item.duration for item in sounding]
            )
        )
        if sounding
        else 0.0
    )
    mean_tonality = float(np.mean(tonalities)) if tonalities else 0.0
    parsed_dominant = next(
        ((region.root, region.mode) for region in key_regions if region.key == dominant_key),
        (0, "major"),
    )
    ratio = _diatonic_duration_ratio(sounding, *parsed_dominant, detail)
    if not sounding:
        warnings.append(
            "No stable chord sequence was found. Try a cleaner recording or manual edits."
        )
    elif mean_confidence < 0.42:
        warnings.append("Overall chord confidence is low; verify the timeline by ear.")
    if mean_tonality < 0.11:
        warnings.append(
            "The recording is harmonically dense or noisy, which weakens chord estimates."
        )
    if abs(tuning_cents) >= 25:
        warnings.append(
            f"The recording is about {abs(tuning_cents):.0f} cents {'sharp' if tuning_cents > 0 else 'flat'} of A=440."
        )
    if drone_root is not None:
        warnings.append(
            f"A sustained {pitch_name(drone_root, key_prefers_flats(dominant_key))} drone was detected and partly removed before chord scoring."
        )
    if len(key_regions) > 1:
        warnings.append(
            "Multiple tonal regions were detected; use the local-key labels rather than assuming one key for the whole song."
        )
    if ratio is not None and ratio < 0.70:
        warnings.append(
            f"Only about {ratio:.0%} of chord duration fits the dominant {dominant_key} region; the song may be modal, modulating, or use borrowed harmony."
        )
    if meter_confidence < 0.28 and meter_override is None:
        warnings.append(
            "The bar-line estimate is uncertain. Set beats per bar manually if the chart groups beats incorrectly."
        )
    return AnalyzerOutput(
        duration=duration,
        tempo_bpm=tempo,
        tempo_confidence=tempo_confidence,
        meter=meter,
        meter_confidence=meter_confidence,
        key=dominant_key,
        key_confidence=key_confidence,
        tuning_cents=tuning_cents,
        segments=segments,
        beats=beats,
        key_regions=key_regions,
        patterns=patterns,
        drone=(
            pitch_name(drone_root, key_prefers_flats(dominant_key))
            if drone_root is not None
            else None
        ),
        drone_confidence=drone_confidence,
        warnings=warnings,
        used_beat_tracking=used_beats,
    )
