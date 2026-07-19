from __future__ import annotations

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
    estimate_key,
    is_diatonic,
    key_name,
    key_prefers_flats,
    transpose_chord,
)
from .models import ChordSegment


class AnalysisError(RuntimeError):
    pass


def _load_audio(path: str | Path, target_rate: int) -> tuple[np.ndarray, int]:
    try:
        y, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AnalysisError(f"Could not decode the normalized audio: {exc}") from exc
    if y.ndim > 1:
        y = np.mean(y, axis=1, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sample_rate != target_rate and y.size:
        divisor = gcd(int(sample_rate), int(target_rate))
        y = signal.resample_poly(y, target_rate // divisor, sample_rate // divisor).astype(
            np.float32, copy=False
        )
        sample_rate = target_rate
    return y, int(sample_rate)


def _estimate_beats(
    onset_envelope: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    min_bpm: float = 45.0,
    max_bpm: float = 220.0,
) -> tuple[float | None, np.ndarray]:
    """Estimate a regular beat grid without librosa's JIT-heavy dynamic-programming tracker."""

    envelope = np.nan_to_num(np.asarray(onset_envelope, dtype=float).reshape(-1))
    if envelope.size < 8 or float(np.max(envelope)) <= 1e-8:
        return None, np.asarray([], dtype=int)
    envelope = np.maximum(envelope - np.median(envelope), 0.0)
    if float(np.sum(envelope)) <= 1e-8:
        return None, np.asarray([], dtype=int)
    envelope = np.convolve(envelope, np.asarray([0.2, 0.6, 0.2]), mode="same")
    envelope /= float(np.max(envelope)) + 1e-12

    minimum_lag = max(1, int(round(60.0 * sample_rate / (hop_length * max_bpm))))
    maximum_lag = min(
        envelope.size - 1,
        int(round(60.0 * sample_rate / (hop_length * min_bpm))),
    )
    if maximum_lag <= minimum_lag:
        return None, np.asarray([], dtype=int)

    autocorrelation = signal.fftconvolve(envelope, envelope[::-1], mode="full")
    autocorrelation = autocorrelation[envelope.size - 1 :]
    lags = np.arange(minimum_lag, maximum_lag + 1)
    bpms = 60.0 * sample_rate / (hop_length * lags)
    preference = np.exp(-0.5 * (np.log2(bpms / 100.0) / 0.85) ** 2)
    scores = autocorrelation[lags] * preference
    if not np.any(np.isfinite(scores)) or float(np.max(scores)) <= 1e-8:
        return None, np.asarray([], dtype=int)
    lag = int(lags[int(np.argmax(scores))])
    tempo = float(60.0 * sample_rate / (hop_length * lag))

    phase_scores = np.zeros(lag, dtype=float)
    for phase in range(lag):
        phase_scores[phase] = float(np.sum(envelope[phase::lag]))
    phase = int(np.argmax(phase_scores))
    nominal = np.arange(phase, envelope.size, lag, dtype=int)
    radius = max(1, int(round(lag * 0.22)))
    refined: list[int] = []
    for frame in nominal:
        left = max(0, frame - radius)
        right = min(envelope.size, frame + radius + 1)
        refined.append(left + int(np.argmax(envelope[left:right])))
    beats = np.asarray(sorted(set(refined)), dtype=int)
    strong = envelope[beats] >= max(0.08, float(np.percentile(envelope, 45)))
    beats = beats[strong]
    return (tempo, beats) if beats.size >= 2 else (None, np.asarray([], dtype=int))


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    sample_rate: int = 22_050
    hop_length: int = 512
    min_audio_seconds: float = 2.0
    fallback_window_seconds: float = 0.75
    max_segment_seconds: float = 1.5
    min_segment_seconds: float = 0.18


@dataclass(slots=True)
class AnalyzerOutput:
    duration: float
    tempo_bpm: float | None
    key: str
    key_confidence: float
    tuning_cents: float
    segments: list[ChordSegment]
    warnings: list[str]
    used_beat_tracking: bool


def _entropy(vector: np.ndarray) -> float:
    values = np.clip(np.asarray(vector, dtype=float), 0.0, None)
    total = float(np.sum(values))
    if total <= 1e-12:
        return 1.0
    probabilities = values / total
    nonzero = probabilities[probabilities > 1e-12]
    return float(-np.sum(nonzero * np.log(nonzero)) / np.log(12.0))


def _chord_score(vector: np.ndarray, spec: ChordSpec) -> float:
    vector = np.clip(np.asarray(vector, dtype=float), 0.0, None)
    total = float(np.sum(vector))
    if total <= 1e-12:
        return 0.0
    l1 = vector / total
    l2 = vector / (float(np.linalg.norm(vector)) + 1e-12)
    tones = [(spec.root + interval) % 12 for interval in spec.intervals]
    coverage = float(np.sum(l1[tones]))
    root_strength = float(l1[spec.root])
    cosine = float(np.dot(l2, chord_template(spec)))
    # Cosine handles the chord shape; coverage penalizes unexplained pitch classes.
    return 0.60 * cosine + 0.48 * coverage + 0.10 * root_strength


def _transition_matrix(specs: Sequence[ChordSpec]) -> np.ndarray:
    state_count = len(specs) + 1  # state zero is N (no chord)
    transitions = np.full((state_count, state_count), -0.44, dtype=float)
    transitions[0, 0] = 0.05
    transitions[0, 1:] = -0.17
    transitions[1:, 0] = -0.17

    for left_index, left in enumerate(specs, start=1):
        for right_index, right in enumerate(specs, start=1):
            if left_index == right_index:
                penalty = 0.06
            elif left.root == right.root:
                penalty = -0.12
            else:
                interval = (right.root - left.root) % 12
                if interval in {5, 7}:  # fourth/fifth movement
                    penalty = -0.17
                elif interval in {3, 4, 8, 9}:  # relative or mediant movement
                    penalty = -0.23
                elif interval in {2, 10}:
                    penalty = -0.29
                else:
                    penalty = -0.38
            transitions[left_index, right_index] = penalty
    return transitions


def classify_chroma_segments(
    vectors: np.ndarray,
    energies: np.ndarray,
    *,
    detail: str = "simple",
    smoothing: float = 0.65,
    key_root: int = 0,
    key_mode: str = "major",
    key_confidence: float = 0.0,
) -> tuple[list[str], list[float], list[float]]:
    """Classify pre-aggregated 12-bin chroma vectors with Viterbi smoothing.

    This function is separated from audio feature extraction so the musical classifier can be
    unit-tested deterministically.
    """

    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 12:
        raise ValueError("vectors must have shape (segments, 12)")
    if matrix.shape[0] == 0:
        return [], [], []
    energy_values = np.asarray(energies, dtype=float).reshape(-1)
    if energy_values.shape[0] != matrix.shape[0]:
        raise ValueError("energies must contain one value per segment")

    specs = chord_vocabulary(detail)
    state_count = len(specs) + 1
    emissions = np.zeros((matrix.shape[0], state_count), dtype=float)
    tonalities: list[float] = []
    positive_energies = energy_values[energy_values > 1e-10]
    median_energy = float(np.median(positive_energies)) if positive_energies.size else 1.0

    for segment_index, vector in enumerate(matrix):
        entropy = _entropy(vector)
        tonality = float(np.clip(1.0 - entropy, 0.0, 1.0))
        tonalities.append(tonality)
        relative_energy = float(energy_values[segment_index] / (median_energy + 1e-12))
        silence = float(np.clip((0.18 - relative_energy) / 0.18, 0.0, 1.0))
        flatness = float(np.clip((entropy - 0.82) / 0.18, 0.0, 1.0))
        emissions[segment_index, 0] = 0.25 + 0.90 * silence + 0.36 * flatness

        for state_index, spec in enumerate(specs, start=1):
            score = _chord_score(vector, spec)
            if key_confidence > 0 and is_diatonic(spec, key_root, key_mode):
                score += 0.045 * key_confidence
            emissions[segment_index, state_index] = score

    smoothing = float(np.clip(smoothing, 0.0, 1.0))
    transitions = _transition_matrix(specs) * (0.25 + 0.75 * smoothing)
    dp = np.full_like(emissions, -np.inf)
    backpointers = np.zeros_like(emissions, dtype=np.int32)
    dp[0] = emissions[0]
    for time_index in range(1, emissions.shape[0]):
        candidate_scores = dp[time_index - 1][:, None] + transitions
        backpointers[time_index] = np.argmax(candidate_scores, axis=0)
        dp[time_index] = emissions[time_index] + np.max(candidate_scores, axis=0)

    states = np.zeros(emissions.shape[0], dtype=np.int32)
    states[-1] = int(np.argmax(dp[-1]))
    for time_index in range(emissions.shape[0] - 1, 0, -1):
        states[time_index - 1] = backpointers[time_index, states[time_index]]

    prefer_flats = key_prefers_flats(key_name(key_root, key_mode))
    labels: list[str] = []
    confidences: list[float] = []
    for segment_index, state in enumerate(states):
        if state == 0:
            labels.append("N")
        else:
            labels.append(transpose_chord(specs[state - 1].label, 0, prefer_flats=prefer_flats))

        row = emissions[segment_index]
        temperature = 0.075
        shifted = (row - np.max(row)) / temperature
        probabilities = np.exp(np.clip(shifted, -60.0, 0.0))
        probabilities /= float(np.sum(probabilities)) + 1e-12
        chosen_probability = float(probabilities[state])
        if state == 0:
            confidence = chosen_probability
        else:
            confidence = chosen_probability * (0.65 + 0.60 * tonalities[segment_index])
        confidences.append(float(np.clip(confidence, 0.0, 1.0)))

    return labels, confidences, tonalities


def _fill_large_gaps(boundaries: list[float], max_gap: float) -> list[float]:
    result: list[float] = [boundaries[0]]
    for end in boundaries[1:]:
        start = result[-1]
        gap = end - start
        if gap > max_gap:
            pieces = int(np.ceil(gap / max_gap))
            for index in range(1, pieces):
                result.append(start + gap * index / pieces)
        result.append(end)
    return result


def _segment_boundaries(
    duration: float,
    beat_times: np.ndarray,
    config: AnalyzerConfig,
) -> tuple[np.ndarray, bool]:
    valid_beats = sorted(
        {
            float(value)
            for value in np.asarray(beat_times, dtype=float).reshape(-1)
            if config.min_segment_seconds < value < duration - config.min_segment_seconds
        }
    )
    used_beats = len(valid_beats) >= 2
    if used_beats:
        raw = [0.0, *valid_beats, duration]
    else:
        raw = list(np.arange(0.0, duration, config.fallback_window_seconds))
        if not raw or raw[0] != 0.0:
            raw.insert(0, 0.0)
        raw.append(duration)

    filled = _fill_large_gaps(raw, config.max_segment_seconds)
    filtered = [filled[0]]
    for value in filled[1:]:
        if value - filtered[-1] >= config.min_segment_seconds or value == duration:
            filtered.append(value)
    if filtered[-1] != duration:
        filtered.append(duration)
    return np.asarray(filtered, dtype=float), used_beats


def _aggregate_features(
    chroma: np.ndarray,
    rms: np.ndarray,
    boundaries: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = min(chroma.shape[1], rms.shape[0])
    chroma = chroma[:, :frame_count]
    rms = rms[:frame_count]
    frame_times = librosa.frames_to_time(
        np.arange(frame_count), sr=sample_rate, hop_length=hop_length
    )

    vectors: list[np.ndarray] = []
    energies: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        left = int(np.searchsorted(frame_times, start, side="left"))
        right = int(np.searchsorted(frame_times, end, side="right"))
        right = max(right, left + 1)
        left = min(left, frame_count - 1)
        right = min(right, frame_count)
        segment_chroma = chroma[:, left:right]
        segment_rms = rms[left:right]
        if segment_chroma.size == 0:
            vectors.append(np.zeros(12, dtype=float))
            energies.append(0.0)
            continue

        weights = segment_rms + max(float(np.max(segment_rms)), 1e-8) * 0.05
        weighted = np.average(segment_chroma, axis=1, weights=weights)
        median = np.median(segment_chroma, axis=1)
        vector = 0.58 * median + 0.42 * weighted
        vectors.append(np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0))
        energies.append(float(np.median(segment_rms)))
    return np.asarray(vectors), np.asarray(energies)


def _merge_segments(segments: Sequence[ChordSegment]) -> list[ChordSegment]:
    merged: list[ChordSegment] = []
    for segment in segments:
        if segment.end <= segment.start:
            continue
        if merged and merged[-1].chord == segment.chord:
            previous = merged[-1]
            total_duration = previous.duration + segment.duration
            confidence = (
                previous.confidence * previous.duration + segment.confidence * segment.duration
            ) / max(total_duration, 1e-9)
            merged[-1] = ChordSegment(previous.start, segment.end, previous.chord, confidence)
        else:
            merged.append(segment)
    return merged


def _diatonic_duration_ratio(
    segments: Sequence[ChordSegment],
    *,
    detail: str,
    key_root: int,
    key_mode: str,
) -> float | None:
    specs = chord_vocabulary(detail)
    prefer_flats = key_prefers_flats(key_name(key_root, key_mode))
    by_label = {
        transpose_chord(spec.label, 0, prefer_flats=prefer_flats): spec
        for spec in specs
    }
    total_duration = 0.0
    diatonic_duration = 0.0
    for segment in segments:
        if segment.chord == "N" or segment.duration <= 0:
            continue
        total_duration += segment.duration
        spec = by_label.get(segment.chord)
        if spec is not None and is_diatonic(spec, key_root, key_mode):
            diatonic_duration += segment.duration
    if total_duration <= 0:
        return None
    return diatonic_duration / total_duration


def analyze_audio(
    audio_path: str | Path,
    *,
    detail: str = "simple",
    smoothing: float = 0.65,
    config: AnalyzerConfig | None = None,
) -> AnalyzerOutput:
    config = config or AnalyzerConfig()
    y, sample_rate = _load_audio(audio_path, config.sample_rate)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    duration = float(len(y) / sample_rate) if sample_rate else 0.0
    if duration < config.min_audio_seconds:
        raise AnalysisError(
            f"The song must be at least {config.min_audio_seconds:.0f} seconds long."
        )
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-5:
        raise AnalysisError("The audio is silent or too quiet to analyze.")

    warnings: list[str] = []
    try:
        harmonic = librosa.effects.harmonic(y, margin=2.5)
    except Exception:
        harmonic = y
        warnings.append("Harmonic/percussive separation failed; the raw mix was analyzed.")

    try:
        tuning_bins_12 = float(
            librosa.estimate_tuning(
                y=harmonic,
                sr=sample_rate,
                bins_per_octave=12,
                resolution=0.01,
            )
        )
        if not np.isfinite(tuning_bins_12):
            tuning_bins_12 = 0.0
    except Exception:
        tuning_bins_12 = 0.0
    tuning_cents = float(np.clip(tuning_bins_12 * 100.0, -50.0, 50.0))

    try:
        # chroma_cqt uses 36 bins/octave, so convert the 12-bin tuning fraction.
        chroma = librosa.feature.chroma_cqt(
            y=harmonic,
            sr=sample_rate,
            hop_length=config.hop_length,
            bins_per_octave=36,
            n_octaves=6,
            tuning=tuning_bins_12 * 3.0,
            norm=2,
            cqt_mode="hybrid",
        )
    except Exception:
        chroma = librosa.feature.chroma_stft(
            y=harmonic,
            sr=sample_rate,
            hop_length=config.hop_length,
            n_fft=4096,
            tuning=tuning_bins_12,
            norm=2,
        )
        warnings.append(
            "Constant-Q analysis failed; a less pitch-selective STFT fallback was used."
        )

    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=config.hop_length,
        center=True,
    )[0]
    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sample_rate,
        hop_length=config.hop_length,
        aggregate=np.median,
    )
    tempo_bpm, beat_frames = _estimate_beats(
        onset_envelope,
        sample_rate=sample_rate,
        hop_length=config.hop_length,
    )
    beat_times = librosa.frames_to_time(
        np.asarray(beat_frames), sr=sample_rate, hop_length=config.hop_length
    )

    boundaries, used_beat_tracking = _segment_boundaries(duration, beat_times, config)
    if not used_beat_tracking:
        warnings.append("A stable beat grid was not found; fixed time windows were used.")

    vectors, energies = _aggregate_features(
        chroma,
        rms,
        boundaries,
        sample_rate=sample_rate,
        hop_length=config.hop_length,
    )
    if vectors.size == 0:
        raise AnalysisError("No analyzable harmonic frames were found.")

    active = energies >= max(float(np.median(energies)) * 0.15, 1e-8)
    global_chroma = (
        np.median(vectors[active], axis=0) if np.any(active) else np.mean(vectors, axis=0)
    )
    key, key_confidence, key_root, key_mode = estimate_key(global_chroma)
    labels, confidences, tonalities = classify_chroma_segments(
        vectors,
        energies,
        detail=detail,
        smoothing=smoothing,
        key_root=key_root,
        key_mode=key_mode,
        key_confidence=key_confidence,
    )

    segments = _merge_segments(
        [
            ChordSegment(
                start=float(boundaries[index]),
                end=float(boundaries[index + 1]),
                chord=labels[index],
                confidence=confidences[index],
            )
            for index in range(len(labels))
        ]
    )
    chord_segments = [segment for segment in segments if segment.chord != "N"]
    mean_confidence = (
        float(
            np.average(
                [item.confidence for item in chord_segments],
                weights=[item.duration for item in chord_segments],
            )
        )
        if chord_segments
        else 0.0
    )
    mean_tonality = float(np.mean(tonalities)) if tonalities else 0.0
    diatonic_ratio = _diatonic_duration_ratio(
        chord_segments,
        detail=detail,
        key_root=key_root,
        key_mode=key_mode,
    )
    if not chord_segments:
        warnings.append(
            "No stable chord sequence was found. Try a cleaner recording or manual edits."
        )
    elif mean_confidence < 0.38:
        warnings.append("Overall chord confidence is low; verify the timeline by ear.")
    if mean_tonality < 0.12:
        warnings.append(
            "The recording is harmonically dense or noisy, which weakens chord estimates."
        )
    if abs(tuning_cents) >= 25:
        direction = "sharp" if tuning_cents > 0 else "flat"
        warnings.append(
            f"The recording appears about {abs(tuning_cents):.0f} cents {direction} of A=440. "
            "Retune the instrument or adjust playback pitch for closer matching."
        )
    if diatonic_ratio is not None and diatonic_ratio < 0.70:
        warnings.append(
            f"Only about {diatonic_ratio:.0%} of detected chord duration fits one {key} "
            "harmony. The song may modulate, be modal, or use borrowed chords; treat the "
            "global key label cautiously."
        )
    if key_confidence < 0.18:
        warnings.append(
            "The key estimate is uncertain; modal folk music may not fit major/minor labels."
        )

    return AnalyzerOutput(
        duration=duration,
        tempo_bpm=tempo_bpm,
        key=key,
        key_confidence=key_confidence,
        tuning_cents=tuning_cents,
        segments=segments,
        warnings=warnings,
        used_beat_tracking=used_beat_tracking,
    )
