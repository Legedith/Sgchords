from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .analyzer import analyze_audio
from .audio import InputError, is_youtube_url, prepare_local_audio, prepare_youtube_audio
from .exports import rows_to_segments, write_exports
from .models import AnalysisResult

DEFAULT_MAX_DURATION_SECONDS = int(os.getenv("SGCHORDS_MAX_DURATION_SECONDS", "1200"))
DEFAULT_JOB_TTL_SECONDS = int(os.getenv("SGCHORDS_JOB_TTL_SECONDS", "86400"))


def default_workspace_root() -> Path:
    configured = os.getenv("SGCHORDS_WORKSPACE")
    return Path(configured).expanduser() if configured else Path(tempfile.gettempdir()) / "sgchords"


def cleanup_old_jobs(root: str | Path, *, ttl_seconds: int = DEFAULT_JOB_TTL_SECONDS) -> None:
    root = Path(root)
    if not root.exists():
        return
    cutoff = time.time() - ttl_seconds
    for path in root.iterdir():
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def create_job_dir(root: str | Path | None = None) -> Path:
    root_path = Path(root) if root is not None else default_workspace_root()
    root_path.mkdir(parents=True, exist_ok=True)
    cleanup_old_jobs(root_path)
    job = root_path / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=False)
    return job


def analyze_request(
    *,
    youtube_url: str | None = None,
    upload_path: str | Path | None = None,
    detail: str = "standard",
    smoothing: float = 0.68,
    meter_override: int | None = None,
    workspace_root: str | Path | None = None,
    max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
) -> tuple[AnalysisResult, Path]:
    url = (youtube_url or "").strip()
    file_value = str(upload_path).strip() if upload_path else ""
    if bool(url) == bool(file_value):
        raise InputError("Provide exactly one source: a YouTube URL or an uploaded file.")
    if url and not is_youtube_url(url):
        raise InputError("Only full YouTube URLs are accepted in the URL field.")
    job_dir = create_job_dir(workspace_root)
    try:
        prepared = (
            prepare_youtube_audio(url, job_dir, max_duration_seconds=max_duration_seconds)
            if url
            else prepare_local_audio(file_value, job_dir, max_duration_seconds=max_duration_seconds)
        )
        analyzed = analyze_audio(
            prepared.path,
            detail=detail,
            smoothing=smoothing,
            meter_override=meter_override,
        )
        result = AnalysisResult(
            title=prepared.title,
            source=prepared.source,
            audio_path=str(prepared.path),
            duration=analyzed.duration,
            tempo_bpm=analyzed.tempo_bpm,
            tempo_confidence=analyzed.tempo_confidence,
            meter=analyzed.meter,
            meter_confidence=analyzed.meter_confidence,
            key=analyzed.key,
            key_confidence=analyzed.key_confidence,
            tuning_cents=analyzed.tuning_cents,
            segments=analyzed.segments,
            beats=analyzed.beats,
            key_regions=analyzed.key_regions,
            patterns=analyzed.patterns,
            drone=analyzed.drone,
            drone_confidence=analyzed.drone_confidence,
            warnings=analyzed.warnings,
            used_beat_tracking=analyzed.used_beat_tracking,
        )
        return result, job_dir
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise


def export_from_state(
    state: dict[str, Any],
    timeline: Any,
    *,
    transpose: int = 0,
    instrument: str = "guitar",
    capo: int = 0,
    notation: str = "standard",
    meter_override: int | None = None,
) -> tuple[AnalysisResult, list[str]]:
    if not state:
        raise InputError("Analyze a song before exporting.")
    result = AnalysisResult.from_state(state)
    segments = rows_to_segments(timeline)
    if not segments:
        raise InputError("The timeline is empty.")
    files = write_exports(
        result,
        Path(result.audio_path).parent / "exports",
        segments=segments,
        transpose=int(transpose),
        instrument=instrument,
        capo=int(capo),
        notation=notation,
        meter_override=meter_override,
    )
    return result, files
