from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtube-nocookie.com",
}


class InputError(ValueError):
    pass


class DependencyError(RuntimeError):
    pass


@dataclass(slots=True)
class PreparedAudio:
    path: Path
    title: str
    source: str
    duration: float


def is_youtube_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and (parsed.hostname or "").lower() in YOUTUBE_HOSTS


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise DependencyError(
            f"{name} is required. Install FFmpeg and put ffmpeg and ffprobe on PATH."
        )
    return path


def probe_duration(path: str | Path) -> float:
    command = [
        require_binary("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    if process.returncode != 0:
        raise InputError(process.stderr.strip() or "ffprobe could not read this file")
    try:
        duration = float(json.loads(process.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InputError("Could not determine media duration.") from exc
    if duration <= 0:
        raise InputError("The media file has no readable audio duration.")
    return duration


def normalize_audio(source: str | Path, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        require_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "22050",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=240)
    if process.returncode != 0 or not destination.exists():
        details = process.stderr.strip().splitlines()
        raise InputError(details[-1] if details else "FFmpeg could not decode the audio.")
    return destination


def _safe_title(value: str) -> str:
    return re.sub(r"[\x00-\x1f]+", " ", value).strip()[:160] or "Untitled song"


def prepare_local_audio(
    source: str | Path,
    workdir: str | Path,
    *,
    max_duration_seconds: int = 1200,
    max_file_bytes: int = 600 * 1024 * 1024,
) -> PreparedAudio:
    source_path = Path(source)
    if not source_path.is_file():
        raise InputError("Choose an existing audio or video file.")
    if source_path.stat().st_size > max_file_bytes:
        raise InputError("The uploaded file is too large (limit: 600 MB).")
    duration = probe_duration(source_path)
    if duration > max_duration_seconds:
        raise InputError(
            f"This file is {duration / 60:.1f} minutes; the limit is {max_duration_seconds / 60:.0f} minutes."
        )
    destination = Path(workdir) / "source.wav"
    normalize_audio(source_path, destination)
    return PreparedAudio(destination, _safe_title(source_path.stem), str(source_path), duration)


def _downloaded_path(info: dict[str, Any], ydl: Any, workdir: Path) -> Path:
    for item in info.get("requested_downloads") or []:
        if item.get("filepath") and Path(item["filepath"]).is_file():
            return Path(item["filepath"])
    prepared = Path(ydl.prepare_filename(info))
    if prepared.is_file():
        return prepared
    candidates = [path for path in workdir.glob("download.*") if path.is_file()]
    if not candidates:
        raise InputError("YouTube audio was downloaded but the output file was not found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def prepare_youtube_audio(
    url: str, workdir: str | Path, *, max_duration_seconds: int = 1200
) -> PreparedAudio:
    if not is_youtube_url(url):
        raise InputError("Enter a full youtube.com or youtu.be URL.")
    try:
        import yt_dlp
    except ImportError as exc:
        raise DependencyError("yt-dlp is not installed. Run `uv sync --locked`.") from exc
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def reject(info: dict[str, Any], *, incomplete: bool) -> str | None:
        del incomplete
        if info.get("is_live"):
            return "Live streams are not supported."
        if info.get("duration") and float(info["duration"]) > max_duration_seconds:
            return f"Video exceeds the {max_duration_seconds / 60:.0f}-minute limit."
        return None

    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "download.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 2,
        "match_filter": reject,
        "restrictfilenames": True,
    }
    cookie_file = os.getenv("SGCHORDS_COOKIES_FILE")
    if cookie_file:
        cookie_path = Path(cookie_file).expanduser()
        if not cookie_path.is_file():
            raise DependencyError("SGCHORDS_COOKIES_FILE is not readable.")
        options["cookiefile"] = str(cookie_path)
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            raw = ydl.extract_info(url.strip(), download=True)
            if not raw or raw.get("_type") == "playlist":
                raise InputError("Provide one video, not a playlist.")
            info = ydl.sanitize_info(raw)
            downloaded = _downloaded_path(info, ydl, workdir)
    except yt_dlp.utils.DownloadError as exc:
        message = str(exc).replace("ERROR:", "").strip()
        raise InputError(
            f"YouTube download failed: {message}. Update the lockfile or use a local file; some videos require cookies."
        ) from exc
    duration = float(info.get("duration") or probe_duration(downloaded))
    if duration > max_duration_seconds:
        raise InputError(f"Video exceeds the {max_duration_seconds / 60:.0f}-minute limit.")
    destination = workdir / "source.wav"
    normalize_audio(downloaded, destination)
    return PreparedAudio(
        destination, _safe_title(str(info.get("title") or "YouTube song")), url.strip(), duration
    )
