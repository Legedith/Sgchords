from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from .chords import key_prefers_flats, transpose_key
from .exports import (
    render_shapes_for_segments,
    render_sheet,
    render_summary,
    rows_to_segments,
    transpose_segments,
    write_exports,
)
from .service import analyze_request, default_workspace_root, export_from_state

HEADERS = ["Start (s)", "End (s)", "Chord", "Confidence"]
APP_CSS = """
.sg-note { max-width: 72rem; }
.sg-output textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
"""
PROGRESS = gr.Progress()


def _timeline_frame(rows: list[list[Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=HEADERS)


def _analyze_ui(
    youtube_url: str,
    upload_path: str | None,
    detail: str,
    smoothing: float,
    instrument: str,
    progress: gr.Progress = PROGRESS,
) -> tuple[str, str, str, pd.DataFrame, str, str, list[str], dict[str, Any]]:
    try:
        progress(0.05, desc="Preparing audio")
        result, job_dir = analyze_request(
            youtube_url=youtube_url,
            upload_path=upload_path,
            detail=detail,
            smoothing=float(smoothing),
        )
        progress(0.88, desc="Building chord sheet")
        files = write_exports(result, job_dir / "exports", instrument=instrument)
        timeline = _timeline_frame([segment.to_row() for segment in result.segments])
        sheet = render_sheet(
            result.title,
            result.segments,
            key=result.key,
            tempo_bpm=result.tempo_bpm,
        )
        summary = render_summary(result, result.segments, instrument=instrument)
        shapes = render_shapes_for_segments(result.segments, instrument)
        progress(1.0, desc="Done")
        status = (
            "Analysis complete. Play the normalized audio against the timeline, edit any wrong "
            "chord directly in the table, then regenerate the downloads."
        )
        return (
            status,
            result.audio_path,
            summary,
            timeline,
            sheet,
            shapes,
            files,
            result.to_state(),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _export_ui(
    timeline: Any,
    state: dict[str, Any],
    transpose: int,
    instrument: str,
) -> tuple[str, str, str, str, list[str]]:
    try:
        result, files = export_from_state(
            state,
            timeline,
            transpose=int(transpose),
            instrument=instrument,
        )
        edited_segments = rows_to_segments(timeline)
        target_key = transpose_key(result.key, int(transpose))
        prefer_flats = key_prefers_flats(target_key)
        target_segments = transpose_segments(
            edited_segments,
            int(transpose),
            prefer_flats=prefer_flats,
        )
        summary = render_summary(
            result,
            target_segments,
            transpose=int(transpose),
            instrument=instrument,
        )
        sheet = render_sheet(
            result.title,
            target_segments,
            key=result.key,
            tempo_bpm=result.tempo_bpm,
            transpose=int(transpose),
        )
        shapes = render_shapes_for_segments(target_segments, instrument)
        return "Downloads regenerated from the edited timeline.", summary, sheet, shapes, files
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="SgChords",
        delete_cache=(86_400, 86_400),
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# SgChords\n"
            "Estimate time-aligned chords from a YouTube song or an uploaded recording. "
            "The result is a starting point, not ground truth: vocals, drums, drones, modal tunes, "
            "live noise, and unusual tuning can confuse automatic transcription."
        )
        gr.Markdown(
            "Use only recordings you are allowed to download or process. YouTube support depends "
            "on yt-dlp, FFmpeg, and the video's access restrictions.",
            elem_classes="sg-note",
        )

        with gr.Row():
            with gr.Column(scale=3):
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    info="Leave blank when uploading a file.",
                )
                upload = gr.File(
                    label="Or upload audio/video",
                    file_types=["audio", "video", ".m4a", ".webm", ".mp4"],
                    type="filepath",
                )
            with gr.Column(scale=2):
                detail = gr.Radio(
                    choices=[
                        ("Simple: major/minor", "simple"),
                        ("Detailed: 7/sus/dim", "detailed"),
                    ],
                    value="simple",
                    label="Chord vocabulary",
                )
                smoothing = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.65,
                    step=0.05,
                    label="Sequence smoothing",
                    info="Higher values remove short chord flicker but can miss quick changes.",
                )
                instrument = gr.Radio(
                    choices=[("Guitar", "guitar"), ("Ukulele", "ukulele")],
                    value="guitar",
                    label="Chord shapes",
                )

        with gr.Row():
            analyze_button = gr.Button("Analyze song", variant="primary")
            clear_button = gr.ClearButton(value="Clear")

        status = gr.Markdown()
        state = gr.State({})
        with gr.Row():
            audio_preview = gr.Audio(
                label="Normalized source",
                type="filepath",
                interactive=False,
            )
            summary = gr.Markdown(label="Analysis summary")

        timeline = gr.Dataframe(
            headers=HEADERS,
            datatype=["number", "number", "str", "number"],
            type="pandas",
            interactive=True,
            wrap=True,
            label="Editable chord timeline",
            show_row_numbers=True,
        )
        with gr.Row():
            transpose = gr.Slider(
                minimum=-11,
                maximum=11,
                value=0,
                step=1,
                label="Export transpose (semitones)",
                info=(
                    "Keep 0 to play with the original audio. Use the capo suggestion "
                    "for easier shapes."
                ),
            )
            export_button = gr.Button("Regenerate downloads from edits", variant="secondary")

        with gr.Row():
            sheet = gr.Code(
                label="Compact chord sheet",
                language="markdown",
                lines=14,
                interactive=False,
                elem_classes="sg-output",
            )
            shapes = gr.Markdown(label="Chord shapes")
        downloads = gr.File(
            label="Downloads: text, CSV, JSON, ChordPro, shapes",
            file_count="multiple",
            interactive=False,
        )

        analyze_event = analyze_button.click(
            fn=_analyze_ui,
            inputs=[youtube_url, upload, detail, smoothing, instrument],
            outputs=[status, audio_preview, summary, timeline, sheet, shapes, downloads, state],
            api_name="analyze",
        )
        youtube_url.submit(
            fn=_analyze_ui,
            inputs=[youtube_url, upload, detail, smoothing, instrument],
            outputs=[status, audio_preview, summary, timeline, sheet, shapes, downloads, state],
            api_name=False,
        )
        export_button.click(
            fn=_export_ui,
            inputs=[timeline, state, transpose, instrument],
            outputs=[status, summary, sheet, shapes, downloads],
            api_name="export",
        )
        clear_button.add(
            [
                youtube_url,
                upload,
                status,
                audio_preview,
                summary,
                timeline,
                sheet,
                shapes,
                downloads,
                state,
            ]
        )
        del analyze_event

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue(max_size=8, default_concurrency_limit=1)
    workspace = default_workspace_root()
    workspace.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        allowed_paths=[str(Path(workspace).resolve())],
        max_file_size="600mb",
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
