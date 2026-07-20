from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from .chords import (
    format_chord_notation,
    key_prefers_flats,
    render_chord_diagrams_html,
    suggest_capo,
    transpose_key,
)
from .exports import (
    capo_shape_segments,
    merge_adjacent_segments,
    quantize_segments,
    render_bar_chart,
    render_summary,
    rows_to_segments,
    transpose_segments,
    write_exports,
)
from .models import AnalysisResult, ChordSegment
from .service import analyze_request, default_workspace_root, export_from_state

HEADERS = ["Start (s)", "End (s)", "Chord", "Confidence", "Local key", "Bar", "Beat"]
APP_CSS = """
:root { --sg-accent:#6574ff; --sg-accent2:#a25cff; }
.gradio-container { max-width: 1480px !important; }
.sg-hero { padding:28px; border-radius:24px; margin-bottom:16px; border:1px solid var(--border-color-primary); background:linear-gradient(135deg,rgba(101,116,255,.18),rgba(162,92,255,.11)); }
.sg-hero h1 { margin:0 0 8px; font-size:clamp(2.2rem,5vw,3.8rem); letter-spacing:-.05em; }
.sg-hero p { margin:0; max-width:78ch; font-size:1.03rem; opacity:.84; }
.sg-card { border:1px solid var(--border-color-primary)!important; border-radius:20px!important; padding:4px!important; }
.sg-note { opacity:.8; max-width:90rem; }
.sg-output textarea { font-family:ui-monospace,SFMono-Regular,Menlo,monospace!important; }
.sg-diagram-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr)); gap:12px; }
.sg-diagram-card { border:1px solid var(--border-color-primary); border-radius:16px; padding:8px; background:var(--block-background-fill); }
.sg-diagram-card svg { width:100%; height:auto; stroke:currentColor; fill:none; }
.sg-diagram-title { fill:currentColor; stroke:none; font:700 14px ui-sans-serif,system-ui; }
.sg-fret-label,.sg-marker { fill:currentColor; stroke:none; font:600 10px ui-sans-serif,system-ui; }
.sg-open { stroke:currentColor; stroke-width:1.5; }
.sg-finger { fill:currentColor; stroke:currentColor; }
.sg-missing-shapes { grid-column:1/-1; opacity:.7; padding:8px; }
"""
PROGRESS = gr.Progress()


class SyncedChordViewer(gr.HTML):
    """A browser-side chord follower synchronized to the Gradio audio element."""

    def __init__(self, value: dict[str, Any] | None = None, **kwargs: Any) -> None:
        html_template = r"""
        <div class="sg-sync-shell">
          <div class="sg-now">
            <div><span class="sg-kicker">CURRENT</span><div class="sg-current" data-role="current">—</div><div class="sg-meta" data-role="meta">Analyze a song to begin.</div></div>
            <div class="sg-next-wrap"><span class="sg-kicker">NEXT</span><div class="sg-next" data-role="next">—</div><div class="sg-countdown" data-role="countdown">—</div></div>
          </div>
          <div class="sg-actions">
            <button type="button" data-action="back">−5s</button>
            <button type="button" data-action="rate" data-rate="0.75">0.75×</button>
            <button type="button" data-action="rate" data-rate="1" class="sg-on">1×</button>
            <button type="button" data-action="rate" data-rate="1.25">1.25×</button>
            <button type="button" data-action="forward">+5s</button>
            <button type="button" data-action="loop">Loop selected chord</button>
          </div>
          <div class="sg-help">Click a chord to seek. Double-click one to select the practice loop.</div>
          <div class="sg-strip">
            ${value && value.segments && value.segments.length
              ? value.segments.map((s,i)=>`<button type="button" class="sg-chord" data-segment="${i}" data-start="${s.start}" data-end="${s.end}"><span class="sg-time">${s.time}</span><strong>${s.display}</strong><span class="sg-secondary">${s.secondary||''}</span><span class="sg-confidence">${Math.round(s.confidence*100)}%</span></button>`).join('')
              : '<div class="sg-empty">The synchronized chord track will appear here.</div>'}
          </div>
        </div>
        """
        css_template = r"""
        .sg-sync-shell{border:1px solid var(--border-color-primary);border-radius:22px;padding:16px;background:var(--block-background-fill)}
        .sg-now{display:grid;grid-template-columns:1.35fr .65fr;gap:14px;padding:16px;border-radius:18px;background:linear-gradient(135deg,rgba(101,116,255,.18),rgba(162,92,255,.11))}
        .sg-kicker{font-size:.68rem;letter-spacing:.14em;font-weight:850;opacity:.6}.sg-current{font-size:clamp(3.2rem,8vw,6.8rem);line-height:.95;font-weight:900;letter-spacing:-.06em;margin-top:7px}.sg-meta{opacity:.72;margin-top:7px}.sg-next-wrap{text-align:right;align-self:end}.sg-next{font-size:clamp(1.7rem,4vw,3rem);font-weight:850}.sg-countdown{opacity:.68;font-variant-numeric:tabular-nums}
        .sg-actions{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 7px}.sg-actions button{border:1px solid var(--border-color-primary);border-radius:999px;padding:7px 12px;background:transparent;color:inherit;cursor:pointer;font-weight:700}.sg-actions button.sg-on,.sg-actions button:hover{background:rgba(101,116,255,.18)}
        .sg-help{font-size:.82rem;opacity:.65;margin-bottom:10px}.sg-strip{display:flex;gap:9px;overflow-x:auto;padding:4px 2px 9px;scroll-behavior:smooth}.sg-chord{min-width:116px;padding:10px 12px;border:1px solid var(--border-color-primary);border-radius:15px;background:transparent;color:inherit;text-align:left;cursor:pointer;transition:.15s}.sg-chord:hover{transform:translateY(-2px)}.sg-chord.sg-active{background:linear-gradient(135deg,rgba(101,116,255,.28),rgba(162,92,255,.22));border-color:#6574ff;box-shadow:0 10px 28px rgba(80,88,190,.17)}.sg-chord.sg-selected{outline:2px solid #a25cff;outline-offset:2px}.sg-chord span{display:block;font-size:.72rem;opacity:.64}.sg-chord strong{display:block;font-size:1.45rem;line-height:1.05;margin:4px 0}.sg-confidence{text-align:right}.sg-empty{padding:24px;opacity:.62}@media(max-width:720px){.sg-now{grid-template-columns:1fr}.sg-next-wrap{text-align:left}}
        """
        js_on_load = r"""
        if(element.__sgCleanup) element.__sgCleanup();
        let selected=null,loop=false,last=-1;
        const audio=()=>document.querySelector('#sg-audio-player audio');
        const cards=()=>Array.from(element.querySelectorAll('[data-segment]'));
        const role=(name)=>element.querySelector(`[data-role="${name}"]`);
        const indexAt=(time)=>{const list=cards();for(let i=0;i<list.length;i++){if(time>=Number(list[i].dataset.start)&&time<Number(list[i].dataset.end))return i;}return list.length?Math.max(0,list.length-1):-1;};
        const render=(index,time)=>{const list=cards();list.forEach((card,i)=>card.classList.toggle('sg-active',i===index));if(index<0||!list[index])return;const current=list[index],next=list[index+1];role('current').textContent=current.querySelector('strong')?.textContent||'—';role('meta').textContent=current.querySelector('.sg-secondary')?.textContent||'Concert chord';role('next').textContent=next?.querySelector('strong')?.textContent||'—';const remaining=Math.max(0,Number(current.dataset.end)-time);role('countdown').textContent=next?`in ${remaining.toFixed(1)}s`:'final chord';if(index!==last){current.scrollIntoView({behavior:'smooth',inline:'center',block:'nearest'});last=index;}};
        const tick=()=>{const player=audio();if(!player)return;const index=indexAt(player.currentTime||0);render(index,player.currentTime||0);if(loop&&selected!==null){const card=cards()[selected];if(card&&player.currentTime>=Number(card.dataset.end)-.025){player.currentTime=Number(card.dataset.start);if(player.paused)player.play().catch(()=>{});}}};
        const click=(event)=>{const player=audio();const action=event.target.closest('[data-action]');if(action&&player){if(action.dataset.action==='back')player.currentTime=Math.max(0,player.currentTime-5);if(action.dataset.action==='forward')player.currentTime=Math.min(player.duration||Infinity,player.currentTime+5);if(action.dataset.action==='rate'){player.playbackRate=Number(action.dataset.rate||1);element.querySelectorAll('[data-action="rate"]').forEach(button=>button.classList.toggle('sg-on',button===action));}if(action.dataset.action==='loop'){loop=!loop;action.classList.toggle('sg-on',loop);}return;}const card=event.target.closest('[data-segment]');if(card&&player){player.currentTime=Number(card.dataset.start);player.play().catch(()=>{});}};
        const dbl=(event)=>{const card=event.target.closest('[data-segment]');if(!card)return;selected=Number(card.dataset.segment);cards().forEach((item,i)=>item.classList.toggle('sg-selected',i===selected));};
        element.addEventListener('click',click);element.addEventListener('dblclick',dbl);const timer=setInterval(tick,100);element.__sgCleanup=()=>{clearInterval(timer);element.removeEventListener('click',click);element.removeEventListener('dblclick',dbl);};
        """
        super().__init__(
            value=value or {"segments": []},
            html_template=html_template,
            css_template=css_template,
            js_on_load=js_on_load,
            **kwargs,
        )

    def api_info(self) -> dict[str, Any]:
        return {"type": "object"}


def _timeline_frame(segments: list[ChordSegment]) -> pd.DataFrame:
    return pd.DataFrame([segment.to_row() for segment in segments], columns=HEADERS)


def _meter(value: str | int | None) -> int | None:
    return int(value) if value in {3, 4, 6, "3", "4", "6"} else None


def _views(
    result: AnalysisResult,
    segments: list[ChordSegment],
    *,
    transpose: int,
    capo: int,
    notation: str,
    instrument: str,
    meter_override: int | None,
) -> tuple[str, dict[str, Any], str, str]:
    target_key = transpose_key(result.key, transpose)
    flats = key_prefers_flats(target_key)
    sounding = transpose_segments(segments, transpose, prefer_flats=flats)
    shape_key = transpose_key(target_key, -capo) if capo else target_key
    shapes = capo_shape_segments(sounding, capo, prefer_flats=key_prefers_flats(shape_key))
    payload: list[dict[str, Any]] = []
    for sound, shape in zip(sounding, shapes, strict=True):
        local_key = transpose_key(sound.local_key, transpose) if sound.local_key else target_key
        display = (
            shape.chord
            if notation == "standard"
            else format_chord_notation(sound.chord, local_key, notation)
        )
        secondary: list[str] = []
        if capo and notation == "standard":
            secondary.append(f"sounds {sound.chord}")
        if sound.local_key:
            secondary.append(local_key)
        if sound.bar is not None:
            secondary.append(
                f"bar {sound.bar}" + (f", beat {sound.beat:.1f}" if sound.beat is not None else "")
            )
        minutes, seconds = divmod(sound.start, 60)
        payload.append(
            {
                "start": round(sound.start, 4),
                "end": round(sound.end, 4),
                "time": f"{int(minutes):02d}:{seconds:04.1f}",
                "display": display,
                "secondary": " · ".join(secondary),
                "confidence": sound.confidence,
            }
        )
    return (
        render_summary(result, segments, transpose=transpose, instrument=instrument, capo=capo),
        {"segments": payload},
        render_bar_chart(
            result,
            segments,
            transpose=transpose,
            capo=capo,
            notation=notation,
            meter_override=meter_override,
        ),
        render_chord_diagrams_html((segment.chord for segment in shapes), instrument),
    )


def _analyze_ui(
    youtube_url: str,
    upload_path: str | None,
    detail: str,
    smoothing: float,
    meter_override: str,
    instrument: str,
    progress: gr.Progress = PROGRESS,
) -> tuple[str, str, str, pd.DataFrame, dict[str, Any], str, str, list[str], dict[str, Any], int]:
    try:
        progress(0.04, desc="Preparing audio")
        result, job_dir = analyze_request(
            youtube_url=youtube_url,
            upload_path=upload_path,
            detail=detail,
            smoothing=float(smoothing),
            meter_override=_meter(meter_override),
        )
        progress(0.88, desc="Building synchronized chart")
        capo, _shapes, _ease = suggest_capo(result.segments, instrument)
        summary, player, chart, diagrams = _views(
            result,
            result.segments,
            transpose=0,
            capo=capo,
            notation="standard",
            instrument=instrument,
            meter_override=_meter(meter_override),
        )
        files = write_exports(
            result,
            job_dir / "exports",
            instrument=instrument,
            capo=capo,
            meter_override=_meter(meter_override),
        )
        progress(1.0, desc="Done")
        return (
            "Analysis complete. The chord strip follows playback. Edit anything that sounds wrong, then apply edits or regenerate downloads.",
            result.audio_path,
            summary,
            _timeline_frame(result.segments),
            player,
            chart,
            diagrams,
            files,
            result.to_state(),
            capo,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _refresh_ui(
    timeline: Any,
    state: dict[str, Any],
    transpose: int,
    capo: int,
    notation: str,
    instrument: str,
    meter_override: str,
) -> tuple[str, dict[str, Any], str, str]:
    try:
        if not state:
            return "Analyze a song first.", {"segments": []}, "", ""
        return _views(
            AnalysisResult.from_state(state),
            rows_to_segments(timeline),
            transpose=int(transpose),
            capo=int(capo),
            notation=notation,
            instrument=instrument,
            meter_override=_meter(meter_override),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _export_ui(
    timeline: Any,
    state: dict[str, Any],
    transpose: int,
    capo: int,
    notation: str,
    instrument: str,
    meter_override: str,
) -> tuple[str, str, dict[str, Any], str, str, list[str]]:
    try:
        result, files = export_from_state(
            state,
            timeline,
            transpose=int(transpose),
            instrument=instrument,
            capo=int(capo),
            notation=notation,
            meter_override=_meter(meter_override),
        )
        summary, player, chart, diagrams = _views(
            result,
            rows_to_segments(timeline),
            transpose=int(transpose),
            capo=int(capo),
            notation=notation,
            instrument=instrument,
            meter_override=_meter(meter_override),
        )
        return (
            "Views and downloads regenerated from the edited timeline.",
            summary,
            player,
            chart,
            diagrams,
            files,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _quantize_ui(timeline: Any, state: dict[str, Any]) -> pd.DataFrame:
    if not state:
        raise gr.Error("Analyze a song first.")
    result = AnalysisResult.from_state(state)
    return _timeline_frame(
        quantize_segments(rows_to_segments(timeline), result.beats, duration=result.duration)
    )


def _merge_ui(timeline: Any) -> pd.DataFrame:
    return _timeline_frame(merge_adjacent_segments(rows_to_segments(timeline)))


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="SgChords", delete_cache=(86_400, 86_400), analytics_enabled=False
    ) as demo:
        gr.HTML(
            "<section class='sg-hero'><h1>SgChords</h1><p>Turn a YouTube link or recording into a synchronized, editable guitar or ukulele play-along chart. Chords, local tonal regions, bars, confidence, capo shapes and transposition stay visible.</p></section>"
        )
        with gr.Row():
            with gr.Column(scale=3, elem_classes="sg-card"):
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
            with gr.Column(scale=2, elem_classes="sg-card"):
                detail = gr.Radio(
                    choices=[
                        ("Simple — major/minor", "simple"),
                        ("Standard — 7, maj7, m7, sus, dim", "standard"),
                        ("Detailed — adds 6, add9, aug and power", "detailed"),
                    ],
                    value="standard",
                    label="Chord vocabulary",
                )
                smoothing = gr.Slider(
                    0,
                    1,
                    value=0.68,
                    step=0.05,
                    label="Sequence smoothing",
                    info="Higher removes flicker; lower preserves fast changes.",
                )
                with gr.Row():
                    meter_override = gr.Dropdown(
                        ["Auto", "3", "4", "6"], value="Auto", label="Beats per bar"
                    )
                    instrument = gr.Radio(
                        [("Guitar", "guitar"), ("Ukulele", "ukulele")],
                        value="guitar",
                        label="Instrument",
                    )
        with gr.Row():
            analyze_button = gr.Button("Analyze song", variant="primary", size="lg")
            clear_button = gr.ClearButton(value="Clear", size="lg")
        gr.Markdown(
            "Automatic transcription is a draft. Local-key and drone tracking reduce common folk-music errors, but modal tunes and noisy field recordings still need your ear.",
            elem_classes="sg-note",
        )
        status = gr.Markdown()
        state = gr.State({})
        with gr.Tab("Play along"):
            with gr.Row():
                with gr.Column(scale=2):
                    audio_preview = gr.Audio(
                        label="Normalized source",
                        type="filepath",
                        interactive=False,
                        editable=False,
                        elem_id="sg-audio-player",
                    )
                with gr.Column(scale=3):
                    summary = gr.Markdown()
            player = SyncedChordViewer()
            with gr.Row():
                transpose = gr.Slider(
                    -11,
                    11,
                    value=0,
                    step=1,
                    label="Concert transpose",
                    info="Keep 0 to play with the original audio.",
                )
                capo = gr.Slider(
                    0,
                    7,
                    value=0,
                    step=1,
                    label="Capo fret",
                    info="Changes shapes, not sounding pitch.",
                )
                notation = gr.Radio(
                    [
                        ("Chord names", "standard"),
                        ("Roman numerals", "roman"),
                        ("Nashville numbers", "nashville"),
                    ],
                    value="standard",
                    label="Display notation",
                )
        with gr.Tab("Chord chart"):
            chart = gr.Code(
                label="Beat-aligned chart",
                language=None,
                lines=20,
                interactive=False,
                elem_classes="sg-output",
            )
            diagrams = gr.HTML(label="Chord diagrams")
        with gr.Tab("Edit timeline"):
            gr.Markdown(
                "Edit concert chord names. Quantize snaps boundaries to detected beats; merge joins adjacent repeats."
            )
            timeline = gr.Dataframe(
                headers=HEADERS,
                datatype=["number", "number", "str", "number", "str", "number", "number"],
                type="pandas",
                interactive=True,
                wrap=True,
                label="Editable timeline",
                show_row_numbers=True,
            )
            with gr.Row():
                quantize_button = gr.Button("Quantize to beats")
                merge_button = gr.Button("Merge repeats")
                apply_button = gr.Button("Apply edits to player", variant="secondary")
        with gr.Tab("Downloads"):
            export_button = gr.Button("Regenerate all downloads", variant="primary")
            downloads = gr.File(
                label="Text, CSV, JSON, ChordPro, shapes, SRT and LAB",
                file_count="multiple",
                interactive=False,
            )

        analyze_outputs = [
            status,
            audio_preview,
            summary,
            timeline,
            player,
            chart,
            diagrams,
            downloads,
            state,
            capo,
        ]
        analyze_button.click(
            _analyze_ui,
            [youtube_url, upload, detail, smoothing, meter_override, instrument],
            analyze_outputs,
            api_name="analyze",
            concurrency_limit=1,
        )
        youtube_url.submit(
            _analyze_ui,
            [youtube_url, upload, detail, smoothing, meter_override, instrument],
            analyze_outputs,
            api_name=False,
            concurrency_limit=1,
        )
        refresh_inputs = [timeline, state, transpose, capo, notation, instrument, meter_override]
        refresh_outputs = [summary, player, chart, diagrams]
        for component in (transpose, capo, notation, instrument, meter_override):
            component.change(
                _refresh_ui,
                refresh_inputs,
                refresh_outputs,
                api_name=False,
                trigger_mode="always_last",
            )
        apply_button.click(_refresh_ui, refresh_inputs, refresh_outputs, api_name="refresh")
        quantize_button.click(_quantize_ui, [timeline, state], timeline, api_name="quantize")
        merge_button.click(_merge_ui, timeline, timeline, api_name="merge")
        export_button.click(
            _export_ui,
            refresh_inputs,
            [status, summary, player, chart, diagrams, downloads],
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
                player,
                chart,
                diagrams,
                downloads,
                state,
            ]
        )
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
