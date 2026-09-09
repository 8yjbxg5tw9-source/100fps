"""Step 8 — Gradio WebUI: 720p videonu kliklə 8K @ 1000 FPS-ə çevir.

Run it with::

    pip install -r requirements-ui.txt   # once
    python app.py                        # or: python main.py --ui
    python app.py --port 7860 --share    # custom port / public link

The heavy pipeline runs in a background thread (:func:`run_pipeline`) while
this frontend polls the :class:`ProgressBus` and streams status, progress,
GPU stats and console lines back to the browser — the interface never
freezes, and ⏹ stops the run after the current step (checkpoint kept, so a
stopped run resumes later). When the run finishes, the original and the
upscaled video appear side by side for before/after comparison.
"""

from __future__ import annotations

import argparse
import shlex
import threading
import time
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple

_STAGE_LABELS = {
    "interpolate": "🔀 İnterpolasiya (Step 3)",
    "upscale": "🔍 8K Upscale (Step 4)",
}


def _progress_html(
    stage: str, done: int, total: Optional[int],
    rate: Optional[float], eta: str,
) -> str:
    if total:
        pct = min(100.0, 100.0 * done / total)
        bar = (
            f"<div style='background:#2b2b2b;border-radius:8px;height:22px;'>"
            f"<div style='background:linear-gradient(90deg,#ff7a18,#ffb347);"
            f"width:{pct:.1f}%;height:22px;border-radius:8px;'></div></div>"
            f"<div style='margin-top:4px'>{stage}: "
            f"<b>{done:,}/{total:,}</b> kadr ({pct:.1f}%)</div>"
        )
    else:
        bar = (
            f"<div style='background:#2b2b2b;border-radius:8px;height:22px;'>"
            f"<div style='background:linear-gradient(90deg,#ff7a18,#ffb347);"
            f"width:100%;height:22px;border-radius:8px;'></div></div>"
            f"<div style='margin-top:4px'>{stage}: <b>{done:,}</b> kadr</div>"
        )
    speed = f"{rate:.1f} kadr/san" if rate else "—"
    return f"{bar}<div>⚡ {speed} · ⏳ ETA: <b>{eta}</b></div>"


def create_app():
    """Build the Gradio Blocks app (imports Gradio lazily)."""
    try:
        import gradio as gr
    except ImportError:
        raise RuntimeError(
            "WebUI üçün Gradio lazımdır: pip install -r requirements-ui.txt "
            "(Gradio is required for the WebUI)"
        ) from None

    from pipeline.webui import (
        FrameProgress,
        LogEvent,
        ProgressBus,
        RateEstimator,
        RunOptions,
        StatusEvent,
        StepEvent,
        format_eta,
        parse_resolution,
        parse_tile_size,
        probe_gpu,
        run_pipeline,
    )

    stop_flags: Dict[int, threading.Event] = {}
    run_counter = {"n": 0}

    with gr.Blocks(title="100fps — 720p → 8K @ 1000 FPS") as demo:
        gr.Markdown(
            "# 🎬 720p → 8K @ 1000 FPS\n"
            "Yerli video pipeline — videonu yükləyin, parametrləri seçin, "
            "**Emalı başlat**-a basın. Tərəqqi, GPU və loglar canlı gəlir; "
            "bitdikdə **Əvvəl/Sonra** müqayisəsi açılır."
        )
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Video(label="📥 Giriş videosu (sürükləyib buraxın)")
                fps = gr.Slider(
                    minimum=30, maximum=1000, value=1000, step=10,
                    label="🎞 Hədəf FPS",
                )
                res = gr.Radio(
                    ["8K", "4K", "1080p", "720p"], value="8K",
                    label="🖥 Hədəf rezolusiya",
                )
                model = gr.Dropdown(
                    ["x4plus", "x4plus-anime"], value="x4plus",
                    label="🧠 Model (Real-ESRGAN)",
                )
                tile = gr.Dropdown(
                    ["auto", "256", "512", "1024"], value="auto",
                    label="🧩 Tile ölçüsü (VRAM)",
                )
                with gr.Row():
                    interp = gr.Radio(
                        ["rife", "blend"], value="rife",
                        label="İnterpolasiya (blend = sürətli sınaq)",
                    )
                    upsc = gr.Radio(
                        ["esrgan", "resize"], value="esrgan",
                        label="Upscale (resize = sürətli sınaq)",
                    )
                codec = gr.Dropdown(
                    ["auto", "hevc_nvenc", "libx265", "av1_nvenc", "libsvtav1"],
                    value="auto", label="📦 Video kodek",
                )
                crf = gr.Slider(
                    minimum=15, maximum=28, value=19, step=1,
                    label="🎚 Keyfiyyət (CRF — kiçik = yaxşı)",
                )
                resume = gr.Radio(
                    ["auto", "resume", "fresh"], value="auto",
                    label="💾 Yarımçıq iş (auto = soruşmadan davam et)",
                )
                cleanup = gr.Checkbox(
                    False, label="🧹 Sonda müvəqqəti faylları sil (Step 6)"
                )
                out = gr.Textbox("", label="Çıxış yolu (boş = avtomatik)")
                ws = gr.Textbox("workspace", label="Workspace qovluğu")
                with gr.Row():
                    run_btn = gr.Button("▶ Emalı başlat", variant="primary")
                    stop_btn = gr.Button("⏹ Dayandır")
                cli_box = gr.Textbox(
                    "", label="⌨ Ekvivalent CLI əmri", interactive=False
                )
            with gr.Column(scale=3):
                status = gr.Markdown("**Hazırdır.** Video seçib başlayın.")
                progress = gr.HTML("")
                gpu = gr.Markdown("")
                console = gr.Textbox(
                    "", label="🖥 Konsol (canlı log)", lines=16,
                    interactive=False, autoscroll=True,
                )
                with gr.Row():
                    before = gr.Video(label="⬅ Əvvəl (orijinal)")
                    after = gr.Video(label="Sonra ➡")

        def _run(
            video: Optional[str], fps_v: float, res_v: str, model_v: str,
            tile_v: str, interp_v: str, upsc_v: str, codec_v: str, crf_v: float,
            resume_v: str, cleanup_v: bool, out_v: str, ws_v: str,
        ) -> Iterator[Tuple[Any, ...]]:
            empty_bar = _progress_html("Gözləyir", 0, None, None, "—")
            try:
                opts = RunOptions(
                    input=video or "",
                    output=out_v.strip() or None,
                    workspace=ws_v.strip() or "workspace",
                    target_fps=float(fps_v),
                    resolution=parse_resolution(res_v),
                    tile_size=parse_tile_size(tile_v),
                    model=model_v,
                    interp_backend=interp_v,
                    upscale_backend=upsc_v,
                    video_codec=codec_v,
                    crf=float(crf_v),
                    resume=resume_v,
                    cleanup=bool(cleanup_v),
                )
            except (ValueError, KeyError) as exc:
                yield (
                    f"❌ **Parametr xətası:** {exc}", empty_bar, "", "", video,
                    None, "",
                )
                return
            problems = opts.validate()
            if problems:
                yield (
                    "❌ **Yoxlama xətası:**\n" + "\n".join(f"- {p}" for p in problems),
                    empty_bar, "", "", video, None, "",
                )
                return

            run_counter["n"] += 1
            run_id = run_counter["n"]
            bus = ProgressBus()
            stop = threading.Event()
            stop_flags[run_id] = stop
            outcome: Dict[str, Any] = {}

            def _worker() -> None:
                try:
                    outcome["result"] = run_pipeline(opts, bus, stop)
                except Exception as exc:  # noqa: BLE001 - surfaced to UI
                    outcome["error"] = exc

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

            logs: deque = deque(maxlen=200)
            est = RateEstimator()
            stage, done, total = "⏳ Başlayır...", 0, None
            step_no: Optional[int] = None
            gpu_md, last_gpu = "", 0.0
            cli_cmd = "python main.py " + " ".join(
                shlex.quote(a) for a in opts.to_cli_args()
            )
            try:
                while thread.is_alive():
                    for event in bus.poll():
                        if isinstance(event, LogEvent):
                            logs.append(f"[{event.level}] {event.message}")
                        elif isinstance(event, StepEvent):
                            step_no = event.step
                            stage = f"🧩 Addım {event.step}: {event.name}"
                        elif isinstance(event, FrameProgress):
                            stage = _STAGE_LABELS.get(event.stage, event.stage)
                            done, total = event.done, event.total or total
                            est.update(done)
                        elif isinstance(event, StatusEvent):
                            logs.append(f"[STATUS] {event.detail or event.status}")
                    now = time.monotonic()
                    if now - last_gpu > 2.0:
                        last_gpu = now
                        stats = probe_gpu()
                        gpu_md = (
                            f"**GPU:** {stats.describe()}"
                            if stats else "**GPU:** NVIDIA tapılmadı (CPU rejimi)"
                        )
                    rate = est.rate
                    status_md = (
                        f"### ▶ Emal gedir ({stage})\n"
                        + (f"**Addım:** {step_no}/{'6' if opts.cleanup else '5'}  ·  " if step_no else "")
                        + f"**Məqsəd:** {opts.resolution_label().upper()} @ "
                        f"{opts.target_fps:g} FPS"
                    )
                    yield (
                        status_md,
                        _progress_html(stage, done, total, rate, format_eta(est.eta(done, total))),
                        gpu_md, "\n".join(logs), video, None, cli_cmd,
                    )
                    time.sleep(0.5)
                thread.join()
                for event in bus.poll():  # final drain
                    if isinstance(event, LogEvent):
                        logs.append(f"[{event.level}] {event.message}")
            finally:
                stop_flags.pop(run_id, None)

            if "error" in outcome:
                exc = outcome["error"]
                yield (
                    f"❌ **Xəta:** `{type(exc).__name__}: {exc}`",
                    _progress_html("Dayandı (xəta)", done, total, None, "—"),
                    gpu_md, "\n".join(logs), video, None, cli_cmd,
                )
                return
            result = outcome["result"]
            if result.status == "cancelled":
                yield (
                    f"⏹ **Dayandırıldı.** {result.message}",
                    _progress_html("Dayandırıldı", done, total, None, "—"),
                    gpu_md, "\n".join(logs), video, None, cli_cmd,
                )
            else:
                yield (
                    f"✅ **Tamamlandı!** {result.message}\n\n"
                    f"Aşağıda **Əvvəl / Sonra** müqayisəsinə baxın.",
                    _progress_html("Tamamlandı 🎉", total or done, total or done, None, "0s"),
                    gpu_md, "\n".join(logs), video, result.output, cli_cmd,
                )

        def _stop() -> str:
            for flag in list(stop_flags.values()):
                flag.set()
            if not stop_flags:
                return "**Hazırdır.** (İşləyən emal yoxdur.)"
            return (
                "⏹ **Dayandırma sorğusu göndərildi** — cari addım bitdikdən "
                "sonra dayanacaq, checkpoint yadda saxlanılacaq."
            )

        run_btn.click(
            _run,
            inputs=[inp, fps, res, model, tile, interp, upsc, codec, crf,
                    resume, cleanup, out, ws],
            outputs=[status, progress, gpu, console, before, after, cli_box],
        )
        stop_btn.click(_stop, outputs=[status])

    return demo


def launch_ui(port: int = 7860, share: bool = False) -> None:
    """Launch the WebUI server (blocks until interrupted)."""
    demo = create_app()
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=share)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="100fps WebUI (Gradio).")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public gradio.live link.")
    args = parser.parse_args(argv)
    try:
        launch_ui(port=args.port, share=args.share)
    except RuntimeError as exc:
        print(f"app.py: error: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
