"""Unified TripoSplat web app.

Three tabs in one UI:
  1. Single image  -> 3D, with the same QA verdict the agent uses (preview + checks).
  2. Batch Agent    -> point at a folder (or upload many images); runs the full
                       auto-QA + self-healing pipeline with LIVE progress, a status
                       table, and success / failed galleries. (The flagship feature,
                       previously CLI-only.)
  3. Results Browser-> load any existing agent output folder and review the
                       success / failed galleries + per-image QA reasons + downloads.

The 3D pipeline is loaded LAZILY (only when a generation/batch run is requested),
so the Results Browser works even on a machine without a GPU / weights.

Run:
    ..\\venv\\Scripts\\python.exe app.py        (or run_app.bat from the project root)
Then open http://127.0.0.1:7860
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import threading
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr

# Reuse the agent's QA + plumbing so the UI and CLI stay in lock-step.
import agent_batch as AB

# ---------------------------------------------------------------------------
# Lazy pipeline (one-time, GPU). Kept out of import so the UI can load anywhere.
# ---------------------------------------------------------------------------

_PIPE = None
_PIPE_LOCK = threading.Lock()


def get_pipe():
    global _PIPE
    if _PIPE is None:
        with _PIPE_LOCK:
            if _PIPE is None:
                import torch
                from triposplat import TripoSplatPipeline
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    print("[warn] CUDA not available - running on CPU will be very slow.")
                _PIPE = TripoSplatPipeline(device=device, **AB.CKPTS)
    return _PIPE


# ---------------------------------------------------------------------------
# In-browser Spark.js viewer (shared by single + results tabs)
# ---------------------------------------------------------------------------

VIEWER_HTML  = Path("static/viewer/viewer.html").resolve()
EXAMPLES_DIR = Path("static/example_inputs").resolve()
SINGLE_OUT   = Path("gradio_outputs").resolve()
SINGLE_OUT.mkdir(parents=True, exist_ok=True)

EXAMPLES = [str(p) for p in (
    EXAMPLES_DIR / "creature_butterfly.webp",
    EXAMPLES_DIR / "building_stone_house.webp",
    EXAMPLES_DIR / "vehicle_pirate_ship.webp",
    EXAMPLES_DIR / "plant_water_lily.webp",
) if p.exists()]

PLACEHOLDER_HTML = (
    "<div style='display:flex;align-items:center;justify-content:center;height:520px;"
    "color:#94a3b8;font:16px system-ui;background:#111318;border-radius:12px'>"
    "3D viewer will appear here after generation</div>"
)


def _gr_file(path: Path) -> str:
    return f"/gradio_api/file={Path(path).resolve().as_posix()}"


def _viewer_iframe(ply_path: Path) -> str:
    ts = int(time.time() * 1000)  # cache-bust (Date.now()-equivalent; fine outside workflow scripts)
    src = f"{_gr_file(VIEWER_HTML)}?ply={_gr_file(ply_path)}&ts={ts}"
    return (f"<iframe src='{src}' style='width:100%;height:520px;border:0;"
            "border-radius:12px;background:#0a0b0e'></iframe>")


def _qa_badge(geo: dict, ren: dict, vision: dict | None) -> str:
    reasons = list(geo["reasons"]) + list(ren["reasons"])
    if vision and vision.get("ok") is False:
        reasons.append(f"vision:{vision.get('reason','')}")
    ok = not reasons
    if ok:
        return ("### ✅ QA passed\n"
                f"- geometry: `{geo['metrics']}`\n"
                f"- render coverage: `{ren['metrics'].get('render_cov_mean')}`"
                + (f"\n- vision: {vision.get('reason','ok')}" if vision and vision.get('ok') else ""))
    return ("### ❌ QA failed\n- " + "\n- ".join(reasons))


# ---------------------------------------------------------------------------
# Tab 1 — single image
# ---------------------------------------------------------------------------

def generate_single(image_path, seed, steps, guidance, num_gaussians, fmt, use_vision,
                    progress=gr.Progress(track_tqdm=True)):
    if not image_path:
        raise gr.Error("Please upload an image first.")
    progress(0, desc="Loading pipeline / generating...")
    pipe = get_pipe()

    t0 = time.time()
    g, prepared = pipe.run(str(image_path), seed=int(seed), steps=int(steps),
                           guidance_scale=float(guidance), shift=3.0,
                           num_gaussians=int(num_gaussians), show_progress=True)
    gen_dt = time.time() - t0

    # Same QA the agent runs, surfaced in the UI.
    geo = AB.geometric_check(g)
    view_imgs, covs = AB.render_views(g)
    ren = AB.render_check(covs)
    preview = AB.make_preview(prepared, view_imgs)

    vision = None
    if use_vision and AB.vision_available():
        vision = AB.vision_judge(prepared, preview, AB.VISION_MODEL_DEFAULT)

    out_dir = SINGLE_OUT / uuid4().hex[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "splat.ply"
    g.save_ply(str(ply_path))
    dl_path = ply_path
    if fmt.lower() == "splat":
        dl_path = out_dir / "splat.splat"
        g.save_splat(str(dl_path))

    info = (f"{g.get_xyz.shape[0]:,} gaussians · {gen_dt:.1f}s · saved {dl_path.name}")
    qa_md = _qa_badge(geo, ren, vision)
    return (prepared, preview, _viewer_iframe(ply_path),
            gr.update(value=str(dl_path), interactive=True), info, qa_md)


# ---------------------------------------------------------------------------
# Tab 2 — batch agent (live)
# ---------------------------------------------------------------------------

TABLE_HEADERS = ["#", "name", "status", "tries", "sec", "reasons"]


def _collect_images(folder: str, files) -> list[Path]:
    if folder and Path(folder).is_dir():
        return sorted(p for p in Path(folder).iterdir()
                      if p.suffix.lower() in AB.IMAGE_EXTS)
    if files:
        staging = SINGLE_OUT / "_uploaded" / uuid4().hex[:8]
        staging.mkdir(parents=True, exist_ok=True)
        out = []
        for f in files:
            src = Path(f.name if hasattr(f, "name") else f)
            if src.suffix.lower() in AB.IMAGE_EXTS:
                dst = staging / src.name
                shutil.copy(src, dst)
                out.append(dst)
        return sorted(out)
    return []


def run_batch(folder, files, out_folder, steps, guidance, num_gaussians,
              max_attempts, qa_vision, limit):
    """Generator: yields (status_md, table, success_gallery, failed_gallery)."""
    images = _collect_images(folder, files)
    if limit and limit > 0:
        images = images[: int(limit)]
    if not images:
        yield ("⚠️ No images found. Give a valid folder path or upload images.",
               [], [], [])
        return

    out_root = Path(out_folder or "agent_outputs").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    use_vision = (qa_vision == "on") or (qa_vision == "auto" and AB.vision_available())
    vis_note = (f"vision QA ON ({AB.VISION_MODEL_DEFAULT})" if use_vision
                else "vision QA off")

    # Resume support: skip names already in the manifest.
    manifest_path = out_root / "manifest.json"
    manifest = {"results": []}
    done = set()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            done = {r["name"] for r in manifest["results"]}
        except Exception:
            manifest = {"results": []}

    preset = dict(seed=42, steps=int(steps), guidance_scale=float(guidance),
                  shift=3.0, num_gaussians=int(num_gaussians))

    fh = open(out_root / "agent.log", "a", encoding="utf-8")
    AB._log(f"=== UI batch start: {len(images)} images, steps={steps} "
            f"guidance={guidance} ng={num_gaussians} max_attempts={max_attempts} "
            f"{vis_note} ===", fh)

    rows, success_gallery, failed_gallery = [], [], []
    n_ok = n_fail = n_skip = 0

    yield (f"⏳ Loading pipeline (one-time)…  ·  {len(images)} images queued  ·  {vis_note}",
           rows, success_gallery, failed_gallery)
    pipe = get_pipe()

    for i, img in enumerate(images, 1):
        if img.stem in done:
            n_skip += 1
            rows.append([i, img.stem, "⏭ skip", "-", "-", "resume"])
            yield (f"Processing {i}/{len(images)} · {n_ok}✅ {n_fail}❌ {n_skip}⏭",
                   rows, success_gallery, failed_gallery)
            continue

        rows.append([i, img.stem, "⚙️ running…", "-", "-", ""])
        yield (f"Processing {i}/{len(images)} · {img.name}", rows,
               success_gallery, failed_gallery)

        info = AB.process_image(pipe, img, out_root, preset, int(max_attempts),
                                use_vision, AB.VISION_MODEL_DEFAULT, fh)

        last = info["attempts"][-1] if info["attempts"] else {}
        reasons = ", ".join(last.get("reasons", [])) if info["status"] == "failed" else ""
        n_tries = len(info["attempts"])
        if info["status"] == "success":
            n_ok += 1
            rows[-1] = [i, info["name"], "✅ success", n_tries, info["total_sec"], ""]
            prev = out_root / "success" / info["name"] / "preview.webp"
            if prev.exists():
                success_gallery = success_gallery + [(str(prev), info["name"])]
        else:
            n_fail += 1
            rows[-1] = [i, info["name"], "❌ failed", n_tries, info["total_sec"], reasons]
            prev = out_root / "failed" / info["name"] / "last_preview.webp"
            if prev.exists():
                failed_gallery = failed_gallery + [(str(prev), f"{info['name']}: {reasons}")]

        # persist manifest (same shape as CLI)
        manifest["results"] = [r for r in manifest["results"] if r["name"] != info["name"]]
        manifest["results"].append({k: info[k] for k in ("name", "status", "source", "total_sec")}
                                   | {"n_attempts": n_tries})
        manifest["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=2))

        yield (f"Processing {i}/{len(images)} · {n_ok}✅ {n_fail}❌ {n_skip}⏭",
               rows, success_gallery, failed_gallery)

    AB._log(f"=== UI batch done: {n_ok} ok, {n_fail} failed, {n_skip} skipped ===", fh)
    fh.close()
    yield (f"### ✅ Done — {n_ok} success · {n_fail} failed · {n_skip} skipped\n"
           f"Output: `{out_root}`",
           rows, success_gallery, failed_gallery)


# ---------------------------------------------------------------------------
# Tab 3 — results browser (no GPU needed)
# ---------------------------------------------------------------------------

def load_results(out_folder):
    out_root = Path(out_folder or "agent_outputs").resolve()
    state = {"out_root": str(out_root), "success_ply": [], "failed": []}
    if not out_root.is_dir():
        return (f"⚠️ Folder not found: `{out_root}`", [], [], [], None, state)

    success_gallery, failed_gallery, rows, downloads = [], [], [], []

    succ_dir = out_root / "success"
    if succ_dir.is_dir():
        for d in sorted(succ_dir.iterdir()):
            if not d.is_dir():
                continue
            prev = d / "preview.webp"
            ply = d / "model.ply"
            if prev.exists():
                # keep success_gallery and state["success_ply"] index-aligned
                success_gallery.append((str(prev), d.name))
                state["success_ply"].append(str(ply) if ply.exists() else None)
            if ply.exists():
                downloads.append(str(ply))
            tries = sec = "-"
            info_f = d / "info.json"
            if info_f.exists():
                try:
                    info = json.loads(info_f.read_text())
                    tries = len(info.get("attempts", []))
                    sec = info.get("total_sec", "-")
                except Exception:
                    pass
            rows.append([d.name, "✅ success", tries, sec, ""])

    fail_dir = out_root / "failed"
    if fail_dir.is_dir():
        for d in sorted(fail_dir.iterdir()):
            if not d.is_dir():
                continue
            reasons = ""
            tries = sec = "-"
            source = None
            info_f = d / "info.json"
            if info_f.exists():
                try:
                    info = json.loads(info_f.read_text())
                    attempts = info.get("attempts", [])
                    tries = len(attempts)
                    sec = info.get("total_sec", "-")
                    source = info.get("source")
                    if attempts:
                        reasons = ", ".join(attempts[-1].get("reasons", []))
                except Exception:
                    pass
            if source and Path(source).exists():
                state["failed"].append({"name": d.name, "source": source})
            prev = d / "last_preview.webp"
            if prev.exists():
                failed_gallery.append((str(prev), f"{d.name}: {reasons}"))
            rows.append([d.name, "❌ failed", tries, sec, reasons])

    n_ok = sum(1 for r in rows if r[1].startswith("✅"))
    n_fail = sum(1 for r in rows if r[1].startswith("❌"))
    summary = (f"### {out_root.name}: {n_ok} success · {n_fail} failed"
               if rows else f"No results under `{out_root}` (expected success/ and failed/).")
    return summary, rows, success_gallery, failed_gallery, downloads or None, state


def view_selected(state, evt: gr.SelectData):
    """Show the picked success item's .ply in the in-browser viewer.
    The .ply is copied into a served path so it works for any source folder."""
    plys = (state or {}).get("success_ply", [])
    i = getattr(evt, "index", None)
    if i is None or i >= len(plys) or not plys[i]:
        return PLACEHOLDER_HTML
    served = SINGLE_OUT / "_view" / f"{uuid4().hex[:8]}.ply"
    served.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(plys[i], served)
    return _viewer_iframe(served)


def retry_failed(state, steps, guidance, num_gaussians, max_attempts, qa_vision):
    """Generator: re-run the agent ONLY on the failed cases in the loaded folder.
    On success, the stale failed/<name> entry is removed. Yields
    (summary, table, success_gallery, failed_gallery, downloads, state)."""
    targets = (state or {}).get("failed", [])
    out_root = Path((state or {}).get("out_root") or "agent_outputs").resolve()
    keep = (gr.update(),) * 5  # table, succ, fail, downloads, state — unchanged until reload
    if not targets:
        yield ("⚠️ No retryable failed cases. Click **Load** first; each failed case "
               "needs an `info.json` with a still-valid source path.", *keep)
        return

    use_vision = (qa_vision == "on") or (qa_vision == "auto" and AB.vision_available())
    preset = dict(seed=123, steps=int(steps), guidance_scale=float(guidance),
                  shift=3.0, num_gaussians=int(num_gaussians))
    fh = open(out_root / "agent.log", "a", encoding="utf-8")
    AB._log(f"=== UI retry-failed: {len(targets)} cases ===", fh)

    yield (f"⏳ Loading pipeline…  ·  retrying {len(targets)} failed case(s)", *keep)
    pipe = get_pipe()

    rows, n_ok, n_still = [], 0, 0
    upd3 = (gr.update(), gr.update(), gr.update(), gr.update())  # succ, fail, downloads, state
    for i, t in enumerate(targets, 1):
        name, src = t["name"], Path(t["source"])
        rows.append([name, "⚙️ retrying…", "-", "-", ""])
        yield (f"Retrying {i}/{len(targets)} · {name}", rows, *upd3)
        info = AB.process_image(pipe, src, out_root, preset, int(max_attempts),
                                use_vision, AB.VISION_MODEL_DEFAULT, fh)
        if info["status"] == "success":
            n_ok += 1
            # clear the now-stale failed/ entry
            stale = out_root / "failed" / name
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            rows[-1] = [name, "✅ recovered", len(info["attempts"]),
                        info["total_sec"], ""]
        else:
            n_still += 1
            last = info["attempts"][-1] if info["attempts"] else {}
            rows[-1] = [name, "❌ still failed", len(info["attempts"]),
                        info["total_sec"], ", ".join(last.get("reasons", []))]
        yield (f"Retrying {i}/{len(targets)} · {n_ok} recovered · {n_still} still failed",
               rows, *upd3)

    AB._log(f"=== retry done: {n_ok} recovered, {n_still} still failed ===", fh)
    fh.close()
    # full refresh so galleries/state/downloads reflect the new state
    summary, full_rows, succ, fail, downloads, new_state = load_results(str(out_root))
    yield (f"### ♻️ Retry done — {n_ok} recovered · {n_still} still failed\n{summary}",
           full_rows, succ, fail, downloads, new_state)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="TripoSplat Studio") as demo:
        gr.Markdown(
            "# 🧊 TripoSplat Studio\n"
            "Single-image **and** batch image→3D with automatic QA + self-healing retries. "
            "Engine: [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) · "
            "Agent/QA: AI FUTURE STREAMER (Park Seong-Woo)."
        )

        # ---- Tab 1: Single ----
        with gr.Tab("Single image"):
            with gr.Row():
                with gr.Column(scale=1):
                    s_img = gr.Image(label="Input image", type="filepath",
                                     image_mode="RGBA", height=320)
                    if EXAMPLES:
                        gr.Examples([[p] for p in EXAMPLES], inputs=[s_img],
                                    label="Examples (click to load)")
                    with gr.Accordion("Sampling settings", open=False):
                        s_seed = gr.Number(label="Seed", value=42, precision=0)
                        s_steps = gr.Slider(1, 50, value=20, step=1, label="Inference steps")
                        s_cfg = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Guidance scale")
                        s_ng = gr.Dropdown(["32768", "65536", "131072", "262144"],
                                           value="262144", label="Number of gaussians")
                        s_fmt = gr.Dropdown(["ply", "splat"], value="ply", label="Download format")
                        s_vision = gr.Checkbox(value=False,
                            label="Run Claude vision QA (needs ANTHROPIC_API_KEY)")
                    s_run = gr.Button("Generate", variant="primary")
                    s_prepared = gr.Image(label="Preprocessed input", interactive=False, height=200)
                    s_info = gr.Markdown()
                    s_qa = gr.Markdown()
                with gr.Column(scale=2):
                    s_viewer = gr.HTML(value=PLACEHOLDER_HTML)
                    s_preview = gr.Image(label="QA preview (reference + 4 views)",
                                         interactive=False)
                    s_dl = gr.DownloadButton(label="Download", value=None, interactive=False)
            s_run.click(generate_single,
                        inputs=[s_img, s_seed, s_steps, s_cfg, s_ng, s_fmt, s_vision],
                        outputs=[s_prepared, s_preview, s_viewer, s_dl, s_info, s_qa])

        # ---- Tab 2: Batch agent ----
        with gr.Tab("Batch Agent"):
            gr.Markdown(
                "Process a whole **folder** (or upload many images). Each image is "
                "generated at high quality, auto-QA'd (geometry + 4-view render "
                "+ optional Claude vision), and **regenerated up to N times** if broken. "
                "Resume is automatic — re-running skips finished images."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    b_folder = gr.Textbox(label="Input folder (path on this machine)",
                                          placeholder=r"C:\my_images")
                    b_files = gr.File(label="…or upload images", file_count="multiple",
                                      file_types=["image"])
                    b_out = gr.Textbox(label="Output folder", value="agent_outputs")
                    with gr.Accordion("Agent settings", open=False):
                        b_steps = gr.Slider(1, 60, value=30, step=1, label="Steps")
                        b_cfg = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="Guidance scale")
                        b_ng = gr.Dropdown(["32768", "65536", "131072", "262144"],
                                           value="262144", label="Number of gaussians")
                        b_max = gr.Slider(1, 6, value=4, step=1, label="Max attempts / image")
                        b_vision = gr.Dropdown(["auto", "on", "off"], value="auto",
                                               label="Vision QA")
                        b_limit = gr.Number(label="Limit (0 = all)", value=0, precision=0)
                    b_run = gr.Button("Run batch agent", variant="primary")
                    b_status = gr.Markdown("Idle.")
                with gr.Column(scale=2):
                    b_table = gr.Dataframe(headers=TABLE_HEADERS, label="Progress",
                                           interactive=False, wrap=True)
                    with gr.Row():
                        b_succ = gr.Gallery(label="✅ Success", columns=3, height=320)
                        b_fail = gr.Gallery(label="❌ Failed", columns=3, height=320)
            b_run.click(run_batch,
                        inputs=[b_folder, b_files, b_out, b_steps, b_cfg, b_ng,
                                b_max, b_vision, b_limit],
                        outputs=[b_status, b_table, b_succ, b_fail])

        # ---- Tab 3: Results browser ----
        with gr.Tab("Results Browser"):
            gr.Markdown("Load an existing agent output folder to review results "
                        "(loading/viewing works without a GPU; retry needs a GPU).")
            r_state = gr.State({})
            with gr.Row():
                r_folder = gr.Textbox(label="Output folder", value="agent_outputs", scale=4)
                r_load = gr.Button("Load", variant="primary", scale=1)
            r_summary = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    r_table = gr.Dataframe(
                        headers=["name", "status", "tries", "sec", "reasons"],
                        label="All results", interactive=False, wrap=True)
                    with gr.Row():
                        r_succ = gr.Gallery(label="✅ Success (click to view 3D)",
                                            columns=3, height=300)
                        r_fail = gr.Gallery(label="❌ Failed", columns=3, height=300)
                    r_downloads = gr.Files(label="Download .ply files")
                with gr.Column(scale=1):
                    r_viewer = gr.HTML(value=PLACEHOLDER_HTML)
                    with gr.Accordion("Retry settings", open=False):
                        r_steps = gr.Slider(1, 60, value=40, step=1, label="Steps")
                        r_cfg = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="Guidance scale")
                        r_ng = gr.Dropdown(["32768", "65536", "131072", "262144"],
                                           value="262144", label="Number of gaussians")
                        r_max = gr.Slider(1, 6, value=4, step=1, label="Max attempts / image")
                        r_vision = gr.Dropdown(["auto", "on", "off"], value="auto",
                                               label="Vision QA")
                    r_retry = gr.Button("♻️ Retry failed cases only", variant="secondary")

            r_load.click(load_results, inputs=[r_folder],
                         outputs=[r_summary, r_table, r_succ, r_fail, r_downloads, r_state])
            r_succ.select(view_selected, inputs=[r_state], outputs=[r_viewer])
            r_retry.click(retry_failed,
                          inputs=[r_state, r_steps, r_cfg, r_ng, r_max, r_vision],
                          outputs=[r_summary, r_table, r_succ, r_fail, r_downloads, r_state])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[
            str(VIEWER_HTML.parent),
            str(SINGLE_OUT),
            str(Path("agent_outputs").resolve()),
        ],
    )
