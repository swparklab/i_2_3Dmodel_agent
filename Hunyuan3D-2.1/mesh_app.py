"""Unified Hunyuan3D Studio web app (image -> 3D MESH .glb).

Three tabs:
  1. Single image  -> 3D mesh, with QA verdict + in-browser GLTF viewer + download.
  2. Batch Agent    -> folder / multi-upload, auto-QA + self-healing retries,
                       live progress table, success / failed galleries, auto-resume.
  3. Results Browser-> load any output folder, review results, click to view 3D,
                       retry failed cases only. (loading/viewing needs no GPU)

Shape only (no texture) — the texture pipeline needs a compiled CUDA rasterizer
(CUDA Toolkit + MSVC); see README. The shape engine is loaded lazily so the
Results Browser works without a GPU.

Run:  ..\\venv_hy\\Scripts\\python.exe mesh_app.py   (or run_mesh_app.bat)
Open: http://127.0.0.1:7861
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

import mesh_agent as MA

# ---------------------------------------------------------------------------
# Lazy engine (GPU). Kept out of import so the Results Browser works anywhere.
# ---------------------------------------------------------------------------

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                _ENGINE = MA.ShapeEngine(device=dev)
    return _ENGINE


VIEWER_HTML  = Path("static/viewer/mesh_viewer.html").resolve()
EXAMPLES_DIR = Path("assets").resolve()
SINGLE_OUT   = Path("gradio_mesh_outputs").resolve()
SINGLE_OUT.mkdir(parents=True, exist_ok=True)
EXAMPLES = [str(p) for p in sorted(EXAMPLES_DIR.glob("*.png"))][:6]

PLACEHOLDER_HTML = (
    "<div style='display:flex;align-items:center;justify-content:center;height:520px;"
    "color:#94a3b8;font:16px system-ui;background:#111318;border-radius:12px'>"
    "3D mesh viewer will appear here after generation</div>"
)


def _gr_file(path: Path) -> str:
    return f"/gradio_api/file={Path(path).resolve().as_posix()}"


def _viewer_iframe(glb_path: Path) -> str:
    ts = int(time.time() * 1000)
    src = f"{_gr_file(VIEWER_HTML)}?glb={_gr_file(glb_path)}&ts={ts}"
    return (f"<iframe src='{src}' style='width:100%;height:520px;border:0;"
            "border-radius:12px;background:#0a0b0e'></iframe>")


def _qa_badge(geo: dict, ren: dict, vision: dict | None) -> str:
    reasons = list(geo["reasons"]) + list(ren["reasons"])
    if vision and vision.get("ok") is False:
        reasons.append(f"vision:{vision.get('reason','')}")
    if not reasons:
        return ("### ✅ QA passed\n"
                f"- geometry: `{geo['metrics']}`\n"
                f"- render coverage: `{ren['metrics'].get('render_cov_mean')}`"
                + (f"\n- vision: {vision.get('reason','ok')}" if vision and vision.get('ok') else ""))
    return "### ❌ QA failed\n- " + "\n- ".join(reasons)


# ---------------------------------------------------------------------------
# Tab 1 — single image
# ---------------------------------------------------------------------------

def generate_single(image_path, seed, steps, guidance, octree, use_vision,
                    progress=gr.Progress(track_tqdm=True)):
    if not image_path:
        raise gr.Error("Please upload an image first.")
    progress(0, desc="Loading engine / generating…")
    eng = get_engine()
    prepared = eng.preprocess(str(image_path))
    t0 = time.time()
    mesh = eng.generate(prepared, seed=int(seed), steps=int(steps),
                        guidance=float(guidance), octree_resolution=int(octree))
    gen_dt = time.time() - t0

    geo = MA.geometric_check(mesh)
    view_imgs, covs = MA.render_views(mesh)
    ren = MA.render_check(covs)
    preview = MA.make_preview(prepared, view_imgs)
    vision = None
    if use_vision and MA.vision_available():
        vision = MA.vision_judge(prepared, preview, MA.VISION_MODEL_DEFAULT)

    out_dir = SINGLE_OUT / uuid4().hex[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "model.glb"
    mesh.export(str(glb_path))

    info = (f"{geo['metrics'].get('faces','?'):,} faces · "
            f"{geo['metrics'].get('verts','?'):,} verts · {gen_dt:.1f}s")
    return (prepared.convert("RGB"), preview, _viewer_iframe(glb_path),
            gr.update(value=str(glb_path), interactive=True), info, _qa_badge(geo, ren, vision))


# ---------------------------------------------------------------------------
# Tab 2 — batch agent (live)
# ---------------------------------------------------------------------------

TABLE_HEADERS = ["#", "name", "status", "tries", "sec", "reasons"]


def _collect_images(folder, files):
    if folder and Path(folder).is_dir():
        return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in MA.IMAGE_EXTS)
    if files:
        staging = SINGLE_OUT / "_uploaded" / uuid4().hex[:8]
        staging.mkdir(parents=True, exist_ok=True)
        out = []
        for f in files:
            src = Path(f.name if hasattr(f, "name") else f)
            if src.suffix.lower() in MA.IMAGE_EXTS:
                dst = staging / src.name
                shutil.copy(src, dst); out.append(dst)
        return sorted(out)
    return []


def run_batch(folder, files, out_folder, steps, guidance, octree, max_attempts, qa_vision, limit):
    images = _collect_images(folder, files)
    if limit and limit > 0:
        images = images[: int(limit)]
    if not images:
        yield ("⚠️ No images found. Give a valid folder path or upload images.", [], [], [])
        return

    out_root = Path(out_folder or "mesh_outputs").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    use_vision = (qa_vision == "on") or (qa_vision == "auto" and MA.vision_available())
    vis_note = f"vision QA {'ON' if use_vision else 'off'}"

    manifest_path = out_root / "manifest.json"
    manifest = {"results": []}; done = set()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            done = {r["name"] for r in manifest["results"]}
        except Exception:
            manifest = {"results": []}

    preset = dict(seed=42, steps=int(steps), guidance=float(guidance),
                  octree_resolution=int(octree))
    fh = open(out_root / "agent.log", "a", encoding="utf-8")
    MA._log(f"=== UI mesh batch: {len(images)} imgs, steps={steps} guidance={guidance} "
            f"octree={octree} max_attempts={max_attempts} {vis_note} ===", fh)

    rows, succ_gallery, fail_gallery = [], [], []
    n_ok = n_fail = n_skip = 0
    yield (f"⏳ Loading engine (one-time)…  ·  {len(images)} images queued  ·  {vis_note}",
           rows, succ_gallery, fail_gallery)
    eng = get_engine()

    for i, img in enumerate(images, 1):
        if img.stem in done:
            n_skip += 1
            rows.append([i, img.stem, "⏭ skip", "-", "-", "resume"])
            yield (f"{i}/{len(images)} · {n_ok}✅ {n_fail}❌ {n_skip}⏭", rows, succ_gallery, fail_gallery)
            continue
        rows.append([i, img.stem, "⚙️ running…", "-", "-", ""])
        yield (f"{i}/{len(images)} · {img.name}", rows, succ_gallery, fail_gallery)

        info = MA.process_image(eng, img, out_root, preset, int(max_attempts),
                                use_vision, MA.VISION_MODEL_DEFAULT, fh)
        last = info["attempts"][-1] if info["attempts"] else {}
        n_tries = len(info["attempts"])
        if info["status"] == "success":
            n_ok += 1
            rows[-1] = [i, info["name"], "✅ success", n_tries, info["total_sec"], ""]
            prev = out_root / "success" / info["name"] / "preview.webp"
            if prev.exists():
                succ_gallery = succ_gallery + [(str(prev), info["name"])]
        else:
            n_fail += 1
            reasons = ", ".join(last.get("reasons", []))
            rows[-1] = [i, info["name"], "❌ failed", n_tries, info["total_sec"], reasons]
            prev = out_root / "failed" / info["name"] / "last_preview.webp"
            if prev.exists():
                fail_gallery = fail_gallery + [(str(prev), f"{info['name']}: {reasons}")]

        manifest["results"] = [r for r in manifest["results"] if r["name"] != info["name"]]
        manifest["results"].append({k: info[k] for k in ("name", "status", "source", "total_sec")}
                                   | {"n_attempts": n_tries})
        manifest["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        yield (f"{i}/{len(images)} · {n_ok}✅ {n_fail}❌ {n_skip}⏭", rows, succ_gallery, fail_gallery)

    MA._log(f"=== UI mesh batch done: {n_ok} ok, {n_fail} failed, {n_skip} skipped ===", fh)
    fh.close()
    yield (f"### ✅ Done — {n_ok} success · {n_fail} failed · {n_skip} skipped\nOutput: `{out_root}`",
           rows, succ_gallery, fail_gallery)


# ---------------------------------------------------------------------------
# Tab 3 — results browser (no GPU needed)
# ---------------------------------------------------------------------------

def load_results(out_folder):
    out_root = Path(out_folder or "mesh_outputs").resolve()
    state = {"out_root": str(out_root), "success_glb": [], "failed": []}
    if not out_root.is_dir():
        return (f"⚠️ Folder not found: `{out_root}`", [], [], [], None, state)
    succ_gallery, fail_gallery, rows, downloads = [], [], [], []

    sdir = out_root / "success"
    if sdir.is_dir():
        for d in sorted(sdir.iterdir()):
            if not d.is_dir():
                continue
            prev = d / "preview.webp"; glb = d / "model.glb"
            if prev.exists():
                succ_gallery.append((str(prev), d.name))
                state["success_glb"].append(str(glb) if glb.exists() else None)
            if glb.exists():
                downloads.append(str(glb))
            tries = sec = "-"
            inf = d / "info.json"
            if inf.exists():
                try:
                    j = json.loads(inf.read_text())
                    tries = len(j.get("attempts", [])); sec = j.get("total_sec", "-")
                except Exception:
                    pass
            rows.append([d.name, "✅ success", tries, sec, ""])

    fdir = out_root / "failed"
    if fdir.is_dir():
        for d in sorted(fdir.iterdir()):
            if not d.is_dir():
                continue
            reasons = ""; tries = sec = "-"; source = None
            inf = d / "info.json"
            if inf.exists():
                try:
                    j = json.loads(inf.read_text())
                    a = j.get("attempts", []); tries = len(a); sec = j.get("total_sec", "-")
                    source = j.get("source")
                    if a:
                        reasons = ", ".join(a[-1].get("reasons", []))
                except Exception:
                    pass
            if source and Path(source).exists():
                state["failed"].append({"name": d.name, "source": source})
            prev = d / "last_preview.webp"
            if prev.exists():
                fail_gallery.append((str(prev), f"{d.name}: {reasons}"))
            rows.append([d.name, "❌ failed", tries, sec, reasons])

    n_ok = sum(1 for r in rows if r[1].startswith("✅"))
    n_fail = sum(1 for r in rows if r[1].startswith("❌"))
    summary = (f"### {out_root.name}: {n_ok} success · {n_fail} failed"
               if rows else f"No results under `{out_root}` (expected success/ and failed/).")
    return summary, rows, succ_gallery, fail_gallery, downloads or None, state


def view_selected(state, evt: gr.SelectData):
    glbs = (state or {}).get("success_glb", [])
    i = getattr(evt, "index", None)
    if i is None or i >= len(glbs) or not glbs[i]:
        return PLACEHOLDER_HTML
    served = SINGLE_OUT / "_view" / f"{uuid4().hex[:8]}.glb"
    served.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(glbs[i], served)
    return _viewer_iframe(served)


def retry_failed(state, steps, guidance, octree, max_attempts, qa_vision):
    targets = (state or {}).get("failed", [])
    out_root = Path((state or {}).get("out_root") or "mesh_outputs").resolve()
    keep = (gr.update(),) * 5
    if not targets:
        yield ("⚠️ No retryable failed cases. Click **Load** first.", *keep)
        return
    use_vision = (qa_vision == "on") or (qa_vision == "auto" and MA.vision_available())
    preset = dict(seed=123, steps=int(steps), guidance=float(guidance),
                  octree_resolution=int(octree))
    fh = open(out_root / "agent.log", "a", encoding="utf-8")
    MA._log(f"=== UI mesh retry-failed: {len(targets)} cases ===", fh)
    yield (f"⏳ Loading engine…  ·  retrying {len(targets)} case(s)", *keep)
    eng = get_engine()

    rows, n_ok, n_still = [], 0, 0
    upd = (gr.update(), gr.update(), gr.update(), gr.update())
    for i, t in enumerate(targets, 1):
        name, src = t["name"], Path(t["source"])
        rows.append([name, "⚙️ retrying…", "-", "-", ""])
        yield (f"Retrying {i}/{len(targets)} · {name}", rows, *upd)
        info = MA.process_image(eng, src, out_root, preset, int(max_attempts),
                                use_vision, MA.VISION_MODEL_DEFAULT, fh)
        if info["status"] == "success":
            n_ok += 1
            stale = out_root / "failed" / name
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            rows[-1] = [name, "✅ recovered", len(info["attempts"]), info["total_sec"], ""]
        else:
            n_still += 1
            last = info["attempts"][-1] if info["attempts"] else {}
            rows[-1] = [name, "❌ still failed", len(info["attempts"]), info["total_sec"],
                        ", ".join(last.get("reasons", []))]
        yield (f"{n_ok} recovered · {n_still} still failed", rows, *upd)

    MA._log(f"=== mesh retry done: {n_ok} recovered, {n_still} still failed ===", fh)
    fh.close()
    summary, full_rows, succ, fail, downloads, new_state = load_results(str(out_root))
    yield (f"### ♻️ Retry done — {n_ok} recovered · {n_still} still failed\n{summary}",
           full_rows, succ, fail, downloads, new_state)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Hunyuan3D Studio") as demo:
        gr.Markdown(
            "# 🧱 Hunyuan3D Studio (mesh)\n"
            "Single-image **and** batch image→3D **mesh (.glb)** with automatic QA + "
            "self-healing retries. Engine: "
            "[Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) shape · "
            "Agent/QA: AI FUTURE STREAMER (Park Seong-Woo). *(shape only — texture needs CUDA build)*"
        )

        with gr.Tab("Single image"):
            with gr.Row():
                with gr.Column(scale=1):
                    s_img = gr.Image(label="Input image", type="filepath", height=320)
                    if EXAMPLES:
                        gr.Examples([[p] for p in EXAMPLES], inputs=[s_img], label="Examples")
                    with gr.Accordion("Settings", open=False):
                        s_seed = gr.Number(label="Seed", value=42, precision=0)
                        s_steps = gr.Slider(5, 50, value=30, step=1, label="Diffusion steps")
                        s_cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance scale")
                        s_oct = gr.Dropdown(["256", "320", "384"], value="384",
                                            label="Octree resolution (detail)")
                        s_vision = gr.Checkbox(value=False,
                            label="Run Claude vision QA (needs ANTHROPIC_API_KEY)")
                    s_run = gr.Button("Generate mesh", variant="primary")
                    s_prepared = gr.Image(label="Preprocessed input", interactive=False, height=200)
                    s_info = gr.Markdown(); s_qa = gr.Markdown()
                with gr.Column(scale=2):
                    s_viewer = gr.HTML(value=PLACEHOLDER_HTML)
                    s_preview = gr.Image(label="QA preview (reference + 4 views)", interactive=False)
                    s_dl = gr.DownloadButton(label="Download .glb", value=None, interactive=False)
            s_run.click(generate_single,
                        inputs=[s_img, s_seed, s_steps, s_cfg, s_oct, s_vision],
                        outputs=[s_prepared, s_preview, s_viewer, s_dl, s_info, s_qa])

        with gr.Tab("Batch Agent"):
            gr.Markdown("Process a folder (or upload many). Each image → mesh, auto-QA "
                        "(geometry + 4-view render + optional vision), regenerated up to N "
                        "times if broken. Resume is automatic.")
            with gr.Row():
                with gr.Column(scale=1):
                    b_folder = gr.Textbox(label="Input folder (path on this machine)",
                                          placeholder=r"C:\my_images")
                    b_files = gr.File(label="…or upload images", file_count="multiple",
                                      file_types=["image"])
                    b_out = gr.Textbox(label="Output folder", value="mesh_outputs")
                    with gr.Accordion("Agent settings", open=False):
                        b_steps = gr.Slider(5, 60, value=30, step=1, label="Steps")
                        b_cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance")
                        b_oct = gr.Dropdown(["256", "320", "384"], value="384", label="Octree resolution")
                        b_max = gr.Slider(1, 6, value=3, step=1, label="Max attempts / image")
                        b_vision = gr.Dropdown(["auto", "on", "off"], value="auto", label="Vision QA")
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
                        inputs=[b_folder, b_files, b_out, b_steps, b_cfg, b_oct, b_max, b_vision, b_limit],
                        outputs=[b_status, b_table, b_succ, b_fail])

        with gr.Tab("Results Browser"):
            gr.Markdown("Load a mesh output folder to review (loading/viewing needs no GPU; retry needs GPU).")
            r_state = gr.State({})
            with gr.Row():
                r_folder = gr.Textbox(label="Output folder", value="mesh_outputs", scale=4)
                r_load = gr.Button("Load", variant="primary", scale=1)
            r_summary = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    r_table = gr.Dataframe(headers=["name", "status", "tries", "sec", "reasons"],
                                           label="All results", interactive=False, wrap=True)
                    with gr.Row():
                        r_succ = gr.Gallery(label="✅ Success (click to view 3D)", columns=3, height=300)
                        r_fail = gr.Gallery(label="❌ Failed", columns=3, height=300)
                    r_downloads = gr.Files(label="Download .glb files")
                with gr.Column(scale=1):
                    r_viewer = gr.HTML(value=PLACEHOLDER_HTML)
                    with gr.Accordion("Retry settings", open=False):
                        r_steps = gr.Slider(5, 60, value=40, step=1, label="Steps")
                        r_cfg = gr.Slider(1.0, 10.0, value=6.0, step=0.5, label="Guidance")
                        r_oct = gr.Dropdown(["256", "320", "384"], value="384", label="Octree resolution")
                        r_max = gr.Slider(1, 6, value=3, step=1, label="Max attempts / image")
                        r_vision = gr.Dropdown(["auto", "on", "off"], value="auto", label="Vision QA")
                    r_retry = gr.Button("♻️ Retry failed cases only", variant="secondary")
            r_load.click(load_results, inputs=[r_folder],
                         outputs=[r_summary, r_table, r_succ, r_fail, r_downloads, r_state])
            r_succ.select(view_selected, inputs=[r_state], outputs=[r_viewer])
            r_retry.click(retry_failed, inputs=[r_state, r_steps, r_cfg, r_oct, r_max, r_vision],
                          outputs=[r_summary, r_table, r_succ, r_fail, r_downloads, r_state])
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(
        server_name="0.0.0.0", server_port=7861,
        allowed_paths=[str(VIEWER_HTML.parent), str(SINGLE_OUT),
                       str(Path("mesh_outputs").resolve())],
    )
