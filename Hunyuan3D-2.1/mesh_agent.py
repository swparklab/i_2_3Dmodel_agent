"""Hunyuan3D-2.1 batch agent — image folder -> high-quality 3D MESHES (.glb)
with automatic "is the mesh broken?" QA and self-healing regeneration.

This mirrors the TripoSplat agent (TripoSplat/agent_batch.py) but for the
Hunyuan3D *shape* pipeline, whose output is a textured-ready triangle MESH
instead of a Gaussian splat. The QA layers are therefore mesh-oriented:

  1. geometric checks  (empty / degenerate / non-finite / collapsed-flat /
                        runaway / over-fragmented)                      [always]
  2. self-rendered 4-view coverage heuristic (surface-sampled z-buffer)  [always]
  3. Claude vision judge: reference image vs renders                     [if API key]

Texture/PBR is a separate Hunyuan pipeline that needs a compiled CUDA
rasterizer (CUDA Toolkit + MSVC); this agent does SHAPE only, which runs on
the stock wheels and fits comfortably on a 24 GB GPU.

Outputs:
  <output>/success/<name>/  model.glb, preprocessed.webp, preview.webp, info.json
  <output>/failed/<name>/   last_preview.webp, info.json
  <output>/manifest.json    full run record (also drives --resume)
  <output>/agent.log

Usage:
  python mesh_agent.py --input <img_dir> --output <out_dir> [options]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# Hunyuan's packages live under these subfolders.
sys.path.insert(0, str(Path(__file__).parent / "hy3dshape"))

import numpy as np
import torch
import trimesh
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults / QA thresholds
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

MODEL_PATH = "tencent/Hunyuan3D-2.1"

QA = dict(
    min_faces        = 200,     # fewer than this -> empty / degenerate
    min_verts        = 100,
    min_extent       = 0.02,    # largest bbox axis below this -> collapsed
    max_extent       = 100.0,   # sanity ceiling on bbox size
    min_axis_ratio   = 0.02,    # smallest/largest bbox axis; ~0 -> a flat sheet (not 3D)
    max_components    = 64,      # wildly fragmented -> broken
    min_render_cov   = 0.004,   # mean foreground coverage over 4 renders; ~0 -> empty
    max_render_cov   = 0.985,   # full-frame fill -> blob
)

VISION_MODEL_DEFAULT = "claude-sonnet-4-6"


def _log(msg: str, fh=None):
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n"); fh.flush()


# ---------------------------------------------------------------------------
# Engine — Hunyuan3D shape pipeline (lazy, loaded once)
# ---------------------------------------------------------------------------

class ShapeEngine:
    def __init__(self, device: str = "cuda"):
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.rembg import BackgroundRemover
        self.device = device
        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_PATH)
        self.rembg = BackgroundRemover()

    def preprocess(self, image_path: str) -> Image.Image:
        img = Image.open(image_path)
        if img.mode != "RGBA":
            img = self.rembg(img.convert("RGB"))
        return img

    def generate(self, prepared: Image.Image, *, seed: int, steps: int,
                 guidance: float, octree_resolution: int) -> trimesh.Trimesh:
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(image=prepared, num_inference_steps=int(steps),
                        guidance_scale=float(guidance),
                        octree_resolution=int(octree_resolution),
                        generator=gen)
        mesh = out[0]
        # Pipeline returns a trimesh.Trimesh (or list); normalise to Trimesh.
        if isinstance(mesh, (list, tuple)):
            mesh = mesh[0]
        return mesh


# ---------------------------------------------------------------------------
# QA — geometric checks (authoritative, no rendering)
# ---------------------------------------------------------------------------

def geometric_check(mesh: trimesh.Trimesh) -> dict:
    reasons, metrics = [], {}

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces)
    nv, nf = len(v), len(f)
    metrics["verts"], metrics["faces"] = int(nv), int(nf)

    if nv < QA["min_verts"] or nf < QA["min_faces"]:
        reasons.append(f"empty(verts={nv},faces={nf})")
        return {"ok": False, "reasons": reasons, "metrics": metrics}

    if not np.isfinite(v).all():
        reasons.append("non_finite:vertices")

    extent = v.max(0) - v.min(0)
    max_ext = float(extent.max())
    min_ext = float(extent.min())
    ratio = (min_ext / max_ext) if max_ext > 0 else 0.0
    metrics["extent"] = [round(float(e), 4) for e in extent]
    metrics["axis_ratio"] = round(ratio, 4)
    if max_ext < QA["min_extent"]:
        reasons.append(f"collapsed(extent={max_ext:.4f})")
    if max_ext > QA["max_extent"]:
        reasons.append(f"runaway(extent={max_ext:.4f})")
    if ratio < QA["min_axis_ratio"]:
        reasons.append(f"flat(axis_ratio={ratio:.4f})")

    try:
        comps = mesh.body_count
    except Exception:
        comps = 1
    metrics["components"] = int(comps)
    if comps > QA["max_components"]:
        reasons.append(f"fragmented(components={comps})")

    metrics["watertight"] = bool(mesh.is_watertight)  # informational, not a hard fail
    return {"ok": len(reasons) == 0, "reasons": reasons, "metrics": metrics}


# ---------------------------------------------------------------------------
# QA — self renderer (surface-sampled point z-buffer, pure torch, headless)
# ---------------------------------------------------------------------------

def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


@torch.no_grad()
def render_views(mesh: trimesh.Trimesh, res=320, n_samples=200000,
                 azimuths=(0, 90, 180, 270), elevation=18.0, bg=0.12):
    """Sample points on the mesh surface and z-buffer render a few views with
    simple normal shading. Headless (no OpenGL). Good enough to judge
    'coherent object vs garbage' and to give a human-eyeballable preview."""
    try:
        pts, fidx = trimesh.sample.sample_surface(mesh, n_samples)
        nrm = mesh.face_normals[fidx]
    except Exception:
        pts = np.asarray(mesh.vertices)
        nrm = np.zeros_like(pts); nrm[:, 2] = 1.0

    xyz = torch.tensor(np.asarray(pts), dtype=torch.float32)
    nrm = torch.tensor(np.asarray(nrm), dtype=torch.float32)
    if xyz.shape[0] == 0:
        empty = np.full((res, res, 3), int(bg * 255), np.uint8)
        return [empty for _ in azimuths], [0.0 for _ in azimuths]

    # normalise to ~[-1, 1]
    lo = torch.quantile(xyz, 0.01, dim=0)
    hi = torch.quantile(xyz, 0.99, dim=0)
    center = (lo + hi) / 2
    scale = float((hi - lo).max()) / 2 + 1e-6
    xyz = (xyz - center) / scale

    el = math.radians(elevation)
    Rx = _rot_x(el)
    light = torch.tensor([0.4, 0.6, 0.7]); light = light / light.norm()
    imgs, covs = [], []
    for az in azimuths:
        Ry = _rot_y(math.radians(az))
        R = Ry @ Rx
        p = xyz @ R.t()
        n = nrm @ R.t()
        margin = 1.3
        sx = ((p[:, 0] / margin) * 0.5 + 0.5) * (res - 1)
        sy = ((-p[:, 1] / margin) * 0.5 + 0.5) * (res - 1)
        px = sx.round().long().clamp(0, res - 1)
        py = sy.round().long().clamp(0, res - 1)
        depth = -p[:, 2]
        idx = py * res + px
        zbuf = torch.full((res * res,), float("inf"))
        zbuf.scatter_reduce_(0, idx, depth, reduce="amin", include_self=True)
        winners = depth <= zbuf[idx] + 1e-6
        shade = (0.25 + 0.75 * (n[:, 2].abs().clamp(0, 1)))  # face toward camera = brighter
        canvas = torch.full((res * res, 3), bg)
        alpha = torch.zeros(res * res)
        w = winners
        col = shade[w].unsqueeze(1).repeat(1, 3) * torch.tensor([0.80, 0.84, 0.92])
        canvas[idx[w]] = col
        alpha[idx[w]] = 1.0
        covs.append(float(alpha.mean()))
        rgb = (canvas.reshape(res, res, 3).clamp(0, 1) * 255).to(torch.uint8).numpy()
        imgs.append(rgb)
    return imgs, covs


def make_preview(ref: Image.Image, view_imgs, res=320) -> Image.Image:
    tiles = [ref.convert("RGB").resize((res, res), Image.LANCZOS)]
    tiles += [Image.fromarray(v) for v in view_imgs]
    strip = Image.new("RGB", (res * len(tiles), res), (24, 24, 28))
    for i, t in enumerate(tiles):
        strip.paste(t, (i * res, 0))
    return strip


def render_check(covs) -> dict:
    mean_cov = float(np.mean(covs)) if covs else 0.0
    reasons = []
    if mean_cov < QA["min_render_cov"]:
        reasons.append(f"render_empty(cov={mean_cov:.4f})")
    if mean_cov > QA["max_render_cov"]:
        reasons.append(f"render_full(cov={mean_cov:.4f})")
    return {"ok": len(reasons) == 0, "reasons": reasons,
            "metrics": {"render_cov_mean": round(mean_cov, 4),
                        "render_cov": [round(c, 4) for c in covs]}}


# ---------------------------------------------------------------------------
# QA — optional Claude vision judge
# ---------------------------------------------------------------------------

def _img_to_b64(img: Image.Image) -> str:
    import base64, io
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def vision_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def vision_judge(ref: Image.Image, preview: Image.Image, model: str):
    try:
        import anthropic
        client = anthropic.Anthropic()
        sys_prompt = (
            "You are a strict 3D reconstruction QA inspector. You see a reference "
            "object image, then 4 rendered views of an UNTEXTURED 3D mesh "
            "reconstruction of that object (left = reference, the rest = renders "
            "from 4 angles, flat-shaded grey). Decide if the mesh shape is BROKEN. "
            "Broken means: empty/nearly empty, collapsed, a flat sheet, melted, "
            "full of large holes, fragmented into disconnected pieces, or clearly "
            "the wrong shape vs the reference. Ignore the lack of color/texture and "
            "minor surface noise. Reply with ONLY compact JSON: "
            '{"broken": true|false, "reason": "<short>"}'
        )
        msg = client.messages.create(
            model=model, max_tokens=200, system=sys_prompt,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Reference object:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                    "data": _img_to_b64(ref.convert("RGB"))}},
                {"type": "text", "text": "Mesh reconstruction renders (4 views):"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                    "data": _img_to_b64(preview)}},
                {"type": "text", "text": "Is the mesh shape broken? JSON only."},
            ]}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        text = text[text.find("{"): text.rfind("}") + 1]
        data = json.loads(text)
        return {"ok": not bool(data.get("broken", False)),
                "reason": str(data.get("reason", "")), "raw": text}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Attempt parameter schedule
# ---------------------------------------------------------------------------

def attempt_params(base_seed: int, attempt: int, preset: dict) -> dict:
    p = dict(preset)
    p["seed"] = base_seed + attempt * 9973
    if attempt >= 2:
        p["steps"] = preset["steps"] + 10
    if attempt == 1:
        p["guidance"] = round(preset["guidance"] + 1.0, 2)
    if attempt >= 3:
        p["guidance"] = max(1.0, round(preset["guidance"] - 1.0, 2))
    return p


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(engine: ShapeEngine, img_path: Path, out_root: Path, preset: dict,
                  max_attempts: int, use_vision: bool, vision_model: str, fh):
    name = img_path.stem
    attempts_log = []
    t_img = time.time()
    last_preview = last_prepared = None

    try:
        prepared = engine.preprocess(str(img_path))
    except Exception as e:
        _log(f"  [{name}] preprocess ERROR: {e}", fh)
        prepared = Image.open(img_path).convert("RGB")

    for attempt in range(max_attempts):
        p = attempt_params(preset["seed"], attempt, preset)
        _log(f"  [{name}] attempt {attempt+1}/{max_attempts} "
             f"(seed={p['seed']}, steps={p['steps']}, guidance={p['guidance']})", fh)
        rec = {"attempt": attempt + 1, "params": {k: p[k] for k in
               ("seed", "steps", "guidance", "octree_resolution")}}
        try:
            t0 = time.time()
            mesh = engine.generate(prepared, seed=p["seed"], steps=p["steps"],
                                   guidance=p["guidance"],
                                   octree_resolution=p["octree_resolution"])
            rec["gen_sec"] = round(time.time() - t0, 1)

            geo = geometric_check(mesh)
            view_imgs, covs = render_views(mesh)
            ren = render_check(covs)
            preview = make_preview(prepared, view_imgs)
            rec["geometric"] = geo
            rec["render"] = ren

            ok = geo["ok"] and ren["ok"]
            reasons = list(geo["reasons"]) + list(ren["reasons"])

            if ok and use_vision:
                vis = vision_judge(prepared, preview, vision_model)
                if vis is not None:
                    rec["vision"] = {"ok": vis["ok"], "reason": vis["reason"]}
                    if not vis["ok"]:
                        ok = False
                        reasons.append(f"vision:{vis['reason']}")
                else:
                    rec["vision"] = {"ok": None, "reason": "unavailable"}

            rec["ok"] = ok
            rec["reasons"] = reasons
            attempts_log.append(rec)

            if ok:
                dst = out_root / "success" / name
                dst.mkdir(parents=True, exist_ok=True)
                mesh.export(str(dst / "model.glb"))
                prepared.convert("RGBA").save(str(dst / "preprocessed.webp"))
                preview.save(str(dst / "preview.webp"))
                _log(f"  [{name}] OK on attempt {attempt+1} "
                     f"({rec['gen_sec']}s, {geo['metrics']})", fh)
                info = {"name": name, "status": "success", "source": str(img_path),
                        "attempts": attempts_log, "total_sec": round(time.time() - t_img, 1)}
                (dst / "info.json").write_text(json.dumps(info, indent=2))
                return info
            else:
                _log(f"  [{name}] attempt {attempt+1} REJECTED -> {reasons}", fh)
                last_preview, last_prepared = preview, prepared

        except Exception as e:
            rec["ok"] = False
            rec["reasons"] = [f"exception:{type(e).__name__}:{e}"]
            rec["traceback"] = traceback.format_exc()
            attempts_log.append(rec)
            _log(f"  [{name}] attempt {attempt+1} ERROR: {e}", fh)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    dst = out_root / "failed" / name
    dst.mkdir(parents=True, exist_ok=True)
    try:
        if last_prepared is not None:
            last_prepared.convert("RGBA").save(str(dst / "last_preprocessed.webp"))
        if last_preview is not None:
            last_preview.save(str(dst / "last_preview.webp"))
    except Exception:
        pass
    _log(f"  [{name}] FAILED after {max_attempts} attempts", fh)
    info = {"name": name, "status": "failed", "source": str(img_path),
            "attempts": attempts_log, "total_sec": round(time.time() - t_img, 1)}
    (dst / "info.json").write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Hunyuan3D batch agent (image -> 3D mesh with auto-QA + retries)")
    ap.add_argument("--input", required=True, help="folder of input images")
    ap.add_argument("--output", default="mesh_outputs", help="output folder")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30, help="diffusion steps")
    ap.add_argument("--guidance", type=float, default=5.0, help="guidance scale")
    ap.add_argument("--octree-resolution", type=int, default=384, help="mesh detail (256/320/384)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qa-vision", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--vision-model", default=VISION_MODEL_DEFAULT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not in_dir.is_dir():
        print(f"input folder not found: {in_dir}", file=sys.stderr); sys.exit(2)
    out_root = Path(args.output); out_root.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"no images found in {in_dir}", file=sys.stderr); sys.exit(2)

    manifest_path = out_root / "manifest.json"
    manifest = {"results": []}
    done = set()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if args.resume:
                done = {r["name"] for r in manifest["results"]}
        except Exception:
            manifest = {"results": []}

    use_vision = (args.qa_vision == "on") or (args.qa_vision == "auto" and vision_available())
    vis_note = f"vision QA {'ON ('+args.vision_model+')' if use_vision else 'OFF'}"

    preset = dict(seed=args.seed, steps=args.steps, guidance=args.guidance,
                  octree_resolution=args.octree_resolution)

    fh = open(out_root / "agent.log", "a", encoding="utf-8")
    _log(f"=== Hunyuan3D mesh agent start: {len(images)} images, "
         f"steps={args.steps} guidance={args.guidance} octree={args.octree_resolution}, "
         f"max_attempts={args.max_attempts}, {vis_note} ===", fh)
    _log("loading shape pipeline (one-time)...", fh)
    engine = ShapeEngine(device="cuda" if torch.cuda.is_available() else "cpu")

    n_ok = n_fail = 0
    for i, img in enumerate(images, 1):
        if img.stem in done:
            _log(f"[{i}/{len(images)}] {img.name} -- skip (resume)", fh); continue
        _log(f"[{i}/{len(images)}] {img.name}", fh)
        info = process_image(engine, img, out_root, preset, args.max_attempts,
                             use_vision, args.vision_model, fh)
        manifest["results"] = [r for r in manifest["results"] if r["name"] != info["name"]]
        manifest["results"].append({k: info[k] for k in ("name", "status", "source", "total_sec")}
                                   | {"n_attempts": len(info["attempts"])})
        manifest["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        n_ok += info["status"] == "success"
        n_fail += info["status"] == "failed"

    _log(f"=== done: {n_ok} success, {n_fail} failed, "
         f"out of {len(images)} (skipped {len(done)}) ===", fh)
    fh.close()


if __name__ == "__main__":
    main()
