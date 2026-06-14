"""TripoSplat batch agent — image folder -> ultra-high-quality 3D Gaussian Splats
with automatic "is the model broken?" QA and self-healing regeneration.

Pipeline per image (processed sequentially):
  1. Generate at a high-quality preset (steps / guidance / 262144 gaussians).
  2. QA the result with up to 3 layers:
       - geometric checks  (NaN/Inf, empty, collapsed, exploded)        [always]
       - self-rendered 4-view coverage heuristic                        [always]
       - Claude vision judge: reference image vs renders                [if API key]
  3. If QA fails, regenerate with a different seed / params.
  4. After MAX_ATTEMPTS failed tries, record a failure case and move on.

Outputs:
  <output>/success/<name>/  model.ply, model.splat, preprocessed.webp, preview.webp, info.json
  <output>/failed/<name>/   last_preprocessed.webp, last_preview.webp, info.json
  <output>/manifest.json    full run record (also drives --resume)
  <output>/agent.log

Usage:
  python agent_batch.py --input <img_dir> --output <out_dir> [options]
See `python agent_batch.py -h`.
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

import numpy as np
import torch
from PIL import Image

from triposplat import TripoSplatPipeline

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CKPTS = dict(
    ckpt_path              = "ckpts/diffusion_models/triposplat_fp16.safetensors",
    decoder_path           = "ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path            = "ckpts/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path = "ckpts/vae/flux2-vae.safetensors",
    rmbg_path              = "ckpts/background_removal/birefnet.safetensors",
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# QA thresholds, calibrated against a healthy model
# (visible-frac ~0.95, extent ~1.0, scale median ~0.0013, p99 ~0.018).
QA = dict(
    opacity_vis_thresh = 0.05,   # a gaussian counts as "visible" above this opacity
    min_visible_frac   = 0.05,   # below this -> essentially empty / broken
    min_extent         = 0.02,   # max-axis spatial extent below this -> collapsed to a point
    max_extent         = 6.0,    # object lives in ~unit cube; far beyond this -> runaway
    max_scale_median   = 0.10,   # healthy ~0.0013; this is ~75x margin -> exploded blob
    max_scale_p99      = 0.60,   # huge tails -> spiky / exploded
    min_render_cov     = 0.004,  # mean foreground coverage over the 4 renders; near 0 -> empty
    max_render_cov     = 0.97,   # full-frame fill -> blob / exploded
)

VISION_MODEL_DEFAULT = "claude-sonnet-4-6"


def _log(msg: str, fh=None):
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n"); fh.flush()


# ---------------------------------------------------------------------------
# QA — geometric checks (authoritative, no dependencies)
# ---------------------------------------------------------------------------

def geometric_check(g) -> dict:
    xyz = g.get_xyz.float()
    op  = g.get_opacity.float().reshape(-1)
    sc  = g.get_scaling.float()
    fdc = g._features_dc.float()

    reasons, metrics = [], {}

    # NaN / Inf anywhere is a hard failure.
    for name, t in (("xyz", xyz), ("opacity", op), ("scaling", sc), ("features", fdc)):
        if not torch.isfinite(t).all():
            reasons.append(f"non_finite:{name}")

    n = xyz.shape[0]
    metrics["count"] = int(n)

    vis = op > QA["opacity_vis_thresh"]
    vis_frac = float(vis.float().mean()) if n else 0.0
    metrics["visible_frac"] = round(vis_frac, 4)
    if vis_frac < QA["min_visible_frac"]:
        reasons.append(f"empty(visible_frac={vis_frac:.4f})")

    if vis.any():
        v = xyz[vis]
        extent = (v.max(0).values - v.min(0).values)
        max_extent = float(extent.max())
        metrics["extent_max"] = round(max_extent, 4)
        if max_extent < QA["min_extent"]:
            reasons.append(f"collapsed(extent={max_extent:.4f})")
        if max_extent > QA["max_extent"]:
            reasons.append(f"runaway(extent={max_extent:.4f})")

        sv = sc[vis].reshape(-1)
        sc_med = float(sv.median())
        sc_p99 = float(torch.quantile(sv, 0.99))
        metrics["scale_median"] = round(sc_med, 5)
        metrics["scale_p99"] = round(sc_p99, 5)
        if sc_med > QA["max_scale_median"]:
            reasons.append(f"exploded(scale_med={sc_med:.4f})")
        if sc_p99 > QA["max_scale_p99"]:
            reasons.append(f"spiky(scale_p99={sc_p99:.4f})")
    else:
        metrics["extent_max"] = 0.0

    return {"ok": len(reasons) == 0, "reasons": reasons, "metrics": metrics}


# ---------------------------------------------------------------------------
# QA — self renderer (painter's z-buffer, pure torch) for heuristic + vision
# ---------------------------------------------------------------------------

_SH_C0 = 0.28209479177387814
_DEFAULT_T = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # match Gaussian._DEFAULT_TRANSFORM (y-up)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


@torch.no_grad()
def render_views(g, res=320, azimuths=(0, 90, 180, 270), elevation=18.0,
                 bg=0.12) -> tuple[list[np.ndarray], list[float]]:
    """Rough nearest-point (z-buffer) render of the gaussians from a few views.
    Good enough to judge 'is this a coherent object or garbage'. Returns RGB
    uint8 arrays and per-view foreground coverage fractions."""
    dev = g.get_xyz.device
    xyz = g.get_xyz.float()
    op  = g.get_opacity.float().reshape(-1)
    col = (g._features_dc.float()[:, 0, :] * _SH_C0 + 0.5).clamp(0, 1)

    m = op > QA["opacity_vis_thresh"]
    if int(m.sum()) < 32:
        m = op > 0
    xyz, col = xyz[m], col[m]
    if xyz.shape[0] == 0:
        empty = (np.full((res, res, 3), int(bg * 255), np.uint8))
        return [empty for _ in azimuths], [0.0 for _ in azimuths]

    T = torch.tensor(_DEFAULT_T, dtype=torch.float32, device=dev)
    xyz = xyz @ T.t()
    lo = torch.quantile(xyz, 0.01, dim=0)
    hi = torch.quantile(xyz, 0.99, dim=0)
    center = (lo + hi) / 2
    scale = float((hi - lo).max()) / 2 + 1e-6
    xyz = (xyz - center) / scale  # ~[-1, 1]

    el = math.radians(elevation)
    Rx = _rot_x(el).to(dev)
    imgs, covs = [], []
    for az in azimuths:
        Ry = _rot_y(math.radians(az)).to(dev)
        p = xyz @ Ry.t() @ Rx.t()
        margin = 1.25
        sx = ((p[:, 0] / margin) * 0.5 + 0.5) * (res - 1)
        sy = ((-p[:, 1] / margin) * 0.5 + 0.5) * (res - 1)
        px = sx.round().long().clamp(0, res - 1)
        py = sy.round().long().clamp(0, res - 1)
        depth = -p[:, 2]
        idx = py * res + px
        zbuf = torch.full((res * res,), float("inf"), device=dev)
        zbuf.scatter_reduce_(0, idx, depth, reduce="amin", include_self=True)
        winners = depth <= zbuf[idx] + 1e-6
        canvas = torch.full((res * res, 3), bg, device=dev)
        alpha = torch.zeros(res * res, device=dev)
        canvas[idx[winners]] = col[winners]
        alpha[idx[winners]] = 1.0
        covs.append(float(alpha.mean()))
        rgb = (canvas.reshape(res, res, 3).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        imgs.append(rgb)
    return imgs, covs


def make_preview(ref: Image.Image, view_imgs: list[np.ndarray], res=320) -> Image.Image:
    """Reference + the rendered views side by side, saved for human eyeballing
    and used as the vision-judge montage."""
    tiles = [ref.convert("RGB").resize((res, res), Image.LANCZOS)]
    tiles += [Image.fromarray(v) for v in view_imgs]
    strip = Image.new("RGB", (res * len(tiles), res), (24, 24, 28))
    for i, t in enumerate(tiles):
        strip.paste(t, (i * res, 0))
    return strip


def render_check(covs: list[float]) -> dict:
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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def vision_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def vision_judge(ref: Image.Image, preview: Image.Image, model: str) -> dict | None:
    """Ask Claude whether the reconstruction looks broken. Returns
    {ok, reason} or None if the call could not be made."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        sys_prompt = (
            "You are a strict 3D reconstruction QA inspector. You are shown a "
            "reference object image, then 4 rendered views of a 3D Gaussian-splat "
            "reconstruction of that object (left = reference, the rest = renders "
            "from 4 angles). Decide if the reconstruction is BROKEN. Broken means: "
            "empty/nearly empty, collapsed, a shapeless noisy point cloud, melted, "
            "full of large holes, fragmented into disconnected blobs, or clearly the "
            "wrong shape vs the reference. Minor blur, slight color shift, or soft "
            "edges are NOT broken. Reply with ONLY compact JSON: "
            '{"broken": true|false, "reason": "<short>"}'
        )
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            system=sys_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reference object:"},
                    {"type": "image", "source": {"type": "base64",
                        "media_type": "image/png", "data": _img_to_b64(ref.convert("RGB"))}},
                    {"type": "text", "text": "Reconstruction renders (4 views):"},
                    {"type": "image", "source": {"type": "base64",
                        "media_type": "image/png", "data": _img_to_b64(preview)}},
                    {"type": "text", "text": "Is the reconstruction broken? JSON only."},
                ],
            }],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        text = text[text.find("{"): text.rfind("}") + 1]
        data = json.loads(text)
        broken = bool(data.get("broken", False))
        return {"ok": not broken, "reason": str(data.get("reason", "")), "raw": text}
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Attempt parameter schedule (vary across retries to escape bad samples)
# ---------------------------------------------------------------------------

def attempt_params(base_seed: int, attempt: int, preset: dict) -> dict:
    p = dict(preset)
    p["seed"] = base_seed + attempt * 9973
    if attempt >= 2:
        p["steps"] = preset["steps"] + 10
    if attempt == 1:
        p["guidance_scale"] = round(preset["guidance_scale"] + 0.5, 2)
    if attempt >= 3:
        p["guidance_scale"] = max(1.5, round(preset["guidance_scale"] - 0.5, 2))
    return p


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(pipe, img_path: Path, out_root: Path, preset: dict,
                  max_attempts: int, use_vision: bool, vision_model: str, fh):
    name = img_path.stem
    attempts_log = []
    t_img = time.time()

    for attempt in range(max_attempts):
        p = attempt_params(preset["seed"], attempt, preset)
        _log(f"  [{name}] attempt {attempt+1}/{max_attempts} "
             f"(seed={p['seed']}, steps={p['steps']}, guidance={p['guidance_scale']})", fh)
        rec = {"attempt": attempt + 1, "params": {k: p[k] for k in
               ("seed", "steps", "guidance_scale", "shift", "num_gaussians")}}
        try:
            t0 = time.time()
            g, prepared = pipe.run(
                str(img_path), seed=p["seed"], steps=p["steps"],
                guidance_scale=p["guidance_scale"], shift=p["shift"],
                num_gaussians=p["num_gaussians"], show_progress=False)
            rec["gen_sec"] = round(time.time() - t0, 1)

            geo = geometric_check(g)
            view_imgs, covs = render_views(g)
            ren = render_check(covs)
            preview = make_preview(prepared, view_imgs)

            rec["geometric"] = geo
            rec["render"] = ren

            ok = geo["ok"] and ren["ok"]
            reasons = list(geo["reasons"]) + list(ren["reasons"])

            vis_res = None
            if ok and use_vision:
                vis_res = vision_judge(prepared, preview, vision_model)
                if vis_res is not None:
                    rec["vision"] = {"ok": vis_res["ok"], "reason": vis_res["reason"]}
                    if not vis_res["ok"]:
                        ok = False
                        reasons.append(f"vision:{vis_res['reason']}")
                else:
                    rec["vision"] = {"ok": None, "reason": "unavailable"}

            rec["ok"] = ok
            rec["reasons"] = reasons
            attempts_log.append(rec)

            if ok:
                dst = out_root / "success" / name
                dst.mkdir(parents=True, exist_ok=True)
                g.save_ply(str(dst / "model.ply"))
                g.save_splat(str(dst / "model.splat"))
                prepared.save(str(dst / "preprocessed.webp"))
                preview.save(str(dst / "preview.webp"))
                _log(f"  [{name}] OK on attempt {attempt+1} "
                     f"({rec['gen_sec']}s, {geo['metrics']})", fh)
                info = {"name": name, "status": "success", "source": str(img_path),
                        "attempts": attempts_log, "total_sec": round(time.time() - t_img, 1)}
                (dst / "info.json").write_text(json.dumps(info, indent=2))
                return info
            else:
                _log(f"  [{name}] attempt {attempt+1} REJECTED -> {reasons}", fh)
                # keep last preview/preprocessed around in case all attempts fail
                last_preview, last_prepared = preview, prepared

        except Exception as e:
            rec["ok"] = False
            rec["reasons"] = [f"exception:{type(e).__name__}:{e}"]
            rec["traceback"] = traceback.format_exc()
            attempts_log.append(rec)
            _log(f"  [{name}] attempt {attempt+1} ERROR: {e}", fh)
            last_preview = last_prepared = None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # all attempts failed
    dst = out_root / "failed" / name
    dst.mkdir(parents=True, exist_ok=True)
    try:
        if 'last_prepared' in dir() and last_prepared is not None:
            last_prepared.save(str(dst / "last_preprocessed.webp"))
        if 'last_preview' in dir() and last_preview is not None:
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
    ap = argparse.ArgumentParser(description="TripoSplat batch agent (image -> 3D with auto-QA + retries)")
    ap.add_argument("--input", required=True, help="folder of input images")
    ap.add_argument("--output", default="agent_outputs", help="output folder")
    ap.add_argument("--max-attempts", type=int, default=4, help="max generation tries per image")
    ap.add_argument("--steps", type=int, default=30, help="sampler steps (quality)")
    ap.add_argument("--guidance", type=float, default=3.5, help="CFG guidance scale")
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--num-gaussians", type=int, default=262144, help="gaussian count (max 262144)")
    ap.add_argument("--seed", type=int, default=42, help="base seed")
    ap.add_argument("--qa-vision", choices=["auto", "on", "off"], default="auto",
                    help="Claude vision QA: auto=use if ANTHROPIC_API_KEY set")
    ap.add_argument("--vision-model", default=VISION_MODEL_DEFAULT)
    ap.add_argument("--resume", action="store_true", help="skip images already in the manifest")
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (0=all)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not in_dir.is_dir():
        print(f"input folder not found: {in_dir}", file=sys.stderr); sys.exit(2)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

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
    if args.qa_vision != "off" and not use_vision:
        vis_note = "vision QA OFF (no ANTHROPIC_API_KEY / anthropic pkg)"
    else:
        vis_note = f"vision QA {'ON ('+args.vision_model+')' if use_vision else 'OFF'}"

    preset = dict(seed=args.seed, steps=args.steps, guidance_scale=args.guidance,
                  shift=args.shift, num_gaussians=args.num_gaussians)

    fh = open(out_root / "agent.log", "a", encoding="utf-8")
    _log(f"=== TripoSplat agent start: {len(images)} images, "
         f"preset steps={args.steps} guidance={args.guidance} ng={args.num_gaussians}, "
         f"max_attempts={args.max_attempts}, {vis_note} ===", fh)

    _log("loading pipeline (one-time)...", fh)
    pipe = TripoSplatPipeline(device="cuda", **CKPTS)

    n_ok = n_fail = 0
    for i, img in enumerate(images, 1):
        if img.stem in done:
            _log(f"[{i}/{len(images)}] {img.name} -- skip (resume)", fh); continue
        _log(f"[{i}/{len(images)}] {img.name}", fh)
        info = process_image(pipe, img, out_root, preset, args.max_attempts,
                             use_vision, args.vision_model, fh)
        # update manifest (replace any prior entry for this name)
        manifest["results"] = [r for r in manifest["results"] if r["name"] != info["name"]]
        manifest["results"].append({k: info[k] for k in ("name", "status", "source", "total_sec")}
                                   | {"n_attempts": len(info["attempts"])})
        manifest["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        n_ok += info["status"] == "success"
        n_fail += info["status"] == "failed"

    _log(f"=== done: {n_ok} success, {n_fail} failed, "
         f"out of {len(images)} (skipped {len(done)}) ===", fh)
    _log(f"manifest: {manifest_path}", fh)
    fh.close()


if __name__ == "__main__":
    main()
