# TripoSplat — local setup (image → 3D Gaussian Splat)

Reference: https://github.com/VAST-AI-Research/TripoSplat

## What's installed
- `venv/` — Python 3.12 virtual env (torch 2.6 + cu124, gradio, etc.)
- `TripoSplat/` — the cloned repo
- `TripoSplat/ckpts/` — model weights (~3.6 GB, downloaded from HuggingFace)

Hardware here: RTX 4090 (24 GB). Runs fully on GPU (`device="cuda"`).

## Run on a single image (CLI)
```
run_image.bat  C:\path\to\your\image.png
```
Optional 2nd arg = number of gaussians (default 262144, max 262144):
```
run_image.bat  C:\path\to\your\image.png  131072
```
Outputs land in `TripoSplat/`:
- `output.ply`   — open in https://superspl.at/editor or https://sparkjs.dev
- `output.splat`
- `preprocessed_image.webp` — background-removed input

No argument = runs the bundled example image.

## Web UI (drag & drop)
```
run_webui.bat
```
Then open http://127.0.0.1:7860

## Manual (inside the repo folder)
```
cd TripoSplat
..\venv\Scripts\python.exe run_example.py      # example
..\venv\Scripts\python.exe run_gradio.py       # web UI
```

## Viewing the result
Drag `output.ply` (or `.splat`) into:
- https://superspl.at/editor
- https://sparkjs.dev

---

# Batch Agent — many images → 3D, with auto-QA + self-healing retries

Give it a folder of images (e.g. 100). For each image it:
1. Generates an **ultra-high-quality** splat (steps 30, 262144 gaussians).
2. **Auto-detects broken models** (empty / collapsed / exploded / noisy /
   wrong shape) via geometric checks + a self-rendered 4-view check
   (+ optional Claude vision judge).
3. **Regenerates** with a different seed/params if broken — **up to 4 tries**.
4. After 4 failures, records the case under `failed/` and moves to the next image.

## Run
```
run_agent.bat  C:\path\to\image_folder
run_agent.bat  C:\path\to\image_folder  C:\path\to\output
```
Results:
```
output\
  success\<name>\  model.ply, model.splat, preprocessed.webp, preview.webp, info.json
  failed\<name>\   last_preprocessed.webp, last_preview.webp, info.json
  manifest.json    run summary (also enables --resume)
  agent.log
```
- `preview.webp` = reference image + 4 rendered angles, so you can eyeball quality.
- **Resume**: re-running the same command skips images already finished.

## Options (append after the output folder)
```
--max-attempts 4      tries per image before marking failed
--steps 30            sampler steps (higher = finer, slower)
--guidance 3.5        CFG strength
--num-gaussians 262144 (max; lower = lighter files)
--limit 10            process only the first N images (testing)
--qa-vision auto|on|off
```

## Optional: AI (Claude) vision QA
The geometric + render checks already catch broken models. For an extra
"does this actually look right vs the photo?" judgment, set an API key first:
```
set ANTHROPIC_API_KEY=sk-ant-...
run_agent.bat C:\path\to\image_folder
```
Without a key it runs fine (vision step is skipped automatically).

## Manual (inside TripoSplat\)
```
..\venv\Scripts\python.exe agent_batch.py --input C:\imgs --output C:\out --resume
```
