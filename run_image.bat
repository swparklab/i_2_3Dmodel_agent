@echo off
REM Usage: run_image.bat path\to\image.png  [num_gaussians]
REM Produces output.ply + output.splat inside the TripoSplat folder.
setlocal
set ROOT=%~dp0
set PY=%ROOT%venv\Scripts\python.exe
cd /d "%ROOT%TripoSplat"

if "%~1"=="" (
  echo No image given - running the bundled example instead.
  "%PY%" run_example.py
  goto :eof
)

set IMG=%~1
set NG=%~2
if "%NG%"=="" set NG=262144

"%PY%" -c "from triposplat import TripoSplatPipeline; p=TripoSplatPipeline(ckpt_path='ckpts/diffusion_models/triposplat_fp16.safetensors',decoder_path='ckpts/vae/triposplat_vae_decoder_fp16.safetensors',dinov3_path='ckpts/clip_vision/dino_v3_vit_h.safetensors',flux2_vae_encoder_path='ckpts/vae/flux2-vae.safetensors',rmbg_path='ckpts/background_removal/birefnet.safetensors',device='cuda'); g,prep=p.run(r'%IMG%',num_gaussians=%NG%,show_progress=True); prep.save('preprocessed_image.webp'); g.save_ply('output.ply'); g.save_splat('output.splat'); print('DONE -> TripoSplat\\output.ply and output.splat')"
endlocal
