@echo off
REM ==========================================================================
REM  TripoSplat batch agent
REM  Usage:   run_agent.bat  INPUT_IMAGE_FOLDER  [OUTPUT_FOLDER]  [extra args]
REM  Example: run_agent.bat  C:\my_images
REM           run_agent.bat  C:\my_images  C:\my_3d_out  --steps 40
REM
REM  - Processes every image in the folder, one by one, at ultra-high quality.
REM  - Auto-detects broken 3D models and regenerates (up to 4 tries each).
REM  - Failed cases are recorded; the run continues to the next image.
REM  - --resume is on by default: re-running skips already-finished images.
REM
REM  Optional Claude vision QA: set ANTHROPIC_API_KEY before running.
REM    set ANTHROPIC_API_KEY=sk-ant-...
REM ==========================================================================
setlocal
set ROOT=%~dp0
set PY=%ROOT%venv\Scripts\python.exe
cd /d "%ROOT%TripoSplat"

if "%~1"=="" (
  echo Usage: run_agent.bat  INPUT_IMAGE_FOLDER  [OUTPUT_FOLDER]  [extra args]
  echo Example: run_agent.bat  C:\my_images
  goto :eof
)

set IN=%~1
set OUT=%~2
if "%OUT%"=="" set OUT=%ROOT%agent_outputs

"%PY%" agent_batch.py --input "%IN%" --output "%OUT%" --resume %3 %4 %5 %6 %7 %8 %9
echo.
echo Done. Results in: %OUT%
echo   success\^<name^>\model.ply ^| model.splat ^| preview.webp
echo   failed\^<name^>\  (cases that failed all attempts)
echo   manifest.json   (summary)
endlocal
