@echo off
REM Launches the TripoSplat Gradio web UI at http://127.0.0.1:7860
setlocal
set ROOT=%~dp0
set PY=%ROOT%venv\Scripts\python.exe
cd /d "%ROOT%TripoSplat"
"%PY%" run_gradio.py
endlocal
