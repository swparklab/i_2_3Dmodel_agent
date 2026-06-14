@echo off
REM Launches Hunyuan3D Studio (mesh) web app at http://127.0.0.1:7861
REM   Tabs: Single image  |  Batch Agent (QA + self-heal)  |  Results Browser
setlocal
set ROOT=%~dp0
set PY=%ROOT%venv_hy\Scripts\python.exe
cd /d "%ROOT%Hunyuan3D-2.1"
REM Optional: enable Claude vision QA -> set ANTHROPIC_API_KEY=sk-ant-...
"%PY%" mesh_app.py
endlocal
