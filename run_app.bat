@echo off
REM Launches the unified TripoSplat Studio web app at http://127.0.0.1:7860
REM   Tabs: Single image  |  Batch Agent (QA + self-heal)  |  Results Browser
setlocal
set ROOT=%~dp0
set PY=%ROOT%venv\Scripts\python.exe
cd /d "%ROOT%TripoSplat"
REM Optional: enable Claude vision QA by setting your key first, e.g.
REM   set ANTHROPIC_API_KEY=sk-ant-...
"%PY%" app.py
endlocal
