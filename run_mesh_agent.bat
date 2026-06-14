@echo off
REM Hunyuan3D mesh batch agent: image folder -> 3D meshes (.glb) with auto-QA + retries
REM Usage:
REM   run_mesh_agent.bat  C:\path\to\image_folder
REM   run_mesh_agent.bat  C:\path\to\image_folder  C:\path\to\output
setlocal
set ROOT=%~dp0
set PY=%ROOT%venv_hy\Scripts\python.exe
cd /d "%ROOT%Hunyuan3D-2.1"
set INPUT=%~1
set OUTPUT=%~2
if "%OUTPUT%"=="" set OUTPUT=mesh_outputs
"%PY%" mesh_agent.py --input "%INPUT%" --output "%OUTPUT%" --resume %3 %4 %5 %6 %7 %8 %9
endlocal
