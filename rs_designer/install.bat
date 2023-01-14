@echo off
rem Root OSGEO4W home dir to the directory
set PYTHONHOME=%~1
set OSGEO4W_ROOT=%~2
echo %OSGEO4W_ROOT%
call "%OSGEO4W_ROOT%\o4w_env.bat"

set RS_ROOT=%~3
echo %RS_ROOT%

cd /d %~dp0
"%PYTHONHOME%\python3.exe" -m pip install -r "%RS_ROOT%\requirements.txt"