@echo off
rem Root OSGEO4W home dir to the directory

set OSGEO4W_ROOT=%~1
echo %OSGEO4W_ROOT%
call "%OSGEO4W_ROOT%\o4w_env.bat"

set RS_ROOT=%~2
echo %RS_ROOT%

cd /d %~dp0
pip install -r "%RS_ROOT%\requirements.txt"