@echo off
rem Root OSGEO4W home dir to the directory
::修改OSGEO4W_ROOT路径为QGIS的安装路径

set OSGEO4W_ROOT=D:\QGIS 3.22.6
call "%OSGEO4W_ROOT%\bin\o4w_env.bat"

cd /d "%~dp0"

pip install -r requirements.txt
pause