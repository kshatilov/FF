@echo off
setlocal
rem hg - homography / device-map utility launcher.
rem HG_PYTHON overrides the interpreter (default: C:\Python314\python.exe).
if defined HG_PYTHON (set "HG_PY=%HG_PYTHON%") else (set "HG_PY=C:\Python314\python.exe")
"%HG_PY%" "%~dp0tools\hg.py" %*
exit /b %ERRORLEVEL%
