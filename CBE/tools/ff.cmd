@echo off
setlocal
rem FF_PYTHON overrides the calibration interpreter.
if defined FF_PYTHON (set "FF_PY=%FF_PYTHON%") else (set "FF_PY=C:\Python314\python.exe")
"%FF_PY%" "%~dp0tools\ff\ff_cli.py" %*
exit /b %ERRORLEVEL%
