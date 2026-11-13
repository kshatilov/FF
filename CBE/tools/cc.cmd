@echo off
setlocal
rem cc - table color calibration launcher.
rem CC_PYTHON overrides the interpreter, then HG_PYTHON, then default.
if defined CC_PYTHON (set "CC_PY=%CC_PYTHON%") else if defined HG_PYTHON (set "CC_PY=%HG_PYTHON%") else (set "CC_PY=C:\Python314\python.exe")
"%CC_PY%" "%~dp0tools\cc\cc.py" %*
exit /b %ERRORLEVEL%
