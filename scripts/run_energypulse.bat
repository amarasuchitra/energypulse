@echo off
setlocal
title EnergyPulse
rem Start EnergyPulse on this computer. First run: creates .venv and installs packages.
cd /d "%~dp0.."
if not exist logs mkdir logs
set "LOG=logs\setup_log.txt"
echo EnergyPulse setup %date% %time% > "%LOG%"

rem ---- Find a working Python (prefer 3.11, which the app is built for) ----
set "PY="
py -3.11 -V >nul 2>&1 && set "PY=py -3.11"
if not defined PY py -3.12 -V >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3 -V >nul 2>&1 && set "PY=py -3"
if not defined PY python -V >nul 2>&1 && set "PY=python"
if not defined PY goto :nopython
echo Using: %PY% >> "%LOG%"
%PY% -V >> "%LOG%" 2>&1

rem ---- Virtual environment and packages (skipped once done) ----
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv >> "%LOG%" 2>&1
)
if not exist ".venv\Scripts\python.exe" goto :novenv
set "VPY=.venv\Scripts\python.exe"

if exist ".venv\setup_ok" goto :run
echo Installing packages. The first run takes several minutes...
"%VPY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
findstr /v /i "tensorflow" requirements.txt > "%TEMP%\energypulse_req.txt"
"%VPY%" -m pip install -r "%TEMP%\energypulse_req.txt" >> "%LOG%" 2>&1
if errorlevel 1 goto :pipfail
echo Installing TensorFlow, optional, the app works without it...
"%VPY%" -m pip install "tensorflow>=2.15" >> "%LOG%" 2>&1
if errorlevel 1 echo TENSORFLOW_SKIPPED>> "%LOG%"
echo done> ".venv\setup_ok"

:run
echo SETUP_DONE>> "%LOG%"
echo.
echo EnergyPulse is starting at http://localhost:8501
echo Keep this window open while you use the app. Close it to stop the app.
"%VPY%" -m streamlit run frontend\app.py --server.headless true --server.port 8501 > logs\streamlit_log.txt 2>&1
goto :eof

:nopython
echo PYTHON_NOT_FOUND>> "%LOG%"
echo Python was not found. Install Python 3.11 from python.org, tick "Add python.exe to PATH", then run this again.
pause
goto :eof

:novenv
echo VENV_FAILED>> "%LOG%"
echo Could not create the virtual environment. See logs\setup_log.txt
pause
goto :eof

:pipfail
echo PIP_FAILED>> "%LOG%"
echo Package install failed. See logs\setup_log.txt
pause
goto :eof
