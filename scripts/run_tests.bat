@echo off
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set "VPY=.venv\Scripts\python.exe"
set TF_ENABLE_ONEDNN_OPTS=0
title EnergyPulse - running tests
echo Running the test suite. Keep this window open. Results: logs\test_log.txt
echo Started %date% %time% > logs\test_log.txt
"%VPY%" -m pip install -q pytest >> logs\test_log.txt 2>&1
"%VPY%" -m pytest -q -p no:cacheprovider >> logs\test_log.txt 2>&1
echo EXIT_CODE=%errorlevel% >> logs\test_log.txt
"%VPY%" -c "import sys; sys.path[:0] = ['frontend', 'backend']; import app" >> logs\test_log.txt 2>&1
echo IMPORT_APP_EXIT=%errorlevel% >> logs\test_log.txt
echo Finished %date% %time% >> logs\test_log.txt
echo Done. You can close this window.
