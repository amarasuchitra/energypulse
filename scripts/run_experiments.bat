@echo off
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set "VPY=.venv\Scripts\python.exe"
set TF_ENABLE_ONEDNN_OPTS=0
title EnergyPulse - model experiments
echo Testing ways to improve the forecast. About 25 minutes. Keep this window open. Log: logs\experiments_log.txt
echo Started %date% %time% > logs\experiments_log.txt
"%VPY%" -u backend\experiments.py >> logs\experiments_log.txt 2>&1
echo EXIT_CODE=%errorlevel% >> logs\experiments_log.txt
echo Finished %date% %time% >> logs\experiments_log.txt
echo Done. You can close this window.
