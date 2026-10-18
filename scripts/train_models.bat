@echo off
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set "VPY=.venv\Scripts\python.exe"
set TF_ENABLE_ONEDNN_OPTS=0
title EnergyPulse - training all forecast models
echo Training all forecast models. About 20 minutes. Keep this window open. Log: logs\train_log.txt
echo Started %date% %time% > logs\train_log.txt
"%VPY%" -u backend\model.py >> logs\train_log.txt 2>&1
echo EXIT_CODE=%errorlevel% >> logs\train_log.txt
"%VPY%" backend\model_report.py >> logs\train_log.txt 2>&1
echo Finished %date% %time% >> logs\train_log.txt
echo Done. You can close this window.
