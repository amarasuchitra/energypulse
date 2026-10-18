@echo off
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set "VPY=.venv\Scripts\python.exe"
set TF_ENABLE_ONEDNN_OPTS=0
title EnergyPulse - retraining the live-meter model
echo Retraining the live-meter forecast model. About 10 minutes. Keep this window open. Log: logs\train_sim_log.txt
echo Started %date% %time% > logs\train_sim_log.txt
"%VPY%" -u backend\train_sim_meter.py >> logs\train_sim_log.txt 2>&1
echo EXIT_CODE=%errorlevel% >> logs\train_sim_log.txt
"%VPY%" backend\model_report.py >> logs\train_sim_log.txt 2>&1
echo Finished %date% %time% >> logs\train_sim_log.txt
echo Done. You can close this window.
