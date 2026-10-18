@echo off
title EnergyPulse - restart
rem Stop the EnergyPulse server on port 8501, then start it again with the latest code and models.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do taskkill /PID %%p /F >nul 2>&1
timeout /t 2 /nobreak >nul
call "%~dp0run_energypulse.bat"
