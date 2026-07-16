@echo off
title RescuOpt AI Launcher
echo =========================================
echo       RESCUOPT AI - System Launcher      
echo =========================================
echo.

echo [1/2] Starting Server (Dashboard)...
start "RescuOpt Server" cmd /k "C:\\Users\\arin\\miniconda3\\python.exe Server.py"

echo [2/2] Starting Main App (Tkinter/YOLO)...
cd Flood-detection
start "RescuOpt Desktop App" cmd /k "C:\\Users\\arin\\miniconda3\\python.exe main.py"

echo.
echo System is running! 
echo You can close this launcher window.
timeout /t 3 >nul
