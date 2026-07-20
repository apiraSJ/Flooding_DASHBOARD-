@echo off
title RescuOpt AI Launcher
echo =========================================
echo       RESCUOPT AI - System Launcher      
echo =========================================
echo.

REM ชื่อ conda environment ที่ใช้รันโปรเจกต์นี้ (แก้ตรงนี้ที่เดียวถ้า env ของคุณชื่ออื่น)
set ENV_NAME=geoai

REM %~dp0 คือโฟลเดอร์ที่ไฟล์ .bat นี้อยู่ (มี \ ปิดท้ายอยู่แล้ว) ใช้แทนการ hardcode path
REM ของ python.exe หรือ path ของโปรเจกต์ ทำให้สคริปต์นี้พกไปรันเครื่องไหนก็ได้โดยไม่ต้องแก้ไข
REM ขอแค่เครื่องนั้นติดตั้ง conda และมี environment ชื่อ %ENV_NAME% ไว้แล้ว
REM (สร้างด้วย: conda create -n geoai python=3.10 -y)

echo [1/2] Starting Server (Dashboard)...
start "RescuOpt Server" cmd /k "cd /d "%~dp0" && call conda activate %ENV_NAME% && python Server.py"

echo [2/2] Starting Main App (Tkinter/YOLO)...
start "RescuOpt Desktop App" cmd /k "cd /d "%~dp0Flood-detection" && call conda activate %ENV_NAME% && python main.py"

echo.
echo System is running! 
echo You can close this launcher window.
timeout /t 3 >nul