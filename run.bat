@echo off
setlocal

set "ROOT=%~dp0"

echo ============================================
echo           Starting MindVoice AI
echo ============================================

REM ============================
REM Backend
REM ============================

start "MindVoice Backend" cmd /k ^
"cd /d "%ROOT%backend" && ^
call .venv\Scripts\activate.bat && ^
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000"

REM ============================
REM Frontend
REM ============================

start "MindVoice Frontend" cmd /k ^
"cd /d "%ROOT%frontend" && ^
npm run dev"

exit
