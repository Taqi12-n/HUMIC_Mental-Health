@echo off
setlocal

set "ROOT=%~dp0"

echo ==================================================
echo             MindVoice AI - Initial Setup
echo ==================================================
echo.

REM ==================================================
REM Check Python
REM ==================================================

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not added to PATH.
    pause
    exit /b
)

REM ==================================================
REM Check Node.js
REM ==================================================

node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not added to PATH.
    pause
    exit /b
)

echo Python and Node.js detected.
echo.

REM ==================================================
REM Backend Setup
REM ==================================================

cd /d "%ROOT%backend"

if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo.
echo Installing backend dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Backend setup completed.
echo.

REM ==================================================
REM Frontend Setup
REM ==================================================

cd /d "%ROOT%frontend"

if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install
) else (
    echo node_modules already exists. Skipping installation.
)

echo.
echo ==================================================
echo             Setup Completed Successfully!
echo ==================================================
echo.
echo Next step:
echo Run run.bat
echo.
pause