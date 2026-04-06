@echo off
echo ============================================================
echo  VLSI Interview Agent — Setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Python found. Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Could not create virtual environment.
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies (this may take 5-10 minutes)...
pip install --upgrade pip -q
pip install -r requirements.txt

echo [4/4] Checking for .env file...
if not exist .env (
    copy .env.example .env
    echo.
    echo *** IMPORTANT ***
    echo .env file created from template.
    echo Please open .env in a text editor and fill in your API keys!
    echo.
    echo   GROQ_API_KEY    — from https://console.groq.com (free)
    echo   AWS_ACCESS_KEY_ID   — from AWS IAM console
    echo   AWS_SECRET_ACCESS_KEY — from AWS IAM console
    echo.
    echo After filling in keys, run: python main.py
    echo.
) else (
    echo .env file already exists.
    echo.
    echo Setup complete! Run: python main.py
    echo Then open: http://localhost:8001
    echo.
)

pause
