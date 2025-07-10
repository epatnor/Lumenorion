@echo off
setlocal

echo 🔧 Starting Lumenorion...
git reset --hard >nul
git pull

:: Setup environment
if not exist ".venv" (
    echo 🐍 Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

echo 📦 Installing requirements...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

:menu
cls
echo ====================================
echo         🌌 LUMENORION MENU
echo ====================================
echo 1. Generate new dream & reflect
echo 2. Talk to Lumenorion
echo 3. Exit
echo.

set /p choice=Choose an option [1-3]:

if "%choice%"=="1" (
    echo.
    echo 💤 Dreaming and reflecting...
    python main.py
    pause
    goto menu
) else if "%choice%"=="2" (
    echo.
    echo 🧠 Connecting to Lumenorion...
    python agent.py
    pause
    goto menu
) else if "%choice%"=="3" (
    echo 🛑 Exiting.
    exit
) else (
    echo ❌ Invalid choice.
    pause
    goto menu
)

endlocal
