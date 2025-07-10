@echo off
setlocal enabledelayedexpansion

:: == Initial startup ==
echo 🔧 Starting Lumenorion...
git reset --hard >nul
git pull >nul

:: == Setup virtual environment ==
if not exist ".venv" (
    echo 🐍 Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

:: == Install dependencies ==
echo 📦 Installing requirements...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt >nul

:menu
cls
echo ====================================
echo        🌌  LUMENORION MENU
echo ====================================
echo 1. Generate new dream & reflect
echo 2. Talk to Lumenorion
echo 3. Exit
echo.

set /p choice=Choose an option [1-3]: 

if "!choice!"=="1" (
    echo.
    echo 💤 Generating new dream...
    python main.py
    pause
    goto menu
)

if "!choice!"=="2" (
    echo.
    echo 🧠 Talking to Lumenorion...
    python agent.py
    pause
    goto menu
)

if "!choice!"=="3" (
    echo 🛑 Exiting Lumenorion.
    goto end
)

:: Invalid option
echo ❌ Invalid choice. Please enter 1, 2 or 3.
pause
goto menu

:end
endlocal
