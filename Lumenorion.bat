@echo off
setlocal enabledelayedexpansion

:: == Initial startup ==
echo 🔧 Starting Lumenorion...
git reset --hard >nul 2>&1
git pull >nul 2>&1

:: == Check Python ==
where python >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python and try again.
    pause
    exit /b
)

:: == Setup virtual environment ==
if not exist ".venv" (
    echo 🐍 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment.
        pause
        exit /b
    )
)

:: == Activate venv ==
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ❌ Could not find venv activation script.
    pause
    exit /b
)

:: == Install dependencies ==
echo 📦 Installing requirements...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo ❌ Failed to install requirements.
    pause
    exit /b
)

:menu
cls
echo ====================================
echo        🌌  LUMENORION MENU
echo ====================================
echo 1. Generate new dream & reflect
echo 2. Talk to Lumenorion
echo 3. 🔬 Train LoRA from memory
echo 4. ⚡ Activate LoRA model
echo 5. Exit
echo.

set /p choice=Choose an option [1-5]: 

if "!choice!"=="1" (
    echo.
    echo 💤 Generating new dream and reflecting...
    python main.py || echo ❌ Error running main.py
    pause
    goto menu
)

if "!choice!"=="2" (
    echo.
    echo 🧠 Talking to Lumenorion...
    python agent.py || echo ❌ Error running agent.py
    pause
    goto menu
)

if "!choice!"=="3" (
    echo.
    echo 🔄 Exporting dataset and training LoRA...
    python logs\export_lora_dataset.py || echo ❌ Failed to export dataset.
    ollama create lumenorion-lora -f train_lora_modelfile || echo ❌ Failed to train LoRA model.
    pause
    goto menu
)

if "!choice!"=="4" (
    echo.
    echo ⚡ Activating LoRA model: lumenorion-lora...
    powershell -Command "Set-Content .\core\model.txt 'lumenorion-lora'" || echo ❌ Failed to write model.txt
    echo ✅ Model switched. Restart app to take effect.
    pause
    goto menu
)

if "!choice!"=="5" (
    echo 🛑 Exiting Lumenorion.
    goto end
)

:: Invalid option
echo ❌ Invalid choice. Please enter 1 to 5.
pause
goto menu

:end
endlocal
