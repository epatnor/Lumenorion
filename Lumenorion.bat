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
echo 3. 🔬 Train LoRA from memory
echo 4. ⚡ Activate LoRA model
echo 5. Exit
echo.

set /p choice=Choose an option [1-5]: 

if "!choice!"=="1" (
    echo.
    echo 💤 Generating new dream and reflecting...
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
    echo.
    echo 🔄 Exporting dataset and training LoRA...
    python export_lora_dataset.py
    ollama create lumenorion-lora -f train_lora_modelfile
    pause
    goto menu
)

if "!choice!"=="4" (
    echo.
    echo ⚡ Activating LoRA model: lumenorion-lora...
    powershell -Command "Set-Content .\core\model.txt 'lumenorion-lora'"
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
