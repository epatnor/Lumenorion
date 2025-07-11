@echo off
setlocal enabledelayedexpansion

:: == Initial startup ==
echo 🔧 Starting Lumenorion...
echo 🔄 Resetting local changes and pulling latest...
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

:: == Upgrade pip ==
echo 🔼 Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: == Install core dependencies ==
echo 📦 Installing/updating requirements with CUDA support...
if not exist requirements.txt (
    echo ❌ requirements.txt not found!
    pause
    exit /b
)

:: Try CUDA pip install first, fall back to default if it fails
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121 || pip install -r requirements.txt

:: Force compatible NumPy version
pip install -q "numpy<2"

if errorlevel 1 (
    echo ❌ Failed to install requirements.
    pause
    exit /b
)

:: == Ensure vision dependencies ==
echo 🧩 Verifying vision packages (Pillow + timm)...
python -c "import PIL, timm" 2>nul
if errorlevel 1 (
    echo 📥 Installing missing vision packages...
    pip install pillow timm
)

echo.
echo ✅ Environment ready.

:menu
cls
echo.
echo  ⠀⠀⠀⠀⠀⠀  .  *        ✨             *
echo  *       ✨        .     *        ✨     .
echo      .
echo ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
echo ▓                                              ▓
echo ▓   🌿   ╔══════════════════════=═════╗   🌿   ▓
echo ▓   🪨   ║    🌌  LUMENORION GATE    ║   🪨   ▓
echo ▓   🌿   ╚═════════════════════=══════╝   🌿   ▓
echo ▓                                              ▓
echo ▓          🪨 [①] 🌙  Dream and Reflect        ▓
echo ▓          🌿 [②] 🧠  Talk to Lumenorion       ▓
echo ▓          🪨 [③] 🔬  Train LoRA Model         ▓
echo ▓          🌿 [④] 🚪  Exit the Portal          ▓
echo ▓                                              ▓
echo ▓  ✨  Stars shimmer through ancient stone...  ▓
echo ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
echo.
set /p choice=        🌠 Choose your path [1-4]: 


if "!choice!"=="1" (
    echo.
    echo 💤 Generating new dream and reflecting...
    python main.py || echo ❌ Error running main.py
    echo.
    pause
    goto menu
)

if "!choice!"=="2" (
    echo.
    echo 🧠 Talking to Lumenorion...
    python agent.py || echo ❌ Error running agent.py
    echo.
    pause
    goto menu
)

if "!choice!"=="3" (
    echo.
    if exist "train_lora.py" (
        echo 🔬 Training LoRA model...
        python train_lora.py || echo ❌ Failed to train LoRA
    ) else (
        echo ❌ train_lora.py not found!
    )
    echo.
    pause
    goto menu
)

if "!choice!"=="4" (
    echo 🛑 Exiting Lumenorion.
    exit /b
)

:: Invalid option
echo ❌ Invalid choice. Please enter 1 to 4.
echo.
pause
goto menu

:end
endlocal
