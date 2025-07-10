@echo off
setlocal

echo 🔧 Checking environment...

:: Skapa virtuell miljö om den inte finns
if not exist ".venv" (
    echo 🐍 Creating virtual environment...
    python -m venv .venv
)

:: Aktivera miljön
call .venv\Scripts\activate.bat

:: Installera beroenden om de inte redan finns
echo 📦 Installing required packages...
pip install --upgrade pip >nul
pip install -r requirements.txt

:: Kör huvudprogrammet
echo 🌙 Launching Lumenorion dream loop...
python main.py

endlocal
pause
