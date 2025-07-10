@echo off
setlocal

echo 🔧 Checking environment...

:: Rensa lokala ändringar och dra senaste från GitHub
echo 🌐 Syncing with GitHub (resetting local changes)...
git reset --hard >nul
git pull

:: Skapa virtuell miljö om den inte finns
if not exist ".venv" (
    echo 🐍 Creating virtual environment...
    python -m venv .venv
)

:: Aktivera miljön
call .venv\Scripts\activate.bat

:: Installera beroenden
echo 📦 Installing required packages...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

:: Kör huvudprogrammet
echo 🌙 Launching Lumenorion dream loop...
python main.py

endlocal
pause
