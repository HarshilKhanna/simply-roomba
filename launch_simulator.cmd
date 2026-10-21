@echo off
setlocal
cd /d "%~dp0"
echo Opening the visual cleaning robot simulator...
python -B visual_sim.py --gui %*
if errorlevel 1 (
    echo.
    echo If dependencies are missing, run: python -m pip install -r requirements.txt
    pause
)
