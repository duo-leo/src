@echo off
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    SemiAutoFixer Setup                       ║
echo ║                                                              ║
echo ║  This script will set up SemiAutoFixer on Windows           ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Run the Python setup script
echo 🚀 Running setup script...
python setup.py

if errorlevel 1 (
    echo.
    echo ❌ Setup failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Run: python how_to_use.py
echo 2. Prepare your data files
echo 3. Run: python runner.py
echo.
pause
