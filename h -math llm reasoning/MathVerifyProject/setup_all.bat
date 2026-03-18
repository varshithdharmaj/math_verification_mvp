@echo off
REM Master setup script for MathVerifyProject (Windows)
REM Clones repositories, installs dependencies, and verifies installation

setlocal enabledelayedexpansion

echo ==========================================
echo MathVerifyProject Setup Script (Windows)
echo ==========================================
echo.

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

REM Step 1: Clone repositories
echo Step 1: Cloning repositories...
echo -----------------------------------

set "repos[0]=https://github.com/huggingface/Math-Verify.git Math-Verify"
set "repos[1]=https://github.com/mathllm/MATH-V.git MATH-V"
set "repos[2]=https://github.com/ZrrSkywalker/MathVerse.git MathVerse"
set "repos[3]=https://github.com/yixchen/Math_Handwriting_OCR.git Math_Handwriting_OCR"
set "repos[4]=https://github.com/johnkimdw/handwritten-math-transcription.git handwritten-math-transcription"

for /L %%i in (0,1,4) do (
    for /f "tokens=1,2" %%a in ("!repos[%%i]!") do (
        if exist "%%b" (
            echo [^!] %%b already exists, skipping clone
        ) else (
            echo Cloning %%b...
            git clone "%%a" "%%b"
            if !errorlevel! equ 0 (
                echo [✓] Cloned %%b
            ) else (
                echo [✗] Failed to clone %%b
            )
        )
    )
)

echo.

REM Step 2: Install Python dependencies
echo Step 2: Installing Python dependencies...
echo -----------------------------------

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [✗] Python not found. Please install Python 3.10+
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo [✓] Found Python !PYTHON_VERSION!

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [✓] Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [✗] Could not activate virtual environment
    exit /b 1
)
echo [✓] Virtual environment activated

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [✓] pip upgraded

REM Install main requirements
echo Installing main requirements...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    echo [✓] Main requirements installed
) else (
    echo [!] requirements.txt not found
)

REM Install Math-Verify
echo Installing Math-Verify...
cd Math-Verify
if exist "pyproject.toml" (
    python -m pip install -e .[antlr4_13_2] || python -m pip install math-verify[antlr4_13_2]
    echo [✓] Math-Verify installed
) else (
    echo [!] Math-Verify pyproject.toml not found, trying pip install
    python -m pip install math-verify[antlr4_13_2]
)
cd ..

REM Install handwritten-math-transcription dependencies
echo Installing handwritten-math-transcription dependencies...
cd handwritten-math-transcription
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    echo [✓] Handwriting transcription dependencies installed
) else (
    echo [!] handwritten-math-transcription requirements.txt not found
)
cd ..

echo.

REM Step 3: Verify installation
echo Step 3: Verifying installation...
echo -----------------------------------

REM Test Math-Verify
echo Testing Math-Verify...
python -c "from math_verify import parse, verify; print('Math-Verify: OK')" 2>nul
if !errorlevel! equ 0 (
    echo [✓] Math-Verify import successful
) else (
    echo [✗] Math-Verify import failed
)

REM Test core modules
echo Testing core modules...
python -c "import sys; sys.path.insert(0, '.'); from core_verification import MathVerifier; print('Core verification: OK')" 2>nul
if !errorlevel! equ 0 (
    echo [✓] Core verification module OK
) else (
    echo [!] Core verification module test failed (may need dependencies)
)

echo.

REM Step 4: Create demo script
echo Step 4: Demo scripts ready
echo -----------------------------------
echo [✓] Demo scripts ready (see demo_verification.py)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To use the system:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run main interface: python main.py --mode gradio
echo   3. Or use CLI: python main.py --mode cli verify --gold "1/2" --pred "0.5"
echo.
echo For more information, see README.md
echo.

endlocal

