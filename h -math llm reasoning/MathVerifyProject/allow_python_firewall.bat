@echo off
echo ========================================
echo Allow Python Through Windows Firewall
echo ========================================
echo.
echo This will add Python to Windows Firewall exceptions
echo.
pause

:: Find Python executable
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python313\python.exe

if not exist "%PYTHON_PATH%" (
    echo Python not found at: %PYTHON_PATH%
    echo Please find your Python.exe location and update this script
    pause
    exit /b 1
)

echo Found Python at: %PYTHON_PATH%
echo.
echo Adding firewall rule...
echo.

:: Add firewall rule for Python
netsh advfirewall firewall add rule name="Python - MathVerify" dir=in action=allow program="%PYTHON_PATH%" enable=yes

if %errorlevel% equ 0 (
    echo.
    echo SUCCESS! Python has been allowed through firewall.
    echo.
    echo You can now try: python simple_launch.py
) else (
    echo.
    echo ERROR: Could not add firewall rule.
    echo You may need to run this as Administrator.
    echo Right-click and select "Run as administrator"
)

echo.
pause

