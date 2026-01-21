@echo off
echo Starting MVM2 System - All Services
echo =====================================
echo.

echo [1/4] Starting OCR Service (Port 8001)...
start "OCR Service" cmd /k "cd /d %~dp0 && python services\ocr_service.py"
timeout /t 3 /nobreak >nul

echo [2/4] Starting SymPy Service (Port 8005)...
start "SymPy Service" cmd /k "cd /d %~dp0 && python services\sympy_service.py"
timeout /t 3 /nobreak >nul

echo [3/4] Starting LLM Service (Port 8003)...
start "LLM Service" cmd /k "cd /d %~dp0 && python services\llm_service.py"
timeout /t 3 /nobreak >nul

echo [4/4] Starting Streamlit Dashboard (Port 8501)...
echo.
echo =====================================
echo All services started!
echo =====================================
echo.
echo Access the dashboard at: http://localhost:8501
echo.
streamlit run app.py
