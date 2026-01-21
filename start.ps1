# Quick Start Script for MVM²
# This script helps you start all services easily

Write-Host "🔢 MVM² - Multi-Modal Math Verifier" -ForegroundColor Cyan
Write-Host "VNR VJIET Major Project 2025" -ForegroundColor Gray
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "⚠️  Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

# Check if requirements are installed
Write-Host "📋 Checking dependencies..." -ForegroundColor Cyan
$pip_list = pip list
if ($pip_list -notmatch "streamlit") {
    Write-Host "⚠️  Dependencies not installed. Installing..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "🚀 Starting MVM² System..." -ForegroundColor Green
Write-Host ""
Write-Host "Choose an option:" -ForegroundColor Yellow
Write-Host "1. Start Full System (4 services in separate windows)"
Write-Host "2. Start Dashboard Only (quick demo)"
Write-Host "3. Exit"
Write-Host ""

$choice = Read-Host "Enter your choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host "Starting all services..." -ForegroundColor Cyan
        
        # Start OCR Service
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; python services\ocr_service.py"
        Start-Sleep -Seconds 2
        
        # Start SymPy Service
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; python services\sympy_service.py"
        Start-Sleep -Seconds 2
        
        # Start LLM Service
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; python services\llm_service.py"
        Start-Sleep -Seconds 2
        
        # Start Streamlit Dashboard
        Write-Host "✅ All microservices started!" -ForegroundColor Green
        Write-Host "🌐 Starting dashboard..." -ForegroundColor Cyan
        streamlit run app.py
    }
    "2" {
        Write-Host "Starting dashboard only..." -ForegroundColor Cyan
        streamlit run app.py
    }
    "3" {
        Write-Host "Goodbye! 👋" -ForegroundColor Gray
        exit
    }
    default {
        Write-Host "Invalid choice. Please run the script again." -ForegroundColor Red
    }
}
