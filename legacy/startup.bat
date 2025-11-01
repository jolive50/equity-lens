@echo off
REM StockSense Automated Startup Script for Windows
REM This script starts both the backend and frontend servers

echo ========================================
echo StockSense Startup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

echo [OK] Python and Node.js detected
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv .venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if requirements are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing Python dependencies...
    pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install Python dependencies
        pause
        exit /b 1
    )
    echo [OK] Python dependencies installed
)

REM Check if frontend dependencies are installed
if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo [OK] Frontend dependencies installed
)

REM Check if .env file exists
if not exist ".env" (
    echo [WARNING] .env file not found
    echo Please create a .env file with your API keys
    echo You can copy .env.example if it exists
    echo.
    echo The application will start but some features may not work
    timeout /t 3
)

echo.
echo ========================================
echo Starting StockSense Services
echo ========================================
echo.
echo Backend will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:3000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C in either window to stop the services
echo.

REM Start backend in a new window
echo [INFO] Starting FastAPI backend...
start "StockSense Backend" cmd /k "call .venv\Scripts\activate.bat && python -m pipelines.realtime.api"

REM Wait a moment for backend to initialize
timeout /t 3 /nobreak >nul

REM Start frontend in a new window
echo [INFO] Starting Next.js frontend...
start "StockSense Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo StockSense Started Successfully!
echo ========================================
echo.
echo Two terminal windows have been opened:
echo   1. Backend server (FastAPI)
echo   2. Frontend server (Next.js)
echo.
echo Open your browser to: http://localhost:3000
echo.
echo To stop the servers, close both terminal windows
echo or press Ctrl+C in each window
echo.
pause
