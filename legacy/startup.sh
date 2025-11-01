#!/bin/bash
# StockSense Automated Startup Script for macOS/Linux
# This script starts both the backend and frontend servers

set -e

echo "========================================"
echo "StockSense Startup Script"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.11+ from https://www.python.org/"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed"
    echo "Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

echo "[OK] Python and Node.js detected"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[INFO] Virtual environment not found. Creating one..."
    python3 -m venv .venv
    echo "[OK] Virtual environment created"
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
if ! python -c "import fastapi" &> /dev/null; then
    echo "[INFO] Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "[OK] Python dependencies installed"
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "[INFO] Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    echo "[OK] Frontend dependencies installed"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "[WARNING] .env file not found"
    echo "Please create a .env file with your API keys"
    echo "You can copy .env.example if it exists"
    echo ""
    echo "The application will start but some features may not work"
    sleep 3
fi

echo ""
echo "========================================"
echo "Starting StockSense Services"
echo "========================================"
echo ""
echo "Backend will be available at: http://localhost:8000"
echo "Frontend will be available at: http://localhost:3000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "Services stopped"
    exit 0
}

trap cleanup INT TERM

# Start backend
echo "[INFO] Starting FastAPI backend..."
python -m pipelines.realtime.api &
BACKEND_PID=$!

# Wait for backend to initialize
sleep 3

# Start frontend
echo "[INFO] Starting Next.js frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================"
echo "StockSense Started Successfully!"
echo "========================================"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Open your browser to: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait
