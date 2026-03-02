#!/bin/bash
# SolarMind AI — Start both backend and frontend with one command

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "☀️  Starting SolarMind AI..."
echo ""

# Start backend in background
echo "⚡ Starting Backend (port 8000)..."
cd "$PROJECT_DIR/backend"
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend in background
echo "🖥️  Starting Frontend (port 5173)..."
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ SolarMind AI is running!"
echo "   🌐 Dashboard: http://localhost:5173"
echo "   📡 API:       http://localhost:8000"
echo ""
echo "   Press Ctrl+C to stop both servers"
echo ""

# Handle Ctrl+C — kill both processes
trap "echo ''; echo '🛑 Stopping SolarMind AI...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

# Wait for both processes
wait
