#!/bin/bash
# Stop all Realm of Training services

# Auto-detect base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$BASE_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           REALM OF TRAINING - Shutdown                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Stop The Weaver first
WEAVER_PID_FILE="$BASE_DIR/.pids/weaver.pid"
if [ -f "$WEAVER_PID_FILE" ]; then
    PID=$(cat "$WEAVER_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "🧵 Stopping The Weaver (PID: $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$WEAVER_PID_FILE"
fi

# Stop other known services
echo "🛑 Stopping services..."

# Training daemon
if [ -f "$BASE_DIR/.daemon.pid" ]; then
    PID=$(cat "$BASE_DIR/.daemon.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo "   Stopping training daemon..."
        kill "$PID" 2>/dev/null || true
    fi
    rm -f "$BASE_DIR/.daemon.pid"
fi

# Tavern
if [ -f "$BASE_DIR/.pids/tavern.pid" ]; then
    PID=$(cat "$BASE_DIR/.pids/tavern.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo "   Stopping Tavern..."
        kill "$PID" 2>/dev/null || true
    fi
    rm -f "$BASE_DIR/.pids/tavern.pid"
fi

# VaultKeeper
if [ -f "$BASE_DIR/.pids/vault.pid" ]; then
    PID=$(cat "$BASE_DIR/.pids/vault.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo "   Stopping VaultKeeper..."
        kill "$PID" 2>/dev/null || true
    fi
    rm -f "$BASE_DIR/.pids/vault.pid"
fi

# Kill by port as fallback
echo "🔌 Releasing ports..."
fuser -k 8888/tcp 2>/dev/null || true
fuser -k 8767/tcp 2>/dev/null || true

# Kill any lingering training processes using GPU
echo "🔫 Killing lingering training processes..."
# Kill by name pattern - training_daemon specifically
pkill -f "training_daemon.py" 2>/dev/null || true
pkill -f "train.py.*--dataset" 2>/dev/null || true

# Give GPU time to release
sleep 2

# Verify GPU is clear
if command -v nvidia-smi &>/dev/null; then
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_PROCS" -gt 0 ]; then
        echo "⚠️  Warning: $GPU_PROCS process(es) still using GPU"
        nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null || true
    else
        echo "✅ GPU is clear"
    fi
fi

echo ""
echo "✅ All services stopped."
echo ""
