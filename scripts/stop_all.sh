#!/bin/bash
# Stop all Realm of Training services

BASE_DIR="/path/to/training"
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

echo ""
echo "✅ All services stopped."
echo ""
