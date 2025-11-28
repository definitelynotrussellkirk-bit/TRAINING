#!/bin/bash
# Start the Realm of Training
#
# The Weaver manages all daemon threads:
# - Training daemon (the heart)
# - Tavern server (the face)
# - VaultKeeper (the memory)
# - Data flow (the fuel)

set -e

BASE_DIR="/path/to/training"
cd "$BASE_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           REALM OF TRAINING - Startup                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Weaver is already running
WEAVER_PID_FILE="$BASE_DIR/.pids/weaver.pid"
if [ -f "$WEAVER_PID_FILE" ]; then
    PID=$(cat "$WEAVER_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "⚠️  The Weaver is already running (PID: $PID)"
        echo "   Use: python3 weaver/weaver.py --status"
        exit 0
    else
        echo "🧹 Cleaning stale PID file..."
        rm -f "$WEAVER_PID_FILE"
    fi
fi

# Ensure log directory exists
mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/.pids"

# First, do a single check-and-mend to start all services
echo "🔍 Checking tapestry and starting services..."
python3 weaver/weaver.py 2>&1 | head -30

echo ""
echo "🧵 Starting The Weaver daemon..."
nohup python3 weaver/weaver.py --daemon > "$BASE_DIR/logs/weaver.log" 2>&1 &
WEAVER_PID=$!
echo "$WEAVER_PID" > "$WEAVER_PID_FILE"
sleep 2

# Verify
if kill -0 "$WEAVER_PID" 2>/dev/null; then
    echo "✅ The Weaver is now watching over the realm (PID: $WEAVER_PID)"
else
    echo "❌ The Weaver failed to start - check logs/weaver.log"
    exit 1
fi

echo ""
echo "🎮 Access points:"
echo "   Tavern (Game UI):  http://localhost:8888"
echo "   VaultKeeper API:   http://localhost:8767"
echo ""
echo "📊 Check status:"
echo "   python3 weaver/weaver.py --status"
echo ""
echo "🛑 Stop all:"
echo "   ./scripts/stop_all.sh"
echo ""
