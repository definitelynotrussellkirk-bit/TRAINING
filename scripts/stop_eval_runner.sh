#!/bin/bash
# Stop the Eval Runner daemon

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$BASE_DIR"

PID_FILE="$BASE_DIR/.pids/eval_runner.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  Eval Runner not running (no PID file)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "⚠️  Eval Runner not running (stale PID)"
    rm -f "$PID_FILE"
    exit 0
fi

echo "🛑 Stopping Eval Runner (PID: $PID)..."
kill "$PID"

# Wait for graceful shutdown
for i in {1..10}; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "✅ Eval Runner stopped"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "⚠️  Force killing Eval Runner..."
kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "✅ Eval Runner stopped (forced)"
