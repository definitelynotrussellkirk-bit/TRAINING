#!/bin/bash
# Start the Eval Runner daemon
#
# Monitors evaluation queues and processes evals in the background

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$BASE_DIR"

PID_FILE="$BASE_DIR/.pids/eval_runner.pid"
LOG_FILE="$BASE_DIR/logs/eval_runner.log"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "✅ Eval Runner already running (PID: $PID)"
        echo "   Check status: python3 core/eval_runner.py --status"
        echo "   View logs: tail -f $LOG_FILE"
        exit 0
    else
        echo "🧹 Cleaning stale PID file..."
        rm -f "$PID_FILE"
    fi
fi

# Ensure directories exist
mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/.pids"

# Get inference server config from hosts.json (or use defaults)
INFERENCE_HOST="${INFERENCE_HOST:-192.168.x.x}"
INFERENCE_PORT="${INFERENCE_PORT:-8765}"

echo "🔬 Starting Eval Runner daemon..."
echo "   Inference: http://$INFERENCE_HOST:$INFERENCE_PORT"
echo "   Log file: $LOG_FILE"

# Start daemon
nohup python3 core/eval_runner.py \
    --daemon \
    --inference-host "$INFERENCE_HOST" \
    --inference-port "$INFERENCE_PORT" \
    --interval 60 \
    > "$LOG_FILE" 2>&1 &

EVAL_PID=$!
echo "$EVAL_PID" > "$PID_FILE"

# Verify
sleep 2
if kill -0 "$EVAL_PID" 2>/dev/null; then
    echo "✅ Eval Runner started (PID: $EVAL_PID)"
    echo ""
    echo "📊 Monitor:"
    echo "   Status: python3 core/eval_runner.py --status"
    echo "   Logs:   tail -f $LOG_FILE"
    echo ""
    echo "🛑 Stop:"
    echo "   kill $EVAL_PID"
    echo "   Or: ./scripts/stop_eval_runner.sh"
else
    echo "❌ Eval Runner failed to start - check $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
