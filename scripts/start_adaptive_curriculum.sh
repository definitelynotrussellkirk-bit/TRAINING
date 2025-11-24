#!/bin/bash
# Start adaptive curriculum orchestrator as background daemon

set -e

BASE_DIR="/path/to/training"
LOG_FILE="$BASE_DIR/logs/curriculum.log"
PID_FILE="$BASE_DIR/control/curriculum.pid"

# Create directories
mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/control"
mkdir -p "$BASE_DIR/eval_sets"
mkdir -p "$BASE_DIR/eval_results"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Adaptive curriculum orchestrator already running (PID: $PID)"
        exit 0
    else
        echo "Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Default generator config
GEN_CONFIG="$BASE_DIR/generators.json"

# Create default config if it doesn't exist
if [ ! -f "$GEN_CONFIG" ]; then
    echo "Creating default generator config..."
    cd "$BASE_DIR"
    python3 -m tools.adaptive_curriculum.cli init-config --output "$GEN_CONFIG"
fi

# Start orchestrator
echo "Starting adaptive curriculum orchestrator..."
cd "$BASE_DIR"

nohup python3 -m tools.adaptive_curriculum.cli start \
    --generators "$GEN_CONFIG" \
    --batch-size 1000 \
    --queue-threshold 2 \
    --check-interval 300 \
    --target-accuracy 0.8 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"

# Wait a moment to check if it started
sleep 2

if ps -p $PID > /dev/null; then
    echo "✓ Adaptive curriculum orchestrator started (PID: $PID)"
    echo "  Log: $LOG_FILE"
    echo "  Config: $GEN_CONFIG"
    echo ""
    echo "Check status:"
    echo "  python3 -m tools.adaptive_curriculum.cli status"
    echo ""
    echo "Stop:"
    echo "  kill \$(cat $PID_FILE)"
else
    echo "✗ Failed to start orchestrator"
    echo "Check log: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
