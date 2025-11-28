#!/bin/bash
# Monitor Training Output - Shows what's happening with the current training

# Auto-detect base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

echo "=== Training Monitor ==="
echo ""

# Find training process
TRAIN_PID=$(pgrep -f "python3.*train.py" | head -1)

if [ -z "$TRAIN_PID" ]; then
    echo "No training process running"
    echo ""
    echo "Daemon status:"
    if pgrep -f "training_daemon.py" > /dev/null; then
        echo "✓ Daemon is running"
        echo ""
        echo "Last 20 lines of daemon log:"
        tail -20 "$BASE_DIR"/logs/daemon_*.log | tail -20
    else
        echo "✗ Daemon is NOT running"
    fi
    exit 0
fi

echo "✓ Training process found (PID: $TRAIN_PID)"
echo ""

# Show GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""

# Show process info
echo "Process info:"
ps aux | grep $TRAIN_PID | grep -v grep
echo ""

# Try to show stderr/stdout if available
echo "Checking for training output..."
if [ -f "/proc/$TRAIN_PID/fd/1" ]; then
    echo "stdout: (last 50 lines)"
    timeout 2 tail -50 /proc/$TRAIN_PID/fd/1 2>/dev/null || echo "Unable to read stdout"
fi
echo ""

# Show daemon log
echo "Daemon log (last 30 lines):"
tail -30 "$BASE_DIR"/logs/daemon_*.log | tail -30
echo ""

# Show training status if it exists
if [ -f "$BASE_DIR/training_status.json" ]; then
    echo "Training status:"
    cat "$BASE_DIR/training_status.json" | python3 -m json.tool 2>/dev/null || cat "$BASE_DIR/training_status.json"
fi
