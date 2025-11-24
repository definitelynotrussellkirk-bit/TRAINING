#!/bin/bash
# Start Curriculum Progression Optimizer on RTX 3090
# Monitors checkpoints and A/B tests different curriculum strategies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting Curriculum Progression Optimizer..."
echo "Base dir: $BASE_DIR"

# Check if already running
if pgrep -f "python3.*curriculum_optimizer.py" > /dev/null; then
    echo "⚠️  Curriculum optimizer already running!"
    echo "PIDs:"
    pgrep -af "python3.*curriculum_optimizer.py"
    exit 1
fi

# Check for validation datasets
VALIDATION_DIR="$BASE_DIR/data/validation"
if [ ! -f "$VALIDATION_DIR/easy.jsonl" ] || \
   [ ! -f "$VALIDATION_DIR/medium.jsonl" ] || \
   [ ! -f "$VALIDATION_DIR/hard.jsonl" ]; then
    echo "❌ Missing validation datasets!"
    echo "Required files:"
    echo "  - $VALIDATION_DIR/easy.jsonl"
    echo "  - $VALIDATION_DIR/medium.jsonl"
    echo "  - $VALIDATION_DIR/hard.jsonl"
    exit 1
fi

# Start optimizer in background
LOG_FILE="$BASE_DIR/logs/curriculum_optimizer.log"
mkdir -p "$(dirname "$LOG_FILE")"

nohup python3 "$BASE_DIR/monitoring/curriculum_optimizer.py" \
    --base-dir "$BASE_DIR" \
    --interval 300 \
    --samples 50 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✓ Started curriculum optimizer (PID: $PID)"
echo "  Log: $LOG_FILE"
echo "  Results: $BASE_DIR/status/curriculum_optimization.json"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  python3 $BASE_DIR/monitoring/curriculum_optimizer.py --status"
