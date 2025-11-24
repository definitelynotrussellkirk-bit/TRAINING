#!/bin/bash
#
# Start Self-Correction Loop
# Continuous learning from model mistakes
#

set -e

TRAINING_DIR="/path/to/training"
cd "$TRAINING_DIR"

# Create queue directories
mkdir -p queue/unvalidated
mkdir -p queue/corrections
mkdir -p logs/error_patterns

echo "================================================================================"
echo "🧠 STARTING SELF-CORRECTION LOOP"
echo "================================================================================"
echo ""

# Check if already running
if pgrep -f "self_correction_loop.py" > /dev/null; then
    echo "⚠️  Self-correction loop already running"
    echo "   Kill with: pkill -f self_correction_loop.py"
    exit 1
fi

echo "🔄 Starting self-correction loop..."
nohup python3 monitoring/self_correction_loop.py \
    --continuous \
    --interval 300 \
    --error-threshold 50 \
    > logs/self_correction.log 2>&1 &

LOOP_PID=$!

sleep 2

if ps -p $LOOP_PID > /dev/null; then
    echo "✅ Self-correction loop started (PID: $LOOP_PID)"
else
    echo "❌ Failed to start self-correction loop"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ SELF-CORRECTION LOOP RUNNING"
echo "================================================================================"
echo ""
echo "📊 What it does:"
echo "   - Watches queue/unvalidated/ for new data"
echo "   - Tests with checkpoint on 3090"
echo "   - Captures wrong answers"
echo "   - Analyzes errors (What went wrong?)"
echo "   - Generates correction training examples"
echo "   - Mines error patterns"
echo "   - Saves corrections to queue/corrections/"
echo ""
echo "📁 Directories:"
echo "   Input:  queue/unvalidated/"
echo "   Output: queue/corrections/"
echo "   Valid:  queue/normal/"
echo "   Logs:   logs/error_patterns/"
echo ""
echo "📊 View logs:"
echo "   tail -f logs/self_correction.log"
echo ""
echo "🛑 Stop:"
echo "   pkill -f self_correction_loop.py"
echo ""
echo "================================================================================"
