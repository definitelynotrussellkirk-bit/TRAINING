#!/bin/bash
#
# Quick Training Summary
#
# Shows current training status in a nice format.
# Run anytime to see progress.

echo "================================================================================"
echo "                        TRAINING STATUS SUMMARY"
echo "================================================================================"
echo ""

STATUS_FILE="status/training_status.json"

if [ ! -f "$STATUS_FILE" ]; then
    echo "❌ No training status file found"
    echo "   Looking for: $STATUS_FILE"
    exit 1
fi

# Extract key metrics using jq
CURRENT_STEP=$(jq -r '.current_step // 0' "$STATUS_FILE")
TOTAL_STEPS=$(jq -r '.total_steps // 0' "$STATUS_FILE")
LOSS=$(jq -r '.loss // 0' "$STATUS_FILE")
CURRENT_FILE=$(jq -r '.current_file // "N/A"' "$STATUS_FILE")
STATUS=$(jq -r '.status // "unknown"' "$STATUS_FILE")
TOTAL_EVALS=$(jq -r '.total_evals // 0' "$STATUS_FILE")

# Calculate progress percentage
if [ "$TOTAL_STEPS" -gt 0 ]; then
    PROGRESS=$(echo "scale=1; $CURRENT_STEP * 100 / $TOTAL_STEPS" | bc)
else
    PROGRESS="0.0"
fi

echo "📊 Training Progress"
echo "   Status: $STATUS"
echo "   Step: $CURRENT_STEP / $TOTAL_STEPS ($PROGRESS%)"
echo "   Loss: $LOSS"
echo "   File: $CURRENT_FILE"
echo ""

echo "🔍 Evaluations"
echo "   Total evaluations: $TOTAL_EVALS"

# Calculate accuracy from recent examples
RECENT_TOTAL=$(jq '.recent_examples | length' "$STATUS_FILE")
RECENT_CORRECT=$(jq '[.recent_examples[] | select(.matches == true)] | length' "$STATUS_FILE")

if [ "$RECENT_TOTAL" -gt 0 ]; then
    RECENT_ACCURACY=$(echo "scale=1; $RECENT_CORRECT * 100 / $RECENT_TOTAL" | bc)
    echo "   Recent accuracy: $RECENT_CORRECT / $RECENT_TOTAL ($RECENT_ACCURACY%)"
else
    echo "   Recent accuracy: No data yet"
fi

echo ""

# GPU info
echo "🎮 GPU Status"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk '{printf "   Utilization: %s%%\n   Memory: %s MB / %s MB\n", $1, $2, $3}'

echo ""

# Model size
if [ -f "current_model/adapter_model.safetensors" ]; then
    ADAPTER_SIZE=$(du -h current_model/adapter_model.safetensors | awk '{print $1}')
    echo "💾 Current Adapter"
    echo "   Size: $ADAPTER_SIZE"
    echo ""
fi

# Time estimate
if [ "$TOTAL_STEPS" -gt "$CURRENT_STEP" ] && [ "$CURRENT_STEP" -gt 0 ]; then
    REMAINING=$((TOTAL_STEPS - CURRENT_STEP))
    # Assume ~6 seconds per step
    REMAINING_SECONDS=$((REMAINING * 6))
    REMAINING_HOURS=$((REMAINING_SECONDS / 3600))
    REMAINING_MINS=$(((REMAINING_SECONDS % 3600) / 60))

    echo "⏱️  Time Estimate"
    echo "   Remaining: ~${REMAINING_HOURS}h ${REMAINING_MINS}m"
    echo ""
fi

echo "================================================================================"
