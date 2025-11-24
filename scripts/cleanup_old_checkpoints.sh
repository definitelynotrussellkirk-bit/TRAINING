#!/bin/bash
# Auto-cleanup: Keep only latest 10 checkpoints
cd /path/to/training
KEEP=10
CHECKPOINTS=$(ls -1d current_model/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
TOTAL=$(echo "$CHECKPOINTS" | wc -l)
DELETE=$((TOTAL - KEEP))

if [ $DELETE -gt 0 ]; then
    echo "$(date): Found $TOTAL checkpoints, deleting oldest $DELETE"
    echo "$CHECKPOINTS" | head -n $DELETE | xargs rm -rf
    echo "$(date): Cleanup complete. Kept latest $KEEP checkpoints"
else
    echo "$(date): Only $TOTAL checkpoints exist, nothing to delete"
fi
