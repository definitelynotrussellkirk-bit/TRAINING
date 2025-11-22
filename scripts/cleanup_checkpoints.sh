#!/bin/bash
# Checkpoint Cleanup Script - 3-Tier Graduated Retention Policy
#
# Strategy:
#   Tier 1: Keep last 20 checkpoints (every 100 steps) - Dense recent
#   Tier 2: Keep every other checkpoint for next batch (every 200 steps) - Medium
#   Tier 3: Keep every 1000th checkpoint for oldest (every 1000 steps) - Sparse
#
# Example with checkpoints 100-16700:
#   Tier 1 (16500-16700): 16500, 16600, 16700 ... (last 20, all)
#   Tier 2 (14500-16400): 14600, 14800, 15000 ... 16000, 16200, 16400 (every 200)
#   Tier 3 (100-14400):   1000, 2000, 3000 ... 13000, 14000 (every 1000)
#   Delete: All others

CHECKPOINT_DIR="current_model"
KEEP_LAST=20          # Tier 1: Keep last N checkpoints (100% density)
TIER2_COUNT=20        # Tier 2: Keep next N checkpoints at 50% density (every other)
TIER3_INTERVAL=1000   # Tier 3: Keep every Nth step (sparse oldest history)

cd "$(dirname "$0")"

echo "🧹 Checkpoint Cleanup Script (3-Tier Graduated Retention)"
echo "==========================================================="
echo "Directory: $CHECKPOINT_DIR"
echo "Policy:"
echo "  Tier 1 (recent): Keep last $KEEP_LAST checkpoints (every 100 steps)"
echo "  Tier 2 (medium): Keep next $TIER2_COUNT checkpoints (every 200 steps)"
echo "  Tier 3 (oldest): Keep every ${TIER3_INTERVAL}th step"
echo ""

# Get all checkpoints sorted by step number
ALL_CHECKPOINTS=$(ls -1 "$CHECKPOINT_DIR" | grep "checkpoint-" | sort -V)
TOTAL=$(echo "$ALL_CHECKPOINTS" | wc -l)

echo "Found: $TOTAL checkpoints"

if [ $TOTAL -le $KEEP_LAST ]; then
    echo "✓ No cleanup needed (have $TOTAL, keeping $KEEP_LAST)"
    exit 0
fi

# Separate into three tiers
TIER1_CHECKPOINTS=$(echo "$ALL_CHECKPOINTS" | tail -n $KEEP_LAST)
REMAINING=$(echo "$ALL_CHECKPOINTS" | head -n -$KEEP_LAST)

if [ $(echo "$REMAINING" | wc -l) -gt $TIER2_COUNT ]; then
    TIER2_CHECKPOINTS=$(echo "$REMAINING" | tail -n $TIER2_COUNT)
    TIER3_CHECKPOINTS=$(echo "$REMAINING" | head -n -$TIER2_COUNT)
else
    TIER2_CHECKPOINTS="$REMAINING"
    TIER3_CHECKPOINTS=""
fi

echo ""
echo "Analyzing checkpoints..."
echo "  Tier 1 (keep all): $(echo "$TIER1_CHECKPOINTS" | wc -l)"
echo "  Tier 2 (keep 50%): $(echo "$TIER2_CHECKPOINTS" | wc -l)"
if [ -n "$TIER3_CHECKPOINTS" ]; then
    echo "  Tier 3 (keep 1:$TIER3_INTERVAL): $(echo "$TIER3_CHECKPOINTS" | wc -l)"
fi
echo ""

# Build list of checkpoints to keep and delete
KEEP_LIST=()
DELETE_LIST=()

# Tier 1: Keep all recent checkpoints
while IFS= read -r checkpoint; do
    [ -n "$checkpoint" ] && KEEP_LIST+=("$checkpoint")
done <<< "$TIER1_CHECKPOINTS"

# Tier 2: Keep every other checkpoint (every 200 steps)
TIER2_INDEX=0
while IFS= read -r checkpoint; do
    [ -z "$checkpoint" ] && continue
    STEP=$(echo "$checkpoint" | sed 's/checkpoint-//')
    # Keep if step is even hundred (14600, 14800, 15000 not 14700, 14900, 15100)
    LAST_DIGIT=$(echo "$STEP" | rev | cut -c1-3 | rev)
    if [ $((LAST_DIGIT % 200)) -eq 0 ]; then
        KEEP_LIST+=("$checkpoint")
    else
        DELETE_LIST+=("$checkpoint")
    fi
done <<< "$TIER2_CHECKPOINTS"

# Tier 3: Keep every 1000th checkpoint
if [ -n "$TIER3_CHECKPOINTS" ]; then
    while IFS= read -r checkpoint; do
        [ -z "$checkpoint" ] && continue
        STEP=$(echo "$checkpoint" | sed 's/checkpoint-//')
        # Keep if divisible by TIER3_INTERVAL (1000, 2000, 3000...)
        if [ $((STEP % TIER3_INTERVAL)) -eq 0 ]; then
            KEEP_LIST+=("$checkpoint")
        else
            DELETE_LIST+=("$checkpoint")
        fi
    done <<< "$TIER3_CHECKPOINTS"
fi

# Calculate space to be freed
SPACE_TO_FREE=0
for checkpoint in "${DELETE_LIST[@]}"; do
    SIZE=$(du -sb "$CHECKPOINT_DIR/$checkpoint" 2>/dev/null | cut -f1)
    if [ -n "$SIZE" ]; then
        SPACE_TO_FREE=$((SPACE_TO_FREE + SIZE))
    fi
done

SPACE_GB=$(echo "scale=2; $SPACE_TO_FREE / 1024 / 1024 / 1024" | bc)

echo "Summary:"
echo "  Will keep: ${#KEEP_LIST[@]} checkpoints"
echo "  Will delete: ${#DELETE_LIST[@]} checkpoints"
echo "  Space to free: ${SPACE_GB} GB"
echo ""

# Show what will be kept
echo "Checkpoints to keep:"
if [ -n "$TIER1_CHECKPOINTS" ]; then
    T1_START=$(echo "$TIER1_CHECKPOINTS" | head -n 1)
    T1_END=$(echo "$TIER1_CHECKPOINTS" | tail -n 1)
    echo "  Tier 1: $T1_START → $T1_END (all)"
fi
if [ -n "$TIER2_CHECKPOINTS" ]; then
    T2_KEPT=$(printf '%s\n' "${KEEP_LIST[@]}" | grep -F "$TIER2_CHECKPOINTS" | wc -l || echo 0)
    if [ $T2_KEPT -gt 0 ]; then
        T2_FIRST=$(printf '%s\n' "${KEEP_LIST[@]}" | grep -F "$TIER2_CHECKPOINTS" | head -n 1 || echo "")
        T2_LAST=$(printf '%s\n' "${KEEP_LIST[@]}" | grep -F "$TIER2_CHECKPOINTS" | tail -n 1 || echo "")
        echo "  Tier 2: $T2_FIRST → $T2_LAST (every 200, $T2_KEPT kept)"
    fi
fi
if [ -n "$TIER3_CHECKPOINTS" ]; then
    T3_KEPT=$(printf '%s\n' "${KEEP_LIST[@]}" | grep -F "$TIER3_CHECKPOINTS" | wc -l || echo 0)
    if [ $T3_KEPT -gt 0 ]; then
        T3_FIRST=$(printf '%s\n' "${KEEP_LIST[@]}" | grep -F "$TIER3_CHECKPOINTS" | head -n 1 || echo "")
        T3_LAST=$(printf '%s\n' "${KEEP_LIST[@]}" | grep -F "$TIER3_CHECKPOINTS" | tail -n 1 || echo "")
        echo "  Tier 3: $T3_FIRST → $T3_LAST (every ${TIER3_INTERVAL}, $T3_KEPT kept)"
    fi
fi
echo ""

if [ ${#DELETE_LIST[@]} -eq 0 ]; then
    echo "✓ No checkpoints to delete"
    exit 0
fi

# Confirm before deletion
read -p "Proceed with deletion? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Delete checkpoints
echo "Deleting checkpoints..."
DELETED=0
for checkpoint in "${DELETE_LIST[@]}"; do
    echo "  Deleting: $checkpoint"
    rm -rf "$CHECKPOINT_DIR/$checkpoint"
    DELETED=$((DELETED + 1))
done

echo ""
echo "✓ Deleted $DELETED checkpoints"
echo "✓ Freed approximately ${SPACE_GB} GB"
echo ""

# Show remaining checkpoints
REMAINING=$(ls -1 "$CHECKPOINT_DIR" | grep -c "checkpoint-")
echo "Remaining: $REMAINING checkpoints"

# Show range
OLDEST=$(ls -1 "$CHECKPOINT_DIR" | grep "checkpoint-" | sort -V | head -n 1)
NEWEST=$(ls -1 "$CHECKPOINT_DIR" | grep "checkpoint-" | sort -V | tail -n 1)
echo "Range: $OLDEST → $NEWEST"
echo ""
echo "Retention policy applied:"
echo "  ✓ Tier 1 (most recent): 100% density (every 100 steps)"
echo "  ✓ Tier 2 (medium): 50% density (every 200 steps)"
echo "  ✓ Tier 3 (oldest): Sparse (every ${TIER3_INTERVAL} steps)"
