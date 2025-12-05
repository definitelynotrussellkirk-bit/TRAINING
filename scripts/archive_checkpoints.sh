#!/bin/bash
# archive_checkpoints.sh - Sync checkpoints to Synology NAS
#
# Usage:
#   ./scripts/archive_checkpoints.sh [hero]
#   ./scripts/archive_checkpoints.sh ojas-qwen3-8b
#   ./scripts/archive_checkpoints.sh dio-qwen3-0.6b
#   ./scripts/archive_checkpoints.sh   # Archives all heroes

set -e

BASE_DIR="/home/russ/Desktop/TRAINING"
NAS_MOUNT="/mnt/synology/data/llm_training/checkpoints"
REMOTE_HOST="russ@192.168.88.149"

# Check if NAS is mounted locally (trainer) or needs remote access
if mountpoint -q /mnt/synology/data 2>/dev/null; then
    USE_LOCAL=true
    DEST_BASE="/mnt/synology/data/llm_training/checkpoints"
else
    USE_LOCAL=false
    DEST_BASE="$REMOTE_HOST:$NAS_MOUNT"
fi

archive_hero() {
    local hero="$1"
    local src_dir="$BASE_DIR/campaigns/$hero"
    
    if [ ! -d "$src_dir" ]; then
        echo "WARNING: Campaign dir not found: $src_dir"
        return 1
    fi
    
    echo "=== Archiving $hero ==="
    echo "Source: $src_dir"
    echo "Destination: $DEST_BASE/$hero/"
    
    # Create dest dir if using local mount
    if $USE_LOCAL; then
        mkdir -p "$DEST_BASE/$hero"
    fi
    
    # Rsync only checkpoint directories, not temp files
    rsync -avz --progress \
        --include='*/' \
        --include='checkpoint-*/***' \
        --exclude='*' \
        "$src_dir/" "$DEST_BASE/$hero/"
    
    echo "Done archiving $hero"
}

# Main
if [ -n "$1" ]; then
    # Archive specific hero
    archive_hero "$1"
else
    # Archive all heroes
    for campaign_dir in "$BASE_DIR/campaigns"/*/; do
        hero=$(basename "$campaign_dir")
        archive_hero "$hero"
    done
fi

echo ""
echo "=== Archive complete ==="
if $USE_LOCAL; then
    du -sh "$DEST_BASE"/*/ 2>/dev/null || echo "No checkpoints archived yet"
else
    ssh "$REMOTE_HOST" "du -sh $NAS_MOUNT/*/ 2>/dev/null" || echo "No checkpoints archived yet"
fi
