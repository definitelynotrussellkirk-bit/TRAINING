#!/bin/bash
# Sync training checkpoints to RTX 3090 for curriculum optimization
# Runs continuously, syncing latest checkpoint every 5 minutes

# Auto-detect base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

TRAINING_CHECKPOINT_DIR="$BASE_DIR/models/current_model"

# Get remote host from config or env var
REMOTE_HOST="${INFERENCE_HOST:-$(python3 -c 'from core.hosts import get_host; print(get_host("3090").host)' 2>/dev/null || echo "inference.local")}"
REMOTE_USER="${INFERENCE_SSH_USER:-$(python3 -c 'from core.hosts import get_host; print(get_host("3090").ssh_user)' 2>/dev/null || echo "$USER")}"
REMOTE_DIR="~/TRAINING/models/current_model"
SYNC_INTERVAL=300  # 5 minutes

echo "=== Checkpoint Sync to 3090 ==="
echo "Source: $TRAINING_CHECKPOINT_DIR"
echo "Destination: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
echo "Sync interval: ${SYNC_INTERVAL}s"
echo ""

# Create remote directory
ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p ~/TRAINING/models/current_model"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing checkpoints..."

    # Sync only essential files (config, model weights, tokenizer)
    # Exclude optimizer states and large intermediate files
    rsync -avz --delete \
        --include="config.json" \
        --include="*.bin" \
        --include="*.safetensors" \
        --include="tokenizer*" \
        --include="special_tokens_map.json" \
        --include="tokenizer_config.json" \
        --include="checkpoint-*/" \
        --exclude="optimizer.pt" \
        --exclude="scheduler.pt" \
        --exclude="trainer_state.json" \
        "$TRAINING_CHECKPOINT_DIR/" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

    if [ $? -eq 0 ]; then
        echo "  ✓ Sync complete"
    else
        echo "  ✗ Sync failed"
    fi

    echo "  Next sync in ${SYNC_INTERVAL}s..."
    sleep $SYNC_INTERVAL
done
