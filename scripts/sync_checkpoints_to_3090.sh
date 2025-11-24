#!/bin/bash
# Sync training checkpoints to RTX 3090 for curriculum optimization
# Runs continuously, syncing latest checkpoint every 5 minutes

TRAINING_CHECKPOINT_DIR="/path/to/training/models/current_model"
REMOTE_USER="user"
REMOTE_HOST="192.168.x.x"
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
