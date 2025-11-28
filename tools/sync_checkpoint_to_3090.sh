#!/bin/bash
# Sync trained checkpoint to 3090 for testing

set -e

# Get training dir from env or auto-detect
TRAINING_DIR="${TRAINING_BASE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
REMOTE_HOST="${INFERENCE_HOST:-localhost}"
REMOTE_DIR="~/llm/models"

# Parse arguments
CHECKPOINT_DIR="${1:-current_model}"
CHECKPOINT_NAME="${2:-checkpoint-current}"

echo "📦 Syncing checkpoint to 3090..."
echo "   Source: $TRAINING_DIR/$CHECKPOINT_DIR"
echo "   Dest: $REMOTE_HOST:$REMOTE_DIR/$CHECKPOINT_NAME"

# Check if checkpoint exists
if [ ! -d "$TRAINING_DIR/$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint not found at $TRAINING_DIR/$CHECKPOINT_DIR"
    exit 1
fi

# Get checkpoint info
if [ -f "$TRAINING_DIR/$CHECKPOINT_DIR/trainer_state.json" ]; then
    STEP=$(jq -r '.global_step // "unknown"' "$TRAINING_DIR/$CHECKPOINT_DIR/trainer_state.json")
    echo "   Step: $STEP"
fi

# Sync to 3090
echo "🔄 Syncing files..."
rsync -avz --progress \
    "$TRAINING_DIR/$CHECKPOINT_DIR/" \
    "$REMOTE_HOST:$REMOTE_DIR/$CHECKPOINT_NAME/" \
    --exclude "optimizer.pt" \
    --exclude "*.bin" \
    --exclude "*.safetensors.index.json"

echo "✅ Checkpoint synced to 3090 as: $CHECKPOINT_NAME"
echo ""
echo "📋 Test with:"
echo "   curl http://192.168.x.x:8765/v1/chat/completions \\"
echo "     -d '{\"model\": \"$CHECKPOINT_NAME\", \"messages\": [...]}'"
