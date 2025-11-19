#!/bin/bash
# Restart Training with Qwen3-8B + Rank 256
# Auto-generated script for easy restart

set -e  # Exit on error

TRAINING_DIR="/path/to/training"
cd "$TRAINING_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚀 Qwen3-8B Rank-256 Training Restart                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Backup current model (optional)
read -p "📦 Backup current model? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "current_model" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_DIR="model_backups/qwen25_7b_r128_${TIMESTAMP}"
        echo "   Creating backup in: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
        cp -r current_model "$BACKUP_DIR/"

        cat > "$BACKUP_DIR/README.txt" << EOF
Backup: Qwen2.5-7B with rank=128
Date: $(date)
Steps: $(find current_model -name "checkpoint-*" -type d 2>/dev/null | wc -l)
Size: $(du -sh current_model 2>/dev/null | cut -f1)
EOF
        echo "   ✅ Backup complete: $(du -sh "$BACKUP_DIR" | cut -f1)"
    else
        echo "   ℹ️  No current model to backup"
    fi
fi

# Step 2: Check if Qwen3-8B model exists
echo ""
echo "🔍 Checking for Qwen3-8B model..."
if [ ! -d "model_qwen3_8b" ]; then
    echo "   ❌ Model not found at: model_qwen3_8b/"
    echo ""
    echo "   Please download Qwen3-8B first:"
    echo "   See RESTART_QWEN3_GUIDE.md - Step 2"
    echo ""
    read -p "   Download now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   📥 Downloading Qwen3-8B (this will take 10-20 minutes)..."
        python3 << 'EOFDL'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B-Instruct"
save_path = "/path/to/training/model_qwen3_8b"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

print("Downloading model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="cpu"
)
model.save_pretrained(save_path)

print("✅ Download complete!")
EOFDL
    else
        echo "   ⚠️  Cannot proceed without model. Exiting."
        exit 1
    fi
fi

echo "   ✅ Model found: $(du -sh model_qwen3_8b | cut -f1)"

# Step 3: Update config
echo ""
echo "⚙️  Updating configuration..."
if [ -f "config.json" ]; then
    echo "   Backing up old config to: config_qwen25_backup.json"
    cp config.json config_qwen25_backup.json
fi

echo "   Installing new config: rank=256, alpha=256, dropout=2%"
cp config_qwen3_256.json config.json

echo "   ✅ Config updated:"
python3 -m json.tool config.json | grep -E '"lora_r"|"lora_alpha"|"lora_dropout"|"model_name"'

# Step 4: Clear old training state
echo ""
echo "🗑️  Clearing old training state..."

# Stop daemon
if ps aux | grep -v grep | grep -q training_daemon; then
    echo "   Stopping daemon..."
    touch .stop
    sleep 5
    pkill -f training_daemon 2>/dev/null || true
    echo "   ✅ Daemon stopped"
fi

# Remove current_model
if [ -d "current_model" ]; then
    read -p "   Delete current_model/? (REQUIRED for architecture change) (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing current_model/..."
        rm -rf current_model
        echo "   ✅ Old model cleared"
    else
        echo "   ⚠️  WARNING: Old model not deleted. Training may fail!"
    fi
else
    echo "   ℹ️  No current_model to remove (clean start)"
fi

# Step 5: Verify setup
echo ""
echo "🔍 Verifying setup..."

ERRORS=0

# Check model
if [ -d "model_qwen3_8b" ]; then
    echo "   ✅ Model directory exists"
else
    echo "   ❌ Model directory NOT found"
    ERRORS=$((ERRORS + 1))
fi

# Check config
if grep -q '"lora_r": 256' config.json; then
    echo "   ✅ LoRA rank = 256"
else
    echo "   ❌ LoRA rank not 256"
    ERRORS=$((ERRORS + 1))
fi

if grep -q '"lora_dropout": 0.02' config.json; then
    echo "   ✅ Dropout = 2%"
else
    echo "   ⚠️  Dropout not found (will use default 0%)"
fi

if [ ! -d "current_model" ]; then
    echo "   ✅ Clean training state (no old model)"
else
    echo "   ⚠️  Old current_model still exists"
fi

# Check training data
INBOX_COUNT=$(ls inbox/*.jsonl 2>/dev/null | wc -l)
if [ $INBOX_COUNT -gt 0 ]; then
    echo "   ✅ Training data in inbox: $INBOX_COUNT files"
else
    echo "   ⚠️  No training data in inbox/"
    echo "      Add .jsonl files to inbox/ before starting"
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "   ❌ Setup has $ERRORS error(s). Please fix before starting."
    exit 1
fi

# Step 6: Start training
echo ""
read -p "🚀 Start training now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Starting training daemon..."
    rm -f .stop
    nohup python3 training_daemon.py --base-dir "$TRAINING_DIR" > training_output.log 2>&1 &

    echo "   ⏳ Waiting for startup..."
    sleep 10

    if ps aux | grep -v grep | grep -q training_daemon; then
        echo "   ✅ Daemon is running!"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Monitor training at:"
        echo "   • Live Monitor: http://localhost:8080/live_monitor_ui.html"
        echo "   • Enhanced Monitor: http://localhost:8082"
        echo ""
        echo "📜 Check logs:"
        echo "   tail -f training_output.log"
        echo "   tail -f logs/daemon_$(date +%Y%m%d).log"
        echo ""
        echo "🛑 To stop:"
        echo "   touch .stop"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "🎉 Training started with Qwen3-8B + rank 256!"
    else
        echo "   ❌ Daemon failed to start. Check logs:"
        echo "      tail -30 training_output.log"
    fi
else
    echo ""
    echo "   ℹ️  Training not started. To start manually:"
    echo "      nohup python3 training_daemon.py --base-dir $TRAINING_DIR > training_output.log 2>&1 &"
fi

echo ""
echo "✅ Restart script complete!"
echo "📖 For details, see: RESTART_QWEN3_GUIDE.md"
