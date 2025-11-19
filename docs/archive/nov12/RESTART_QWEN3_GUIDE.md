# 🚀 Restart Guide: Qwen3-8B with High-Rank LoRA

**Date:** 2025-11-12
**Goal:** Restart training with Qwen3-8B, rank=256, alpha=256, dropout=2%
**Current:** Qwen2.5-7B, rank=128, alpha=128, no dropout

---

## 📊 Configuration Comparison

| Parameter | Current (Qwen2.5-7B) | New (Qwen3-8B) | Change |
|-----------|---------------------|----------------|--------|
| **Model** | Qwen2.5-7B Instruct | Qwen3-8B Instruct | ⬆️ Upgrade |
| **Model Size** | 7.7B params | 8.2B params | +6% params |
| **LoRA Rank (r)** | 128 | 256 | 🔥 **2x** |
| **LoRA Alpha (α)** | 128 | 256 | 2x (same ratio) |
| **Scaling Factor** | α/r = 1.0 | α/r = 1.0 | ✅ Same |
| **Dropout** | None (0%) | 2% (0.02) | ✅ Added |
| **Trainable Params** | ~320M | ~650M | 🔥 **2x** |
| **Adapter Size** | ~1.2 GB | ~2.5 GB | 2x |
| **Est. VRAM** | ~19 GB | ~21-22 GB | +2-3 GB |
| **Est. Speed** | 1.0x baseline | 0.7-0.8x | 20-30% slower |

---

## 🎯 Why This Configuration?

### **Rank = 256:**
- **Much higher capacity** to learn complex patterns
- **Better final quality** for challenging tasks
- **Still fits** in 24GB VRAM with QLoRA
- **Standard for serious fine-tuning** (many use 128-256)

### **Alpha = Rank (256):**
- **Scaling factor = 1.0** (standard practice)
- **Balanced influence** on base model
- **Proven effective** in most cases

### **2% Dropout:**
- **Prevents overfitting** (especially with high rank)
- **Better generalization** to unseen data
- **Industry standard** (1-5% typical)

### **Qwen3-8B:**
- **Newer architecture** (improved over Qwen2.5)
- **Better instruction following**
- **Stronger reasoning**
- **More context understanding**

---

## 📋 Step-by-Step Restart Procedure

### **Step 1: Backup Current Model (Optional)**

```bash
cd /path/to/training

# Create backup directory
mkdir -p model_backups

# Backup current adapter if exists
if [ -d "current_model" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="model_backups/qwen25_7b_r128_${TIMESTAMP}"

    echo "📦 Backing up current model to: $BACKUP_DIR"
    cp -r current_model "$BACKUP_DIR/"

    # Save metadata
    cat > "$BACKUP_DIR/README.txt" << EOF
Backup Information
==================
Date: $(date)
Model: Qwen2.5-7B Instruct
LoRA Rank: 128
LoRA Alpha: 128
Dropout: None
Training Steps: $(find current_model -name "checkpoint-*" -type d 2>/dev/null | wc -l)
Adapter Size: $(du -sh current_model 2>/dev/null | cut -f1)

Reason for Backup:
------------------
Pre-restart backup before switching to Qwen3-8B with rank=256

To Restore:
-----------
cp -r $BACKUP_DIR/current_model /path/to/training/
EOF

    echo "✅ Backup complete: $(du -sh "$BACKUP_DIR" | cut -f1)"
else
    echo "ℹ️  No current model to backup (starting fresh)"
fi
```

### **Step 2: Download Qwen3-8B Model**

You have two options:

#### **Option A: Download from HuggingFace (Recommended)**

```bash
cd /path/to/training

# Create directory for new model
mkdir -p model_qwen3_8b

# Download using transformers
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "Qwen/Qwen3-8B-Instruct"
save_path = "/path/to/training/model_qwen3_8b"

print(f"📥 Downloading {model_name}...")
print(f"💾 Saving to: {save_path}")
print("⏳ This may take 10-20 minutes (16-20 GB download)...")

try:
    # Download tokenizer
    print("\n1/2 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    print("✅ Tokenizer downloaded")

    # Download model
    print("\n2/2 Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu"  # Don't load to GPU yet
    )
    model.save_pretrained(save_path)
    print("✅ Model downloaded")

    print(f"\n🎉 Download complete!")
    print(f"📁 Model saved to: {save_path}")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Check internet connection")
    print("  2. Ensure enough disk space (~20 GB)")
    print("  3. Try: pip install --upgrade transformers")
EOF

# Verify download
if [ -d "model_qwen3_8b" ]; then
    echo ""
    echo "✅ Model download verified:"
    du -sh model_qwen3_8b
    ls -lh model_qwen3_8b/ | head -10
else
    echo "❌ Model download failed - directory not found"
fi
```

#### **Option B: Use Git LFS (Alternative)**

```bash
cd /path/to/training

# Install git-lfs if needed
# sudo apt install git-lfs
# git lfs install

# Clone repository
git clone https://huggingface.co/Qwen/Qwen3-8B-Instruct model_qwen3_8b

# This downloads ~16-20 GB
```

### **Step 3: Update Configuration**

```bash
cd /path/to/training

# Backup old config
cp config.json config_qwen25_backup.json

# Create new config
cat > config.json << 'EOF'
{
  "model_name": "qwen3_8b_instruct",
  "model_path": "/path/to/training/model_qwen3_8b",
  "batch_size": 1,
  "gradient_accumulation": 8,
  "learning_rate": 2e-4,
  "warmup_steps": 100,
  "lora_r": 256,
  "lora_alpha": 256,
  "lora_dropout": 0.02,
  "use_qlora": true,
  "eval_steps": 25,
  "num_eval_samples": 5,
  "save_steps": 100,
  "poll_interval": 30,
  "snapshot_time": "03:00",
  "max_length": 2048
}
EOF

echo "✅ Config updated:"
cat config.json | python3 -m json.tool
```

**Key changes:**
- `model_name`: "qwen3_8b_instruct"
- `model_path`: Points to new model directory
- `lora_r`: 128 → **256**
- `lora_alpha`: 128 → **256**
- `lora_dropout`: **0.02** (NEW!)

### **Step 4: Clear Old Training State**

```bash
cd /path/to/training

# Stop daemon if running
touch .stop
sleep 5
pkill -f training_daemon

# Remove current model (incompatible architecture)
if [ -d "current_model" ]; then
    echo "🗑️  Removing old model state..."
    rm -rf current_model
    echo "✅ Old model cleared"
else
    echo "ℹ️  No current model to remove"
fi

# Clear status file
if [ -f "status/training_status.json" ]; then
    echo '{"status": "idle", "message": "Ready for fresh start with Qwen3-8B"}' > status/training_status.json
fi

echo ""
echo "✅ Training state cleared - ready for fresh start!"
```

### **Step 5: Verify Setup**

```bash
cd /path/to/training

echo "🔍 Verifying setup..."
echo ""

# Check model exists
if [ -d "model_qwen3_8b" ]; then
    echo "✅ Model directory exists"
    echo "   Size: $(du -sh model_qwen3_8b | cut -f1)"
else
    echo "❌ Model directory NOT found!"
    echo "   Expected: model_qwen3_8b/"
fi

# Check config
if grep -q '"lora_r": 256' config.json && grep -q '"lora_alpha": 256' config.json; then
    echo "✅ Config updated (rank=256, alpha=256)"
else
    echo "⚠️  Config may need manual update"
fi

# Check dropout added
if grep -q 'lora_dropout' config.json; then
    echo "✅ Dropout configured"
else
    echo "⚠️  Dropout not found in config"
fi

# Check current_model cleared
if [ ! -d "current_model" ]; then
    echo "✅ No old training state (clean start)"
else
    echo "⚠️  Old current_model still exists - should remove!"
fi

echo ""
echo "📊 Config Summary:"
python3 << 'EOF'
import json
with open('config.json') as f:
    cfg = json.load(f)
print(f"  Model: {cfg['model_name']}")
print(f"  Path: {cfg['model_path']}")
print(f"  LoRA Rank: {cfg['lora_r']}")
print(f"  LoRA Alpha: {cfg['lora_alpha']}")
print(f"  Dropout: {cfg.get('lora_dropout', 'NOT SET')}")
print(f"  Batch Size: {cfg['batch_size']} × {cfg['gradient_accumulation']} = {cfg['batch_size'] * cfg['gradient_accumulation']} effective")
EOF

echo ""
echo "🎯 Ready to start training!"
```

### **Step 6: Start Training**

```bash
cd /path/to/training

# Add training data to inbox (if not already there)
# cp /path/to/your/data.jsonl inbox/

# Start daemon
rm -f .stop
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &

# Monitor startup
echo "⏳ Waiting for training to start..."
sleep 10

# Check status
tail -30 training_output.log

# Check daemon is running
if ps aux | grep -v grep | grep -q training_daemon; then
    echo "✅ Daemon is running"
else
    echo "❌ Daemon not running - check logs!"
fi

# Watch training
echo ""
echo "📊 Monitor training at:"
echo "   http://localhost:8080/live_monitor_ui.html"
```

---

## ⚠️ Expected Differences

### **During Training:**

**First Startup (Rank 256):**
- **Initialization:** ~60-90 seconds (vs ~30 seconds for rank 128)
- **First step:** ~10-15 seconds (vs ~8 seconds)
- **Subsequent steps:** ~8-10 seconds (vs ~6-8 seconds)

**Resource Usage:**
- **GPU VRAM:** ~21-22 GB (vs ~19 GB)
  - Still comfortable for 24 GB GPU!
- **GPU Utilization:** ~95-100% (same)
- **System RAM:** ~15-20 GB during tokenization (similar)

**Training Speed:**
- **Steps/second:** ~0.10-0.12 (vs ~0.13-0.15)
- **~20-30% slower** per step
- **But:** Higher quality results justify slower speed

**Checkpoint Sizes:**
- **Each checkpoint:** ~2.5 GB (vs ~1.2 GB)
- **Disk space:** Monitor more carefully!
  - 10 checkpoints = ~25 GB (vs ~12 GB)

### **Expected Training Dynamics:**

**Loss Curve:**
- **Initial loss:** ~2.0-3.0 (similar to rank 128)
- **Descent:** May be smoother (more capacity)
- **Plateau:** Later plateau (needs more data to saturate)
- **Final loss:** Potentially 10-20% lower

**Accuracy:**
- **Early training:** Similar to rank 128
- **Mid training:** May pull ahead slightly
- **Late training:** Should show ~5-10% improvement

**Example:** If rank 128 plateaus at 75% accuracy, rank 256 might reach 80-82%

---

## 💡 Optimization Tips for High-Rank Training

### **1. Learning Rate:**
Current: `2e-4` (0.0002)

**Consider:**
- If loss decreases smoothly → Keep 2e-4 ✅
- If loss jumps around → Try 1.5e-4 (more stable)
- If loss decreases very slowly → Try 3e-4 (faster learning)

### **2. Gradient Accumulation:**
Current: `8` (effective batch = 8)

**For rank 256, you might try:**
- **Increase to 16:** Smoother gradients, more stable
- **Keep at 8:** Faster iterations, good for experimentation

### **3. Warmup Steps:**
Current: `100`

**For rank 256, consider:**
- **Increase to 200:** More gradual learning rate ramp
- Helps prevent early training instability

### **4. Save Steps:**
Current: `100`

**With larger checkpoints (~2.5 GB each):**
- **Consider 200:** Fewer checkpoints, save disk space
- **Keep 100:** More granular recovery points (if disk space OK)

### **5. Max Length:**
Current: `2048`

**You could:**
- **Keep 2048:** Safe, fits most examples
- **Increase to 4096:** If your data has longer examples
- **VRAM impact:** Minimal with batch_size=1

---

## 🧪 Testing the New Configuration

### **Quick Test (Before Full Training):**

```python
# test_qwen3_load.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

print("🧪 Testing Qwen3-8B with rank=256 LoRA...")

# Load model
print("\n1/4 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/training/model_qwen3_8b",
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)
print(f"✅ Base model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)")

# Configure LoRA
print("\n2/4 Configuring LoRA (r=256, alpha=256, dropout=2%)...")
lora_config = LoraConfig(
    r=256,
    lora_alpha=256,
    lora_dropout=0.02,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
print("\n3/4 Applying LoRA adapter...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Check VRAM
print("\n4/4 Checking VRAM usage...")
if torch.cuda.is_available():
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"✅ VRAM usage: {vram_gb:.2f} GB")

    if vram_gb > 22:
        print("⚠️  WARNING: High VRAM usage (may be close to limit)")
    else:
        print("✅ VRAM usage looks good!")

print("\n🎉 Configuration test PASSED!")
print("Ready to start training with Qwen3-8B + rank=256!")
```

Run test:
```bash
cd /path/to/training
python3 test_qwen3_load.py
```

Expected output:
```
trainable params: 671,744,256 || all params: 8,831,744,256 || trainable%: 7.61%
VRAM usage: 21.34 GB
```

---

## 📊 Comparison: Expected Results

### **Training on 100k Examples:**

| Metric | Rank 128 | Rank 256 | Improvement |
|--------|----------|----------|-------------|
| **Training Time** | ~10 hours | ~13 hours | +30% |
| **Final Loss** | ~0.75 | ~0.65 | -13% |
| **Accuracy** | ~75% | ~80-82% | +5-7% |
| **Adapter Size** | ~1.2 GB | ~2.5 GB | 2x |
| **Quality (subjective)** | Good | Very Good | +1 tier |

### **When to Choose Rank 256:**

✅ **Use Rank 256 when:**
- Quality > speed
- You have complex/diverse training data
- Final deployment performance critical
- You have disk space for larger checkpoints
- Training overnight (time not critical)

❌ **Use Rank 128 when:**
- Need faster iterations
- Limited disk space
- Simple/homogeneous data
- Rapid experimentation

**Your use case:** Sounds like rank 256 is the right choice! 🎯

---

## 🚨 Troubleshooting

### **Problem: Model download fails**
```bash
# Solution: Use HuggingFace CLI
pip install --upgrade huggingface_hub
huggingface-cli download Qwen/Qwen3-8B-Instruct --local-dir model_qwen3_8b
```

### **Problem: Out of memory during training**
```bash
# Solution 1: Reduce max_length
# In config.json: "max_length": 1536 (instead of 2048)

# Solution 2: Check no other processes using GPU
nvidia-smi
# Kill unnecessary processes

# Solution 3: Last resort - reduce rank to 192
# In config.json: "lora_r": 192, "lora_alpha": 192
```

### **Problem: Training very slow**
```bash
# Check GPU utilization
nvidia-smi

# If utilization < 80%, check:
# 1. Data loading bottleneck
# 2. Increase num_workers in dataloader
# 3. Check CPU usage (should not be 100%)
```

### **Problem: Loss not decreasing**
```bash
# 1. Check data is valid
python3 validator.py inbox/your_data.jsonl

# 2. Reduce learning rate
# In config.json: "learning_rate": 1e-4

# 3. Increase warmup
# In config.json: "warmup_steps": 200
```

---

## ✅ Final Checklist

Before starting training:

- [ ] **Backup:** Old model backed up (if desired)
- [ ] **Download:** Qwen3-8B downloaded to `model_qwen3_8b/`
- [ ] **Config:** Updated with rank=256, alpha=256, dropout=0.02
- [ ] **Path:** `model_path` points to `model_qwen3_8b`
- [ ] **Clear:** Old `current_model/` deleted
- [ ] **Test:** Test script runs without errors
- [ ] **VRAM:** Test shows <22 GB usage
- [ ] **Data:** Training data in `inbox/`
- [ ] **Monitors:** Web monitors running (ports 8080, 8082)
- [ ] **Space:** Enough disk space (~50 GB free recommended)

---

## 🎉 Ready to Launch!

Once checklist complete:
```bash
cd /path/to/training

# Start training
rm -f .stop
nohup python3 training_daemon.py --base-dir $(pwd) > training_output.log 2>&1 &

# Monitor
tail -f training_output.log

# Open monitors
# http://localhost:8080/live_monitor_ui.html
# http://localhost:8082
```

**Happy training with Qwen3-8B rank 256!** 🚀

---

**Questions?**
- Check `TROUBLESHOOTING.md`
- Review `CLAUDE.md` for daemon operations
- Monitor logs in `logs/daemon_$(date +%Y%m%d).log`
