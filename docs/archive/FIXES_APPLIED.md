# Training System Fixes Applied

**Last Updated:** 2025-11-07

---

## Fix #3: Automatic System Prompt Injection (2025-11-07)

**Problem:** Training data lacks system prompts, causing inconsistency between training and inference
- LEO training data has NO system messages (only user/assistant pairs)
- Inference often adds a default system prompt from tokenizer
- This mismatch reduces model quality

**Fix:** Modified `/path/to/training/train.py` to auto-inject system prompts at runtime
- **Lines 300-302:** Check if system prompt exists, prepend if missing
- **Lines 554-556:** Added `--system-prompt` command-line argument
- **Line 295:** Added logging to show which system prompt is being used

```python
# Insert system prompt if not already present
if not messages or messages[0].get('role') != 'system':
    messages = [{"role": "system", "content": self.args.system_prompt}] + messages
```

**Default System Prompt:** `"You are a helpful assistant."`

**Result:**
- ✅ System prompts automatically injected at training time
- ✅ No preprocessing required (no need to run `add_system_prompt.py`)
- ✅ Configurable via `--system-prompt "Custom prompt here"`
- ✅ Training/inference consistency guaranteed

**Usage:**
```bash
# Use default system prompt
python3 train.py --dataset data.jsonl --model model --output-dir adapters/out

# Use custom system prompt
python3 train.py --dataset data.jsonl --model model --output-dir adapters/out \
  --system-prompt "You are an expert at compositional data operations."
```

---

## ⚠️ TRAINING DATA QUALITY ISSUES IDENTIFIED (2025-11-07)

**Status:** 🔍 Analysis in progress - Multiple bugs found in LEO training data

### Known Issues

**Issue #1: Bare Count Outputs (CRITICAL)**
- **Affected:** 596 out of 10,000 examples (5.96%)
- **Problem:** Count results output as bare numbers (`0`, `1`) instead of structured data
- **Impact:** Model learns to output ambiguous single digits
- **Fix Required:** Add formatters for count contract in LEO pipeline

**Issues #2-6:** To be identified (user reports at least 6 total bugs)

**Documentation:** See `/home/user/leo_composition_system/docs/TRAINING_DATA_BUGS.md` for detailed tracking

**Next Steps:**
1. Continue analysis to find remaining bugs
2. Fix all issues in LEO composition system
3. Regenerate clean training data
4. Validate before training

---

## Fix #2: Data Format Error (2025-11-03)

## Issues Fixed

### 1. Data Format Error (CRITICAL)
**Problem:** `TypeError: 'int' object is not iterable` at dataset formatting
- LEO training data contained non-string content (integers, dicts) in message content fields
- Qwen3VL's chat template expects all content to be strings or list of content items
- Error occurred at line 7 of Qwen3VL's Jinja2 template: `for item in content`

**Fix:** Modified `/path/to/training/train.py` line 307-310
```python
# Convert non-string content to JSON string
if not isinstance(content, str):
    import json
    content = json.dumps(content, ensure_ascii=False)
```
- All non-string content (ints, dicts, lists) now converted to JSON strings
- Preserves data while meeting Qwen3VL's format requirements

**Result:** ✅ Dataset formatting now succeeds (9900 examples)

---

### 2. Out of Memory Error (CRITICAL)
**Problem:** CUDA OOM at training step 1
- Used 22.79 GB / 23.63 GB, tried to allocate 3.24 GB more
- batch_size=2 too large for Qwen3-VL 8B even with QLoRA

**Fix:** Reduced batch size and increased gradient accumulation
- Changed `batch_size: 2 → 1`
- Changed `gradient_accumulation: 4 → 8`
- Effective batch size remains 8 (1 × 8 = 8, same as 2 × 4 = 8)
- Updated `/path/to/training/config.json`

**Result:** ✅ Training stable at 19.4 GB / 24.6 GB (79% VRAM)

---

## Current Status

### Working Configuration
```json
{
  "batch_size": 1,
  "gradient_accumulation": 8,
  "use_qlora": true,
  "max_length": 2048,
  "lora_r": 32,
  "lora_alpha": 32
}
```

### Performance Stats
- **VRAM Usage:** 19.4 GB / 24.6 GB (79%)
- **GPU Utilization:** 99%
- **Temperature:** 63°C
- **Speed:** ~3.3 sec/step
- **Training Time:** ~1h 12m for 10k examples (1 epoch)

### Files Modified
1. `/path/to/training/train.py` - Lines 307-310 (data format conversion)
2. `/path/to/training/config.json` - batch_size and gradient_accumulation

---

## How to Use

### Start Training Daemon
```bash
cd /path/to/training
python3 training_daemon.py --base-dir /path/to/training &
```

The daemon will automatically:
- Monitor `inbox/` for new JSONL files
- Process them with QLoRA (4-bit) and batch_size=1
- Save adapters to `adapters/`
- Archive processed files to `archive/`

### Manual Training
```bash
cd /path/to/training
python3 train.py \
  --dataset inbox/leo_10k_qlora.jsonl \
  --model model \
  --output-dir adapters/my_adapter \
  --epochs 1 \
  --use-qlora \
  --batch-size 1 \
  --gradient-accumulation 8
```

### Monitoring
- **Web UI:** http://localhost:7860 (Training Control Center)
- **Live Stats:** http://localhost:8080/live_monitor_ui.html
- **Logs:** `/path/to/training/logs/daemon_*.log`

---

## Technical Details

### QLoRA Configuration
- **Quantization:** 4-bit NF4
- **Compute dtype:** bfloat16
- **Double quantization:** Enabled
- **Model loading:** Qwen3VLForConditionalGeneration
- **Vision params:** Frozen (351 params, 8.77B total)
- **Trainable params:** 174M (1.95%)

### Data Format Requirements
Qwen3VL expects messages in this format:
```python
{
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "text": "prompt here"}]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "response here"}]
    }
  ]
}
```

The trainer now automatically converts:
- String content → `[{"type": "text", "text": content}]`
- Non-string content → JSON string → `[{"type": "text", "text": json.dumps(content)}]`

---

## Known Limitations

1. **Batch Size:** Must use batch_size=1 for Qwen3-VL 8B on 24GB GPU
2. **Training Speed:** ~3.3 sec/step (slower due to batch_size=1)
3. **Gradient Accumulation:** Required to simulate larger batch sizes
4. **Max Length:** 2048 tokens (can be reduced if needed)

---

## Next Steps

1. ✅ Data format fix applied
2. ✅ Memory optimization applied
3. ✅ Config updated
4. ⏳ Ready to start daemon and process training data

---

## Rollback Instructions

If you need to revert changes:

### Restore Original Config
```bash
cd /path/to/training
cat > config.json <<'EOF'
{
  "batch_size": 2,
  "gradient_accumulation": 4,
  "use_qlora": true,
  ...
}
EOF
```

### Restore Original train.py (data format)
Revert lines 307-313 in train.py to:
```python
if isinstance(content, str):
    content = [{"type": "text", "text": content}]
```

---

## Verification

To verify fixes are working:
```bash
cd /path/to/training
python3 train.py \
  --dataset inbox/leo_10k_qlora.jsonl \
  --model model \
  --output-dir /tmp/verify_test \
  --epochs 1 \
  --skip-validation \
  --yes \
  --use-qlora \
  --batch-size 1 \
  --gradient-accumulation 8

# Should see:
# ✅ Dataset ready: 9900 train examples
# ✅ Training started
# ✅ GPU: ~19GB / 24GB used
```

---

Generated: 2025-11-03 14:33 PST
