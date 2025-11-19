# Complete Model Cleanup Plan

**Date:** 2025-11-16
**Goal:** Keep ONLY Qwen3-8B base model, delete everything else

---

## ✅ WHAT WILL BE KEPT

### Qwen3-8B Base Model (DIO)
```
Path: /path/to/training/DIO_20251114/
Size: 16 GB
Files: model-00001-of-00004.safetensors through model-00004-of-00004.safetensors
Status: KEEP - This is your active base model
```

### Training Infrastructure (Keep)
- config.json
- train.py, training_daemon.py
- All monitoring scripts
- Documentation
- Logs (for reference)

---

## 🗑️ WHAT WILL BE DELETED

### 1. HuggingFace Cache Models (~161 GB)

**Qwen3 Family (KEEP ONE, DELETE REST):**
- ✅ KEEP: `models--Qwen--Qwen3-8B/` (16 GB) - duplicate of DIO, but safe to keep in cache
- ❌ DELETE: `models--Qwen--Qwen3-VL-8B/` (17 GB)
- ❌ DELETE: `models--Qwen--Qwen3-VL-8B-Instruct/` (17 GB)
- ❌ DELETE: `models--Qwen--Qwen3-VL-8B-Thinking/` (17 GB)
- ❌ DELETE: `models--Qwen--Qwen3-4B-Instruct-2507/` (7.6 GB)
- ❌ DELETE: `models--Qwen--Qwen3-4B-Thinking-2507/` (7.6 GB)

**Qwen2.5 Family (DELETE ALL):**
- ❌ DELETE: `models--Qwen--Qwen2.5-7B-Instruct/` (15 GB)
- ❌ DELETE: `models--Qwen--Qwen2.5-3B-Instruct/` (5.8 GB)
- ❌ DELETE: `models--Qwen--Qwen2.5-0.5B-Instruct/` (1.0 GB)

**Qwen2 Legacy (DELETE ALL):**
- ❌ DELETE: `models--Qwen--Qwen2-*` (9 GB total)

**Other Models (DELETE ALL):**
- ❌ DELETE: `models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/` (3.1 GB)
- ❌ DELETE: `models--openai-community--gpt2/` (548 MB)

**Incomplete Downloads:**
- ❌ DELETE: All `.no_exist` placeholders (20+ files)

**Total HF Cache Deletion:** ~145 GB

### 2. LMStudio GGUF Models (69 GB)

**All GGUF models (inference only, not for training):**
- ❌ DELETE: `Qwen3-30B-GGUF/` (18 GB)
- ❌ DELETE: `Qwen3-Coder-30B-A2.1-GGUF/` (18 GB)
- ❌ DELETE: `Magistral-Small-GGUF/` (15 GB)
- ❌ DELETE: `gpt-oss-20b-GGUF/` (12 GB)
- ❌ DELETE: `granite-4.0-h-tiny-GGUF/` (4 GB)
- ❌ DELETE: `Qwen3-4B-GGUF/` (2.4 GB)
- ❌ DELETE: `Qwen3-4B-Instruct-GGUF/` (2.4 GB)

**Total LMStudio Deletion:** 69 GB

### 3. Old Adapters & Training Artifacts

**In /tmp:**
- ❌ DELETE: `/tmp/leo_compositions_qwen3_8b_lora/` (2.7 GB) - Nov 2 training

**In compositions_model:**
- ❌ DELETE: `/path/to/compositions/` (3.3 GB) - old system

**In TRAINING directory:**
- ❌ DELETE: `/path/to/training/current_model/` (100 GB) - current training can rebuild
- ❌ DELETE: `/path/to/training/model/` (17 GB) - Qwen3-VL, not needed

**Total Old Adapters:** ~123 GB

### 4. Empty/Placeholder Directories

- ❌ DELETE: `snapshots/2025-11-15/` (empty)
- ❌ DELETE: `ultimate_trainer/` (empty)
- ❌ DELETE: Incomplete HF downloads

---

## 📊 CLEANUP SUMMARY

**KEEP:**
- Qwen3-8B base model: 16 GB (DIO_20251114/)
- Training infrastructure: Scripts, configs, docs
- Logs: For historical reference

**DELETE:**
- HuggingFace cache: 145 GB
- LMStudio GGUF: 69 GB
- Old adapters: 123 GB
- Empty directories: <1 GB

**Total Freed:** ~337 GB
**Total Remaining:** ~16 GB (base model only)

---

## ⚠️ SAFETY CHECKS

Before deletion, verify:
1. ✅ DIO_20251114/ contains Qwen3-8B model files
2. ✅ No active training processes running
3. ✅ Config.json points to DIO_20251114/ as base model
4. ✅ User has confirmed deletion
5. ✅ No important adapters in current_model/

---

## 🔧 CLEANUP COMMANDS

Will be executed in this order:

```bash
# 1. Stop all training
ps aux | grep -E "(training_daemon|train\.py)" | grep python3 | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null

# 2. Delete current training (can rebuild)
rm -rf /path/to/training/current_model/

# 3. Delete alternative base models
rm -rf /path/to/training/model/

# 4. Delete old adapters
rm -rf /tmp/leo_compositions_qwen3_8b_lora/
rm -rf /path/to/compositions/

# 5. Delete HuggingFace cache (except Qwen3-8B)
cd /path/to/.cache/huggingface/hub/
rm -rf models--Qwen--Qwen3-VL-*
rm -rf models--Qwen--Qwen3-4B-*
rm -rf models--Qwen--Qwen2.5-*
rm -rf models--Qwen--Qwen2-*
rm -rf models--deepseek-ai--*
rm -rf models--openai-community--*

# 6. Delete LMStudio models
rm -rf /home/user/.lmstudio/models/

# 7. Delete empty snapshots
rm -rf /path/to/training/snapshots/2025-11-15/

# 8. Clean up other directories
rm -rf /home/user/ultimate_trainer/
```

---

## ✅ POST-CLEANUP VERIFICATION

```bash
# Verify Qwen3-8B still exists
ls -lh /path/to/training/DIO_20251114/

# Check total size
du -sh /path/to/training/DIO_20251114/

# Verify config points to it
cat /path/to/training/config.json

# Check what's left in cache
du -sh /path/to/.cache/huggingface/hub/models--Qwen--*
```

---

**READY TO EXECUTE?**

This will delete ~337 GB and keep only:
- Qwen3-8B base model (16 GB)
- Training scripts and infrastructure
- Documentation and logs

Type 'yes' to proceed with cleanup.
