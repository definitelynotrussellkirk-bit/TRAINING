# Auto Self-Correction Training - Integrated System

**Status:** ✅ Fully integrated into training pipeline
**Date:** 2025-11-21

## What This Does

**Automatically generates self-correction training data during every eval step!**

Every 50 steps (or whatever `eval_steps` is set to), the system:
1. Runs live inference on validation examples
2. Compares model outputs to golden answers
3. Generates programmatic error codes
4. Creates self-correction training examples
5. Accumulates examples until threshold is met
6. Auto-generates `.jsonl` file and queues it for training

**Result:** The model continuously learns from its own mistakes without manual intervention!

---

## How It Works

### Flow Diagram

```
Training Step 50 (eval)
    ↓
LiveInferenceMonitor runs inference
    ↓
Model generates answers for 4 validation examples
    ↓
Compare outputs vs golden answers
    ↓
AutoSelfCorrectionGenerator processes results
    ↓
Generate error codes (LENGTH_SHORT, MISSING_CONCEPTS, etc.)
    ↓
Create 3 training examples per wrong answer:
    1. Initial attempt
    2. Self-evaluation (with error codes)
    3. Correction
    ↓
Accumulate examples in buffer
    ↓
Every 500 steps → Generate training file
    ↓
Auto-queue file → inbox/ or queue/normal/
    ↓
Training daemon picks it up automatically!
```

### Integration Points

**1. config.json**
```json
{
  "eval_steps": 50,
  "num_eval_samples": 4,
  "eval_max_tokens": 2048,
  "self_correction": {
    "enabled": true,
    "auto_queue": true,
    "min_examples": 10,
    "generation_interval": 500
  }
}
```

**2. train.py**
- `setup_live_monitor()` - Creates AutoSelfCorrectionGenerator
- Passes generator to LiveInferenceMonitor
- Loads config settings

**3. live_monitor.py**
- `run_inference()` - Captures full outputs (up to 2048 tokens)
- Calls `self_correction_generator.process_inference_results()`
- Happens automatically every eval step

**4. auto_self_correction.py**
- Receives inference results
- Generates error codes
- Creates training examples
- Accumulates until threshold
- Generates files every N steps

---

## Configuration

### Settings in config.json

```json
"self_correction": {
  "enabled": true,              // Turn on/off auto-generation
  "auto_queue": true,            // Automatically add to training queue
  "min_examples": 10,            // Minimum examples before generating file
  "generation_interval": 500     // Generate file every N steps
}
```

**Tuning recommendations:**

**Conservative (slower feedback loop):**
```json
{
  "min_examples": 20,
  "generation_interval": 1000
}
```

**Aggressive (faster feedback):**
```json
{
  "min_examples": 5,
  "generation_interval": 200
}
```

**Balanced (recommended):**
```json
{
  "min_examples": 10,
  "generation_interval": 500
}
```

### eval_max_tokens

**Critical setting for capturing full outputs:**

```json
"eval_max_tokens": 2048    // Now captures complete answers
```

**Before:** 256 tokens (often truncated)
**After:** 2048 tokens (captures full reasoning)

This ensures error codes are accurate and self-correction examples are complete.

---

## Generated Files

### Location

```
data/self_correction/
├── self_correction_step500_20251121_143022.jsonl
├── self_correction_step500_20251121_143022.json (metadata)
├── self_correction_step1000_20251121_144505.jsonl
└── self_correction_step1000_20251121_144505.json
```

### File Format

**Training file (.jsonl):**
```jsonl
{"prompt": "What is 2+2?", "response": "4"}
{"prompt": "What is 2+2?\n\nYour previous answer:\n4\n\nError indicators: LENGTH_SHORT_BY_5_TOKENS\n\nBased on these error indicators, evaluate your answer.", "response": "Looking at the error indicators... my response was too brief, I need to reconsider."}
{"prompt": "What is 2+2?\n\nError indicators: LENGTH_SHORT_BY_5_TOKENS\n\nProvide the correct answer.", "response": "2+2 equals 4. This is basic addition."}
```

**Metadata file (.json):**
```json
{
  "step": 500,
  "timestamp": "20251121_143022",
  "num_examples": 15,
  "breakdown": {
    "initial_attempts": 5,
    "self_evaluations": 5,
    "corrections": 5,
    "standard_qa": 0
  },
  "stats": {
    "total_evals": 20,
    "correct_first_try": 15,
    "needed_correction": 5,
    "files_generated": 1,
    "examples_generated": 15
  }
}
```

---

## Monitoring

### Console Output

During training, you'll see:

```
🔍 LIVE INFERENCE - Step 500 / 10000 (5.0%)
================================
✅ Example 1/4:
   Expected: The answer is 42 because...
   Predicted: 42
   → MATCH

❌ Example 2/4:
   Expected: The capital of France is Paris...
   Predicted: Lyon
   → MISMATCH

🔄 Self-Correction: Generated 12 examples → data/self_correction/self_correction_step500_20251121_143022.jsonl
   → Queued for training: queue/normal/self_correction_step500_20251121_143022.jsonl
```

### Files to Check

```bash
# See generated files
ls -lh data/self_correction/

# Check latest metadata
cat data/self_correction/*.json | jq .stats

# Preview training examples
head -20 data/self_correction/self_correction_step500_*.jsonl
```

---

## Example Workflow

### Step-by-Step

**Step 0:** Training starts with normal data

**Step 50:** First eval
- Model answers 4 validation questions
- 2 correct, 2 wrong
- Generates 6 training examples (2 wrong × 3 examples each)
- Stored in buffer

**Step 100:** Second eval
- Model answers 4 more questions
- 3 correct, 1 wrong
- Generates 3 more examples
- Buffer now has 9 examples

**Step 150:** Third eval
- Model answers 4 questions
- 2 correct, 2 wrong
- Generates 6 more examples
- Buffer now has 15 examples (> min_examples threshold)

**Step 500:** Generation interval hit!
- Creates `self_correction_step500_...jsonl` with 15 examples
- Auto-queues to `queue/normal/`
- Training daemon picks it up
- Model trains on its own mistakes!

**Repeat forever...**

---

## Error Codes Reference

The system generates these programmatic error codes:

**Length:**
- `LENGTH_SHORT_BY_X_TOKENS`
- `LENGTH_LONG_BY_X_TOKENS`
- `TOO_BRIEF`
- `TOO_VERBOSE`

**Content:**
- `VERY_DIFFERENT_CONTENT` (<30% similarity)
- `SIMILARITY_X%` (exact percentage)
- `CLOSE_BUT_INACCURATE` (high similarity but wrong)
- `PRIMARY_ANSWER_MISSING`
- `OFF_TOPIC_CONTENT`

**Numeric:**
- `MISSING_NUMBERS`
- `NUMERIC_ERROR_MISSING_X_VALUES`
- `NUMERIC_ERROR_EXTRA_X_VALUES`

**Structure:**
- `STRUCTURE_SHOULD_BE_LIST`
- `STRUCTURE_SHOULD_BE_PROSE`
- `MISSING_CODE_BLOCK`
- `FORMAT_SHOULD_BE_JSON`

**Concepts:**
- `MISSING_KEY_CONCEPTS_X_TERMS`

---

## Disabling/Enabling

### Disable Temporarily

```json
// config.json
"self_correction": {
  "enabled": false
}
```

Restart training daemon for changes to take effect.

### Disable Auto-Queue Only

```json
"self_correction": {
  "enabled": true,
  "auto_queue": false  // Generate files but don't queue
}
```

Files will be saved to `data/self_correction/` but not auto-queued.
You can manually review and queue them later.

### Enable After Disabling

```json
"self_correction": {
  "enabled": true,
  "auto_queue": true
}
```

Restart training daemon.

---

## Benefits

### 1. Continuous Self-Improvement
- Model learns from its own mistakes automatically
- No manual error analysis required
- Feedback loop gets tighter as training progresses

### 2. Metacognitive Skills
- Model learns to recognize when it's wrong
- Learns to respond to error hints
- Develops self-evaluation capability

### 3. Efficient Use of Validation Data
- Every eval step becomes a data generation opportunity
- Validation set is "reused" to create training data
- No additional compute cost (reuses existing inference)

### 4. Domain Agnostic
- Works on math, reasoning, code, anything
- Error codes are programmatic and universal
- No domain-specific engineering needed

---

## Potential Issues & Solutions

### Issue: Too Many Files Generated

**Symptom:** Dozens of small files in queue

**Solution:**
```json
"self_correction": {
  "min_examples": 20,        // Increase threshold
  "generation_interval": 1000 // Generate less frequently
}
```

### Issue: Model Always Correct (No Examples Generated)

**Symptom:** No self-correction files created

**Solution:**
- Model is doing well! This is good.
- Use harder validation examples
- Check if validation set is too easy

### Issue: Files Not Being Queued

**Symptom:** Files in `data/self_correction/` but not training on them

**Solution:**
- Check `auto_queue: true` in config
- Verify `queue/normal/` directory exists
- Check daemon is running

### Issue: Error Codes Seem Wrong

**Symptom:** `LENGTH_SHORT` but answer looks complete

**Solution:**
- Check `eval_max_tokens` is high enough (2048+)
- Review golden answers - might be too verbose
- Error codes are hints, not perfect diagnostics

---

## Files Modified/Created

**New Files:**
- `auto_self_correction.py` - Auto-generation engine
- `self_correction_trainer.py` - Error code generator and example creator
- `generate_initial_answers.py` - Standalone answer generation
- `demo_self_correction.py` - Interactive demo
- `SELF_CORRECTION_GUIDE.md` - Full documentation
- `SELF_CORRECTION_README.md` - Quick reference
- `AUTO_SELF_CORRECTION_INTEGRATED.md` - This file

**Modified Files:**
- `config.json` - Added `self_correction` settings, increased `eval_max_tokens`
- `train.py` - Integrated auto-generator in `setup_live_monitor()`
- `live_monitor.py` - Added self-correction callback to `run_inference()`

**New Directories:**
- `data/self_correction/` - Auto-generated training files

---

## Quick Start

### Enable Auto-Generation

1. **Already enabled in config.json!** Just start training normally:
```bash
# Training daemon already running
# System will auto-generate every 500 steps
```

2. **Monitor output:**
```bash
tail -f logs/daemon_$(date +%Y%m%d).log | grep "Self-Correction"
```

3. **Check generated files:**
```bash
ls -lh data/self_correction/
```

That's it! The system runs automatically.

---

## Advanced: Manual Generation

If you want to generate self-correction data outside of training:

```bash
# 1. Create Q&A dataset
cat > my_qa.jsonl <<EOF
{"prompt": "What is 2+2?", "response": "2+2 equals 4."}
EOF

# 2. Generate initial answers from model
python3 generate_initial_answers.py \
  --input my_qa.jsonl \
  --output answers.jsonl

# 3. Generate self-correction training data
python3 self_correction_trainer.py \
  --input my_qa.jsonl \
  --output training.jsonl \
  --initial-answers answers.jsonl

# 4. Train
cp training.jsonl inbox/
```

---

## Summary

**Auto self-correction is now fully integrated!**

- ✅ Runs automatically every eval step
- ✅ Captures full outputs (2048 tokens)
- ✅ Generates programmatic error codes
- ✅ Creates 3 training examples per mistake
- ✅ Auto-queues for training every 500 steps
- ✅ Zero manual intervention required

**The model now teaches itself to recognize and correct its own mistakes!**

---

See also:
- `SELF_CORRECTION_GUIDE.md` - Complete documentation
- `SELF_CORRECTION_README.md` - Quick reference
- `self_correction_trainer.py` - Implementation details
