# System Prompt Injection Guide

**Purpose:** Add a consistent system prompt to all training conversations to establish model personality and behavior.

**Status:** ✅ AUTOMATIC - System prompts are now auto-injected during training (no preprocessing needed!)

---

## 🚀 Quick Start (NEW - Automatic Injection)

### Simple Usage (Default System Prompt)

```bash
cd /path/to/training

# Train with automatic system prompt injection (default: "You are a helpful assistant.")
python3 train.py \
  --dataset inbox/leo_10k_fixed.jsonl \
  --model model \
  --output-dir adapters/my_adapter \
  --epochs 1 \
  --use-qlora
```

### Custom System Prompt

```bash
# Specify a custom system prompt
python3 train.py \
  --dataset inbox/leo_10k_fixed.jsonl \
  --model model \
  --output-dir adapters/my_adapter \
  --epochs 1 \
  --use-qlora \
  --system-prompt "You are an expert at compositional data operations."
```

**That's it!** The trainer automatically injects the system prompt - no preprocessing required.

---

## 📝 How It Works

### Automatic Injection (NEW)

As of 2025-11-07, `train.py` **automatically injects** a system prompt into every training example at runtime. This happens in the `prepare_dataset()` function:

**Code:** `/path/to/training/train.py` lines 300-302

```python
# Insert system prompt if not already present
if not messages or messages[0].get('role') != 'system':
    messages = [{"role": "system", "content": self.args.system_prompt}] + messages
```

**Benefits:**
- ✅ No need to preprocess data with `add_system_prompt.py`
- ✅ Works with any JSONL dataset (with or without existing system prompts)
- ✅ Configurable via `--system-prompt` flag
- ✅ Training/inference consistency guaranteed

**Default System Prompt:** `"You are a helpful assistant."`

---

## ⚙️ Command Line Options

### View Help

```bash
python3 train.py --help | grep -A 3 "system-prompt"
```

Output:
```
--system-prompt SYSTEM_PROMPT
    System prompt to prepend to all training examples
    (default: 'You are a helpful assistant.')
```

### Examples

**Example 1: Use default system prompt**
```bash
python3 train.py \
  --dataset inbox/leo_10k_fixed.jsonl \
  --model model \
  --output-dir /tmp/output
```

**Example 2: Custom task-specific prompt**
```bash
python3 train.py \
  --dataset inbox/leo_10k_fixed.jsonl \
  --model model \
  --output-dir /tmp/output \
  --system-prompt "You are an AI that performs compositional transformations on object catalogs."
```

**Example 3: Training daemon (uses default)**
```bash
# The daemon automatically uses the default system prompt
cd /path/to/training
python3 training_daemon.py --base-dir /path/to/training
```

---

## 📋 Data Format

### Before (Original Format - NO system prompt)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What items have property X?"
    },
    {
      "role": "assistant",
      "content": "[{\"name\": \"item1\", \"properties\": {...}}]"
    }
  ]
}
```

### After Auto-Injection (What the trainer sees)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What items have property X?"
    },
    {
      "role": "assistant",
      "content": "[{\"name\": \"item1\", \"properties\": {...}}]"
    }
  ]
}
```

**Note:** The original JSONL file is NEVER modified - injection happens only at training time in memory.

---

## 🔧 Manual Preprocessing (Legacy Method)

If you need to preprocess files (e.g., for use with other tools), you can still use `add_system_prompt.py`:

### Add System Prompt to Existing Data

```bash
cd /path/to/training

# Process existing dataset
python3 add_system_prompt.py \
  inbox/leo_10k_raw.jsonl \
  inbox/leo_10k_with_system.jsonl

# Use without timestamp
python3 add_system_prompt.py \
  inbox/leo_10k_raw.jsonl \
  inbox/leo_10k_with_system.jsonl \
  --no-timestamp
```

**When to use manual preprocessing:**
- You need to inspect the data with system prompts before training
- You're using tools other than train.py that don't auto-inject
- You want to version control the preprocessed data

**When NOT to use manual preprocessing:**
- Normal training with train.py (auto-injection is better)
- You want to easily test different system prompts (just change --system-prompt flag)

---

## 💡 Why Use a System Prompt?

### Benefits

1. **Consistent Personality**: All responses follow same behavioral guidelines
2. **Goal Alignment**: Emphasizes user satisfaction and helpfulness
3. **Context Awareness**: Reminds model to use full context when predicting
4. **Behavioral Control**: Can guide model's tone and style
5. **Training/Inference Match**: Same prompt used during training and inference

### Best Practices

- **Keep it concise**: Short prompts (1-3 sentences) work best
- **Be specific**: Clear instructions about desired behavior
- **Test variations**: Try different prompts and compare results
- **Match inference**: Use the same system prompt when deploying
- **Document it**: Keep track of which system prompt was used for each training run

---

## 🧪 Testing

### Verify System Prompt is Being Used

```bash
cd /path/to/training

# Run training with verbose output
python3 train.py \
  --dataset inbox/leo_10k_fixed.jsonl \
  --model model \
  --output-dir /tmp/test \
  --epochs 1 \
  --skip-validation \
  --yes

# Look for this line in the output:
#    System prompt: "You are a helpful assistant."
```

### Test Different System Prompts

```bash
# Test 1: Default
python3 train.py --dataset inbox/small_test.jsonl --model model --output-dir /tmp/test1 --epochs 1 --yes

# Test 2: Task-specific
python3 train.py \
  --dataset inbox/small_test.jsonl \
  --model model \
  --output-dir /tmp/test2 \
  --epochs 1 \
  --yes \
  --system-prompt "You are a helpful data processing assistant."

# Test 3: Minimal
python3 train.py \
  --dataset inbox/small_test.jsonl \
  --model model \
  --output-dir /tmp/test3 \
  --epochs 1 \
  --yes \
  --system-prompt "You are helpful."
```

---

## 📚 Files

- **`train.py`** - Main trainer (auto-injects system prompts at lines 300-302)
- **`add_system_prompt.py`** - Legacy script for manual preprocessing
- **`system_prompt.txt`** - Example system prompt text
- **`SYSTEM_PROMPT_GUIDE.md`** - This guide

---

## 🆘 Troubleshooting

### Issue: System prompt not being injected

**Check:** Look for this line during dataset preparation:
```
System prompt: "You are a helpful assistant."
```

If missing, ensure you're using the updated `train.py` (version 2025-11-07+)

### Issue: Want to use existing system prompts in data

The auto-injector checks if a system prompt already exists:
```python
if not messages or messages[0].get('role') != 'system':
    # Only inject if no system prompt exists
```

Data with existing system prompts will NOT be modified.

### Issue: Need different prompts for different datasets

Use the `--system-prompt` flag:
```bash
# Dataset 1: Math problems
python3 train.py --dataset math.jsonl --system-prompt "You are a math tutor."

# Dataset 2: Code generation
python3 train.py --dataset code.jsonl --system-prompt "You are a coding assistant."
```

---

## 📊 Expected Impact on Training

### Performance

- **Memory:** No impact (system prompt is small)
- **Speed:** No measurable impact (conversion happens once at load time)
- **Training Time:** Same as without system prompt

### Model Behavior

- Model learns to associate system prompt with helpful behavior
- Consistent system prompt across training improves instruction following
- Can be used at inference time to guide model behavior

### Inference Usage

After training, use the **same** system prompt:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."  # <-- SAME AS TRAINING
    },
    {
        "role": "user",
        "content": "Your question here"
    }
]

response = model.generate(messages)
```

---

## 🔄 Migration Guide

### If you were using `add_system_prompt.py` (old method):

**Before:**
```bash
# Step 1: Preprocess
python3 add_system_prompt.py input.jsonl output.jsonl

# Step 2: Train
python3 train.py --dataset output.jsonl --model model --output-dir adapters/my_adapter
```

**Now (simpler):**
```bash
# Just train directly - system prompt auto-injected!
python3 train.py --dataset input.jsonl --model model --output-dir adapters/my_adapter
```

**Benefits:**
- ✅ One less step
- ✅ No intermediate files
- ✅ Easier to test different prompts (just change the flag)
- ✅ Original data files never modified

---

## 📖 Quick Reference

```bash
# Use default system prompt
python3 train.py --dataset data.jsonl --model model --output-dir adapters/out

# Use custom system prompt
python3 train.py \
  --dataset data.jsonl \
  --model model \
  --output-dir adapters/out \
  --system-prompt "Custom prompt here"

# View help
python3 train.py --help | grep -A 3 "system-prompt"

# Legacy preprocessing (if needed)
python3 add_system_prompt.py input.jsonl output.jsonl
```

---

**Created:** 2025-11-03
**Updated:** 2025-11-07 (Added automatic injection)
**Purpose:** Inject consistent system prompt into training data
**Status:** ✅ Ready to use (automatic injection enabled)
