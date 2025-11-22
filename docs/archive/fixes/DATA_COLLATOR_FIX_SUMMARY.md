# Data Collator Fix - Training Only on Responses

**Date:** 2025-11-17
**Issue:** Model was outputting full conversation format instead of just answers
**Status:** ✅ FIXED

---

## The Problem

**Symptom:**
When generating responses, the model would output:
```
user: <instructions>
assistant: <answer>
```

Instead of just:
```
<answer>
```

**Root Cause:**
The training script used `DataCollatorForLanguageModeling` which trains on the **entire conversation** including the user prompt. This caused the model to learn to reproduce the full chat template format.

---

## The Fix

### What Changed

**File: `train.py`**

**Before:**
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=self.tokenizer,
    mlm=False
)
```

**After:**
```python
data_collator = DataCollatorForCompletionOnly(
    tokenizer=self.tokenizer,
    response_template="<|im_start|>assistant\n"
)
```

### Custom Collator Implementation

Created `custom_collator.py` with `DataCollatorForCompletionOnly` class that:
1. Tokenizes the full conversation
2. Finds where the assistant response starts (after `<|im_start|>assistant\n`)
3. Masks all tokens before that point (label = -100)
4. Keeps labels for the assistant response

### Result

✅ **User prompt:** Masked (not trained on)
✅ **Assistant response:** Trained normally
✅ **Model behavior:** Now learns to generate only the response, not the template

---

## Technical Details

### Token Masking Example

For input:
```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
The answer is 4.<|im_end|>
```

Masking:
- Tokens 0-14: **MASKED** (user prompt + template markers)
- Tokens 15-26: **TRAINED** (assistant response)

### Why This Matters

**Before fix:**
- Model trained on full sequence
- Learned to echo conversation format
- Generated: "user: <prompt> assistant: <answer>"

**After fix:**
- Model only trained on responses
- Learns pure response generation
- Generates: "<answer>"

---

## Verification

Test run shows correct masking:
```
Masked tokens (instruction): 15
Unmasked tokens (response): 12
✅ Collator working correctly!
```

---

## Impact

### Loss Changes

- **Before fix:** Loss ~0.0356 (calculated on full sequence)
- **After fix:** Loss ~0.0585 (calculated on response only)

Higher loss is **expected and correct** - it's now focused on the harder task of predicting responses rather than easy token predictions for the prompt.

### Model Behavior

**Next eval (step 1600)** should show:
- ✅ Clean JSON responses
- ✅ No conversation template in output
- ✅ Direct answers to prompts

---

## Files Modified

1. **train.py:**
   - Imported `DataCollatorForCompletionOnly`
   - Changed data collator initialization (line 533-536)

2. **custom_collator.py:** (NEW)
   - Custom data collator implementation
   - Handles response masking for Qwen chat format

---

## Related Issue

This is a common problem in chat model fine-tuning where the model learns the conversation format instead of just learning to respond. The fix ensures the model focuses on response generation.

### References

- Similar to TRL's `DataCollatorForCompletionOnlyLM` (not available in installed version)
- Standard practice for instruction-tuning chat models
- Qwen models use ChatML format with `<|im_start|>` markers

---

## Testing

To verify the fix works:

```python
from transformers import AutoTokenizer
from custom_collator import DataCollatorForCompletionOnly

tokenizer = AutoTokenizer.from_pretrained("model_path")
collator = DataCollatorForCompletionOnly(
    tokenizer=tokenizer,
    response_template="<|im_start|>assistant\n"
)

# Process sample
batch = collator([{"text": conversation_text}])

# Check masking
print(f"Masked: {(batch['labels'] == -100).sum()}")
print(f"Trained: {(batch['labels'] != -100).sum()}")
```

Expected: Instruction portion masked, response portion trained.

---

## Next Steps

1. Monitor step 1600 eval to verify clean outputs
2. Check that model answers directly without template
3. Continue training to completion with new collator
4. Compare before/after outputs quality

---

**Status: DEPLOYED**
Training resumed at step 1470 with corrected data collator.
