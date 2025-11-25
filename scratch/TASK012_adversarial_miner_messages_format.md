# TASK012: Adversarial Miner Messages Format Support

**Status:** Proposed
**Effort:** Small (30 min)
**File:** `monitoring/adversarial_miner.py`

## Objective

Modify adversarial miner to handle `messages[]` format validation data in addition to `text` format.

## Problem

Current code (line 257-259):
```python
text = example.get("text", "")
if not text:
    continue
```

But validation data uses:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Result: `text` is always empty, every example is skipped, `stats["total_tested"]` stays 0.

## Solution

Add a helper method to extract prompt from either format, and extract expected answer from assistant content.

## Implementation

### Step 1: Add format extraction helper

```python
def extract_prompt_and_expected(self, example: Dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract prompt and expected output from various data formats.

    Supports:
    - {"text": "prompt\nExpected: answer"}
    - {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    - {"input": "...", "output": "..."}

    Returns: (prompt, expected_answer)
    """
    prompt = None
    expected = None

    # Format 1: text field (legacy)
    if "text" in example and example["text"]:
        prompt = example["text"]
        expected = self.extract_expected_output(prompt)
        return prompt, expected

    # Format 2: messages array (chat format)
    if "messages" in example:
        messages = example["messages"]
        for msg in messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
            elif msg.get("role") == "assistant":
                expected = msg.get("content", "")
        return prompt, expected

    # Format 3: input/output fields
    if "input" in example:
        prompt = example["input"]
        expected = example.get("output", example.get("answer"))
        return prompt, expected

    return None, None
```

### Step 2: Update mine_adversarial_examples

Replace lines 254-318 in `mine_adversarial_examples()`:

```python
with torch.no_grad():
    for idx, example in enumerate(test_sample):
        try:
            # NEW: Use helper method
            prompt, expected = self.extract_prompt_and_expected(example)
            if not prompt:
                continue

            # Tokenize input (unchanged)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)

            # Get model prediction (unchanged)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

            # Decode prediction (unchanged)
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Calculate confidence (unchanged)
            if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                first_token_logits = outputs.scores[0][0]
                confidence = self.calculate_confidence(first_token_logits)
            else:
                confidence = 0.5

            confidences.append(confidence)
            stats["total_tested"] += 1

            # Check if adversarial
            is_adversarial = False
            adversarial_type = None

            if confidence < self.confidence_threshold:
                is_adversarial = True
                adversarial_type = "low_confidence"
                stats["low_confidence"] += 1

            # CHANGED: Use pre-extracted expected
            if expected:
                # For JSON outputs, try parsing and comparing
                prediction_check = prediction.lower().strip()
                expected_check = expected.lower().strip()

                # Simple substring check for now
                # (could be enhanced for JSON comparison)
                if expected_check not in prediction_check:
                    is_adversarial = True
                    adversarial_type = "incorrect_prediction"
                    stats["incorrect_predictions"] += 1

            if is_adversarial:
                adversarial_examples.append({
                    "prompt": prompt,  # NEW: store prompt separately
                    "expected": expected,  # NEW: store expected
                    "prediction": prediction,
                    "confidence": confidence,
                    "type": adversarial_type,
                    "checkpoint_step": step,
                    "timestamp": datetime.now().isoformat(),
                    "source_format": "messages" if "messages" in example else "text"
                })

        except Exception as e:
            logger.warning(f"Error processing example {idx}: {e}")
            continue
```

## Verification

```bash
# 1. Run miner with --status first to check test data loaded
python3 monitoring/adversarial_miner.py --status

# Should show:
# Test examples: 600  (not 0)

# 2. Run single iteration manually
python3 monitoring/adversarial_miner.py --samples 20 --interval 999999 &
# Wait a bit, then Ctrl+C

# 3. Check status file
cat status/adversarial_mining.json | jq '.mining_runs[-1].stats'

# Should show total_tested > 0
```

## Testing

Test with both formats:

```python
# Test 1: messages format
example_messages = {
    "messages": [
        {"role": "user", "content": "Is 2+2=4? Answer True or False"},
        {"role": "assistant", "content": "True"}
    ]
}

# Test 2: text format
example_text = {
    "text": "Is 2+2=4?\nExpected: True"
}

# Both should work
miner = AdversarialMiner()
p1, e1 = miner.extract_prompt_and_expected(example_messages)
p2, e2 = miner.extract_prompt_and_expected(example_text)

assert p1 is not None
assert p2 is not None
```

## Success Criteria

- [ ] `extract_prompt_and_expected()` method added
- [ ] Handles `messages[]` format correctly
- [ ] Handles `text` format (backward compatible)
- [ ] Handles `input/output` format
- [ ] `stats["total_tested"] > 0` after running on validation data
- [ ] Adversarial examples actually found and written
