# Self-Correction Training System

**Created:** 2025-11-21

## Overview

This system teaches models to:
1. **Answer questions** (initial attempt)
2. **Self-evaluate** using programmatic error codes
3. **Correct mistakes** based on hints

### Key Features

**Programmatic Feedback:** Consistent error codes that:
- Work across all domains (math, reasoning, code, etc.)
- Require the model to think (hints, not answers)
- Can be computed automatically

**Training Strategy:**
- Generate 3 separate examples per question
- Train on self-evaluation (teaches reflection)
- Compare against golden answers (not self-eval) to trigger corrections

---

## Error Code System

### Available Error Codes

**Length Issues:**
- `LENGTH_SHORT_BY_X_TOKENS` - Answer too brief
- `LENGTH_LONG_BY_X_TOKENS` - Answer too verbose
- `TOO_BRIEF` - Significantly shorter (char-level)
- `TOO_VERBOSE` - Significantly longer (char-level)

**Content Quality:**
- `VERY_DIFFERENT_CONTENT` - <30% similarity to golden
- `SIMILARITY_X%` - Shows exact similarity percentage
- `CLOSE_BUT_INACCURATE` - High similarity but still wrong
- `OFF_TOPIC_CONTENT` - >50% irrelevant words
- `PRIMARY_ANSWER_MISSING` - Empty or nearly empty response

**Numeric Errors:**
- `MISSING_NUMBERS` - Should contain numbers but doesn't
- `NUMERIC_ERROR_MISSING_X_VALUES` - Wrong/missing numbers
- `NUMERIC_ERROR_EXTRA_X_VALUES` - Extra incorrect numbers

**Structure Issues:**
- `STRUCTURE_SHOULD_BE_LIST` - Should be bullet/numbered list
- `STRUCTURE_SHOULD_BE_PROSE` - Should be paragraph form
- `MISSING_CODE_BLOCK` - Should contain code
- `UNEXPECTED_CODE_BLOCK` - Shouldn't contain code
- `FORMAT_SHOULD_BE_JSON` - Should be JSON format

**Content Gaps:**
- `MISSING_KEY_CONCEPTS_X_TERMS` - Missing important terms/concepts

### Example Error Codes in Action

**Too Short:**
```
Answer: "The answer is 42."
Golden: "The answer is 42 because it represents the ultimate answer..."

Error codes:
- LENGTH_SHORT_BY_21_TOKENS
- TOO_BRIEF
- VERY_DIFFERENT_CONTENT
- MISSING_KEY_CONCEPTS_12_TERMS
```

**Wrong Number:**
```
Answer: "The calculation gives us 156."
Golden: "The calculation gives us 142."

Error codes:
- CLOSE_BUT_INACCURATE
- NUMERIC_ERROR_MISSING_1_VALUES
- NUMERIC_ERROR_EXTRA_1_VALUES
```

---

## Usage

### Full Pipeline (Recommended)

```bash
# 1. Prepare golden Q&A dataset
# Format: {"prompt": "...", "response": "..."}
cat > golden_qa.jsonl <<EOF
{"prompt": "What is 2+2?", "response": "2+2 equals 4. This is basic addition."}
{"prompt": "Explain photosynthesis", "response": "Photosynthesis is..."}
EOF

# 2. Generate initial answers from your model
python3 generate_initial_answers.py \
  --input golden_qa.jsonl \
  --output initial_answers.jsonl \
  --model /path/to/training/current_model

# 3. Generate self-correction training data
python3 self_correction_trainer.py \
  --input golden_qa.jsonl \
  --output training_self_correction.jsonl \
  --initial-answers initial_answers.jsonl

# 4. Train on the generated data
cp training_self_correction.jsonl inbox/
# Daemon will process it automatically
```

### What Gets Generated

For each question where the model's initial answer is wrong, you get **3 training examples:**

#### Example 1: Initial Attempt
```json
{
  "prompt": "What is 2+2?",
  "response": "4"
}
```
*Teaches basic answering*

#### Example 2: Self-Evaluation
```json
{
  "prompt": "What is 2+2?\n\nYour previous answer:\n4\n\nError indicators: LENGTH_SHORT_BY_5_TOKENS | TOO_BRIEF\n\nBased on these error indicators, evaluate your answer. Was it correct? What might be wrong?",
  "response": "Looking at the error indicators:\n- LENGTH_SHORT_BY_5_TOKENS\n- TOO_BRIEF\n\nMy answer appears to be incorrect. The indicators suggest: my response was the wrong length, I need to reconsider my answer."
}
```
*Teaches reflection and error recognition*

#### Example 3: Correction
```json
{
  "prompt": "What is 2+2?\n\nError indicators from your previous attempt: LENGTH_SHORT_BY_5_TOKENS | TOO_BRIEF\n\nPlease provide the correct answer, addressing the errors indicated above.",
  "response": "2+2 equals 4. This is basic addition."
}
```
*Teaches fixing mistakes based on hints*

---

## Workflow Details

### Step 1: Generate Initial Answers

```bash
python3 generate_initial_answers.py \
  --input golden_qa.jsonl \
  --output initial_answers.jsonl \
  --model current_model/ \
  --temperature 0.7
```

**Options:**
- `--max-samples N` - Only process first N samples (for testing)
- `--temperature T` - Sampling temperature (default: 0.7)

**Output format:**
```json
{
  "prompt": "...",
  "answer": "... model's first attempt ...",
  "golden": "... correct answer (for reference) ..."
}
```

### Step 2: Generate Self-Correction Data

```bash
python3 self_correction_trainer.py \
  --input golden_qa.jsonl \
  --output training_data.jsonl \
  --initial-answers initial_answers.jsonl
```

**What happens:**
1. Compares each initial answer to golden answer
2. Generates programmatic error codes
3. Creates 3 training examples per incorrect answer
4. Outputs standard training format + metadata

**Output:**
- `training_data.jsonl` - Ready-to-train examples
- `training_data_metadata.json` - Statistics and breakdown

**Metadata example:**
```json
{
  "stats": {
    "total_questions": 100,
    "correct_first_try": 15,
    "needed_correction": 85,
    "examples_generated": 270
  },
  "examples_breakdown": {
    "initial_attempts": 85,
    "self_evaluations": 85,
    "corrections": 85,
    "standard_qa": 15
  }
}
```

---

## Advanced Usage

### Testing Error Codes

```bash
python3 self_correction_trainer.py --test-error-codes
```

Shows how error codes are generated for sample cases.

### Custom Model Inference

If you want to use a different inference method:

```python
from self_correction_trainer import SelfCorrectionPipeline

def my_model_fn(prompt):
    # Your custom inference code
    return generated_answer

pipeline = SelfCorrectionPipeline(model_inference_fn=my_model_fn)
examples = pipeline.generate_from_qa_pair(
    prompt="What is 2+2?",
    golden_answer="2+2 equals 4. This is basic addition."
)
```

### Batch Processing

For large datasets:

```bash
# Split into batches
split -l 1000 large_dataset.jsonl batch_

# Process each batch
for batch in batch_*; do
    python3 generate_initial_answers.py \
      --input $batch \
      --output ${batch}_answers.jsonl
done

# Combine
cat batch_*_answers.jsonl > all_answers.jsonl

# Generate training data
python3 self_correction_trainer.py \
  --input large_dataset.jsonl \
  --output training_data.jsonl \
  --initial-answers all_answers.jsonl
```

---

## Training Strategy

### Recommended Approach

**Phase 1: Standard Training**
- Train on normal Q&A data first
- Build basic answering capability

**Phase 2: Self-Correction Training**
- Generate initial answers from Phase 1 model
- Create self-correction training data
- Train on mixed dataset:
  - 50% standard Q&A
  - 25% self-evaluation examples
  - 25% correction examples

**Phase 3: Evaluation**
- Test on held-out set
- Check if model can:
  - Recognize its own errors
  - Correct based on hints
  - Improve from feedback

### Mixing Ratios

You can adjust the mix of example types:

```python
# After generating examples, filter by type
with open('training_data.jsonl') as f:
    examples = [json.loads(line) for line in f]

# Create custom mix
standard = [ex for ex in examples if ex.get('type') == 'standard_qa']
initial = [ex for ex in examples if ex.get('type') == 'initial_attempt']
self_eval = [ex for ex in examples if ex.get('type') == 'self_evaluation']
correction = [ex for ex in examples if ex.get('type') == 'correction']

# Mix as desired
custom_mix = standard + self_eval + correction  # Skip initial attempts
```

---

## Troubleshooting

### Initial Answers All Wrong

**Problem:** Model generates poor initial answers
**Solution:**
- Increase temperature for more diverse samples
- Train on standard Q&A first (Phase 1)
- Check if prompts need formatting

### Too Many Error Codes

**Problem:** Every answer gets 5+ error codes
**Solution:**
- Answers might be very different from golden
- Review golden answers - are they too specific?
- Consider adjusting error code thresholds in `ErrorCodeGenerator`

### Not Enough Correction Examples

**Problem:** Most answers correct on first try
**Solution:**
- Good problem to have! Model is doing well
- Use harder questions for self-correction training
- Intentionally use a weaker/earlier checkpoint for initial answers

---

## System Architecture

```
Golden Q&A Dataset
        ↓
    [Model Inference]
        ↓
Initial Answers → [Error Code Generator] ← Golden Answers
        ↓
Error Codes
        ↓
[Training Example Generator]
        ↓
    3 Examples Per Question:
    1. Initial attempt
    2. Self-evaluation
    3. Correction
        ↓
Training Data → [Training Daemon]
```

---

## Future Enhancements

**Potential additions:**
- Multi-round corrections (try 3 times)
- Confidence scores ("I'm 70% confident this is right")
- Difficulty-based error codes (harder questions get more hints)
- Domain-specific error codes (code: syntax_error, math: calculation_error)
- Error code explanation training ("What does LENGTH_SHORT_BY_10_TOKENS mean?")

---

## Files

- `self_correction_trainer.py` - Main training data generator
- `generate_initial_answers.py` - Generate initial answers from model
- `SELF_CORRECTION_GUIDE.md` - This guide

---

## Quick Reference

```bash
# Full pipeline
python3 generate_initial_answers.py --input qa.jsonl --output answers.jsonl
python3 self_correction_trainer.py --input qa.jsonl --output train.jsonl --initial-answers answers.jsonl
cp train.jsonl inbox/

# Test error codes
python3 self_correction_trainer.py --test-error-codes

# Process subset
python3 generate_initial_answers.py --input qa.jsonl --output answers.jsonl --max-samples 100
```
