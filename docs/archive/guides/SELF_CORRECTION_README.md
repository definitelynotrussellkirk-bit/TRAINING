# Self-Correction Training System

**Status:** ✅ Complete and tested
**Created:** 2025-11-21

## What It Does

Teaches your model to:
1. Answer questions
2. **Recognize its own mistakes** using programmatic error codes
3. **Self-correct** based on consistent hints

## Key Innovation: Programmatic Error Codes

**Consistent, automatic feedback** that works across all domains:

```
Answer: "180"
Golden: "15 × 12 = 180. This is calculated by multiplying 15 by 12."

Error Codes:
• LENGTH_SHORT_BY_16_TOKENS
• TOO_BRIEF
• MISSING_KEY_CONCEPTS_4_TERMS
```

**Crucially:** These codes can be computed automatically by comparing any answer to any golden answer!

## Quick Start

```bash
# 1. Test the system
./test_self_correction.sh

# 2. Run demo
python3 demo_self_correction.py

# 3. Use on real data
python3 generate_initial_answers.py --input qa.jsonl --output answers.jsonl
python3 self_correction_trainer.py --input qa.jsonl --output train.jsonl --initial-answers answers.jsonl
cp train.jsonl inbox/
```

## Example Output

**For each wrong answer, generates 3 training examples:**

### Example 1: Initial Attempt
```
Prompt: "List three primary colors."
Response: "The three primary colors are red, blue, and yellow."
```

### Example 2: Self-Evaluation
```
Prompt: "List three primary colors.

Your previous answer:
The three primary colors are red, blue, and yellow.

Error indicators: CLOSE_BUT_INACCURATE | STRUCTURE_SHOULD_BE_LIST

Based on these error indicators, evaluate your answer."

Response: "Looking at the error indicators... my formatting was wrong, I need to reconsider my answer."
```

### Example 3: Correction
```
Prompt: "List three primary colors.

Error indicators from your previous attempt: STRUCTURE_SHOULD_BE_LIST

Please provide the correct answer."

Response: "The three primary colors are:
1. Red
2. Blue
3. Yellow"
```

## Error Codes Reference

**Length:** `LENGTH_SHORT_BY_X`, `LENGTH_LONG_BY_X`, `TOO_BRIEF`, `TOO_VERBOSE`

**Content:** `VERY_DIFFERENT_CONTENT`, `SIMILARITY_X%`, `CLOSE_BUT_INACCURATE`, `OFF_TOPIC_CONTENT`, `PRIMARY_ANSWER_MISSING`

**Numeric:** `MISSING_NUMBERS`, `NUMERIC_ERROR_MISSING_X_VALUES`, `NUMERIC_ERROR_EXTRA_X_VALUES`

**Structure:** `STRUCTURE_SHOULD_BE_LIST`, `STRUCTURE_SHOULD_BE_PROSE`, `MISSING_CODE_BLOCK`, `FORMAT_SHOULD_BE_JSON`

**Concepts:** `MISSING_KEY_CONCEPTS_X_TERMS`

## Files

- `self_correction_trainer.py` - Core pipeline (400+ lines)
- `generate_initial_answers.py` - Generate answers from model
- `demo_self_correction.py` - Interactive demo
- `test_self_correction.sh` - Quick test
- `SELF_CORRECTION_GUIDE.md` - Full documentation (500+ lines)

## Design Decisions (from user input)

1. ✅ **Train on self-eval** - Teach reflection as a skill
2. ✅ **Programmatic feedback** - Consistent error codes that require thinking
3. ✅ **Separate examples** - 3 distinct training patterns per mistake
4. ✅ **Check vs golden** - Always compare to golden answer (not self-eval) to trigger corrections

## What Makes This Cool

1. **Fully automatic** - No manual labeling of errors
2. **Domain agnostic** - Works on math, code, reasoning, anything
3. **Consistent feedback** - Model learns standard error patterns
4. **Self-improvement** - Model learns to recognize and fix its own mistakes
5. **Scalable** - Generate 1000s of examples from existing Q&A data

## Future Enhancements

- Multi-round corrections (try again after first fix)
- Confidence scores ("I'm 80% sure this is right")
- Domain-specific error codes (syntax_error, calculation_error, etc.)
- Difficulty-based hints (harder questions get more help)

## Theory

**Why this works:**

Traditional training: `prompt → answer` (learn to answer)

Self-correction training adds:
- `prompt + my_answer + hints → self_eval` (learn to reflect)
- `prompt + hints → corrected_answer` (learn to improve)

**Result:** Model develops metacognitive skills - it can think about its own thinking.

---

**See `SELF_CORRECTION_GUIDE.md` for complete documentation.**
