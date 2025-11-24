# Prediction Viewer - Data Contract & Documentation

**Version:** 1.0
**Last Updated:** 2025-11-24
**Status:** Production

## Overview

The Prediction Viewer provides interactive inspection of model outputs during training, allowing human evaluation of semantic correctness beyond simple format matching.

## Why This Exists

**Problem:** Automated metrics (loss, perplexity) don't tell you if the model's *answers are actually correct*. A model might:
- Have low loss but wrong logic
- Use `<thinking>` tags (we don't want this)
- Format answers differently but correctly
- Miss subtle requirements

**Solution:** Human-in-the-loop evaluation with side-by-side comparison.

---

## Data Contract

### API Endpoint: `/api/predictions`

**URL:** `http://localhost:8090/api/predictions`
**Method:** GET
**Response Format:** JSON

#### Response Schema

```json
{
  "checkpoint": {
    "path": "/home/user/TRAINING/models/current_model/checkpoint-102000",
    "step": 102000,
    "training_step": 102534,
    "last_updated": "2025-11-24T05:43:21Z",
    "model_name": "Qwen3-0.6B",
    "status": "active"
  },
  "predictions": [
    {
      "id": "pred_20251124_054321_001",
      "difficulty": "easy",
      "prompt": "SYLLO Puzzle...",
      "expected_answer": "{\n  \"solutions\": [...],\n  \"inventory_check\": {...}\n}",
      "model_output": "{\n  \"solutions\": [...],\n  \"inventory_check\": {...}\n}",
      "extracted_answer": {
        "solutions": ["answer1", "answer2"],
        "inventory": ["unused1"]
      },
      "metrics": {
        "exact_match": false,
        "semantic_match": true,
        "has_thinking_tags": false,
        "format_valid": true,
        "completion_time_ms": 342
      },
      "manual_grade": null,
      "timestamp": "2025-11-24T05:43:22Z"
    }
  ],
  "statistics": {
    "total_predictions": 20,
    "by_difficulty": {
      "easy": {"total": 8, "graded": 5, "correct": 4},
      "medium": {"total": 7, "graded": 4, "correct": 2},
      "hard": {"total": 5, "graded": 3, "correct": 1}
    },
    "manual_accuracy": 0.583,
    "auto_accuracy": 0.450,
    "has_thinking_tags_count": 0
  },
  "last_updated": "2025-11-24T05:43:25Z"
}
```

---

## Field Definitions

### `checkpoint` Object

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `path` | string | Absolute path to checkpoint directory | `/home/user/TRAINING/models/current_model/checkpoint-102000` |
| `step` | integer | Checkpoint save step (from checkpoint name) | `102000` |
| `training_step` | integer | Current training step (may be ahead of checkpoint) | `102534` |
| `last_updated` | ISO 8601 | When checkpoint was last modified | `2025-11-24T05:43:21Z` |
| `model_name` | string | Base model architecture | `Qwen3-0.6B` |
| `status` | enum | Checkpoint status: `active`, `syncing`, `stale`, `missing` | `active` |

### `predictions[]` Array

Each prediction contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique prediction ID (format: `pred_YYYYMMDD_HHMMSS_NNN`) |
| `difficulty` | enum | Puzzle difficulty: `easy`, `medium`, `hard` |
| `prompt` | string | Full input prompt sent to model |
| `expected_answer` | string | Ground truth answer from validation data |
| `model_output` | string | Raw model generation (unprocessed) |
| `extracted_answer` | object | Parsed/extracted answer (JSON or structured data) |
| `metrics` | object | Automated quality metrics |
| `manual_grade` | enum | Human grade: `correct`, `wrong`, `partial`, `null` (not graded) |
| `timestamp` | ISO 8601 | When prediction was generated |

### `metrics` Object

| Field | Type | Calculation | Meaning |
|-------|------|-------------|---------|
| `exact_match` | boolean | `model_output == expected_answer` (string comparison) | Perfect character-by-character match |
| `semantic_match` | boolean | JSON parse + deep equals (ignores whitespace/order) | Logically equivalent answers |
| `has_thinking_tags` | boolean | Regex search for `<thinking>` or `</thinking>` | Model is using CoT tags (undesirable) |
| `format_valid` | boolean | JSON.parse() succeeds | Output is valid JSON |
| `completion_time_ms` | integer | Inference latency in milliseconds | How long generation took |

---

## Difficulty Definitions

Based on SYLLO puzzle generation (see `singleSKILL/skill_syllo_variant`):

| Difficulty | Target Words | Complexity | Zipf Range | Overlap/Red Herrings |
|-----------|--------------|------------|------------|---------------------|
| **Easy** | 4 words | Low ambiguity | Higher frequency | Minimal |
| **Medium** | 5-6 words | Moderate | Mid-range | Moderate |
| **Hard** | 7 words | High ambiguity | Lower frequency | Significant |

**Note:** Difficulty is embedded in the prompt text (e.g., "Difficulty: Easy").

---

## Accuracy Calculation

### Auto Accuracy
```
auto_accuracy = count(semantic_match == true) / total_predictions
```
- Based on automated JSON comparison
- **Limitation:** May miss formatting variations, may false-positive on lucky guesses

### Manual Accuracy
```
manual_accuracy = count(manual_grade == "correct") / count(manual_grade != null)
```
- Based on human evaluation
- **Ground truth** for model quality
- Only includes graded predictions

### By-Difficulty Accuracy
```
difficulty_accuracy = correct_for_difficulty / total_for_difficulty
```
- Tracks performance across difficulty levels
- Useful for curriculum optimization

---

## Grading Guidelines

When manually grading predictions:

| Grade | Criteria | Example |
|-------|----------|---------|
| **correct** | Answer is logically correct, even if formatting differs | Solutions match, inventory correct |
| **partial** | Some answers correct, but incomplete or minor errors | 3/4 solutions correct |
| **wrong** | Logically incorrect answer | Wrong word assignments |
| *no grade* | Leave null if unclear or need more context | Ambiguous edge case |

**Special Cases:**
- ✅ Ignore whitespace differences
- ✅ Ignore JSON key order
- ❌ Mark wrong if uses `<thinking>` tags (we're training to NOT use them)
- ❌ Mark wrong if format is invalid (can't parse answer)

---

## Integration Points

### Dashboard Integration
- **Master Dashboard:** Link from `http://localhost:8090/ui/master_dashboard.html`
- **API Backend:** Served by `monitoring/servers/launch_live_monitor.py`
- **Data Source:** Generated by `monitoring/preview_engine.py`

### Data Flow
```
1. Training Step N completed
2. preview_engine.py triggers (every 1000 steps)
3. Sends prompts to RTX 3090 inference API (192.168.x.x:8765)
4. Receives model outputs
5. Saves to status/latest_predictions.json
6. Dashboard polls /api/predictions every 10s
7. User grades predictions via UI
8. Grades saved to status/prediction_grades.json
```

---

## API Endpoints

### GET `/api/predictions`
- Returns last 20 predictions
- Optional query params:
  - `?difficulty=easy` - Filter by difficulty
  - `?checkpoint=102000` - Filter by checkpoint
  - `?graded=false` - Show only ungraded

### POST `/api/predictions/grade`
**Request Body:**
```json
{
  "prediction_id": "pred_20251124_054321_001",
  "grade": "correct",
  "notes": "Answer correct but used different key names"
}
```

**Response:**
```json
{
  "success": true,
  "prediction_id": "pred_20251124_054321_001",
  "grade": "correct"
}
```

---

## File Locations

| File | Purpose |
|------|---------|
| `status/latest_predictions.json` | Last 20 predictions (source of truth) |
| `status/prediction_grades.json` | Human grades database |
| `data/prediction_history/pred_YYYYMMDD.json` | Daily prediction archives |
| `monitoring/ui/predictions.html` | Prediction viewer UI |
| `monitoring/preview_engine.py` | Generates predictions |

---

## Performance Notes

- **Inference Latency:** ~300-500ms per prediction (RTX 3090)
- **Batch Size:** 5 predictions per checkpoint evaluation
- **Frequency:** Every 1000 training steps
- **Storage:** ~50KB per prediction batch
- **Retention:** Keep last 20 predictions in memory, archive daily

---

## Future Enhancements

- [ ] Confidence scores (model's certainty about answer)
- [ ] Multi-annotator agreement tracking
- [ ] Automatic adversarial example mining from "wrong" predictions
- [ ] Trend analysis: accuracy over training steps
- [ ] Export graded examples for fine-tuning

---

## Troubleshooting

### "Checkpoint: Unknown"
- Checkpoint sync daemon not running
- Check: `ps aux | grep checkpoint_sync`
- Fix: Restart sync daemon

### No predictions showing
- Preview engine not running
- Check: `cat status/latest_predictions.json`
- Generate manually: `python3 monitoring/preview_engine.py --step current --count 5`

### All predictions show "has_thinking_tags: true"
- Model is using Chain-of-Thought format
- This is undesirable for your use case
- May need to add penalty for `<thinking>` tokens

---

## Contact & Updates

This contract is maintained alongside the codebase.
For questions or updates: Check `CLAUDE.md` for system overview.
