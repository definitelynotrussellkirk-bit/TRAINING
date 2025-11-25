# TASK013: Adversarial Miner Status Structure Alignment

**Status:** Proposed
**Effort:** Small (20 min)
**File:** `monitoring/adversarial_miner.py`

## Objective

Align the status file structure written by adversarial miner with what `AdversarialPlugin` expects.

## Problem

### What the miner writes (line 338-345):
```python
return {
    "step": step,
    "checkpoint": str(checkpoint_path),
    "timestamp": datetime.now().isoformat(),
    "stats": stats,  # {total_tested, low_confidence, incorrect_predictions, avg_confidence}
    "adversarial_count": len(adversarial_examples),
    "output_file": str(output_file) if adversarial_examples else None
}
```

### What the plugin expects (lines 60-61):
```python
summary = {
    'total_examples': latest.get('total_examples_mined', 0),  # <-- doesn't exist
    'categories': {}  # <-- doesn't exist
}
categories = latest.get('categories', {})  # <-- doesn't exist
```

### Result
Plugin's `latest_summary` shows:
- `total_examples: 0` (always)
- `categories: {}` (always empty)

## Solution

Add `total_examples_mined` and `categories` fields to mining run records.

## Implementation

### Step 1: Update the return dict in mine_adversarial_examples

Replace lines 338-345:

```python
# Categorize adversarial examples
categories = defaultdict(lambda: {"count": 0, "avg_confidence": 0.0, "confidences": []})

for ex in adversarial_examples:
    cat = ex.get("type", "unknown")
    categories[cat]["count"] += 1
    categories[cat]["confidences"].append(ex.get("confidence", 0.5))

# Compute avg confidence per category
for cat_name, cat_data in categories.items():
    if cat_data["confidences"]:
        cat_data["avg_confidence"] = sum(cat_data["confidences"]) / len(cat_data["confidences"])
    del cat_data["confidences"]  # Remove intermediate data

return {
    "step": step,
    "checkpoint": str(checkpoint_path),
    "timestamp": datetime.now().isoformat(),
    "stats": stats,
    "adversarial_count": len(adversarial_examples),
    "total_examples_mined": len(adversarial_examples),  # NEW: matches plugin expectation
    "categories": dict(categories),  # NEW: breakdown by type
    "output_file": str(output_file) if adversarial_examples else None
}
```

### Step 2: Add import for defaultdict

At the top of the file, ensure:
```python
from collections import defaultdict
```

## Expected Status Format After Fix

`status/adversarial_mining.json`:

```json
{
  "mining_runs": [
    {
      "step": 157000,
      "checkpoint": "/path/to/checkpoint-157000",
      "timestamp": "2025-11-25T10:00:00",
      "stats": {
        "total_tested": 100,
        "low_confidence": 7,
        "incorrect_predictions": 23,
        "avg_confidence": 0.71
      },
      "adversarial_count": 26,
      "total_examples_mined": 26,
      "categories": {
        "low_confidence": {
          "count": 7,
          "avg_confidence": 0.42
        },
        "incorrect_prediction": {
          "count": 23,
          "avg_confidence": 0.78
        }
      },
      "output_file": "/path/to/adversarial_step_157000.jsonl"
    }
  ],
  "adversarial_examples_found": 26,
  "total_examples_tested": 100,
  "last_updated": "2025-11-25T10:00:00"
}
```

## Plugin Compatibility

After this change, `AdversarialPlugin.fetch()` will return:

```json
{
  "mining_runs": [...],
  "latest_summary": {
    "timestamp": "2025-11-25T10:00:00",
    "total_examples": 26,
    "categories": {
      "low_confidence": {"count": 7, "avg_loss": 0.42},
      "incorrect_prediction": {"count": 23, "avg_loss": 0.78}
    }
  }
}
```

Note: Plugin uses `avg_loss` but we're providing `avg_confidence`. This is a semantic mismatch but the plugin will at least have data to display. A future improvement could align terminology.

## Verification

```bash
# 1. Run miner and wait for one mining run
python3 monitoring/adversarial_miner.py --samples 50 &
sleep 60  # Wait for one run

# 2. Check status file structure
cat status/adversarial_mining.json | jq '.mining_runs[-1] | keys'
# Should include: categories, total_examples_mined

# 3. Check categories populated
cat status/adversarial_mining.json | jq '.mining_runs[-1].categories'
# Should show low_confidence and/or incorrect_prediction with counts

# 4. Test plugin (if API server running)
curl http://localhost:PORT/api/adversarial_mining | jq '.latest_summary'
# Should show non-zero total_examples and populated categories
```

## Success Criteria

- [ ] `total_examples_mined` field added to mining run records
- [ ] `categories` field added with breakdown by adversarial type
- [ ] Plugin returns non-zero `total_examples` in `latest_summary`
- [ ] Plugin returns populated `categories` dict
- [ ] Backward compatible (old status files still load)
