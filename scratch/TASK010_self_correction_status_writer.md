# TASK010: Self-Correction Status Writer

**Status:** Proposed
**Effort:** Small (30-45 min)
**File:** `monitoring/self_correction_loop.py`

## Objective

Add `update_status()` method to `SelfCorrectionLoop` that writes `status/self_correction.json` after each pipeline run, matching the format expected by `SelfCorrectionPlugin`.

## Current State

```python
# SelfCorrectionLoop currently:
# - Tracks stats in self.stats dict (memory only)
# - Writes corrections to queue/corrections/*.jsonl
# - Writes pattern reports to logs/error_patterns/*.json
# - NEVER writes status/self_correction.json
```

## Expected Status Format

From `monitoring/api/plugins/self_correction.py` lines 55-74:

```json
{
  "correction_runs": [
    {
      "timestamp": "2025-11-25T03:30:00",
      "errors_captured": 50,
      "corrections_generated": 50,
      "error_patterns": [
        {"type": "Misclassifying valid syllogisms", "count": 20, "description": "..."}
      ],
      "tested": 120,
      "correct": 70,
      "incorrect": 50
    }
  ],
  "total_errors_captured": 200,
  "total_corrections_generated": 180,
  "last_updated": "2025-11-25T03:30:00"
}
```

## Implementation

### Step 1: Add status file path

```python
# In __init__(), add:
self.status_file = self.base_dir / "status" / "self_correction.json"
```

### Step 2: Add load/save status methods

```python
def _load_status(self) -> Dict:
    """Load previous status if exists"""
    if self.status_file.exists():
        with open(self.status_file) as f:
            return json.load(f)
    return {
        "correction_runs": [],
        "total_errors_captured": 0,
        "total_corrections_generated": 0,
        "last_updated": None
    }

def _save_status(self):
    """Save status to JSON"""
    self.status["last_updated"] = datetime.now().isoformat()
    self.status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(self.status_file, 'w') as f:
        json.dump(self.status, f, indent=2)
```

### Step 3: Add update_status method

```python
def update_status(self, patterns: List[Dict] = None):
    """Update status file after pipeline run"""

    # Build error_patterns list for this run
    error_patterns = []
    if patterns:
        for p in patterns[:10]:  # Top 10 patterns
            error_patterns.append({
                "type": p.get("error_type", "Unknown"),
                "count": p.get("frequency", 0),
                "description": p.get("sample_problems", [""])[0][:100]
            })

    # Create run record
    run_record = {
        "timestamp": datetime.now().isoformat(),
        "errors_captured": self.stats["incorrect"],
        "corrections_generated": self.stats["corrections_generated"],
        "error_patterns": error_patterns,
        "tested": self.stats["tested"],
        "correct": self.stats["correct"],
        "incorrect": self.stats["incorrect"]
    }

    # Update status
    self.status["correction_runs"].append(run_record)
    self.status["total_errors_captured"] += self.stats["incorrect"]
    self.status["total_corrections_generated"] += self.stats["corrections_generated"]

    self._save_status()
    logger.info(f"Status updated: {self.status_file}")
```

### Step 4: Call update_status in pipeline

In `run_validation_pipeline()`, after logging stats (line 438):

```python
# Add at end of run_validation_pipeline, after stats logging:
patterns = self.mine_patterns() if self.pattern_db else []
self.update_status(patterns)
```

Also call after `process_error_batch()`:

```python
# In process_error_batch(), after mine_patterns():
patterns = self.mine_patterns()
if patterns:
    self.report_patterns(patterns)
self.update_status(patterns)  # <-- Add this
```

### Step 5: Initialize status in __init__

```python
# In __init__(), after directories setup:
self.status = self._load_status()
```

## Verification

```bash
# 1. Run with a test file
python3 monitoring/self_correction_loop.py \
  --file /path/to/training/data/validation/val_easy_200.jsonl \
  --error-threshold 5

# 2. Check status file created
cat status/self_correction.json | jq .

# 3. Verify plugin can read it
curl http://localhost:PORT/api/self_correction | jq .latest_summary
```

## Success Criteria

- [ ] `status/self_correction.json` created after pipeline run
- [ ] Contains `correction_runs` array with expected fields
- [ ] `total_errors_captured` and `total_corrections_generated` accumulate across runs
- [ ] Plugin returns non-empty `latest_summary`
