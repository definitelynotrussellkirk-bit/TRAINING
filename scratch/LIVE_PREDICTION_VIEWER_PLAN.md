# Live Prediction Viewer Implementation Plan

**Created:** 2025-11-24
**Status:** Ready to implement
**Priority:** High - User wants live prediction viewing

---

## Current State (Verified)

### 3090 Inference Server
- ✅ API server running at `http://192.168.x.x:8765`
- ✅ Process: PID 807787 `python3 main.py`
- ✅ `/health` endpoint working
- ✅ `/generate` endpoint working (async job queue)
- ✅ Models on disk:
  - `Qwen3-0.6B` (base model) - REGISTERED & ACTIVE
  - `Qwen3-0.6B-step148k` (trained) - EXISTS BUT NOT REGISTERED
- 📍 API Documentation: `REMOTE_INFERENCE.md`

### 4090 Training Machine
- ✅ Training at step 148850
- ✅ Checkpoints syncing to 3090 via `checkpoint_sync_daemon.py`
- ✅ UI exists at `http://localhost:8080/monitoring/ui/predictions.html`
- ✅ Viewer engine exists: `monitoring/prediction_viewer_engine.py`
- ❌ No live predictions (static old data from manual run)
- ❌ No auto-generation daemon

---

## Problem

User wants **live, automatically updating predictions** in the viewer UI to see model performance during training.

Currently:
- Predictions must be generated manually
- Viewer shows stale data
- Using WRONG model (base model instead of trained checkpoint)

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  4090 Training Machine                                       │
│                                                              │
│  ┌──────────────────┐       ┌─────────────────────────┐    │
│  │ Training Daemon  │──────►│ current_model/ (step N) │    │
│  └──────────────────┘       └─────────────────────────┘    │
│                                      │                       │
│                                      │ rsync                 │
│  ┌──────────────────────────────────▼───────────────┐       │
│  │ Live Prediction Daemon (NEW)                     │       │
│  │  - Runs every 5 minutes                          │       │
│  │  - Reads status/training_status.json             │       │
│  │  - Calls 3090 API /generate                      │       │
│  │  - Polls job status                              │       │
│  │  - Saves status/latest_predictions.json          │       │
│  └──────────────────┬───────────────────────────────┘       │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────┐            │
│  │ predictions.html (auto-refreshes every 10s) │            │
│  │ http://localhost:8080/monitoring/ui/predictions.html     │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                      │
                      │ HTTP POST /generate
                      │ HTTP GET /jobs/{id}
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3090 Inference Machine (192.168.x.x)                    │
│                                                              │
│  ┌────────────────────────────────────────┐                 │
│  │ FastAPI Server (main.py) :8765         │                 │
│  │  - /generate - Queue inference job     │                 │
│  │  - /jobs/{id} - Check job status       │                 │
│  │  - /models/register - Add model        │                 │
│  │  - /models/set_active - Switch model   │                 │
│  └────────────────────────────────────────┘                 │
│                                                              │
│  Models:                                                     │
│  - Qwen3-0.6B (base) ✓ registered                           │
│  - Qwen3-0.6B-step148k (trained) ✗ not registered           │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Register Trained Model with 3090 API

**Goal:** Make the trained checkpoint available for inference

**Commands:**
```bash
# 1. Register the model
curl -X POST http://192.168.x.x:8765/models/register \
  -H "Content-Type: application/json" \
  -d '{"id": "Qwen3-0.6B-step148k", "source": "4090", "tags": "step148k,trained"}'

# 2. Set as active model
curl -X POST http://192.168.x.x:8765/models/set_active \
  -H "Content-Type: application/json" \
  -d '{"id": "Qwen3-0.6B-step148k"}'

# 3. Verify
curl http://192.168.x.x:8765/models/active | jq .
# Should show: "Qwen3-0.6B-step148k" as active
```

**Verification:**
- `curl http://192.168.x.x:8765/models` shows both models
- Active model is step148k, not base model

---

### Step 2: Create Live Prediction Daemon

**Goal:** Automatically generate predictions every 5 minutes

**File:** `monitoring/live_prediction_daemon.py`

**Key Features:**
- Runs continuously with 5-minute interval
- Reads current checkpoint step from `status/training_status.json`
- Loads 5 validation prompts from `data/validation/syllo_validation_20.jsonl`
- Calls 3090 API `/generate` endpoint for each prompt
- Polls job status until complete
- Saves results to `status/latest_predictions.json`
- Logs to `logs/live_prediction_daemon.log`

**Data Contract:**
Output format must match `PREDICTION_VIEWER_CONTRACT.md`:
```json
{
  "checkpoint": {
    "path": "/path/to/checkpoint",
    "step": 148000,
    "generated_at": "2025-11-24T06:42:00Z"
  },
  "predictions": [
    {
      "id": "pred_xyz",
      "difficulty": "hard",
      "prompt": "...",
      "expected": "...",
      "generated": "...",
      "exact_match": false,
      "semantic_match": true,
      "has_thinking_tags": false,
      "inference_time_ms": 5432
    }
  ]
}
```

**Implementation Notes:**
- Use `urllib` or `requests` for HTTP calls
- Handle connection errors gracefully (3090 may be offline)
- Use exponential backoff for job polling
- Max wait time: 2 minutes per job, then mark as timeout

---

### Step 3: Deploy Daemon

**Commands:**
```bash
cd /path/to/training

# Test run (manual, see output)
python3 monitoring/live_prediction_daemon.py --once

# If successful, start as background daemon
nohup python3 monitoring/live_prediction_daemon.py \
  --interval 300 \
  --count 5 \
  > logs/live_prediction_daemon.log 2>&1 &

# Save PID
echo $! > /tmp/live_prediction_daemon.pid

# Verify running
ps aux | grep live_prediction_daemon | grep -v grep

# Tail logs
tail -f logs/live_prediction_daemon.log
```

---

### Step 4: Verify UI Auto-Refresh

**Goal:** Confirm viewer shows live data

**Steps:**
1. Open `http://localhost:8080/monitoring/ui/predictions.html`
2. Check that:
   - Checkpoint info shows current step (~148850)
   - Predictions are recent (generated_at timestamp)
   - Data refreshes every 10 seconds (watch for updates)
3. Wait 5 minutes, verify new predictions appear

**Expected Behavior:**
- UI polls `/status/latest_predictions.json` every 10 seconds
- Shows latest checkpoint step
- Displays model outputs vs expected
- Accuracy metrics calculated

---

## File Reference

### Existing Files (DO NOT MODIFY)
- `monitoring/ui/predictions.html` - Viewer UI ✅
- `monitoring/prediction_viewer_engine.py` - Engine (reference only)
- `monitoring/PREDICTION_VIEWER_CONTRACT.md` - Data contract ✅
- `REMOTE_INFERENCE.md` - 3090 API docs ✅
- `data/validation/syllo_validation_20.jsonl` - Test data ✅

### New File to Create
- `monitoring/live_prediction_daemon.py` - Auto-prediction daemon ⚠️ NEW

### Modified Files
- None (daemon is standalone, writes to `status/latest_predictions.json`)

---

## API Reference (3090)

**Base URL:** `http://192.168.x.x:8765`

### Key Endpoints

**POST /generate** - Queue inference job
```bash
curl -X POST http://192.168.x.x:8765/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "SYLLO Puzzle...",
    "max_tokens": 2048,
    "temperature": 0.1,
    "mode": "normal"
  }'
```
Response:
```json
{"job_id": "gen_20251124_064300", "status": "pending"}
```

**GET /jobs/{job_id}** - Check job status
```bash
curl http://192.168.x.x:8765/jobs/gen_20251124_064300
```
Response (pending):
```json
{"job_id": "gen_...", "status": "pending"}
```
Response (done):
```json
{
  "job_id": "gen_...",
  "status": "done",
  "result": {
    "generated_text": "...",
    "inference_time_ms": 5234
  }
}
```

---

## Testing Plan

### Test 1: Model Registration
```bash
curl http://192.168.x.x:8765/models/active
# Expected: {"id": "Qwen3-0.6B-step148k", "is_active": true}
```

### Test 2: Single Inference
```bash
curl -X POST http://192.168.x.x:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test: 2+2=", "max_tokens": 10}'
# Expected: {"job_id": "gen_...", "status": "pending"}

# Poll until done
curl http://192.168.x.x:8765/jobs/gen_XXXXX
# Expected: {"status": "done", "result": {"generated_text": "..."}}
```

### Test 3: Daemon Single Run
```bash
python3 monitoring/live_prediction_daemon.py --once --count 2
# Expected:
# - 2 predictions generated
# - status/latest_predictions.json created
# - Checkpoint info shows current step
```

### Test 4: UI Refresh
```bash
# Open browser: http://localhost:8080/monitoring/ui/predictions.html
# Watch network tab for:
# - GET /status/latest_predictions.json every 10s
# - Data updates in UI
```

---

## Rollback Plan

If something breaks:

1. **Stop daemon:**
   ```bash
   kill $(cat /tmp/live_prediction_daemon.pid)
   ```

2. **Restore base model:**
   ```bash
   curl -X POST http://192.168.x.x:8765/models/set_active \
     -H "Content-Type: application/json" \
     -d '{"id": "Qwen3-0.6B"}'
   ```

3. **Manual prediction generation:**
   ```bash
   python3 monitoring/prediction_viewer_engine.py \
     --base-dir /path/to/training \
     --count 10
   ```

---

## Success Criteria

- ✅ Step148k model registered and active on 3090
- ✅ Daemon running as background process
- ✅ `status/latest_predictions.json` updates every 5 minutes
- ✅ UI shows live data with current checkpoint
- ✅ Predictions use trained model (not base model)
- ✅ Accuracy metrics visible in UI

---

## Notes

- **Interval:** 5 minutes chosen to balance freshness vs GPU load
- **Count:** 5 predictions per run (fast, keeps GPU responsive)
- **Concurrency:** 3090 handles curriculum optimizer + this daemon fine
- **Error handling:** Daemon continues on errors, logs issues
- **Model updates:** When checkpoint syncs new step, daemon will detect and use it

---

## Next Steps for AI

1. Register step148k model with 3090 API
2. Create `monitoring/live_prediction_daemon.py`
3. Test with `--once` flag first
4. Deploy as background daemon
5. Verify UI shows live updates

Reference docs:
- `REMOTE_INFERENCE.md` for API details
- `PREDICTION_VIEWER_CONTRACT.md` for data format
- `monitoring/prediction_viewer_engine.py` for example code
