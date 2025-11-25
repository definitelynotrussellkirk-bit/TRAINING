# Learning Visualization Plan

**Created:** 2025-11-25
**Goal:** Comprehensive system to visualize and record learning in the model

## Current Infrastructure (Already Have)

### Analytics Modules (in `monitoring/analytics/`)
- `layer_drift_monitor.py` - Track weight changes per layer
- `parameter_stability.py` - Detect exploding/vanishing weights, dead layers
- `data_file_impact.py` - Track which training files had impact

### Status Files (in `status/`)
- `training_status.json` - Loss, step, learning rate
- `curriculum_eval.json` - Skill-specific accuracy
- `model_comparisons.json` - Checkpoint rankings
- `deployment_status.json` - Which model is deployed

### Missing Pieces
- No historical time series storage (only current snapshots)
- No representation geometry tracking
- No logprob trajectories
- No calibration tracking
- No unified visualization dashboard for learning dynamics

---

## Implementation Plan (Prioritized)

### Phase 1: Quick Wins (Tonight)

#### 1.1 Time Series Database
**File:** `monitoring/analytics/learning_history.py`

Store historical data for all metrics:
```python
@dataclass
class LearningSnapshot:
    step: int
    timestamp: str
    metrics: Dict[str, float]  # loss, accuracy, etc.
    layer_drifts: List[float]  # drift per layer
    weight_norms: List[float]  # norm per layer
    skill_scores: Dict[str, float]  # per-skill accuracy
```

Storage: `status/learning_history.jsonl` (append-only)

#### 1.2 Skill Radar Generator
**File:** `monitoring/analytics/skill_radar.py`

Generate radar/spider plots:
- Axes = skill categories (SYLLO levels, Binary difficulty)
- Each checkpoint = polygon overlay
- Output: PNG per checkpoint + animated GIF

```python
def generate_radar(skill_scores: Dict[str, float], checkpoint_name: str) -> Path:
    # matplotlib radar chart
    # Save to status/visualizations/radar_{checkpoint}.png
```

#### 1.3 Hard Example Board
**File:** `monitoring/analytics/hard_example_tracker.py`

Track canonical "rogues gallery" prompts:
```yaml
# config/hard_examples.yaml
examples:
  - id: "negation_trap_1"
    prompt: "No cats are dogs. All dogs bark. Do cats bark?"
    expected: "No"
    category: "negation"
  - id: "double_neg_1"
    prompt: "It is not true that no birds can fly..."
    expected: "Yes"
    category: "double_negation"
```

Track per checkpoint: correct/incorrect/error_type
Output: `status/hard_example_board.json`

---

### Phase 2: Representation Geometry (This Week)

#### 2.1 Probe Set Manager
**File:** `monitoring/analytics/probe_set.py`

Fixed set of ~500 prompts for consistent tracking:
- 100 easy SYLLO
- 100 medium SYLLO
- 100 hard SYLLO
- 100 edge cases (negation, double negation, quantifiers)
- 100 random from training distribution

Save hidden states at each checkpoint for these probes.

#### 2.2 Embedding Tracker
**File:** `monitoring/analytics/embedding_tracker.py`

For each checkpoint:
1. Load model
2. Run probe set through model
3. Extract hidden states (layer -1 and mid-layer)
4. Save mean-pooled embeddings per probe
5. Run UMAP to 2D
6. Save coordinates + metadata

Output:
- `status/embeddings/checkpoint_{step}.npz` - raw embeddings
- `status/visualizations/umap_{step}.png` - scatter plot

#### 2.3 Representational Similarity Analysis
**File:** `monitoring/analytics/rsa_tracker.py`

For each checkpoint:
1. Get embeddings for probe set
2. Compute pairwise cosine similarity matrix
3. Compare to previous checkpoint's matrix
4. Output "geometry drift" scalar

Track: How stable is the model's internal "concept space"?

---

### Phase 3: Token-Level Behavior (Next Week)

#### 3.1 Logprob Trajectory Tracker
**File:** `monitoring/analytics/logprob_tracker.py`

For fixed "canonical answers":
- Track per-token logprob across checkpoints
- Identify which tokens the model learns first/last
- Heatmap: x=token position, y=checkpoint, color=logprob

#### 3.2 Error Localization
**File:** `monitoring/analytics/error_localizer.py`

For benchmark failures:
- Align model output vs expected (token-level)
- Track where errors occur (position in answer)
- Over time: which positions improve?

---

### Phase 4: Calibration & Uncertainty

#### 4.1 Calibration Tracker
**File:** `monitoring/analytics/calibration_tracker.py`

Reliability diagrams:
- Bucket predictions by confidence (0.0-0.1, 0.1-0.2, ...)
- Plot predicted vs actual accuracy
- Track ECE (Expected Calibration Error) over time

#### 4.2 Entropy Tracker
**File:** `monitoring/analytics/entropy_tracker.py`

For probe set:
- Track output distribution entropy per position
- Detect overconfidence (entropy too low) or uncertainty (too high)

---

### Phase 5: Unified Dashboard

#### 5.1 Learning Dashboard
**File:** `monitoring/ui/learning_dashboard.html`

Multi-panel view:
```
┌─────────────────────────────────────────────────────────────────────┐
│ LEARNING DYNAMICS DASHBOARD                                         │
├────────────────────────┬────────────────────────────────────────────┤
│ Loss & Metrics         │ Skill Radar (current vs base)             │
│ [line chart]           │ [radar plot]                               │
├────────────────────────┼────────────────────────────────────────────┤
│ Weight Drift by Layer  │ Representation Space (UMAP)               │
│ [heatmap]              │ [scatter plot, colored by skill]          │
├────────────────────────┼────────────────────────────────────────────┤
│ Hard Example Board     │ Calibration Diagram                       │
│ [table: ✓/✗ per ckpt]  │ [reliability curve]                       │
└────────────────────────┴────────────────────────────────────────────┘
```

---

## Data Storage Schema

### Learning History (append-only log)
```json
// status/learning_history.jsonl
{"step": 164000, "timestamp": "2025-11-25T10:00:00", "train_loss": 1.23, "val_loss": 1.45, "syllo_easy": 0.85, "syllo_medium": 0.72, "syllo_hard": 0.54, "layer_drifts": [0.01, 0.02, ...], "geometry_drift": 0.03}
```

### Embeddings Archive
```
status/embeddings/
├── probe_set.json          # Fixed probe prompts + metadata
├── checkpoint_164000.npz   # Embeddings for this checkpoint
├── checkpoint_165000.npz
└── umap_coords.json        # 2D coords per checkpoint per probe
```

### Visualizations
```
status/visualizations/
├── loss_curve.png
├── skill_radar_latest.png
├── skill_radar_animated.gif
├── umap_latest.png
├── umap_animated.gif
├── layer_drift_heatmap.png
├── calibration_latest.png
└── hard_example_board.png
```

---

## Integration with GPU Task Scheduler

New task types:
- `collect_embeddings` - Extract embeddings for probe set (GPU required)
- `generate_visualizations` - Create all PNG/GIF outputs (CPU only)
- `update_learning_history` - Append latest metrics (CPU only)

Scheduler integration:
```python
# In gpu_task_scheduler.py
TASK_TYPES = {
    ...
    "collect_embeddings": TaskPriority.LOW,      # 3
    "generate_visualizations": TaskPriority.IDLE, # 4
    "update_learning_history": TaskPriority.IDLE, # 4
}
```

---

## Implementation Order

### Tonight (2-3 hours)
1. [ ] Create `learning_history.py` - Time series storage
2. [ ] Create `skill_radar.py` - Radar plots
3. [ ] Create `hard_example_tracker.py` - Rogues gallery
4. [ ] Add collection hooks to existing daemons

### This Week
5. [ ] Create probe set (500 canonical prompts)
6. [ ] Create `embedding_tracker.py` - UMAP snapshots
7. [ ] Create `rsa_tracker.py` - Geometry drift
8. [ ] Integrate with scheduler as task types

### Next Week
9. [ ] Create `logprob_tracker.py` - Token-level behavior
10. [ ] Create `calibration_tracker.py` - Reliability diagrams
11. [ ] Create unified learning dashboard

---

## Quick Start Commands

```bash
# Generate skill radar for current checkpoint
python3 monitoring/analytics/skill_radar.py --checkpoint latest

# Collect embeddings (runs on 3090)
python3 monitoring/analytics/embedding_tracker.py --checkpoint latest

# Generate UMAP visualization
python3 monitoring/analytics/embedding_tracker.py --visualize

# Update hard example board
python3 monitoring/analytics/hard_example_tracker.py --checkpoint latest

# Generate all visualizations
python3 monitoring/analytics/generate_all_visualizations.py
```

---

## Notes

- All heavy computation (embedding extraction) runs on 3090 via scheduler
- Visualization generation is CPU-only, can run anywhere
- Historical data is append-only for durability
- Probe set must be FIXED across all checkpoints for comparability
- Animation GIFs generated on-demand, not continuously
