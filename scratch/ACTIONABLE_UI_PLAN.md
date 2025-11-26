# Actionable Dashboard UI Plan

**Last Updated:** 2025-11-26
**Goal:** Make every card answer "What changed, and what should I do about it?"

---

## Current State Assessment

### What Already Exists

**18 Unified API Plugins:**
- training_status, gpu_stats (4090/3090), curriculum, regression, model_comparison
- adversarial, confidence, testing, self_correction, checkpoints
- skill_metrics, training_analytics (layer_drift, stability, file_impact)
- system_health, retention, storage

**GPU Scheduler API (port 8766):**
- `/api/health` - scheduler running status
- `/api/metrics` - queue_length, active_task, gpu_utilization, tasks_completed, etc.
- `/api/task-types` - available task types

**Master Dashboard:**
- Training, GPU, Curriculum, Regression, Model Comparison cards
- Transfer Learning card (skill_metrics)
- Layer Drift, Parameter Stability cards (training_analytics)

### Gaps Identified

| Gap | Backend Status | UI Status |
|-----|----------------|-----------|
| GPU Scheduler visibility | ✅ API exists | ❌ No card |
| Retention/Storage warnings | ✅ Plugin exists | ❌ No card |
| Data File Impact triage | ✅ Plugin provides data | ❌ No card |
| Curriculum data flow | ⚠️ Partial (no queue status) | ⚠️ Partial card |
| Self-correction "did it help?" | ❌ Backend missing | ❌ No card |
| Action labels/thresholds | N/A | ❌ Not implemented |

---

## Implementation Plan

### Phase 1: Wire Missing Data to UI (Backend → UI)

**1.1 GPU Scheduler Plugin + Card**

Create: `monitoring/api/plugins/scheduler.py`
```python
class SchedulerPlugin(BasePlugin):
    """Fetches GPU scheduler metrics from 8766 API."""

    def fetch(self):
        # GET http://192.168.x.x:8766/api/metrics
        # GET http://192.168.x.x:8766/api/health
        return {
            'status': 'running',
            'queue_length': 17,
            'active_task': 'automated_test',
            'gpu_utilization': 27,
            'in_target_band': True,  # 20-80%
            'tasks_completed_last_hour': 15,
            'queue_by_priority': {
                'critical': 0, 'high': 2, 'normal': 10, 'low': 3, 'idle': 2
            },
            'warnings': []
        }
```

Add to master_dashboard.html (3090 column):
```html
<div class="card card-scheduler">
    <h3>🎛️ GPU Scheduler</h3>
    <div class="metric-item">
        <span class="metric-label">Queue</span>
        <span class="metric-value" id="schedulerQueue">--</span>
    </div>
    <div class="metric-item">
        <span class="metric-label">Active Task</span>
        <span class="metric-value" id="schedulerActiveTask">--</span>
    </div>
    <div class="metric-item">
        <span class="metric-label">GPU Target</span>
        <span class="metric-value" id="schedulerUtilBand">--</span>
    </div>
    <div class="action-hint" id="schedulerAction">--</div>
</div>
```

**1.2 Retention/Storage Card**

Plugin already exists at `monitoring/api/plugins/retention.py`.
Provides: usage_pct, health (good/warning/critical), warnings[], checkpoints.count

Add to master_dashboard.html (bottom):
```html
<div class="card card-retention">
    <h3>💾 Storage & Retention</h3>
    <div class="metric-item">
        <span class="metric-label">Usage</span>
        <span class="metric-value" id="retentionUsage">--</span>
        <div class="progress-bar"><div id="retentionBar"></div></div>
    </div>
    <div class="metric-item">
        <span class="metric-label">Checkpoints</span>
        <span class="metric-value" id="retentionCheckpoints">--</span>
    </div>
    <div class="warning-list" id="retentionWarnings"></div>
    <div class="action-hint" id="retentionAction">--</div>
</div>
```

**1.3 Data File Impact Card**

Plugin: `training_analytics.py` → data_file_impact.recent_impacts[]
Surface top 5 positive/negative files.

Add to master_dashboard.html (analytics section):
```html
<div class="card card-data-impact">
    <h3>📊 Data File Impact</h3>
    <div class="impact-columns">
        <div class="impact-column positive">
            <h4>✅ Top Positive</h4>
            <ul id="positiveImpactFiles"></ul>
        </div>
        <div class="impact-column negative">
            <h4>❌ Top Negative</h4>
            <ul id="negativeImpactFiles"></ul>
        </div>
    </div>
    <div class="action-hint" id="dataImpactAction">--</div>
</div>
```

---

### Phase 2: Add Action Thresholds to Existing Cards

**2.1 Training Card Thresholds**

```javascript
function updateTrainingCard(data) {
    const gap = data.val_train_gap;
    const lossElement = document.getElementById('trainingGap');

    if (gap > 0.5) {
        lossElement.classList.add('warning-severe');
        setActionHint('trainingAction', 'Overfitting likely. Consider: reduce epochs, add regularization, check data quality.');
    } else if (gap > 0.3) {
        lossElement.classList.add('warning-mild');
        setActionHint('trainingAction', 'Monitor closely for overfitting.');
    } else {
        lossElement.classList.add('status-good');
        setActionHint('trainingAction', 'Generalizing well.');
    }
}
```

**2.2 Regression Card Thresholds**

```javascript
function updateRegressionCard(data) {
    if (data.regression_detected) {
        setActionHint('regressionAction',
            `Regression detected! Consider reverting to ${data.best_checkpoint}. See model comparison card.`);
    }
}
```

**2.3 Scheduler Card Thresholds**

```javascript
function updateSchedulerCard(data) {
    const util = data.gpu_utilization;
    const queue = data.queue_by_priority;

    if (queue.critical > 0 || queue.high > 3) {
        setActionHint('schedulerAction', 'High-priority tasks waiting. Check 3090 logs.');
    } else if (util < 20 && data.queue_length > 0) {
        setActionHint('schedulerAction', 'GPU underutilized with pending tasks. Scheduler may be stuck.');
    } else if (util > 80) {
        setActionHint('schedulerAction', 'GPU saturated. Queue processing may slow.');
    } else {
        setActionHint('schedulerAction', 'Operating normally (20-80% utilization).');
    }
}
```

---

### Phase 3: Curriculum + Data Flow Visibility

**3.1 Extend Curriculum Card**

Add to existing curriculum card:
- Current target level
- Last generation time
- Queue status (files pending in training queue)

Plugin change needed: Add training queue status to curriculum plugin or add new endpoint.

```javascript
// In curriculum card update
function updateCurriculumCard(data) {
    // Existing accuracy display...

    // Add data flow status
    document.getElementById('curriculumLevel').textContent = data.current_level;
    document.getElementById('lastGenTime').textContent = data.last_generation || 'Never';

    if (data.auto_gen_active === false && data.queue_empty) {
        setActionHint('curriculumAction', 'Data production stopped. Queue empty, auto-gen inactive.');
    } else if (data.queue_empty) {
        setActionHint('curriculumAction', 'Queue empty. Waiting for auto-generation.');
    }
}
```

---

### Phase 4: "Did It Help?" Tracking (Future)

**Backend Work Required:**

1. **Self-Correction Impact Tracker:**
   - Track error rate on corrected examples before/after
   - Store in `status/self_correction_impact.json`
   - Compare accuracy on corrected set vs uncorrected baseline

2. **Adversarial Impact Tracker:**
   - Track which adversarial categories improved
   - Join mined categories with benchmark results
   - Show "still failing" vs "improved" categories

This requires new analytics daemons that compare model performance on specific subsets over time.

---

## Implementation Order

| Priority | Task | Complexity | Time Est |
|----------|------|------------|----------|
| 1 | Scheduler plugin + card | Medium | 2-3 hrs |
| 2 | Retention/Storage card | Low | 1-2 hrs |
| 3 | Data File Impact card | Low | 1-2 hrs |
| 4 | Action thresholds (all cards) | Medium | 2-3 hrs |
| 5 | Curriculum data flow expansion | Medium | 2-3 hrs |
| 6 | Self-correction impact (backend) | High | 4-6 hrs |
| 7 | Adversarial impact (backend) | High | 4-6 hrs |

**Total Phase 1-4:** ~12 hours of implementation
**Phase 5-7 (backend analytics):** ~10+ hours

---

## CSS Additions Needed

```css
/* Action hint styling */
.action-hint {
    margin-top: 8px;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.85em;
    background: var(--bg-secondary);
}

.action-hint.action-good {
    background: rgba(46, 204, 113, 0.15);
    border-left: 3px solid #2ecc71;
}

.action-hint.action-warning {
    background: rgba(241, 196, 15, 0.15);
    border-left: 3px solid #f1c40f;
}

.action-hint.action-critical {
    background: rgba(231, 76, 60, 0.15);
    border-left: 3px solid #e74c3c;
}

/* Warning states for metrics */
.warning-severe { color: #e74c3c; font-weight: bold; }
.warning-mild { color: #f1c40f; }
.status-good { color: #2ecc71; }

/* Data impact columns */
.impact-columns {
    display: flex;
    gap: 16px;
}
.impact-column { flex: 1; }
.impact-column.positive ul li { color: #2ecc71; }
.impact-column.negative ul li { color: #e74c3c; }
```

---

## JS Helper Functions

```javascript
function setActionHint(elementId, text, level = 'neutral') {
    const el = document.getElementById(elementId);
    if (!el) return;

    el.textContent = text;
    el.className = 'action-hint';

    if (level === 'good') el.classList.add('action-good');
    else if (level === 'warning') el.classList.add('action-warning');
    else if (level === 'critical') el.classList.add('action-critical');
}

function formatTimeAgo(timestamp) {
    if (!timestamp) return 'Never';
    const diff = Date.now() - new Date(timestamp).getTime();
    const minutes = Math.floor(diff / 60000);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}
```

---

## Docs ↔ UI Symmetry

After implementation, update docs with "Where to see this":

| System | Doc Section | UI Location |
|--------|-------------|-------------|
| GPU Scheduler | CLAUDE.md "Autonomous Systems" | Master Dashboard → 3090 → Scheduler card |
| Retention | CLAUDE.md (to add) | Master Dashboard → Storage card |
| Data Impact | ARCHITECTURE.md | Master Dashboard → Analytics → Data Impact card |
| Curriculum | ARCHITECTURE.md | Master Dashboard → Curriculum card |

---

## Success Criteria

After implementation:
- [ ] Every card has an "action hint" showing what to do (or "all good")
- [ ] Scheduler visibility: queue depth, active task, utilization band status
- [ ] Storage visibility: usage %, headroom, warnings, next deletions
- [ ] Data impact visibility: top positive/negative files with triage guidance
- [ ] Curriculum visibility: current level, data flow status, queue status
- [ ] No daemon/system exists that isn't visible on dashboard
