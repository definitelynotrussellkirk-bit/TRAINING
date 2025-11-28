# RPG Layer - Future Features

**Created:** 2025-11-28
**Purpose:** Document worthwhile RPG-themed features for later implementation

These features passed the "does it actually help?" test but aren't urgent now.

---

## 1. Quest Templates - Job Orchestration

**What:** Wrap common job sequences into named "quests" that expand to multiple jobs.

**Why it helps:**
- Instead of manually submitting data_gen → train → eval jobs, you trigger "Trial of Binary Wits L3"
- Repeatable, tested sequences with clear names
- UI shows quest progress, not raw job IDs

**Implementation:**

```python
@dataclass
class QuestTemplate:
    id: str
    display_name: str
    description: str
    quest_type: Literal["train", "eval", "mixed", "maintenance"]
    skill_id: Optional[str]
    level: Optional[int]
    jobs: list[JobSpec]  # Expanded job sequence

# Example templates:
QUEST_TEMPLATES = {
    "trial_binary_l3": QuestTemplate(
        id="trial_binary_l3",
        display_name="Trial of Binary Wits (L3)",
        quest_type="eval",
        skill_id="bin",
        level=3,
        jobs=[
            data_gen_job("binary_arithmetic", count=10000),
            # train job would be added after data_gen completes
            eval_job("bin", level=3, batch_size=256),
        ]
    ),
    "vault_cleanup": QuestTemplate(
        id="vault_cleanup",
        display_name="Clean the Vault",
        quest_type="maintenance",
        jobs=[
            archive_job("hot", "warm"),
            JobSpec(job_type=JobType.RETENTION, payload={"zone": "hot"}),
        ]
    ),
}
```

**Files to create:**
- `guild/quest_templates.py` - Template definitions and expansion
- Add to `/jobs` UI - "Start Quest" dropdown

**Wait until:** Job system is battle-tested (1-2 weeks of use)

---

## 2. Status Effects from Monitors

**What:** Represent monitor outputs (regression detector, confidence calibrator) as discrete "effects" on skills/hero.

**Why it helps:**
- Scheduler logic becomes cleaner: "if skill has critical debuff, prioritize its quests"
- UI shows actionable state, not raw metric soup
- Effects can trigger automatic responses

**Implementation:**

```python
@dataclass
class StatusEffect:
    id: str
    target: Literal["hero", "skill"]
    skill_id: Optional[str]
    kind: Literal["buff", "debuff", "warning"]
    severity: Literal["minor", "major", "critical"]
    reason: str
    created_at: datetime

# Examples:
# Debuff: "Confused about Bits" - binary accuracy dropped 10%+
# Buff: "In the Zone" - last N evals all trending up
# Warning: "Primitive Weakness" - specific primitive below 60%
```

**Scheduler integration:**
```python
# In scheduler decision logic
effects = get_active_effects(skill_id="bin")
if any(e.severity == "critical" for e in effects):
    boost_priority("bin_quests")
```

**Files to create:**
- `guild/effects.py` - StatusEffect model and manager
- Integrate with existing monitors in `watchtower/`
- Add effects display to hero panel

**Wait until:** Monitor infrastructure is more mature

---

## 3. Device Map View

**What:** Visual representation of devices as "locations" - Training Grounds, Arena, Vault.

**Why it helps:**
- Quick glance at system topology
- See queue depths and job distribution
- Makes device roles intuitive

**Implementation:**
```
┌─────────────────┐     ┌─────────────────┐
│ Training Grounds│────▶│     Arena       │
│   (4090)        │     │   (3090)        │
│ Jobs: 2 queued  │     │ Jobs: 1 running │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│     Vault       │
│   (Synology)    │
│ 47 checkpoints  │
└─────────────────┘
```

**Files to create:**
- `tavern/templates/map.html` - Simple SVG/CSS map
- API endpoint for device status aggregation

**Wait until:** Have 3+ active devices to visualize

---

## 4. Dungeon Evals - Integrated Stress Tests

**What:** Special evaluation suites that combine multiple skills and are explicitly harder.

**Why it helps:**
- Gate major decisions (checkpoint freeze, baseline selection)
- Track "real-world" performance vs isolated skill evals
- Named events for historical comparison

**Implementation:**

```python
DUNGEONS = {
    "cave_of_composed_reasoning": {
        "name": "The Cave of Composed Reasoning",
        "description": "Multi-step problems combining binary, arithmetic, and logic",
        "skills": ["bin", "logic"],
        "dataset_spec": "dungeon_composed_reasoning",  # Special dataset
        "passing_threshold": 0.7,
        "recommended_frequency": "weekly",
    }
}
```

**Files to create:**
- `guild/dungeons.py` - Dungeon definitions
- `data/dungeons/` - Curated multi-skill datasets
- Add to scheduler as periodic "boss fights"

**Wait until:** Single-skill evals are solid and consistent

---

## 5. Regression Alerts (Notification System)

**What:** Prominent alerts when skill accuracy drops significantly from peak.

**Why it helps:**
- Don't miss regressions buried in logs
- Immediate visibility into problems
- Can trigger automatic responses (boost training priority)

**Implementation:**
- Track peak accuracy per skill/primitive in SkillState
- On each eval, compare to peak
- If drop > threshold (5%?), emit alert
- Alert destinations: Tavern UI banner, status file, optional webhook

**Integration points:**
- `trainer/monitoring/callbacks.py` - Emit on eval
- `tavern/server.py` - `/api/alerts` endpoint
- `game.html` - Alert banner/toast area

**Wait until:** Have more eval history to detect regressions reliably

---

## 6. Training Cost Tracking

**What:** Track GPU hours, power estimates, and cost per skill/level.

**Why it helps:**
- Real operational metric
- Identify expensive skills
- Budget planning

**Implementation:**
- Track job durations in JobResult
- Aggregate by skill, job_type
- Estimate power (4090 ≈ 450W, 3090 ≈ 350W)
- Store in status/cost_tracker.json

**Files to create:**
- `management/cost_tracker.py`
- Add cost display to settings/dashboard

---

## Rejected Ideas (for reference)

| Idea | Why Rejected |
|------|--------------|
| Factions/Classes for model variants | Only training ONE model (DIO) |
| Buff/Debuff expiration times | Over-engineering; status effects don't need TTL |
| Caravan route visualization | Arrows on map is noise; device list is clearer |
| Security as "Wards and Seals" | Security needs clarity, not metaphor |
| HP/Morale bars | Worse versions of loss/accuracy metrics |
| Layer Tower / Attention viz | Interpretability work, not core to training ops |
| "Immersion level" toggle | Premature - add when there's more to toggle |

---

## Already Implemented

| Feature | Location | Date |
|---------|----------|------|
| Titles System | `configs/titles.yaml`, `guild/titles.py` | 2025-11-28 |
| Lore Dictionary | `tavern/lore.py` | 2025-11-28 |
| Primitive Accuracy Tracking | `guild/skills/types.py` (SkillState) | 2025-11-27 |
| Device Registry | `core/devices.py` | 2025-11-28 |
| Job System | `guild/job_types.py`, `jobs/store.py` | 2025-11-28 |
