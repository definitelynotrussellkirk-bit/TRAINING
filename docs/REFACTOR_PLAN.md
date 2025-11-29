# Training System Refactor Plan

**Created:** 2025-11-28
**Last Updated:** 2025-11-28 22:30
**Status:** COMPLETE - All phases done
**Priority Order:** Phase 1 → 2 → 3 → 4 → 5 → 6

## Completed Files
- `trainer/config/locked.py` - NEW: Single source for locked architecture
- `core/job.py` - NEW: First-class job abstraction
- `core/config_builder.py` - Uses locked.py, reads arch from hero YAML
- `trainer/config/loader.py` - Uses locked.py for fallback
- `core/training_queue.py` - Added get_next_job(), is_healthy(), complete_job(), fail_job()
- `trainer/core/engine.py` - Added verbose parameter and _log() method

---

## Overview

This plan addresses architectural drift and scattered concepts identified in code review. The goal is to unify config handling, create a first-class job abstraction, and decouple the training engine for testing.

---

## Phase 1: Unified Config & Locked Architecture (HIGH IMPACT)

### Problem
`locked` architecture values (model_architecture, vocab_size) are produced in 3 places:
1. `core/config_builder.py:179-185` - hardcodes `Qwen3ForCausalLM`, `151936`
2. `trainer/config/loader.py:253-260` - infers from base config, defaults to `AutoModelForCausalLM`
3. User can specify in `config.json` directly

This causes drift when adding new heroes with different architectures.

### Solution
Hero profiles already have `model.architecture` and `model.vocab_size` in YAML. Use them as single source.

### Tasks

#### 1.1 Create `trainer/config/locked.py`
```python
"""Single source of truth for locked architecture fields."""

from typing import Optional
from guild.heroes import HeroProfile

def build_locked_config(
    hero: Optional[HeroProfile],
    model_path: str,
    max_length: int,
    model_version: str = "v1"
) -> dict:
    """
    Build locked config from hero profile.

    Args:
        hero: HeroProfile with model specs (preferred)
        model_path: Fallback model path if no hero
        max_length: Training context length
        model_version: Version string for tracking

    Returns:
        Dict with base_model, model_architecture, max_context_length, vocab_size
    """
    if hero:
        return {
            'base_model': hero.model.hf_name,
            'model_architecture': hero.model.architecture,
            'max_context_length': max_length,
            'vocab_size': hero.model.vocab_size,
            'model_version': model_version,
        }
    else:
        # Fallback for non-campaign usage
        return {
            'base_model': model_path,
            'model_architecture': 'AutoModelForCausalLM',
            'max_context_length': max_length,
            'vocab_size': 151936,  # Qwen3 default
            'model_version': model_version,
        }
```

#### 1.2 Update `core/config_builder.py`
Replace lines 179-185 in `to_trainer_config_dict()`:
```python
# Before:
'locked': {
    'base_model': self.model_path,
    'model_architecture': 'Qwen3ForCausalLM',  # TODO: Make configurable
    'max_context_length': self.max_length,
    'vocab_size': 151936,  # Qwen3 vocab size
    'model_version': f'campaign-{self.campaign_id}',
},

# After:
from trainer.config.locked import build_locked_config
# In to_trainer_config_dict():
'locked': build_locked_config(
    hero=self._hero,  # Store hero in build()
    model_path=self.model_path,
    max_length=self.max_length,
    model_version=f'campaign-{self.campaign_id}',
),
```

Also store hero reference in `build()` method for use in `to_trainer_config_dict()`.

#### 1.3 Update `trainer/config/loader.py`
Replace lines 250-261 in `_merge_config()`:
```python
# Before:
if 'locked' not in merged:
    model_path = merged.get('model', {}).get('model_path', '')
    merged['locked'] = {
        'base_model': base.get('base_model', model_path),
        'model_architecture': base.get('model_architecture', 'AutoModelForCausalLM'),
        ...
    }

# After:
if 'locked' not in merged:
    from trainer.config.locked import build_locked_config
    model_path = merged.get('model', {}).get('model_path', '')
    merged['locked'] = build_locked_config(
        hero=None,  # No hero in config.json path
        model_path=model_path,
        max_length=merged.get('hyperparams', {}).get('max_length', 4096),
    )
```

#### 1.4 Add precedence docs to ARCHITECTURE.md
```markdown
## Config Precedence

Final training args (later overrides earlier):
1. Schema Defaults → TrainerConfig dataclass defaults
2. Hero Defaults → configs/heroes/{hero_id}.yaml training_defaults
3. Campaign Overrides → campaigns/{hero}/{campaign}/campaign.json
4. config.json → Project-level overrides
5. CLI Flags → --batch-size, --learning-rate, etc.
```

---

## Phase 2: First-Class Job Abstraction (HIGH IMPACT)

### Problem
Job concept scattered across:
- `TrainingQueue` entries (file path + metadata)
- `JobLogger` events (job_id, timestamps, state)
- `TrainingResult` (success, loss, global_step)

### Solution
Unified `Job` dataclass that all systems use.

### Tasks

#### 2.1 Create `core/job.py`
```python
"""First-class job abstraction."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Dict, Any

JobStatus = Literal['queued', 'processing', 'completed', 'failed', 'skipped']

@dataclass
class Job:
    """
    Complete training job representation.

    Single source of truth for job identity and state.
    Used by TrainingQueue, JobLogger, and monitoring API.
    """
    # Identity
    id: str                                    # YYYY-MM-DDTHH:MM:SS_filename
    dataset_path: str                          # Full path to .jsonl

    # Context
    hero_id: Optional[str] = None
    campaign_id: Optional[str] = None

    # Queue
    priority: str = 'normal'                   # high, normal, low
    status: JobStatus = 'queued'
    attempts: int = 0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    # Training metrics
    start_step: Optional[int] = None
    final_step: Optional[int] = None
    final_loss: Optional[float] = None
    final_val_loss: Optional[float] = None

    # Error
    last_error: Optional[str] = None

    # Metadata
    num_examples: Optional[int] = None
    estimated_tokens: Optional[int] = None
    config_hash: Optional[str] = None
    git_commit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str, priority: str = 'normal',
                  hero_id: str = None, campaign_id: str = None) -> 'Job':
        """Create job from dataset path."""
        filename = Path(path).name
        job_id = f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}_{filename}"
        return cls(
            id=job_id,
            dataset_path=str(path),
            priority=priority,
            hero_id=hero_id,
            campaign_id=campaign_id,
        )

    def start(self, step: int = None) -> 'Job':
        """Mark job as started."""
        self.status = 'processing'
        self.started_at = datetime.now().isoformat()
        self.start_step = step
        self.attempts += 1
        return self

    def complete(self, final_step: int, final_loss: float,
                 final_val_loss: float = None) -> 'Job':
        """Mark job as completed."""
        self.status = 'completed'
        self.finished_at = datetime.now().isoformat()
        self.final_step = final_step
        self.final_loss = final_loss
        self.final_val_loss = final_val_loss
        return self

    def fail(self, error: str, final_step: int = None) -> 'Job':
        """Mark job as failed."""
        self.status = 'failed'
        self.finished_at = datetime.now().isoformat()
        self.last_error = error
        self.final_step = final_step
        return self

    def skip(self, reason: str) -> 'Job':
        """Mark job as skipped."""
        self.status = 'skipped'
        self.finished_at = datetime.now().isoformat()
        self.last_error = reason
        return self

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        """Deserialize from JSON."""
        return cls(**data)

    @property
    def duration_hours(self) -> Optional[float]:
        """Calculate duration if finished."""
        if self.started_at and self.finished_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.finished_at)
            return (end - start).total_seconds() / 3600
        return None
```

#### 2.2 Update `core/training_queue.py`
Add job metadata storage alongside files:
- When adding to queue, create `{job_id}.meta.json` with Job data
- `get_next_file()` returns Job object instead of just Path
- Update `mark_completed/failed/skipped` to work with Job

#### 2.3 Simplify `core/job_logger.py`
- Replace `JobRecord` with `Job`
- `append()` takes Job directly
- Remove duplicate fields that are now in Job

---

## Phase 3: Engine Decoupling (MEDIUM IMPACT)

### Problem
`TrainerEngine.run_job()` has 30+ `print()` calls, making tests noisy.

### Tasks

#### 3.1 Add logging parameter to TrainerEngine
```python
# trainer/core/engine.py
import logging

class TrainerEngine:
    def __init__(
        self,
        status_writer: TrainingStatusWriter,
        verbose: bool = True
    ):
        self.status_writer = status_writer
        self.logger = logging.getLogger("trainer.engine")
        if not verbose:
            self.logger.setLevel(logging.WARNING)
        self._log = self.logger.info if verbose else lambda x: None
```

Replace all `print(...)` with `self._log(...)`.

#### 3.2 Split MonitorContext
```python
# trainer/monitoring/context.py

@dataclass
class ProgressContext:
    current_file: Optional[str] = None
    batch_number: Optional[int] = None
    batch_queue_size: Optional[int] = None

@dataclass
class EvalContext:
    fixed_val_dataset: Any = None
    micro_eval_inputs: Any = None
    micro_eval_interval: int = 500

@dataclass
class ControlContext:
    controller: Any = None

@dataclass
class MonitorContext:
    """Combined context - maintains backwards compatibility."""
    progress: ProgressContext = field(default_factory=ProgressContext)
    eval: EvalContext = field(default_factory=EvalContext)
    control: ControlContext = field(default_factory=ControlContext)

    # Keep core monitors at top level
    live_monitor: Any = None
    evolution_tracker: Any = None
    layer_monitor: Any = None
    raw_train_examples: List[Dict] = field(default_factory=list)
    logits_processor: Any = None
    remote_eval_config: Dict[str, Any] = field(default_factory=dict)
    status_writer: Any = None
```

---

## Phase 4: Campaign Versioning (MEDIUM IMPACT)

### Tasks

#### 4.1 Add version tracking to campaigns
In `guild/campaigns/cli.py` `create_campaign()`:
```python
import subprocess
import hashlib

def get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            text=True, timeout=5
        ).strip()
    except:
        return None

def hash_config(config: dict) -> str:
    content = json.dumps(config, sort_keys=True)
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()[:16]}"

# In create_campaign():
campaign_data = {
    # ... existing fields
    "code_version": get_git_commit(),
    "config_hash": hash_config(config_overrides),
}
```

#### 4.2 Add --plan mode to daemon
In `core/training_daemon.py`:
```python
def plan_queue(self):
    """Preview queue without training."""
    queue = TrainingQueue(self.base_dir)
    files = queue.list_queue()

    print("\n" + "="*60)
    print("TRAINING PLAN")
    print("="*60)

    total_examples = 0
    for f in files:
        path = Path(f['file'])
        # Count lines
        with open(self.base_dir / "queue" / f['priority'] / path) as fp:
            examples = sum(1 for _ in fp)
        total_examples += examples
        print(f"  [{f['priority']:6}] {path.name}: {examples:,} examples")

    print("="*60)
    print(f"TOTAL: {len(files)} files, {total_examples:,} examples")

# Add CLI arg:
parser.add_argument('--plan', action='store_true', help='Preview queue')
```

---

## Phase 5: Orchestration (LOWER IMPACT)

### Tasks

#### 5.1 Create service registry
Create `configs/services.json`:
```json
{
  "training": {
    "name": "Training Daemon",
    "command": ["python3", "core/training_daemon.py"],
    "pid_file": ".daemon.pid",
    "required": true,
    "health_check": {"type": "pid"}
  },
  "tavern": {
    "name": "Tavern Server",
    "command": ["python3", "tavern/server.py", "--port", "8888"],
    "port": 8888,
    "required": true,
    "health_check": {"type": "http", "path": "/health"}
  },
  "vault": {
    "name": "VaultKeeper",
    "command": ["python3", "vault/server.py", "--port", "8767"],
    "port": 8767,
    "required": true,
    "health_check": {"type": "http", "path": "/api/health"}
  }
}
```

#### 5.2 Add is_healthy() to TrainingQueue
```python
def is_healthy(self, min_depth: int = 5) -> bool:
    """Check queue has sufficient depth."""
    status = self.get_queue_status()
    return status['total_queued'] + status['processing'] >= min_depth
```

---

## Phase 6: Cleanup (LOWER IMPACT)

### Tasks

#### 6.1 Move test harness from loader.py
Move `trainer/config/loader.py:473-491` to `scripts/test_config_loader.py`

#### 6.2 Unify packing flag
In `trainer/core/engine.py:655`:
```python
# Config is primary, env is override for debugging
enable_packing = config.data.enable_packing
if os.environ.get("ENABLE_PACKING") == "0":
    enable_packing = False
```

#### 6.3 Create CLI output helper
Create `trainer/utils/output.py`:
```python
def banner(title: str, width: int = 80):
    print("\n" + "=" * width)
    print(title.upper())
    print("=" * width + "\n")

def step(n: int, msg: str):
    print(f"Step {n}: {msg}")
```

---

## Implementation Order

| Task | Files | Effort | Status |
|------|-------|--------|--------|
| 1.1 Create locked.py | trainer/config/locked.py | 30m | DONE |
| 1.2 Update config_builder.py | core/config_builder.py | 30m | DONE |
| 1.3 Update loader.py | trainer/config/loader.py | 30m | DONE |
| 2.1 Create job.py | core/job.py | 45m | DONE |
| 2.2 Update training_queue.py | core/training_queue.py | 1h | DONE |
| 2.3 Simplify job_logger.py | core/job_logger.py | 30m | DONE |
| 3.1 Engine logging | trainer/core/engine.py | 30m | DONE |
| 3.2 Split MonitorContext | trainer/monitoring/context.py | 30m | DONE |
| 4.1 Campaign versioning | guild/campaigns/cli.py | 30m | DONE |
| 4.2 Daemon --plan | core/training_daemon.py | 30m | DONE (already existed) |
| 5.1 Service registry | configs/services.json | 45m | DONE |
| 5.2 Queue is_healthy | core/training_queue.py | 15m | DONE (already existed) |
| 6.1 Move test harness | scripts/test_config_loader.py | 15m | DONE |
| 6.2 Unify packing flag | trainer/core/engine.py | 15m | DONE |
| 6.3 Create CLI output helper | trainer/utils/output.py | 15m | DONE |

---

## Validation

After each phase:
1. Run `python3 -m training doctor`
2. Test campaign creation: `python3 -m guild.campaigns.cli create test-refactor --hero dio-qwen3-0.6b`
3. Verify config loading: `python3 -c "from trainer.config import ConfigLoader; print('OK')"`
4. Check queue: `python3 core/training_queue.py status`
