# System Architecture

**Last Updated:** 2025-11-25

## System Split

Two-GPU architecture with distinct responsibilities:

| GPU | IP | Role | Key Processes |
|-----|-----|------|---------------|
| **RTX 4090** | 192.168.x.x | Training, evaluation, orchestration | training_daemon, model_comparison_engine, deployment_orchestrator |
| **RTX 3090** | 192.168.x.x | Inference only | FastAPI server (port 8765) |

## Data Flow

```
                          4090 (Training)                           3090 (Inference)
┌────────────────────────────────────────────────────────────┐    ┌─────────────────┐
│                                                            │    │                 │
│  inbox/ ─► SpecValidator ─► DataValidator ─► queue/       │    │   /v1/chat/     │
│                 │                 │                        │    │   completions   │
│            (schema)          (content)                     │    │        │        │
│                                  │                         │    │        ▼        │
│                                  ▼                         │    │  Deployed Model │
│                         training_daemon                    │    │  (checkpoint-N) │
│                              │                             │    │                 │
│                              ▼                             │    └────────▲────────┘
│                    checkpoints/checkpoint-N                │             │
│                              │                             │             │
│                              ▼                             │     rsync   │
│                  model_comparison_engine                   │      (8s)   │
│                              │                             │             │
│                              ▼                             │             │
│                  deployment_orchestrator ─────────────────────────────────┘
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Deployment latency:** < 15 minutes from checkpoint creation to serving

## Validation System (Two-Layer)

Training data passes through two validation gates before training:

### Layer 1: SpecValidator (Schema Gate)
- **Location:** `core/validation/spec.py`
- **Purpose:** Deny-by-default schema enforcement
- **Checks:** Job must map to known spec (chat_sft_v1, syllo_v1, completion_v1)
- **Outcome:** Unknown schemas rejected before reaching queue

### Layer 2: DataValidator (Content Gate)
- **Location:** `core/validation/validator.py`
- **Purpose:** Content quality validation at three levels
- **Levels:**
  - `QUICK`: Basic format checks (< 1 sec)
  - `STANDARD`: Token stats, balance analysis (< 30 sec)
  - `DEEP`: Answer leakage detection, full quality audit (minutes)

```python
# Usage in daemon
from core.validation import SpecValidator, DataValidator, DATASET_SPECS

spec_validator = SpecValidator(DATASET_SPECS)
data_validator = DataValidator(tokenizer)

# On file arrival
spec = spec_validator.validate_job(job_config, metadata)  # Raises on unknown schema
issues = data_validator.validate_file(file_path, level='QUICK')  # Raises on bad data
```

## Key Modules

### Training Core
| Module | Location | Responsibility |
|--------|----------|----------------|
| TrainerEngine | `trainer/core/engine.py` | High-level training API |
| ConfigLoader | `trainer/config/loader.py` | JSON + CLI config merging |
| DataProfile | `trainer/profiles/base.py` | Data transformation interface |
| UltimateTrainer | `core/train.py` | Main training script |
| training_daemon | `core/training_daemon.py` | Queue processing orchestrator |

### Daemon Services (Extracted)
| Module | Location | Responsibility |
|--------|----------|----------------|
| PIDManager | `core/daemon/pid_manager.py` | Single-instance enforcement |
| FileWatcher | `core/daemon/file_watcher.py` | Inbox monitoring |
| SnapshotService | `core/daemon/snapshot_service.py` | Checkpoint snapshots |
| BackgroundWorker | `core/daemon/background_worker.py` | Non-blocking tasks |

### Monitoring API
| Module | Location | Responsibility |
|--------|----------|----------------|
| aggregator.py | `monitoring/api/aggregator.py` | Plugin-based data aggregation |
| skill_metrics.py | `monitoring/api/plugins/skill_metrics.py` | Baseline test results |
| model_comparison.py | `monitoring/api/plugins/model_comparison.py` | Checkpoint rankings |

## Configuration

Primary config: `config.json`

```json
{
  "model_name": "qwen2.5_0.5b",
  "model_path": "/path/to/training/models/Qwen3-0.6B",
  "batch_size": 19,
  "learning_rate": 0.0002,
  "max_length": 4096,
  "schema_id": "chat_sft_v1"    // Optional, defaults to chat_sft_v1
}
```

See `trainer/config/schema.py` for full TrainerConfig dataclass.

## Module Contracts

All modules follow contract standards defined in `MODULE_CONTRACTS.md`:
- Type hints required on all public functions
- Docstrings with Args/Returns/Raises/Side Effects
- Data formats documented for I/O modules

## Profiles

Data transformation profiles in `trainer/profiles/`:
- `emoji_think.py`: Emoji thinking patterns + stop signals
- `regime3.py`: Symbolic reasoning format

Switch via config.json:
```json
{"profile": {"name": "emoji_think"}}
```

## Data Generation (Curriculum-Based)

Auto-generation uses local singleSKILL APIs with curriculum-based difficulty:

```
                 DataManager
                      │
                      ▼
              CurriculumManager
              (active_skill, level)
                      │
         ┌───────────┴───────────┐
         ▼                       ▼
   SkillAPIClient          SkillAPIClient
   (SYLLO :8080)           (Binary :8090)
         │                       │
         ▼                       ▼
   singleSKILL/            singleSKILL/
   skill_syllo_variant/    skill_basic_math/
```

### Components
| Module | Location | Responsibility |
|--------|----------|----------------|
| DataManager | `data_manager/manager.py` | Orchestrates generation + quality + queueing |
| CurriculumManager | `data_manager/curriculum_manager.py` | Tracks skill levels, progression |
| SkillAPIClient | `data_manager/skill_api_client.py` | HTTP client for skill APIs |

### Curriculum Levels
- **SYLLO:** 5 levels (4-8 word puzzles)
- **Binary:** 7 levels (magnitude 1-10 to 10K-100K)
- **Progression:** 80% accuracy over 3 evaluations

### State Files
- `data_manager/curriculum_state.json` - Current levels, accuracy history
- `config.json` → `auto_generate` section - Enable/disable, count, cooldown

### Usage
```bash
# Check status
python3 data_manager/manager.py status

# Generate manually (100 examples at current level)
python3 data_manager/manager.py generate --force --count 100

# Check curriculum progress
python3 data_manager/curriculum_manager.py status
```
