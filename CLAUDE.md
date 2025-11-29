# REALM OF TRAINING - Game Design Document

**Last Updated:** 2025-11-29
**Update Frequency:** Every ~50k tokens or when significant changes occur

---

## 🎮 THE GAME VISION

**This is an RPG Idler game about training an AI hero.**

A **Campaign** is a hero's journey to maximum potential - one attempt to push a model as far as it can go. The goal: discover the level cap. How much can this hero learn?

### The Core Mental Model

| RPG Concept | Training Reality |
|-------------|------------------|
| **Hero** | A model (DIO=Qwen3-0.6B, FLO=Qwen3-4B) |
| **Campaign** | One playthrough/attempt to reach max potential |
| **Continue** | Keep training, keep learning, push further |
| **Level Cap** | The theoretical limit of what this model can learn |
| **Skill Levels** | Curriculum progression (L1→L50 per skill) |
| **Personal Best** | Peak metrics achieved in this campaign |

**Different heroes have different potentials:**
- A 0.6B model might cap at skill level 20
- A 4B model might reach level 50
- We discover the cap by PLAYING (training)

**Starting points:**
- `starting_checkpoint=None` → Fresh start (new game)
- `starting_checkpoint="checkpoint-X"` → Continue from save (or New Game+)

### The Experience

```
1. Open the Tavern (http://localhost:8888)
2. See DIO - your hero - with stats, level, XP
3. Watch battles (training runs) in real-time
4. Track quest completion and skill progression
5. Manage the Vault (checkpoints, models)
6. Everything feels like playing a game
```

### Core Game Loop

```
DROP QUEST → DIO BATTLES → GAIN XP → LEVEL UP → UNLOCK SKILLS → REPEAT
   (inbox)      (training)    (steps)   (1000 steps)  (curriculum)
```

### Key URLs

| Location | URL | Purpose |
|----------|-----|---------|
| **Tavern** | http://localhost:8888 | Main game UI - DIO, battles, stats |
| **Quests** | http://localhost:8888/quests | Quest board - manage training queue |
| **Jobs** | http://localhost:8888/jobs | Distributed job queue (eval, sparring) |
| **Oracle** | http://localhost:8888/oracle | Talk to DIO - chat with any checkpoint |
| **Vault** | http://localhost:8888/vault | Browse checkpoints, zones, assets |
| **Settings** | http://localhost:8888/settings | Config, VRAM calc, scheduler |
| **Guild Hall** | http://localhost:8888/guild | Skill progression dashboard |
| VaultKeeper API | http://localhost:8767/api/stats | Asset & Ledger API |

### Start Playing

```bash
# Bootstrap (first time only)
./scripts/bootstrap_dev.sh

# Check environment
python3 -m training doctor

# Start everything
python3 -m training start-all
# Or: ./scripts/start_all.sh

# Stop everything
python3 -m training stop-all
```

### Training Modes

| Mode | Command | Description |
|------|---------|-------------|
| **DIO Training** | `./scripts/start_all.sh` | Full daemon system - Tavern, VaultKeeper, queue |
| **4B Experiments** | `python3 scripts/train_4b_full.py` | Standalone script - no daemon needed |

---

## 🦸 THE HERO: DIO

**DIO** is your AI hero - a Qwen3-0.6B model learning to reason.

| Stat | Meaning | Source |
|------|---------|--------|
| **Level** | Training progress | Steps / 1000 |
| **XP** | Experience points | Total training steps |
| **ATK** | Damage dealt | 1 / Loss (lower loss = higher ATK) |
| **DEF** | Defense | 1 / Validation loss |
| **ACC** | Accuracy | Curriculum eval accuracy |
| **Gold** | Currency | Completed quests × 100 |

### Skills

| Skill | Icon | Description | Levels | API Port |
|-------|------|-------------|--------|----------|
| **SY** | 🧩 | Syllacrostic word puzzles (signal degradation) | L1-50 | 8080 |
| **BIN** | 🔢 | Binary arithmetic with circled notation | L1-30 | 8090 |

Skills are defined in `configs/skills/*.yaml` - YAML is the single source of truth.

### Progression

- Every **1000 steps** = Level up
- **80% accuracy** over 3 evals = Skill level up
- Complete quests = Earn gold
- Lower loss = Higher damage

---

## 🏰 THE REALM (Architecture)

| Game Location | Technical System | Port |
|--------------|------------------|------|
| **The Weaver** | Daemon orchestrator (`weaver/weaver.py`) | - |
| **Tavern** | Game UI (`tavern/`) | 8888 |
| **Arena** | Training daemon (`core/training_daemon.py`) | - |
| **Guild** | Skills & progression (`guild/`) | - |
| **Vault** | VaultKeeper API (`vault/server.py`) | 8767 |
| **Oracle** | Inference server (3090) | 8765 |
| **Watchtower** | Monitoring (`watchtower/`) | 8081 |

### RPG → Technical Mapping

| RPG Term | Technical Equivalent |
|----------|---------------------|
| Quest | Training data file |
| Battle | Training run |
| Damage | Loss (lower = better) |
| Champion | Best checkpoint |
| Oracle | Inference server |
| Stronghold | Storage location |
| VaultKeeper | Asset registry |
| Tavern | Game UI |

---

## 🎯 GAME ROADMAP (What's Missing)

### Phase 1: Actions from the Game
- [ ] **Start Quest** - Drop files in inbox from the UI
- [ ] **Pause/Resume Battle** - Control training from Tavern
- [ ] **Promote Champion** - Deploy best checkpoint from UI
- [x] **View Quest Board** - See pending quests, priorities ✅ `/quests`

### Phase 2: More Game Feel
- [ ] **Notifications** - "Quest complete!", "Level up!", "New champion!"
- [ ] **Sound Effects** - Battle sounds, level up chimes (optional)
- [ ] **Animations** - Damage numbers, XP floating up
- [ ] **Achievement System** - Milestones tracked

### Phase 3: Graphics
- [ ] **Hero Portrait** - Replace ASCII with actual image
- [ ] **Skill Icons** - Visual icons for SY, BIN
- [ ] **Battle Animations** - Visual feedback during training

### Phase 4: Full Control
- [x] **Armory Screen** - Edit config.json from UI ✅ `/settings` + VRAM calc
- [ ] **Vault Browser** - Browse/delete/export checkpoints
- [x] **Guild Management** - Adjust curriculum, skill priorities ✅ `/settings` scheduler

---

## 📦 RECENT UPDATES

**See [CHANGELOG.md](CHANGELOG.md) for full history.**

Latest (2025-11-28):
- **Shareable Project** - Bootstrap script, `python -m training doctor`, pre-commit hooks
- **Hero Titles System** - `configs/titles.yaml`, `guild/titles.py`
- **Distributed Job System V2** - SQLite job store, worker registration, backpressure
- **Skill Engine** - 11 passives, 39 primitives, per-primitive accuracy tracking

Previous (2025-11-27):
- **Muon Optimizer** - Alternative to AdamW (`trainer/optimizers/muon.py`)
- **Sparring System** - Self-correction training (`guild/sparring.py`)
- **Task Master** - GPU-aware scheduler (`guild/task_master.py`)
- **The Weaver** - Daemon orchestrator (`weaver/weaver.py`)
- **Checkpoint Ledger** - Stats at save time (`core/checkpoint_ledger.py`)

---

## 📋 COMMUNICATION STYLE

**Default mode: Factual technical documentation**

- State facts about system behavior, configuration, and current state
- Do not include recommendations, suggestions, or opinions unless explicitly asked
- Do not add phrases like "I recommend", "you should", "it's best to", "consider"
- Present options without bias when multiple approaches exist
- Omit evaluative language ("excellent", "better", "perfect", "brilliant")
- When asked "how does X work", describe the mechanism without suggesting changes
- When asked "what are the options", list them without ranking

**Example:**
- ❌ "I recommend using batch_size=30 because it's more efficient"
- ✅ "batch_size=30 uses ~21GB VRAM. batch_size=16 uses ~14GB VRAM"

**Only add recommendations when:**
- Explicitly asked ("what should I do?", "which is better?")
- Critical safety issue (data loss, system damage)
- User makes factual error that needs correction

---

## 🚨 CRITICAL RULES

### Documentation Policies

1. **7 Canonical Docs** - Only write to these 7 files:
   - `README.md` - System overview
   - `QUICKSTART.md` - Getting started guide
   - `ARCHITECTURE.md` - How the system works
   - `TROUBLESHOOTING.md` - Common problems and solutions
   - `REMOTE_INFERENCE.md` - Remote RTX 3090 inference server
   - `DEVELOPMENT.md` - Working on the codebase
   - `CHANGELOG.md` - Track changes

2. **CLAUDE.md Updates** - Update this file every ~50k tokens or when significant changes occur

3. **No Other Docs** - Do NOT create any other .md files without explicit user permission

4. **Remote Inference Focus** - See `REMOTE_INFERENCE.md` for all inference/generation tasks. This training machine does NOT run inference.

### Safety Policies

1. **NEVER delete `models/current_model/` without explicit user permission**
2. **ALWAYS create backup before risky operations**
3. **NEVER modify `config.json` critical parameters without user approval:**
   - `max_length`
   - `model_name`
   - `base_model`
4. **ASK FIRST** before making system-wide changes

---

## 🗃️ KEY SYSTEMS REFERENCE

### VaultKeeper (port 8767)

Central asset registry. Pattern: **Ask Vault First**

```python
from vault import ask_vault_first
model_path = ask_vault_first("checkpoint_175000", fallback="models/checkpoint-175000")
```

Key endpoints: `/api/stats`, `/api/ledger`, `/api/checkpoints`, `/api/jobs`

### Checkpoint Ledger

Records stats when checkpoints are saved. Canonical name: `checkpoint-{step}-{YYYYMMDD}-{HHMM}`

```python
from core.checkpoint_ledger import get_ledger
ledger = get_ledger()
best = ledger.get_best(metric="train_loss")
```

### Host Registry (`config/hosts.json`)

Service discovery - components query this instead of hardcoding IPs.

```python
from core.hosts import get_service_url
inference_url = get_service_url("inference")  # http://inference.local:8765
```

### Job System V2

Distributed task execution with workers.

```bash
# Submit job
curl -X POST http://localhost:8767/api/jobs -d '{"job_type": "eval", "payload": {...}}'

# Start worker
python3 -m workers.claiming_worker --device local_gpu_1 --server http://localhost:8767
```

### Skill Engine

Unified interface for skill operations.

```python
from guild.skills import get_engine
engine = get_engine()
skill = engine.get("bin")
batch = engine.generate_eval_batch("bin", level=1, count=10)
```

Skills defined in `configs/skills/*.yaml`

### TrainerEngine

Full HuggingFace training pipeline.

```bash
USE_ENGINE=1 python3 core/train.py --dataset data/train.jsonl --model qwen3_0.6b --yes
```

Key files: `trainer/core/engine.py`, `trainer/monitoring/callbacks.py`

### Muon Optimizer

Alternative to AdamW. Enable in `config.json`:

```json
{"optimizer": {"type": "muon", "muon": {"hidden_lr": 0.02, "aux_lr": 0.0003}}}
```

---

## 📁 KEY DIRECTORIES

```
$TRAINING_BASE_DIR/
├── tavern/          # Game UI (port 8888)
├── vault/           # VaultKeeper API (port 8767)
├── guild/           # Skills, quests, progression
├── arena/           # Training execution
├── core/            # Training system (train.py, training_daemon.py)
├── trainer/         # TrainerEngine, optimizers, callbacks
├── workers/         # Distributed job workers
├── weaver/          # Daemon orchestrator
├── config/          # hosts.json, devices.json, storage_zones.json
├── configs/skills/  # Skill YAML definitions
├── models/          # Model storage
├── queue/           # Priority queues (high/, normal/, low/)
├── inbox/           # Drop zone for training files
├── status/          # Runtime status JSON files
└── scripts/         # Shell scripts (start_all.sh, bootstrap_dev.sh)
```

---

## 🔧 COMMON TASKS

### Check System Health

```bash
python3 -m training doctor
# Or: python3 safety/comprehensive_health_check.py
```

### Control Training

```bash
python3 core/training_controller.py status|pause|resume|stop
```

### Queue Management

```bash
python3 core/training_queue.py status
```

### Run Sparring

```bash
python3 guild/sparring.py --skill binary --count 100
```

### Hardcode Audit

```bash
python3 scripts/check_hardcodes.py --ignore-ips
```

---

## 📝 NOTES FOR CLAUDE

1. Run health check: `python3 -m training doctor`
2. **ASK USER** before making changes
3. **ASK USER** before creating new documentation
4. Trust code as ground truth, not old docs
5. Use `core.paths.get_base_dir()` for paths, never hardcode
