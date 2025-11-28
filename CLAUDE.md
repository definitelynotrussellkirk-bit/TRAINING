# REALM OF TRAINING - Game Design Document

**Last Updated:** 2025-11-28 (TrainerEngine Refactor Complete)
**Update Frequency:** Every ~50k tokens or when significant changes occur

---

## 🎮 THE GAME VISION

**This is an RPG Idler game about training an AI hero.**

Your hero **DIO** (a Qwen3-0.6B model) battles through quests (training data), learning skills (SY, BIN), and growing stronger. You watch from the **Tavern** as DIO fights, levels up, and becomes a champion.

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
| **Oracle** | http://localhost:8888/oracle | Talk to DIO - chat with any checkpoint |
| **Vault** | http://localhost:8888/vault | Browse checkpoints, zones, assets |
| **Settings** | http://localhost:8888/settings | Config, VRAM calc, scheduler |
| **Scheduler** | http://localhost:8888/scheduler | Curriculum scheduling |
| Guild Hall | http://localhost:8888/guild | Skill progression dashboard |
| VaultKeeper API | http://localhost:8767/api/stats | Asset & Ledger API |

### Start Playing

```bash
# Start everything (The Weaver manages all services)
./scripts/start_all.sh

# Or manually check/start services
python3 weaver/weaver.py --status   # Check tapestry
python3 weaver/weaver.py            # Check and mend broken threads
python3 weaver/weaver.py --daemon   # Run continuously

# Stop everything
./scripts/stop_all.sh
```

### Training Modes

| Mode | Command | Description |
|------|---------|-------------|
| **DIO Training** | `./scripts/start_all.sh` | Full daemon system - Tavern, VaultKeeper, queue |
| **4B Experiments** | `python3 scripts/train_4b_full.py` | Standalone script - no daemon needed |

**Important:** DIO training requires the full system running. Standalone scripts (like 4B experiments) can run directly without services.

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

The realm is the entire game world - every location has an RPG name:

```
REALM (realm.py) - The game world
│
├── weaver/         # 🧵 THE WEAVER - Daemon orchestrator (manages all threads)
│   └── weaver.py           # Watches and restarts all services
│
├── tavern/         # 🍺 MAIN GAME UI - Where you watch DIO (port 8888)
│   ├── server.py           # Tavern server
│   ├── templates/game.html # Game interface
│   └── static/             # CSS, JS assets
│
├── guild/          # 🏰 Skills & progression
│   ├── skills/             # Skill system (loads from configs/skills/*.yaml)
│   ├── quests/             # Quest templates
│   ├── progression/        # XP, level-up logic
│   ├── sparring.py         # ⚔️ Sparring with Trainers (self-correction)
│   ├── sparring_validator.py # Validator for sparring data
│   └── task_registry.py    # Available tasks for Task Master
│
├── arena/          # ⚔️ Where battles happen (training)
│   ├── quest_board.py      # Training queue
│   ├── battle_control.py   # Pause/resume/stop
│   └── battle_log.py       # Battle status
│
├── watchtower/     # 👁️ Monitoring & observation
│   ├── scrying_pool.py     # Real-time training view
│   ├── champion_board.py   # Best checkpoint rankings
│   └── oracle_client.py    # Inference client
│
├── vault/          # 🗃️ Treasure storage
│   ├── keeper.py           # VaultKeeper (knows where everything is)
│   ├── server.py           # VaultKeeper API (port 8767)
│   ├── zones.py            # Zone federation (4090, 3090, NAS)
│   ├── branch_officer.py   # Remote zone daemon (port 8768)
│   ├── archivist.py        # Backup management
│   └── treasury.py         # Disk management
│
├── core/           # 🧠 Core systems
│   ├── checkpoint_ledger.py # Checkpoint stats & canonical naming
│   ├── hosts.py            # Host registry & service discovery
│   └── train.py            # Training script
│
├── sentinels/      # 🛡️ System protection
│   ├── guardian.py         # Daemon watchdog
│   └── health_inspector.py # Health checks
│
├── armory/         # ⚙️ Equipment (config, profiles)
└── scrolls/        # 📜 Utility scripts
```

### Game Locations → Technical Mapping

| Game Location | Technical System | Port |
|--------------|------------------|------|
| **The Weaver** | Daemon orchestrator | - |
| **Tavern** | Game UI | 8888 |
| **Arena** | Training daemon | - |
| **Sparring Ring** | Self-correction (guild/sparring.py) | 8765 (3090) |
| **Watchtower** | Monitoring API | 8081 |
| **Vault** | VaultKeeper API | 8767 |
| **Oracle** | Inference server (3090) | 8765 |
| **Scheduler** | GPU Task Scheduler (3090) | 8766 |

---

## 🎯 GAME ROADMAP (What's Missing)

The Tavern displays data. To feel like a **complete game**, we need:

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

Latest updates (2025-11-28):
- **Skill Engine** - Unified interface for skill training, eval, and per-primitive accuracy
  - 11 passives (7 new), 39 primitives in catalog
  - Per-primitive accuracy tracking identifies specific weaknesses
  - Targeted sparring focuses on weak primitives
  - Tavern API + Guild UI visualization
- **TrainerEngine Refactor** - Complete 5-phase refactor of training architecture
  - TrainerEngine now owns full HF training pipeline (model loading, dataset prep, training)
  - LiveMonitorCallback extracted to `trainer/monitoring/callbacks.py`
  - `core/train.py` reduced from ~2400 to 1881 lines
  - `USE_ENGINE=1` flag for gradual migration

Previous updates (2025-11-27):
- **Muon Optimizer** - Alternative to AdamW with orthogonalized momentum (faster convergence)
- **Packing + Masking Bug Fix** - Critical fix for training on instruction text
- **Sparring with Trainers** - Self-correction training system
- **Task Master** - GPU-aware background task scheduler
- **The Weaver** - Daemon orchestrator (one daemon to rule them all)
- **Oracle** - Strict version checking for checkpoint chat
- **Tavern UI** - Quests page, VRAM calculator, scheduler in settings
- **Checkpoint Ledger** - Single source of truth for checkpoint stats
- **Host Registry** - Service discovery for distributed operation

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
   - `REMOTE_INFERENCE.md` - **⭐ Remote RTX 3090 inference server (primary reference)**
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

## 🏰 RPG ARCHITECTURE

**NEW (2025-11-26)** - Full RPG-themed wrapper modules

The training system now has RPG-themed wrappers providing intuitive naming:

```
REALM (realm.py) - Unified entry point
├── guild/          # Skills, quests, progression, dispatch
├── arena/          # Training execution (battles)
├── watchtower/     # Monitoring and observation
├── vault/          # Storage and versioning
├── sentinels/      # System protection
├── armory/         # Equipment/profiles
├── scrolls/        # Utility scripts
└── tavern/         # Game UI (hero view, battle status) - port 8888
```

**Quick Start:**
```python
from realm import Realm
r = Realm()

# Check training status
vision = r.watchtower.pool.gaze()
print(f"Battle: {vision.battle_state}")

# Check system health
patrol = r.sentinels.inspector.full_patrol()
print(f"Healthy: {patrol.is_all_clear}")

# Register storage
from vault import StorageRegistry, StrongholdType
registry = StorageRegistry(base_dir)
registry.register_stronghold(
    name="synology_main",
    stronghold_type=StrongholdType.NAS,
    host="nas.local"
)
```

**RPG → Technical Mapping:**
| RPG Term | Technical Equivalent |
|----------|---------------------|
| Quest | Training data file |
| Battle | Training run |
| Damage | Loss (lower = better) |
| Champion | Best checkpoint |
| Oracle | Inference server |
| Stronghold | Storage location |
| Scroll | Utility script |
| VaultKeeper | Asset registry (knows what's where) |
| Tavern | Game UI (where adventurers gather) |

---

## 🗃️ VAULTKEEPER - ASK VAULT FIRST

**NEW (2025-11-26)** - Central asset registry across all devices

The VaultKeeper tracks where every asset (checkpoint, model, data, config) lives across all strongholds. Before loading anything, ask the VaultKeeper first.

### The Pattern: Ask Vault First

```python
from vault import ask_vault_first

# OLD WAY (hardcoded path):
model_path = "$TRAINING_BASE_DIR/models/checkpoint-175000"

# NEW WAY (ask vault first):
model_path = ask_vault_first(
    "checkpoint_175000",
    fallback="$TRAINING_BASE_DIR/models/checkpoint-175000"
)

# The keeper will:
# 1. Look up the asset in its catalog
# 2. Find the best available location (local > NAS > remote)
# 3. Fetch it locally if needed
# 4. Return the local path
```

### Quick Start

```python
from vault import VaultKeeper, VaultDiscovery

# Initialize keeper
keeper = VaultKeeper()

# Scan and register all local assets (run once)
discovery = VaultDiscovery()
discovery.scan_all()

# Now locate any asset
result = keeper.locate("checkpoint_175000")
print(f"Found at: {result.best_location.path}")

# Fetch from best source (may be NAS or 3090)
result = keeper.fetch("checkpoint_175000", "/tmp/local_copy")
```

### API Server (for 3090 and other devices)

```bash
# Start VaultKeeper server on 4090
nohup python3 vault/server.py --port 8767 > logs/vault_keeper.log 2>&1 &

# Query from 3090
curl http://trainer.local:8767/api/locate/checkpoint_175000
curl http://trainer.local:8767/api/stats
curl http://trainer.local:8767/api/checkpoints
```

### Client Library (for remote devices)

```python
from vault.client import VaultKeeperClient

# On 3090, connect to 4090's VaultKeeper
client = VaultKeeperClient("trainer.local:8767")

# Find checkpoint
result = client.locate("checkpoint_175000")
if result.found:
    print(f"Found at: {result.best_location.path}")

# Ensure local (fetch if needed)
local_path = client.ensure_local("checkpoint_175000")
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/stats` | GET | Catalog statistics |
| `/api/locate/{asset_id}` | GET | Find asset locations |
| `/api/asset/{asset_id}` | GET | Full asset details |
| `/api/checkpoints` | GET | List all checkpoints |
| `/api/search?type=X` | GET | Search assets |
| `/api/fetch` | POST | Fetch asset to local |
| `/api/register` | POST | Register new asset |
| `/api/scan` | POST | Scan directory |
| `/api/zones` | GET | List all zones |
| `/api/zones/{id}` | GET | Get zone details |
| `/api/transfer` | POST | Transfer between zones |
| `/api/ledger` | GET | List checkpoints with stats |
| `/api/ledger/{step}` | GET | Get checkpoint by step |
| `/api/ledger/best` | GET | Best checkpoint by metric |
| `/api/ledger/summary` | GET | Ledger statistics |
| `/api/training/status` | GET | Training status |
| `/api/training/queue` | GET | Queue status |
| `/api/training/control` | POST | Pause/resume/stop |

### Asset IDs

```
checkpoint_175000     # Training checkpoint
model_qwen3_0.6b      # Model
data_train_syllo_abc  # Training data
config_config         # Configuration
```

### Key Files

```
vault/
├── keeper.py        # VaultKeeper core (SQLite catalog)
├── server.py        # REST API server (port 8767)
├── client.py        # Client library for remote queries
├── discovery.py     # Auto-scan and register assets
├── zones.py         # Zone federation (ZoneRegistry, ZoneTransfer)
├── branch_officer.py # Remote zone daemon
├── assets.py        # Asset types and schemas
├── handlers.py      # Local/Remote/NAS handlers
├── catalog.db       # SQLite database (created automatically)
└── __init__.py      # Exports all components
```

---

## 📖 CHECKPOINT LEDGER - STATS AT SAVE TIME

**NEW (2025-11-27)** - Single source of truth for checkpoint metadata

The Checkpoint Ledger records exact stats when each checkpoint is saved. Every checkpoint has:
- **Canonical name**: `checkpoint-{step}-{YYYYMMDD}-{HHMM}`
- **Stats at save time**: loss, val_loss, learning_rate, epoch
- **Training context**: skill_name, skill_level, training_file
- **Sidecar file**: `.ledger.json` inside each checkpoint directory

### Usage

```python
from core.checkpoint_ledger import get_ledger, record_checkpoint

# Get the ledger
ledger = get_ledger()

# Query checkpoints
latest = ledger.get_latest()
best = ledger.get_best(metric="train_loss")
all_records = ledger.list_all(limit=20)

# Get specific checkpoint
record = ledger.get(190000)
print(f"Loss: {record.train_loss}")
print(f"Canonical: {record.canonical_name}")  # checkpoint-190000-20251127-1430

# Parse any checkpoint name
from core.checkpoint_ledger import extract_step
step = extract_step("checkpoint-190000-20251127-1430")  # 190000
```

### Canonical Name Format

```
checkpoint-190000-20251127-1430
         │        │        └── Time (HHMM)
         │        └── Date (YYYYMMDD)
         └── Step number
```

### API Endpoints (on VaultKeeper :8767)

```bash
# List all checkpoints with stats
curl http://localhost:8767/api/ledger

# Get specific checkpoint
curl http://localhost:8767/api/ledger/190000

# Get best by metric
curl http://localhost:8767/api/ledger/best?metric=train_loss
```

### Key Files

```
core/checkpoint_ledger.py  # Ledger system
status/checkpoint_ledger.json  # Central index
current_model/checkpoint-*/.ledger.json  # Sidecar files
```

### RemoteLedgerClient (for remote hosts)

When running on a remote host (3090, NAS), use `RemoteLedgerClient` to query the ledger API:

```python
from core.checkpoint_ledger import RemoteLedgerClient, get_ledger_client

# Option 1: Auto-detect (local vs remote)
ledger = get_ledger_client()  # Uses local ledger on 4090, remote API elsewhere

# Option 2: Explicit remote client
client = RemoteLedgerClient("http://trainer.local:8767/api/ledger")

# Same interface as local ledger
latest = client.get_latest()
best = client.get_best(metric="train_loss")
all_records = client.list_all(limit=20)
record = client.get(190000)

# Check if API is reachable
if client.is_available():
    print("Ledger API online")
```

---

## 🌐 HOST REGISTRY - SERVICE DISCOVERY

**NEW (2025-11-27)** - Location-independent service access

The Host Registry defines where all services run. Components query it instead of hardcoding IPs.

**Components using Host Registry:**
- **Tavern** (`tavern/server.py`) - For Oracle/inference hosts
- **RemoteLedgerClient** (`core/checkpoint_ledger.py`) - For ledger API URL
- **Zone Federation** (`vault/zones.py`) - For zone endpoints

### Usage

```python
from core.hosts import get_service_url, get_host, is_trainer_local

# Get service URL
ledger_url = get_service_url("ledger")  # http://trainer.local:8767/api/ledger
inference_url = get_service_url("inference")  # http://inference.local:8765

# Check if we're on the trainer
if is_trainer_local():
    ledger = get_ledger()  # Local file access
else:
    # Use remote API
    pass

# Get host config
host = get_host("3090")
print(host.name)  # "Inference Server"
print(host.models_dir)  # ~/llm/models
```

### Configuration (`config/hosts.json`)

```json
{
  "hosts": {
    "4090": {
      "name": "Training Server",
      "host": "trainer.local",
      "role": "trainer",
      "services": {
        "vault": {"port": 8767, "path": "/api"},
        "ledger": {"port": 8767, "path": "/api/ledger"},
        "training": {"port": 8767, "path": "/api/training"},
        "tavern": {"port": 8888}
      }
    },
    "3090": {
      "name": "Inference Server",
      "host": "inference.local",
      "role": "inference",
      "services": {
        "inference": {"port": 8765}
      }
    }
  }
}
```

---

## 🔮 ORACLE - TALK TO DIO

**UPDATED (2025-11-27)** - Strict version checking, no fallbacks

The Oracle lets you chat with any checkpoint loaded on the inference server.

### Access

Visit: **http://localhost:8888/oracle**

### Features

1. **View Loaded Models** - See ALL models currently in the inference pool
2. **Select Checkpoint** - Pick from ledger OR from loaded models
3. **Strict Chat** - Step parameter REQUIRED, no fallback to wrong model
4. **Version Verification** - Response confirms which model was actually used

### Strict Version Checking

The Oracle enforces strict version verification:
- **Step required** - Chat requests MUST specify which step to use
- **No fallback** - If requested step not loaded, fails with 404 (not silent wrong model)
- **Server verification** - Response includes `model` and `model_path` confirming what ran
- **Accurate status** - Uses `/models/info` (actual loaded models) not stale `/health`

### API

```bash
# Get oracle status (shows ALL loaded models)
curl http://localhost:8888/oracle/status
# Returns: {"hosts": {"3090": {"loaded_models": [...], "model_count": 3}}}

# Load checkpoint on host
curl -X POST http://localhost:8888/oracle/load \
  -H "Content-Type: application/json" \
  -d '{"step": 181865, "host": "3090"}'

# Chat (step REQUIRED)
curl -X POST http://localhost:8888/oracle/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello DIO!", "host": "3090", "step": 181865}'
# Returns: {"success": true, "model": "checkpoint-181865", "model_path": "..."}

# Chat with base model (step 0)
curl -X POST http://localhost:8888/oracle/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "host": "3090", "step": 0}'

# Request unloaded step (fails clearly)
curl -X POST http://localhost:8888/oracle/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi", "host": "3090", "step": 148000}'
# Returns: {"success": false, "error": "Step 148000 not loaded. Available: [...]"}
```

---

## ⚔️ SPARRING WITH THE TRAINERS

**NEW (2025-11-27)** - Learn from combat mistakes

When DIO spars with skill trainers, every wrong answer generates 3 training examples:

| # | Type | Prompt | Golden Answer |
|---|------|--------|---------------|
| 1 | Identify Wrong | `[problem + wrong answer] Is this correct?` | `It is incorrect.` |
| 2 | Correct It | `[problem + wrong answer] This is incorrect. Find the correct solution.` | `[golden answer]` |
| 3 | Confirm Right | `[fresh problem + golden] Is this correct?` | `It is correct.` |

### Usage

```bash
# Run sparring session (100 problems)
python3 guild/sparring.py --skill binary --count 100

# Specific checkpoint
python3 guild/sparring.py --skill binary --checkpoint checkpoint-180000

# Validate sparring files
python3 guild/sparring_validator.py --check-all
```

### Task Registry

Tasks available for future Task Master scheduling:

```bash
# List all tasks
python3 guild/task_registry.py list

# Show tasks ready to run on 3090
python3 guild/task_registry.py available --gpu 3090

# Run a task manually
python3 guild/task_registry.py run --task sparring_binary
```

### Key Points

- **Always HIGH priority queue** - Sparring data is checkpoint-specific, becomes stale when model advances
- **Requires 3090** - Uses inference API for testing
- **Auto-queued** - Results go directly to `queue/high/`
- **Status**: `status/sparring.json`

### Files

```
guild/
├── sparring.py           # Main sparring system
├── sparring_validator.py # Dedicated validator
├── task_registry.py      # Task definitions
└── task_master.py        # GPU-aware scheduler
```

---

## 🤖 TASK MASTER - GPU-AWARE SCHEDULER

**NEW (2025-11-27)** - Runs tasks when GPU resources are available

The Task Master monitors GPU utilization on both 4090 and 3090, running tasks opportunistically when resources are idle (<40% utilization).

### Usage

```bash
# Check status
python3 guild/task_master.py --status

# Run one task (if GPU idle)
python3 guild/task_master.py --once

# Run as daemon (continuous)
python3 guild/task_master.py --daemon --interval 60

# Force run specific task
python3 guild/task_master.py --run sparring_binary
```

### Registered Tasks (by priority)

| Pri | Task | GPU | Description |
|-----|------|-----|-------------|
| **10** | `process_eval_queue` | 3090 | Process pending evals (from checkpoint saves) |
| 9 | `sparring_binary` | 3090 | Sparring session (100 problems) |
| 9 | `sparring_sy` | 3090 | SYLLO sparring (disabled) |
| 8 | `sparring_binary_large` | 3090 | Extended sparring (500 problems) |
| 6 | `generate_binary` | none | Generate training data |
| 4 | `health_check` | none | System health check |
| 4 | `validate_queue` | none | Validate queue files |
| 3 | `checkpoint_cleanup` | none | Clean old checkpoints |
| 3 | `vault_scan` | none | Scan VaultKeeper assets |
| 2 | `model_warmup` | 3090 | Warm up inference model |
| 2 | `lineage_report` | none | Data lineage report |

**Note**: Evals are ONLY run once per checkpoint. The `process_eval_queue` task checks for pending evals (queued when checkpoints are saved) and processes them. It does NOT re-run evals.

### Task Registry CLI

```bash
# List all tasks
python3 guild/task_registry.py list

# Show available tasks for GPU
python3 guild/task_registry.py available --gpu 3090

# Check task status/cooldowns
python3 guild/task_registry.py status
```

### Status Files

- `status/task_master.json` - Current GPU status, last task run
- `status/task_state.json` - Task execution history, cooldowns

---

## 🏰 ZONE FEDERATION - BRANCH OFFICERS

**NEW (2025-11-27)** - Distributed asset management

Each zone (4090, 3090, NAS) can run a Branch Officer daemon that tracks local assets.

### Start Branch Officer

```bash
# On 3090
python3 vault/branch_officer.py --zone 3090 --port 8768 --base-dir ~/llm/models

# On NAS
python3 vault/branch_officer.py --zone nas --port 8768 --base-dir /volume1/data/llm_training
```

### Branch Officer API (port 8768)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Zone health and stats |
| `/assets` | GET | List local assets |
| `/assets/{id}` | GET | Get asset details |
| `/fetch/{id}` | GET | Get transfer info |
| `/scan` | POST | Scan for new assets |
| `/receive` | POST | Prepare to receive asset |

### Transfer Between Zones

```python
from vault import push_to_zone, pull_from_zone

# Push checkpoint to 3090
result = push_to_zone(
    "$TRAINING_BASE_DIR/current_model/checkpoint-190000-20251127-1430",
    "3090"
)

# Pull from 3090
result = pull_from_zone(
    "~/llm/models/checkpoint-190000-20251127-1430",
    "3090"
)
```

---

## 📁 DIRECTORY STRUCTURE

**Reorganized 2025-11-22** - All files organized + RPG wrappers added 2025-11-26

```
$TRAINING_BASE_DIR/
│
├── CLAUDE.md                    # This file (Claude instructions)
├── config.json                  # Active configuration
├── realm.py                     # 🏰 Unified RPG entry point
│
├── guild/                       # 🏰 RPG: Skills, quests, dispatch, progression
│   ├── skills/                  # Skill definitions (SY, BIN)
│   ├── quests/                  # Quest templates and forge
│   ├── dispatch/                # Quest coordination (QuestDispatcher)
│   ├── progression/             # XP, levels, effects
│   └── ...
│
├── arena/                       # 🏰 RPG: Training execution
│   ├── quest_board.py           # QuestBoard (wraps TrainingQueue)
│   ├── battle_control.py        # BattleControl (wraps TrainingController)
│   ├── battle_log.py            # BattleLog (training status)
│   └── gatekeepers/             # ScrollInspector, ContentWarden
│
├── watchtower/                  # 🏰 RPG: Monitoring
│   ├── scrying_pool.py          # Real-time training observation
│   ├── champion_board.py        # Model checkpoint rankings
│   └── oracle_client.py         # Inference client
│
├── vault/                       # 🏰 RPG: Storage management
│   ├── keeper.py                # 🆕 VaultKeeper (central asset registry)
│   ├── server.py                # 🆕 API server (port 8767)
│   ├── client.py                # 🆕 Client library for remote queries
│   ├── discovery.py             # 🆕 Auto-discovery and registration
│   ├── assets.py                # 🆕 Asset types and schemas
│   ├── handlers.py              # 🆕 Location handlers (local/remote/NAS)
│   ├── archivist.py             # Backup management
│   ├── chronicle.py             # Version history
│   ├── treasury.py              # Retention/disk management
│   └── storage_registry.py      # Multi-stronghold storage
│
├── sentinels/                   # 🏰 RPG: System protection
│   ├── guardian.py              # Daemon watchdog
│   └── health_inspector.py      # Comprehensive health checks
│
├── armory/                      # 🏰 RPG: Equipment (wraps trainer/)
├── scrolls/                     # 🏰 RPG: Utilities (wraps tools/)
│
├── trainer/                     # Refactored training modules (3-layer architecture)
│   ├── config/                  # Layer 2: Configuration system
│   │   ├── schema.py            # 8 dataclasses (Hyperparams, ProfileConfig, etc.)
│   │   └── loader.py            # ConfigLoader (JSON + CLI merging)
│   ├── profiles/                # Layer 3: Pluggable data profiles
│   │   ├── base.py              # DataProfile ABC interface
│   │   ├── emoji_think.py       # Emoji thinking/stop profile
│   │   └── regime3.py           # Symbolic reasoning profile
│   ├── monitoring/              # Layer 3: Monitoring callbacks
│   │   ├── status_writer.py     # TrainingStatusWriter
│   │   └── callbacks.py         # LiveMonitorCallback (with on_save, ledger, remote eval)
│   ├── core/                    # Layer 1: Engine API
│   │   └── engine.py            # TrainerEngine - full HF training pipeline
│   ├── optimizers/              # Optimizer factory (Muon + AdamW)
│   │   ├── factory.py           # create_optimizer() entry point
│   │   ├── param_groups.py      # split_transformer_params()
│   │   └── muon.py              # SingleDeviceMuonWithAuxAdam
│   └── cli_main.py              # CLI demonstration
│
├── README.md                    # System overview
├── QUICKSTART.md                # Getting started
├── ARCHITECTURE.md              # System design
├── TROUBLESHOOTING.md           # Problem solving
├── DEVELOPMENT.md               # Development guide
├── CHANGELOG.md                 # Change tracking
│
├── pyproject.toml               # 🆕 Package config (pip install -e .)
│
├── core/                        # Core training system
│   ├── train.py                 # UltimateTrainer - CLI orchestrator (delegates to TrainerEngine)
│   ├── training_daemon.py       # File watcher + orchestrator
│   ├── training_controller.py   # Control commands (pause/resume/stop)
│   ├── training_queue.py        # Queue management
│   ├── training_status.py       # Status writer
│   ├── paths.py                 # 🆕 Path auto-detection (get_base_dir)
│   ├── daemon/                  # 🆕 Extracted daemon services
│   │   ├── pid_manager.py       # Single-instance enforcement
│   │   ├── file_watcher.py      # Directory monitoring + inbox flattening
│   │   ├── snapshot_service.py  # Checkpoint snapshots
│   │   └── background_worker.py # Non-blocking task runner
│   ├── training/                # 🆕 Extracted training components
│   │   ├── model_loader.py      # Model loading with precision config
│   │   ├── dataset_preparer.py  # Dataset preparation
│   │   └── monitoring_bundle.py # Training monitoring
│   ├── validation/              # 🆕 Two-layer validation system
│   │   ├── spec.py              # SpecValidator + DatasetSpec registry (deny-by-default)
│   │   └── validator.py         # DataValidator (QUICK/STANDARD/DEEP content checks)
│   ├── custom_collator.py       # Data collator
│   ├── logit_penalty.py         # Penalty processors
│   ├── validator.py             # Legacy validator (deprecated)
│   ├── model_db.py              # Model database
│   └── time_estimator.py        # Time estimation
│
├── monitoring/                  # Monitoring + Web UI
│   ├── servers/                 # API servers
│   │   ├── live_monitor.py      # Main monitor server
│   │   ├── memory_stats_api.py  # Memory stats API
│   │   └── enhanced_monitor.py  # Enhanced monitoring
│   ├── ui/                      # HTML files
│   │   └── *.html
│   ├── js/                      # JavaScript modules
│   └── css/                     # Stylesheets
│
├── management/                  # Model/backup management
│   ├── backup_manager.py        # Backup system
│   ├── model_versioner.py       # Version control
│   ├── consolidate_model.py     # Model consolidation
│   ├── checkpoint_retention.py  # Checkpoint cleanup
│   ├── safe_checkpoint_cleanup.py
│   ├── daily_snapshot.py        # Daily snapshots
│   └── auto_disk_manager.py     # Auto disk cleanup
│
├── safety/                      # Watchdogs + health checks
│   ├── daemon_watchdog.py       # Auto-restart daemon
│   ├── anti_stuck_monitor.py    # Detect hangs
│   ├── crash_detector.py        # Crash analysis
│   ├── comprehensive_health_check.py
│   ├── config_validator.py      # Config validation
│   └── verify_checkpoint_resume.py
│
├── tools/                       # Utilities
│   ├── data/                    # Data processing
│   │   ├── generate_syllo_batch.py
│   │   ├── validate_data.py
│   │   ├── convert_*.py
│   │   └── analyze_training_data.py
│   ├── config/
│   │   └── edit_config.py
│   └── analysis/
│       ├── state_tracker.py     # System state tracking
│       ├── calculate_data_value.py
│       └── compare_models.py
│
├── tests/                       # Test files
│   └── test_*.py
│
├── scripts/                     # Shell scripts
│   ├── start_all.sh             # Start all services
│   ├── check_health.sh          # Health check
│   └── bin/                     # Launcher scripts
│
├── data/                        # Training data
│   ├── validation/              # Fixed validation set
│   └── flagged_examples/        # Flagged outputs
│
├── models/                      # Model storage
│   ├── Qwen3-0.6B/              # Base model (1.5GB)
│   ├── current_model/           # Active checkpoint (EMPTY - needs setup)
│   └── current_model_small/     # Small model checkpoint
│
├── backups/                     # Backups
│   └── consolidated/            # Consolidated backups
│
├── logs/                        # Training logs (daily rotation)
├── status/                      # Status JSON files
├── control/                     # Control files (.stop, .pause, etc.)
├── inbox/                       # Drop zone for training files
├── queue/                       # Priority queues
│   ├── high/
│   ├── normal/
│   ├── low/
│   ├── processing/              # Currently training
│   ├── failed/                  # Failed files
│   └── recently_completed/
│
├── scratch/                     # Working space for design docs & experiments
│   ├── DAEMON_REFACTOR_PLAN.md  # Current work: daemon refactor planning
│   ├── IMPLEMENTATION_PLAN_TASKS.md  # Task breakdowns
│   ├── MONITORING_V2_DESIGN.md  # Monitoring system designs
│   ├── RETENTION_POLICY_DESIGN.md   # Policy documents
│   └── *.md                     # Other design/planning docs
│
└── archive/                     # Archived code & completed work
    ├── refactor_2025_11_22/     # Nov 22 trainer/ refactor
    │   ├── code/                # Backup train.py versions
    │   ├── docs/                # Refactor documentation
    │   └── tests/               # Profile & engine tests
    ├── configs/                 # Old config files
    ├── experiments/             # Old experiments
    └── PERMANENT_ERROR_TRAINING/

# IGNORED (user data/notes):
GOTCHA_BUSINESS_MODEL/
OBSERVATIONS/
```

---

## 🎯 SKILL SYSTEM (Updated 2025-11-28)

**YAML configs are the single source of truth for skills.**

### Skill Engine (NEW)

The Skill Engine provides a unified interface for all skill operations:

```python
from guild.skills import get_engine

engine = get_engine()

# Get a skill (unified training + eval interface)
skill = engine.get("bin")

# Generate eval problems locally (no API needed)
batch = engine.generate_eval_batch("bin", level=1, count=10)

# Score with per-primitive tracking
result, state = engine.run_eval("bin", model_answers, level=1)
print(result.per_primitive_accuracy)
# {'binary_add_no_carry': 0.95, 'bitwise_and': 0.80, ...}
```

### Skill Configs

Location: `configs/skills/*.yaml`

| File | Skill | Icon | Levels | Primitives | Passive |
|------|-------|------|--------|------------|---------|
| `sy.yaml` | Word Weaving | 🧩 | 50 | 7 | word_puzzles |
| `bin.yaml` | Binary Alchemy | 🔢 | 30 | 10 | binary_arithmetic |
| `_template.yaml` | Template | - | - | - | - |

### Passives (11 total)

| Passive | Category | Primitives | Description |
|---------|----------|------------|-------------|
| `arithmetic` | arithmetic | 5 | Basic number sense |
| `binary_arithmetic` | math | 10 | BIN skill eval |
| `code_trace` | code | 6 | Code execution tracing |
| `combinatorics` | math | 5 | Permutations, combinations |
| `counting` | counting | 4 | Letter/word counting |
| `language_basics` | reasoning | 5 | Synonyms, antonyms, plurals |
| `logic` | logic | 4 | Boolean operations |
| `memory` | memory | 5 | Fact retention |
| `sequence` | reasoning | 5 | Pattern recognition |
| `string_craft` | string_craft | 4 | String manipulation |
| `word_puzzles` | reasoning | 5 | SY skill eval |

### Primitives

Primitives are atomic testable concepts within skills. Each problem is tagged with a `primitive_id`:

```python
from guild.skills.primitives import list_tracks, PRIMITIVE_CATALOG

# 5 tracks: arithmetic, binary, logic, string, code
print(list_tracks())

# 39 primitives in catalog
for track, prims in PRIMITIVE_CATALOG.items():
    print(f"{track}: {len(prims)} primitives")
```

### Targeted Sparring

Focus training on weak primitives:

```bash
# Report weak primitives (< 80% accuracy)
python3 guild/targeted_sparring.py --report

# Generate problems targeting weak primitives
python3 guild/targeted_sparring.py --skill bin --threshold 0.8 --count 100
```

### Tavern API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/engine/health` | Engine status |
| `/api/engine/primitives` | All primitives |
| `/api/engine/primitive-summary` | Per-primitive accuracy |
| `/api/engine/skill/{id}/primitives` | Skill primitives |
| `/api/engine/skill/{id}/state` | Skill state |

### Files

```
configs/skills/
├── _template.yaml   # Template with primitives section
├── sy.yaml          # SY - 7 primitives, passive_id: word_puzzles
└── bin.yaml         # BIN - 10 primitives, passive_id: binary_arithmetic

guild/skills/
├── engine.py        # SkillEngine - central manager
├── primitives.py    # PrimitiveId, PRIMITIVE_CATALOG
├── eval_types.py    # EvalProblem, EvalBatch, EvalResult
├── skill.py         # Skill ABC
├── composite.py     # CompositeSkill (generator + passive)
├── adapters/        # GeneratorAdapter, PassiveAdapter
├── types.py         # SkillConfig, SkillState
├── loader.py        # get_trainer(), load_skill_config()
└── registry.py      # SkillRegistry

guild/passives/
├── binary_arithmetic.py  # BIN skill (10 primitives)
├── word_puzzles.py       # SY skill (5 primitives)
├── code_trace.py         # Code tracing (6 primitives)
├── combinatorics.py      # Math counting (5 primitives)
├── language_basics.py    # Language (5 primitives)
├── sequence.py           # Patterns (5 primitives)
├── memory.py             # Facts (5 primitives)
└── ... (4 more)
```

---

## 🔧 TRAINERENGINE - TRAINING PIPELINE (Updated 2025-11-28)

**Complete refactor** - TrainerEngine now owns the full HuggingFace training pipeline.

### Architecture

```
UltimateTrainer (core/train.py)
├── CLI parsing, config loading
├── Data validation
├── Time estimation
└── Delegates to TrainerEngine (USE_ENGINE=1)

TrainerEngine (trainer/core/engine.py)
├── run_job(config) → TrainingResult
├── Model loading (Qwen3VL, Flash Attention 2)
├── Dataset preparation (profiles, packing)
├── Checkpoint resumption
├── Trainer creation (Muon/AdamW, collators)
├── Masking validation
└── Training execution

LiveMonitorCallback (trainer/monitoring/callbacks.py)
├── on_step_end() - Progress, alerts, throughput
├── on_evaluate() - Validation loss tracking
└── on_save() - Checkpoint ledger, eval queuing, remote sync
```

### Usage

```bash
# Enable new engine path (gradual migration)
USE_ENGINE=1 python3 core/train.py \
    --dataset data/train.jsonl \
    --model qwen3_0.6b \
    --output-dir outputs/run \
    --yes
```

### Key Files

| File | Lines | Description |
|------|-------|-------------|
| `trainer/core/engine.py` | ~970 | TrainerEngine - full HF pipeline |
| `trainer/monitoring/callbacks.py` | ~630 | LiveMonitorCallback with all features |
| `core/train.py` | ~1880 | UltimateTrainer - CLI orchestrator |
| `tests/test_trainer_engine.py` | ~350 | 19 unit tests |

### TrainingResult

```python
from trainer.core.engine import TrainerEngine, TrainingResult

result = engine.run_job(config)
if result.success:
    print(f"Completed at step {result.global_step}")
    print(f"Final loss: {result.final_loss}")
    print(f"Checkpoint: {result.last_checkpoint_path}")
else:
    print(f"Failed: {result.error_message}")
```

---

## ⚡ MUON OPTIMIZER (New 2025-11-27)

**Muon** (MomentUm Orthogonalized by Newton-schulz) is an alternative optimizer that can be faster than AdamW for transformer training.

### How It Works

1. Runs standard SGD-momentum internally
2. Orthogonalizes each 2D parameter's update using Newton-Schulz iteration
3. Only applies to hidden weight matrices (attention, MLP projections)
4. Embeddings, output heads, and biases still use AdamW

### Enable Muon

Edit `config.json`:

```json
{
  "optimizer": {
    "type": "muon",
    "muon": {
      "hidden_lr": 0.02,
      "aux_lr": 0.0003,
      "momentum": 0.95
    }
  }
}
```

### Parameter Grouping

| Parameters | Optimizer | Default LR |
|------------|-----------|------------|
| q_proj, k_proj, v_proj, o_proj | Muon | 0.02 |
| gate_proj, up_proj, down_proj | Muon | 0.02 |
| embed_tokens, lm_head | AdamW | 0.0003 |
| layernorms, biases | AdamW | 0.0003 |

### Memory

Muon uses ~25% less optimizer memory than AdamW:
- AdamW: 8 bytes/param (m + v states)
- Muon: 4 bytes/param (single momentum buffer)

For 0.6B model: ~1.2 GB savings

### Files

```
trainer/optimizers/
├── __init__.py      # Module exports
├── factory.py       # create_optimizer() entry point
├── param_groups.py  # split_transformer_params()
└── muon.py          # SingleDeviceMuonWithAuxAdam
```

### Usage

```python
from trainer.optimizers import create_optimizer, split_transformer_params

# Via config
optimizer, scheduler, info = create_optimizer(model, config, num_training_steps=10000)

# Manual
param_groups = split_transformer_params(model, hidden_lr=0.02, aux_lr=3e-4)
optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
```

### Attribution

Muon optimizer by Keller Jordan: https://github.com/KellerJordan/Muon

---

## 🚀 4B FULL FINE-TUNE (New 2025-11-27)

Full fine-tuning of **Qwen3-4B** on a single 24GB RTX 4090 is now possible.

### VRAM Budget

| Component | Memory |
|-----------|--------|
| Model (bf16) | 8 GB |
| Gradients (bf16) | 8 GB |
| Activations | ~3 GB |
| Paged 8-bit Adam | Offloads to CPU |
| **Peak VRAM** | **18.6 GB** |

### Key Optimizations

1. **Paged 8-bit Adam** (`paged_adamw_8bit`) - Offloads optimizer states to CPU when GPU fills
2. **Liger Kernel** - Fused ops, logits never materialized (saves ~2GB)
3. **Gradient Checkpointing** - Trades compute for VRAM
4. **bf16 Precision** - Half the memory footprint

### Usage

```bash
python3 scripts/train_4b_full.py
```

### Configuration

```python
# Key settings in train_4b_full.py
MODEL_PATH = "models/Qwen3-4B-Instruct-2507"
MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM = 32  # Effective batch = 32
optim = "paged_adamw_8bit"
```

### Why Not DeepSpeed?

DeepSpeed ZeRO-3 was tested but triggered Linux OOM killer due to 178GB virtual memory allocation during initialization. The simpler bitsandbytes paged optimizer works better for single-GPU scenarios.

### Files

```
scripts/train_4b_full.py           # Training script
configs/ds_zero2_offload.json      # DeepSpeed config (alternative)
configs/ds_zero3_offload.json      # DeepSpeed config (not recommended)
```

---

## 🎯 CURRENT STATE

**Training:** Step ~182000 | **Latest Checkpoint:** ~181800

### Key Services

| Service | Port | Location |
|---------|------|----------|
| Tavern UI | 8888 | 4090 |
| VaultKeeper | 8767 | 4090 |
| Inference | 8765 | 3090 |
| GPU Scheduler | 8766 | 3090 |

### Quick Commands

```bash
# Check health
curl http://localhost:8888/health
curl http://localhost:8767/api/stats

# Control training
python3 core/training_controller.py status|pause|resume|stop

# Queue management
python3 core/training_queue.py status
```

---

## 📝 NOTES FOR CLAUDE

1. Run health check: `python3 safety/comprehensive_health_check.py`
2. **ASK USER** before making changes
3. **ASK USER** before creating new documentation
4. Trust code as ground truth, not old docs
