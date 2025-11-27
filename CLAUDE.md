# REALM OF TRAINING - Game Design Document

**Last Updated:** 2025-11-27 (Quests Page + VRAM Calculator + Scheduler UI)
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
# Start the Tavern (game UI)
python3 tavern/server.py --port 8888

# Or via Python
from realm import open_tavern
open_tavern()
```

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
├── tavern/         # 🍺 MAIN GAME UI - Where you watch DIO (port 8888)
│   ├── server.py           # Tavern server
│   ├── templates/game.html # Game interface
│   └── static/             # CSS, JS assets
│
├── guild/          # 🏰 Skills & progression
│   ├── skills/             # Skill system (loads from configs/skills/*.yaml)
│   ├── quests/             # Quest templates
│   └── progression/        # XP, level-up logic
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
| **Tavern** | Game UI | 8888 |
| **Arena** | Training daemon | - |
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

**Tavern UI Expansion (2025-11-27 Late)**
- ✅ **Quests Page** (`/quests`) - Full quest board with queue management
  - View queued/processing/completed/failed quests
  - Change priority, delete, retry failed quests
  - Auto-refresh every 10 seconds
- ✅ **VRAM Calculator** (`/settings`) - Estimate GPU memory usage
  - Based on batch size, max length, precision, gradient checkpointing
  - GPU presets (RTX 4090/3090/4080/4070)
  - Visual breakdown: model weights, optimizer, gradients, activations
- ✅ **Scheduler in Settings** (`/settings`) - Full curriculum scheduler integration
  - Quick presets (8 options)
  - Strategy selection (equal, focus, weighted, catch-up)
  - Per-skill enable/disable, priority, weight
  - Next decision preview

**Checkpoint Ledger & Distributed Architecture (2025-11-27)**
- ✅ **Checkpoint Ledger** (`core/checkpoint_ledger.py`) - Single source of truth for checkpoint stats
- ✅ **Canonical Naming** - Checkpoints named `checkpoint-{step}-{YYYYMMDD}-{HHMM}`
- ✅ **Sidecar Files** - Each checkpoint has `.ledger.json` with stats at save time
- ✅ **Host Registry** (`core/hosts.py`) - Service discovery for distributed operation
- ✅ **Branch Officers** (`vault/branch_officer.py`) - Zone daemons for 3090/NAS
- ✅ **Zone Federation** (`vault/zones.py`) - Transfer assets between zones
- ✅ **Oracle Page** (`/oracle`) - "Talk to DIO" - chat with any checkpoint
- ✅ **Ledger API** on VaultKeeper - `/api/ledger/*` endpoints
- ✅ **Training API** on VaultKeeper - `/api/training/*` endpoints
- ✅ **Save frequency** - Now every 10k steps (was saving every file)

**Tavern Game UI (2025-11-26)**
- ✅ Web-based game interface at port 8888
- ✅ DIO hero display with ASCII art
- ✅ Real-time battle status
- ✅ Idle game mechanics (XP ticks, floating numbers)
- ✅ Skill cards, vault summary, forge status

**VaultKeeper (2025-11-26)**
- ✅ Central asset registry across all devices
- ✅ "Ask Vault First" pattern for asset location
- ✅ API server on port 8767
- ✅ SQLite catalog at `vault/catalog.db`

**RPG Architecture (2025-11-26)**
- ✅ Full RPG-themed module system implemented
- ✅ `realm.py` - Unified entry point to all systems
- ✅ `guild/` - Skills, quests, progression, dispatch
- ✅ `arena/` - Training execution (QuestBoard, BattleControl, BattleLog)
- ✅ `watchtower/` - Monitoring (ScryingPool, ChampionBoard, OracleClient)
- ✅ `vault/` - Storage (Archivist, Chronicle, Treasury, StorageRegistry)
- ✅ `sentinels/` - Protection (Guardian, HealthInspector)
- ✅ `armory/` - Equipment/profiles wrapper
- ✅ `scrolls/` - Utility scripts wrapper
- ✅ StorageRegistry supports multiple Synology NAS strongholds

**PREVIOUS UPDATE:** Data Lineage System (2025-11-26)
- ✅ Generator versioning: `GENERATOR_ID` + `GENERATOR_VERSION` in all generators
- ✅ Validator versioning: `VALIDATOR_NAME` + `VALIDATOR_VERSION` in all validators
- ✅ `.meta.json` sidecar files for data provenance tracking
- ✅ LineageTracker aggregates per-generator/validator rejection stats
- ✅ New `/api/lineage` endpoint for dashboard
- ✅ New "Data Lineage" card on master dashboard
- ✅ Answers: "Which generator produces most rejections?", "Which validator is most aggressive?"

**PREVIOUS UPDATE:** Code Review Validated Monitoring Systems (2025-11-25)
- ✅ API authentication added to inference server
- ✅ Test infrastructure cleaned up for CI
- ✅ RetentionManager wired into daemon
- ✅ Extracted daemon services: PIDManager, FileWatcher, SnapshotService, BackgroundWorker
- ✅ Extracted training components: ModelLoader, DatasetPreparer, MonitoringBundle
- ✅ Created pyproject.toml - GPU deps now optional `[training]` extra
- ✅ DataValidator (QUICK/STANDARD/DEEP) - integrated into daemon for early rejection
- ✅ Path auto-detection via get_base_dir() with resolution logging

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
    host="192.168.x.x"
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
model_path = "/path/to/training/models/checkpoint-175000"

# NEW WAY (ask vault first):
model_path = ask_vault_first(
    "checkpoint_175000",
    fallback="/path/to/training/models/checkpoint-175000"
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
curl http://192.168.x.x:8767/api/locate/checkpoint_175000
curl http://192.168.x.x:8767/api/stats
curl http://192.168.x.x:8767/api/checkpoints
```

### Client Library (for remote devices)

```python
from vault.client import VaultKeeperClient

# On 3090, connect to 4090's VaultKeeper
client = VaultKeeperClient("192.168.x.x:8767")

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
client = RemoteLedgerClient("http://192.168.x.x:8767/api/ledger")

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
ledger_url = get_service_url("ledger")  # http://192.168.x.x:8767/api/ledger
inference_url = get_service_url("inference")  # http://192.168.x.x:8765

# Check if we're on the trainer
if is_trainer_local():
    ledger = get_ledger()  # Local file access
else:
    # Use remote API
    pass

# Get host config
host = get_host("3090")
print(host.name)  # "Inference Server"
print(host.models_dir)  # /path/to/models
```

### Configuration (`config/hosts.json`)

```json
{
  "hosts": {
    "4090": {
      "name": "Training Server",
      "host": "192.168.x.x",
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
      "host": "192.168.x.x",
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

**NEW (2025-11-27)** - Chat with any checkpoint version

The Oracle lets you load any checkpoint and chat with it.

### Access

Visit: **http://localhost:8888/oracle**

### Features

1. **Select Checkpoint** - Pick from ledger (shows loss, skill, age)
2. **Select Host** - Choose inference host (3090, etc.)
3. **Load** - Load checkpoint on the selected host
4. **Chat** - Talk to that version of DIO

### API

```bash
# List inference hosts
curl http://localhost:8888/oracle/hosts

# Get oracle status (what's loaded where)
curl http://localhost:8888/oracle/status

# Load checkpoint on host
curl -X POST http://localhost:8888/oracle/load \
  -H "Content-Type: application/json" \
  -d '{"step": 190000, "host": "3090"}'

# Chat
curl -X POST http://localhost:8888/oracle/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello DIO!", "host": "3090"}'
```

---

## 🏰 ZONE FEDERATION - BRANCH OFFICERS

**NEW (2025-11-27)** - Distributed asset management

Each zone (4090, 3090, NAS) can run a Branch Officer daemon that tracks local assets.

### Start Branch Officer

```bash
# On 3090
python3 vault/branch_officer.py --zone 3090 --port 8768 --base-dir /path/to/models

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
    "/path/to/training/current_model/checkpoint-190000-20251127-1430",
    "3090"
)

# Pull from 3090
result = pull_from_zone(
    "/path/to/models/checkpoint-190000-20251127-1430",
    "3090"
)
```

---

## 📁 DIRECTORY STRUCTURE

**Reorganized 2025-11-22** - All files organized + RPG wrappers added 2025-11-26

```
/path/to/training/
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
│   │   └── regime3.py           # Symbolic reasoning profile (NEW!)
│   ├── monitoring/              # Layer 3: Monitoring callbacks
│   │   ├── status_writer.py     # TrainingStatusWriter
│   │   └── callbacks.py         # LiveMonitorCallback
│   ├── core/                    # Layer 1: Engine API
│   │   └── engine.py            # TrainerEngine.run_job() (proof-of-concept)
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
│   ├── train.py                 # Main training script (HuggingFace Trainer)
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

## 🎯 SKILL SYSTEM (Updated 2025-11-27)

**YAML configs are the single source of truth for skills.**

### Skill Configs

Location: `configs/skills/*.yaml`

| File | Skill | Icon | Levels | API |
|------|-------|------|--------|-----|
| `sy.yaml` | Word Weaving | 🧩 | 50 | localhost:8080 |
| `bin.yaml` | Binary Alchemy | 🔢 | 30 | localhost:8090 |
| `_template.yaml` | Template for new skills | - | - | - |

### Usage

```python
from guild.skills import get_skill, get_trainer, list_skills

# List available skills
skills = list_skills()  # ['bin', 'sy']

# Load skill config (all metadata from YAML)
skill = get_skill('sy')
print(skill.icon)       # 🧩
print(skill.color)      # #8B5CF6
print(skill.max_level)  # 50
print(skill.api_url)    # http://localhost:8080

# Get trainer (API client) for generating samples
trainer = get_trainer('sy')
batch = trainer.sample(level=5, count=100)
```

### Adding a New Skill

1. Copy `configs/skills/_template.yaml` to `configs/skills/{id}.yaml`
2. Fill in all required fields (see template for structure)
3. Implement the API server in singleSKILL
4. That's it - no code changes needed in TRAINING

### Key Design Principles

- **Combinatorially Infinite**: Every skill MUST have infinite problem space
- **Static Eval Sets**: 5 samples per level, seeded for reproducibility
- **Signal Degradation**: Higher levels = weaker hints (for SY) or harder params (for BIN)

### Files

```
configs/skills/
├── _template.yaml   # Template with all required fields
├── sy.yaml          # SY - Syllacrostic puzzles (50 levels)
└── bin.yaml         # BIN - Binary arithmetic (30 levels)

guild/skills/
├── types.py         # SkillConfig, SkillDisplay, SkillAPI, SkillEval
├── loader.py        # get_trainer(), load_skill_config()
├── registry.py      # SkillRegistry
├── contract.py      # SkillClient (API client)
└── state_manager.py # Runtime state tracking
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
