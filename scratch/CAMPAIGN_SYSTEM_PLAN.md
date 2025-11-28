# THE CAMPAIGN SYSTEM - Implementation Plan

**Date:** 2025-11-27
**Scope:** Multi-hero, multi-campaign architecture with clean switching and archival

---

## VISION: "New Game+" for AI Training

Just like a save file in an RPG, you can:
- **Start New Campaign** with a fresh hero (new model, fresh training)
- **Switch Campaigns** without losing progress
- **Archive Campaigns** to the Hall of Legends
- **Compare Heroes** across campaigns

One click to RESET. Never lose your old save.

---

## RPG VOCABULARY

| Game Term | Technical Meaning |
|-----------|------------------|
| **Hero** | A model architecture (Qwen3-0.6B, Muon-0.6B, etc.) |
| **Campaign** | A training run with checkpoints, metrics, progression |
| **Saga** | Complete history of a campaign |
| **Scroll of Destiny** | Active campaign pointer |
| **Hall of Legends** | Archive of completed/old campaigns |
| **Battle Scars** | Metrics, losses, accuracy history |

---

## PHASE 1: HERO REGISTRY (configs/heroes/*.yaml)

### 1.1 Hero Profile Schema

Create `configs/heroes/_template.yaml`:

```yaml
# =============================================================================
# HERO TEMPLATE
# =============================================================================

id: hero-id                    # Unique identifier (lowercase, hyphens)
name: "Hero Display Name"      # Display name in UI
rpg_name: "The Title"          # RPG epithet ("The Skeptic", "The Scholar")

# =============================================================================
# MODEL IDENTITY
# =============================================================================
model:
  hf_name: "Qwen/Qwen3-0.6B"   # HuggingFace model ID
  family: "qwen3"              # Model family (qwen3, llama, mistral, phi)
  architecture: "Qwen3ForCausalLM"
  size_b: 0.6                  # Size in billions of parameters
  vocab_size: 151936
  context_length: 4096         # Max context window
  rope_scaling: "dynamic"      # RoPE style if applicable

# =============================================================================
# DEFAULT TRAINING SETTINGS
# =============================================================================
training_defaults:
  precision: "bf16"            # fp16, bf16, fp32
  load_in_4bit: false          # Quantization
  batch_size: 1                # Micro-batch (VRAM-bound)
  gradient_accumulation: 16    # Effective batch = 16
  learning_rate: 0.0004
  warmup_steps: 100
  max_length: 2048             # Training context
  gradient_checkpointing: true

# =============================================================================
# QLORA DEFAULTS (if using adapter training)
# =============================================================================
qlora:
  enabled: false
  r: 64
  alpha: 16
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# =============================================================================
# CHAT TEMPLATE
# =============================================================================
chat:
  template: "qwen3_chat"       # Template name or path
  system_token: "<|im_start|>system"
  user_token: "<|im_start|>user"
  assistant_token: "<|im_start|>assistant"
  end_token: "<|im_end|>"

# =============================================================================
# VRAM PROFILE (for estimator)
# =============================================================================
vram:
  base_memory_gb: 1.2          # Model weights in bf16
  per_batch_gb: 0.8            # Per micro-batch (at max_length=2048)
  optimizer_overhead_gb: 0.6   # AdamW states

# =============================================================================
# DISPLAY
# =============================================================================
display:
  icon: "icon path or emoji"
  portrait: "portraits/dio.png"
  color: "#8B5CF6"

# =============================================================================
# NOTES
# =============================================================================
notes: |
  Additional notes about this hero, quirks, known issues, etc.
```

### 1.2 Current Hero: DIO

Create `configs/heroes/dio-qwen3-0.6b.yaml`:

```yaml
id: dio-qwen3-0.6b
name: "DIO"
rpg_name: "The Skeptic"

model:
  hf_name: "Qwen/Qwen3-0.6B"
  family: "qwen3"
  architecture: "Qwen3ForCausalLM"
  size_b: 0.6
  vocab_size: 151936
  context_length: 4096
  rope_scaling: "dynamic"

training_defaults:
  precision: "bf16"
  load_in_4bit: false
  batch_size: 1
  gradient_accumulation: 16
  learning_rate: 0.0004
  warmup_steps: 100
  max_length: 2048
  gradient_checkpointing: true

qlora:
  enabled: false

chat:
  template: "qwen3_chat"
  system_token: "<|im_start|>system"
  user_token: "<|im_start|>user"
  assistant_token: "<|im_start|>assistant"
  end_token: "<|im_end|>"

vram:
  base_memory_gb: 1.2
  per_batch_gb: 0.8
  optimizer_overhead_gb: 0.6

display:
  icon: "portraits/dio_icon.png"
  portrait: "portraits/dio.png"
  color: "#8B5CF6"

notes: |
  DIO is our first hero - a Qwen3-0.6B model learning Binary Alchemy
  and Word Weaving (Syllacrostic puzzles). Started training 2025-11-22.
```

### 1.3 Hero Registry Module

Create `guild/heroes/registry.py`:

```python
"""
Hero Registry - Load and manage hero profiles.

Usage:
    from guild.heroes import get_hero, list_heroes, HeroProfile

    # List available heroes
    heroes = list_heroes()  # ['dio-qwen3-0.6b', 'muon-0.6b']

    # Get hero profile
    hero = get_hero('dio-qwen3-0.6b')
    print(hero.model.hf_name)  # "Qwen/Qwen3-0.6B"
    print(hero.training_defaults.learning_rate)  # 0.0004
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml

@dataclass
class ModelSpec:
    hf_name: str
    family: str
    architecture: str
    size_b: float
    vocab_size: int
    context_length: int
    rope_scaling: Optional[str] = None

@dataclass
class TrainingDefaults:
    precision: str
    load_in_4bit: bool
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    warmup_steps: int
    max_length: int
    gradient_checkpointing: bool

@dataclass
class QLoRAConfig:
    enabled: bool
    r: int = 64
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = None

@dataclass
class HeroProfile:
    id: str
    name: str
    rpg_name: str
    model: ModelSpec
    training_defaults: TrainingDefaults
    qlora: QLoRAConfig
    # ... other fields

class HeroRegistry:
    """Registry of all available heroes."""

    def __init__(self, base_dir: Path):
        self.heroes_dir = base_dir / "configs" / "heroes"
        self._cache: Dict[str, HeroProfile] = {}

    def list_heroes(self) -> List[str]:
        """List all hero IDs."""
        return [f.stem for f in self.heroes_dir.glob("*.yaml")
                if not f.name.startswith("_")]

    def get_hero(self, hero_id: str) -> HeroProfile:
        """Load a hero profile."""
        if hero_id not in self._cache:
            self._cache[hero_id] = self._load_hero(hero_id)
        return self._cache[hero_id]

    def _load_hero(self, hero_id: str) -> HeroProfile:
        path = self.heroes_dir / f"{hero_id}.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        return HeroProfile(...)  # Parse YAML to dataclass
```

---

## PHASE 2: CAMPAIGN STRUCTURE

### 2.1 Directory Layout

```
campaigns/
├── active -> dio-qwen3-0.6b/campaign-001  # Symlink to active campaign
│
├── dio-qwen3-0.6b/
│   ├── campaign-001/                      # First campaign
│   │   ├── checkpoints/
│   │   │   ├── checkpoint-182000/
│   │   │   ├── checkpoint-183000/
│   │   │   └── ...
│   │   ├── status/
│   │   │   ├── training_status.json
│   │   │   ├── curriculum_state.json
│   │   │   ├── checkpoint_ledger.json
│   │   │   └── ...
│   │   ├── logs/
│   │   │   └── training.log
│   │   ├── campaign.json                  # Campaign metadata
│   │   └── saga.json                      # Full history (optional)
│   │
│   └── campaign-002/                      # Fresh start
│       └── ...
│
├── muon-0.6b/
│   └── campaign-001/
│       └── ...
│
└── archive/                               # Hall of Legends
    └── dio-qwen3-0.6b/
        └── campaign-001-archived/
```

### 2.2 Campaign Metadata (campaign.json)

```json
{
  "id": "campaign-001",
  "hero_id": "dio-qwen3-0.6b",
  "name": "Binary Alchemy Training",
  "description": "Teaching DIO to compute in binary",

  "created_at": "2025-11-22T04:00:00Z",
  "status": "active",

  "starting_checkpoint": null,
  "starting_step": 0,

  "current_step": 183520,
  "total_examples": 450000,

  "skills_focus": ["bin", "sy"],

  "config_overrides": {
    "learning_rate": 0.0004,
    "batch_size": 1
  },

  "milestones": [
    {"step": 100000, "note": "First 100k steps", "date": "2025-11-24"},
    {"step": 180000, "note": "Switched to Binary focus", "date": "2025-11-27"}
  ],

  "archived_at": null
}
```

### 2.3 Active Campaign Pointer (control/active_campaign.json)

```json
{
  "hero_id": "dio-qwen3-0.6b",
  "campaign_id": "campaign-001",
  "campaign_path": "campaigns/dio-qwen3-0.6b/campaign-001",

  "activated_at": "2025-11-22T04:00:00Z",

  "_comment": "Scroll of Destiny - Points to the currently active campaign"
}
```

### 2.4 Campaign Manager Module

Create `guild/campaigns/manager.py`:

```python
"""
Campaign Manager - Create, switch, archive campaigns.

Usage:
    from guild.campaigns import CampaignManager, Campaign

    mgr = CampaignManager(base_dir)

    # List campaigns for a hero
    campaigns = mgr.list_campaigns("dio-qwen3-0.6b")

    # Get active campaign
    active = mgr.get_active()
    print(active.hero_id, active.id)  # "dio-qwen3-0.6b", "campaign-001"

    # Create new campaign
    campaign = mgr.create_campaign(
        hero_id="dio-qwen3-0.6b",
        name="Fresh Start",
        starting_checkpoint=None  # From base model
    )

    # Switch campaigns
    mgr.activate(campaign)

    # Archive old campaign
    mgr.archive("dio-qwen3-0.6b", "campaign-001")
"""

class CampaignManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.campaigns_dir = base_dir / "campaigns"
        self.control_dir = base_dir / "control"

    def get_active(self) -> Optional[Campaign]:
        """Get currently active campaign."""
        pointer = self.control_dir / "active_campaign.json"
        if not pointer.exists():
            return None
        with open(pointer) as f:
            data = json.load(f)
        return self.load_campaign(data["hero_id"], data["campaign_id"])

    def activate(self, campaign: Campaign) -> None:
        """Make a campaign the active one."""
        # Update pointer
        pointer = {
            "hero_id": campaign.hero_id,
            "campaign_id": campaign.id,
            "campaign_path": str(campaign.path.relative_to(self.base_dir)),
            "activated_at": datetime.now().isoformat()
        }
        with open(self.control_dir / "active_campaign.json", "w") as f:
            json.dump(pointer, f, indent=2)

        # Update symlink
        active_link = self.campaigns_dir / "active"
        if active_link.exists():
            active_link.unlink()
        active_link.symlink_to(campaign.path)

    def create_campaign(
        self,
        hero_id: str,
        name: str,
        starting_checkpoint: Optional[str] = None,
        config_overrides: Optional[Dict] = None
    ) -> Campaign:
        """Create a new campaign for a hero."""
        # Generate campaign ID
        existing = self.list_campaigns(hero_id)
        next_num = len(existing) + 1
        campaign_id = f"campaign-{next_num:03d}"

        # Create directory structure
        campaign_dir = self.campaigns_dir / hero_id / campaign_id
        (campaign_dir / "checkpoints").mkdir(parents=True)
        (campaign_dir / "status").mkdir()
        (campaign_dir / "logs").mkdir()

        # Initialize campaign.json
        metadata = {
            "id": campaign_id,
            "hero_id": hero_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "starting_checkpoint": starting_checkpoint,
            "starting_step": 0,
            "current_step": 0,
            "config_overrides": config_overrides or {}
        }
        with open(campaign_dir / "campaign.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Copy base checkpoint if specified
        if starting_checkpoint:
            # Copy checkpoint to campaign/checkpoints/
            pass

        return Campaign(campaign_dir, metadata)

    def archive(self, hero_id: str, campaign_id: str) -> None:
        """Move campaign to the Hall of Legends."""
        src = self.campaigns_dir / hero_id / campaign_id
        dst = self.campaigns_dir / "archive" / hero_id / f"{campaign_id}-archived"

        # Update metadata
        meta_path = src / "campaign.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = "archived"
        meta["archived_at"] = datetime.now().isoformat()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Move to archive
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
```

---

## PHASE 3: CONFIG LAYERING

### 3.1 Config Precedence

```
Final Config = merge(
    global_defaults,           # defaults.yaml (logging, paths, etc.)
    hero.training_defaults,    # From HeroProfile
    campaign.config_overrides, # Campaign-specific tweaks
    machine_overrides          # Local hardware (from hosts.json)
)
```

### 3.2 Config Builder

Create `core/config_builder.py`:

```python
"""
Config Builder - Merge layered configs for training.

Usage:
    from core.config_builder import build_training_config

    config = build_training_config(
        campaign=active_campaign,
        hero=hero_profile
    )
    # Returns fully merged TrainingConfig
"""

def build_training_config(campaign: Campaign, hero: HeroProfile) -> TrainingConfig:
    """Build final training config from layers."""

    # Start with global defaults
    config = load_defaults()

    # Apply hero training defaults
    config.update(hero.training_defaults)

    # Apply campaign overrides
    config.update(campaign.config_overrides)

    # Apply machine-specific (from hosts.json)
    machine = get_local_machine_config()
    config.update(machine)

    return TrainingConfig(**config)
```

---

## PHASE 4: MIGRATION STRATEGY

### 4.1 Migrate Current State

Script: `scripts/migrate_to_campaigns.py`

```python
"""
Migrate existing setup to campaign system.

This will:
1. Create campaigns/ directory structure
2. Move current_model/ -> campaigns/dio-qwen3-0.6b/campaign-001/checkpoints/
3. Move status/*.json -> campaigns/dio-qwen3-0.6b/campaign-001/status/
4. Create campaign.json with current stats
5. Create active_campaign.json pointer
6. Update symlinks for backward compatibility
"""

def migrate():
    # 1. Create hero profile (if not exists)
    create_hero_profile("dio-qwen3-0.6b", ...)

    # 2. Create campaign directory
    campaign_dir = campaigns_dir / "dio-qwen3-0.6b" / "campaign-001"

    # 3. Move checkpoints (preserve current_model symlink temporarily)
    # Option A: Move all checkpoints
    # Option B: Keep current_model as symlink to campaign checkpoints

    # 4. Move status files
    for status_file in ["training_status.json", "curriculum_state.json", ...]:
        shutil.move(status_dir / status_file, campaign_dir / "status" / status_file)

    # 5. Create legacy symlinks (backward compat)
    # current_model -> campaigns/active/checkpoints
    # status/training_status.json -> campaigns/active/status/training_status.json

    # 6. Activate campaign
    create_active_pointer(...)
```

### 4.2 Backward Compatibility

For a smooth transition:
- Keep `current_model` as symlink to `campaigns/active/checkpoints`
- Keep `status/training_status.json` as symlink to active campaign
- Update paths gradually in daemon and UI

---

## PHASE 5: TAVERN UI - CAMPAIGN WIZARD

### 5.1 New Page: `/campaign`

Campaign management screen with:

1. **Current Campaign Card**
   - Hero portrait, name, RPG title
   - Campaign name, started date
   - Current step, total XP
   - Skills being trained

2. **Campaign Actions**
   - "New Campaign" button
   - "Switch Campaign" dropdown
   - "Archive Campaign" button (with confirmation)

3. **New Campaign Wizard**
   - Step 1: Choose Hero (dropdown of HeroProfile)
   - Step 2: Choose Starting Point
     - Fresh Start (base model)
     - Continue from Checkpoint (select checkpoint)
   - Step 3: Campaign Name & Focus Skills
   - Step 4: Config Overrides (optional)
   - Step 5: Confirm & Create

4. **Hall of Legends**
   - Archived campaigns list
   - View-only metrics
   - "Restore" option (unarchive)

### 5.2 API Endpoints

```
GET  /api/campaigns                # List all campaigns
GET  /api/campaigns/active         # Get active campaign
POST /api/campaigns                # Create new campaign
PUT  /api/campaigns/activate       # Switch active campaign
POST /api/campaigns/archive        # Archive a campaign

GET  /api/heroes                   # List all heroes
GET  /api/heroes/{id}              # Get hero profile
```

---

## PHASE 6: SCOPED STATUS FILES

### 6.1 Files That Need Campaign Scoping

| File | Current Location | New Location |
|------|-----------------|--------------|
| training_status.json | status/ | campaigns/active/status/ |
| curriculum_state.json | status/ | campaigns/active/status/ |
| checkpoint_ledger.json | status/ | campaigns/active/status/ |
| task_state.json | status/ | status/ (global) |
| task_master.json | status/ | status/ (global) |

### 6.2 Update Code References

Files to update:
- `core/training_daemon.py` - Read/write status to campaign dir
- `core/checkpoint_ledger.py` - Ledger in campaign dir
- `tavern/server.py` - Read from active campaign
- `guild/skills/state_manager.py` - Progression in campaign

Pattern:
```python
# OLD
status_path = base_dir / "status" / "training_status.json"

# NEW
campaign = get_active_campaign()
status_path = campaign.path / "status" / "training_status.json"
```

---

## PHASE 7: CLEANUP & ARCHIVE

### 7.1 Archive System

The **Hall of Legends** (`campaigns/archive/`) stores:
- Full checkpoint directory (or just final checkpoint)
- All status files at archive time
- Final metrics summary

### 7.2 Cleanup Wizard

"Clean Up Old Data" button in Settings:
1. List archived campaigns with sizes
2. Options:
   - Delete checkpoints, keep metrics only
   - Full delete (with confirmation)
   - Export to NAS before delete

### 7.3 Retention Policy for Campaigns

```yaml
# configs/retention.yaml
campaigns:
  active:
    keep_checkpoints: 40           # Rolling window
    keep_all_ledger_entries: true
  archived:
    keep_best_checkpoint: true     # Always keep best
    keep_final_checkpoint: true
    delete_intermediate: true
    keep_metrics: forever          # Metrics are small
```

---

## IMPLEMENTATION ORDER

### Week 1: Foundation
1. [ ] Create `configs/heroes/` with template and DIO profile
2. [ ] Create `guild/heroes/` module (registry, types)
3. [ ] Create `campaigns/` directory structure
4. [ ] Write migration script (dry-run first!)

### Week 2: Campaign Manager
5. [ ] Create `guild/campaigns/` module
6. [ ] Implement campaign CRUD operations
7. [ ] Add active campaign pointer
8. [ ] Update training daemon to use campaign paths

### Week 3: Config Layering
9. [ ] Create `core/config_builder.py`
10. [ ] Update train.py to use built config
11. [ ] Test config inheritance

### Week 4: UI
12. [ ] Add `/campaign` page to Tavern
13. [ ] Create Campaign Wizard UI
14. [ ] Add API endpoints
15. [ ] Hall of Legends display

### Week 5: Polish
16. [ ] Archive system
17. [ ] Cleanup wizard
18. [ ] Documentation
19. [ ] Test switching campaigns

---

## QUESTIONS TO CLARIFY

1. **Checkpoint Storage**: Move all checkpoints to campaign dir, or keep in `current_model/` with symlinks?

2. **Status Files**: Move all status to campaign, or only campaign-specific ones?

3. **Queue System**: Should the queue be global or per-campaign?

4. **Inference**: When switching campaigns, should Oracle auto-switch to that campaign's best checkpoint?

5. **Data Files**: Keep training data (`queue/`, `inbox/`) global or per-campaign?

---

## SUMMARY

The Campaign System transforms TRAINING from a single-hero game to a save-file system:

```
BEFORE:
  One hero (DIO), one set of checkpoints, one status file
  RESET = scary, might lose everything

AFTER:
  Multiple heroes, multiple campaigns per hero
  RESET = "New Campaign" button, old data archived
  Compare heroes, switch freely, never lose progress
```

**Key Files to Create:**
- `configs/heroes/_template.yaml`
- `configs/heroes/dio-qwen3-0.6b.yaml`
- `guild/heroes/registry.py`
- `guild/campaigns/manager.py`
- `core/config_builder.py`
- `scripts/migrate_to_campaigns.py`
- `tavern/templates/campaign.html`

**Key Directories:**
- `campaigns/{hero_id}/{campaign_id}/`
- `campaigns/archive/`
- `configs/heroes/`

One click to start fresh. Never lose your old saves.
