# The Realm of Training

> **Contributing/Forking?** See [`SHARING.md`](SHARING.md) before pushing to public repos.

> *We propose a surjective morphism* **T: ML** *&rarr;* **RPG** *which maps the high-dimensional manifold of machine learning operations onto the more cognitively tractable space of role-playing game mechanics. This transformation preserves essential structure while minimizing the Kolmogorov complexity of the user's mental model. Empirically, we observe that framing gradient descent as "battle damage" and checkpoint management as "treasure vaults" yields a mass reduction in the probability of the developer staring blankly at terminal output wondering if anything is actually happening.*
>
> *&mdash; Definitely Not A Real Paper, 2025*

**Translation:** We turned ML training into an RPG because staring at loss curves is boring, but watching your hero level up is fun.

---

## The Core Thesis

**This repository is the method, not the results.**

If you clone this repo, you don't get a static paper or a pretrained model—you get the lab:
- The way campaigns are structured
- The way skills and curricula are defined
- The way evaluation, peaks, and "maxed out" behavior are tracked
- The way the human is pulled into forward momentum loops

The idea: if you copy the method faithfully and point it at similar data/models, you should be able to recreate the results I'd otherwise just write in a PDF.

**The main point is to have fun learning.**

---

## What Is This?

**Your hero DIO** (a Qwen3 model) battles through **quests** (training data), learning **skills** (SY, BIN), and growing stronger. You watch from the **Tavern** (http://localhost:8888) as DIO fights, levels up, and becomes a champion.

![The Tavern - Main Game UI](docs/images/tavern-screenshot.png)
*The Tavern: Watch your hero train, track skills, manage the vault, and monitor GPU status in real-time.*

### The RPG → ML Mapping

| RPG Term | ML Equivalent | Notes |
|----------|---------------|-------|
| **Hero** (DIO, FLO) | The model being trained | Different heroes = different architectures/sizes |
| **Campaign** | One training playthrough | Goal: discover this hero's level cap |
| **Quest** | Training data file | Dropped into `inbox/` |
| **Battle** | Training run | Gradient descent in action |
| **Damage Dealt** | 1 / Loss | Lower loss = more damage (Weber-Fechner) |
| **Level** | Training steps / 1000 | Visual progress marker |
| **Skill** | Curriculum domain (SY, BIN) | What the hero is learning |
| **Champion** | Best checkpoint by eval | Peak performance snapshot |
| **Tavern** | Web UI dashboard | http://localhost:8888 |
| **Vault** | Checkpoint storage | Managed by VaultKeeper |
| **Oracle** | Inference server | Chat with the model |
| **Temple** | Validation system | Blesses effort into experience |
| **Strain** | Loss - floor | How "stretched" the model is now |
| **Effort** | Cumulative strain | Total work done |

### Why 1/Loss → Damage?

The transformation exploits the **Weber-Fechner law of perception**:
- Users struggle to intuit the difference between loss 0.01 and 0.001
- But "100 Damage" vs "1,000 Damage" provides visceral feedback
- This keeps the human engaged during the "long tail" where gains appear marginal

---

## The Six Training Schools

How the Hero learns—each school has its own philosophy:

| School | Icon | Motto | Method | Status |
|--------|------|-------|--------|--------|
| **Scribe** | 📜 | "Copy the master's form until it becomes your own." | SFT | ✓ Implemented |
| **Mirror** | 🪞 | "See your flaws reflected, then correct them." | Sparring | ✓ Implemented |
| **Judge** | ⚖️ | "Between two paths, always choose the better." | DPO | Future |
| **Champion** | 🏆 | "Seek the reward, master the arena." | RLHF | Future |
| **Whisper** | 👻 | "The wisdom of giants flows to those who listen." | Distillation | Future |
| **Oracle** | 🔮 | "Focus where uncertainty dwells; ignore what is already known." | Fortune Teller | ✓ Enhancer |

The **Oracle** is special—it's an *enhancer* that works with any base school, weighting gradients by surprise.

---

## The Strain/Effort Metaphor

Training viewed through a **materials science** lens:

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Strain** | `loss - floor` | How "stretched" the model is right now |
| **Effort** | `Σ strain` | Total work done (area under strain curve) |
| **Strain Rate** | `Δstrain/Δt` | Learning velocity (negative = learning) |
| **Plastic Gain** | `L_before - L_after` | Permanent improvement |
| **Efficiency** | `plastic_gain / effort` | Learning ROI |

### Strain Zones (like heart rate zones)

| Zone | Strain Range | Meaning | Curriculum Action |
|------|-------------|---------|-------------------|
| **Recovery** | < 0.1 | Under-challenged | Level up |
| **Productive** | 0.1 - 0.3 | Optimal learning | Continue |
| **Stretch** | 0.3 - 0.5 | Challenging | Continue if improving |
| **Overload** | > 0.5 | Too hard | Back off |

This provides **automatic curriculum guidance**:
- High strain + not improving → Decrease difficulty
- Low strain + stable → Level up (too easy)
- Moderate strain + improving → Sweet spot (continue)

---

## The Campaign Model

A **Campaign** is a hero's journey to maximum potential—one attempt to push a model as far as it can go. The goal: discover the level cap.

**Different heroes have different potentials:**
- A 0.6B model might cap at skill level 20
- A 4B model might reach level 50
- We discover the cap by PLAYING (training)

**What "Maxed Out" means:**
A hero is effectively maxed when the experience required to gain a new skill level causes too much regression in previously mastered skills. The maintenance multiplier blows up—you spend more training just keeping old skills from decaying.

At that point:
1. Archive the journey
2. Keep the method
3. Start a new campaign with a different hero

---

## Humans as Neural Nets: The HITL Twist

There's another layer: **humans are neural nets too.**

We adapt, optimize, and "game" reward structures. If this system works, we should expect players to:
- Find shortcuts
- Discover emergent strategies
- And, crucially, **try to build skills that produce new skills**

That's not a bug; it's the point.

### The Meta-Skill: Skill Forge

The ultimate expression of HITL pressure is a skill that creates skills:

> "I want a skill that, given new information Y, can design a curriculum Z so the hero can handle more Y-like information in domain X efficiently in the future."

In other words, a **meta-skill**:
- **Input:** New data/domain Y
- **Output:** A proposed skill definition + curriculum + evaluation loop

So the hero can bootstrap competence in new areas without the human hand-crafting every step.

**This is the human-in-the-loop AGI twist:**
1. The human is naturally motivated to "game" the system
2. The system gives them a language for that: skills, quests, curricula, campaigns
3. The obvious high-level play becomes: **"Invent a skill that generates more skills"**

At that point, the project becomes a sandbox where humans and models co-evolve a pipeline that extends itself.

---

## Who This Is For

- **ML practitioners** who want a more engaging way to run training experiments
- **LocalLLM enthusiasts** with 16-24GB VRAM GPUs
- **Researchers** interested in curriculum learning and HITL systems
- **Anyone curious** about gamifying the training feedback loop

## What This Is Not

- **Not a one-click fine-tuning tool** — you'll configure campaigns, skills, and data pipelines
- **Not production-hardened** — this is research tooling, expect rough edges
- **Not a hosted service** — runs locally on your hardware
- **Not model weights** — bring your own base model (Qwen3-0.6B, Qwen3-4B, etc.)
- **Not finished research** — this is the lab, not the paper

---

## Current Status

**Work in Progress.** The system is functional but under active development.

See [`SHARING.md`](SHARING.md) for sharing guidelines and [`CHANGELOG.md`](CHANGELOG.md) for recent updates.

---

## The Hero Loop

Heroes are autonomous. They never sleep—always training or seeking new adventures.

```
┌─────────────────────────────────────────────────────────┐
│                     HERO LOOP                           │
│                                                         │
│    ┌──────────┐                                         │
│    │  Idle?   │◄────────────────────────────────┐       │
│    └────┬─────┘                                 │       │
│         │                                       │       │
│         ▼                                       │       │
│    ┌──────────────┐                             │       │
│    │ Check Queue  │                             │       │
│    └──────┬───────┘                             │       │
│           │                                     │       │
│           ▼                                     │       │
│    ┌──────────────┐                             │       │
│    │ Data exists? │                             │       │
│    └──────┬───────┘                             │       │
│           │                                     │       │
│     ┌─────┴─────┐                               │       │
│     │           │                               │       │
│     ▼           ▼                               │       │
│  ┌──────┐   ┌───────────────┐                   │       │
│  │  No  │   │      Yes      │                   │       │
│  └──┬───┘   └───────┬───────┘                   │       │
│     │               │                           │       │
│     ▼               ▼                           │       │
│  ┌──────────┐   ┌──────────┐                    │       │
│  │ Generate │   │  Train   │                    │       │
│  │  (skill  │   │  (gain   │                    │       │
│  │priorities│   │   XP)    │                    │       │
│  └────┬─────┘   └────┬─────┘                    │       │
│       │              │                          │       │
│       └──────────────┴──────────────────────────┘       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The hero's profile defines behavior:
- **skill_priorities**: Which skills to grind when idle (SY, BIN, etc.)
- **idle_generation**: How much training data to create
- **training_defaults**: Hyperparameters (batch size, learning rate, etc.)

**Note on Model Collapse:** Idle generation (training on self-generated data) risks "model collapse" if not carefully managed. The system uses:
- **Diversity filters** to maintain output variance
- **RAG injection** from the Vault to ground generations in external knowledge
- **Strain monitoring** to detect degradation

---

## Quick Start (Single Machine)

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **GPU** | 16GB VRAM | 24GB VRAM (RTX 3090/4090) |
| **RAM** | 16GB | 32GB+ |
| **Disk** | 50GB free | 100GB+ free |
| **OS** | Linux | Ubuntu 22.04+ |

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/definitelynotrussellkirk-bit/TRAINING.git
cd TRAINING

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[training,guild,dev]"

# 4. Run bootstrap (creates directories, checks config)
./scripts/bootstrap_dev.sh

# 5. Run diagnostics
python3 -m training doctor
```

### Start Playing

```bash
# Start all services (Tavern, VaultKeeper, daemon)
python3 -m training start-all

# Open the Tavern in your browser
xdg-open http://localhost:8888  # Linux
# or: open http://localhost:8888  # macOS
```

**That's it.** You should see the Tavern UI with your hero's stats.

### What You'll See

When you open http://localhost:8888:
- **Hero Card** - Your model (DIO/FLO) with level, XP, and stats
- **Training Status** - Current battle progress, loss, step count
- **Strain Zone** - Recovery/Productive/Stretch/Overload indicator
- **GPU Monitor** - VRAM usage, temperature
- **Navigation** - Battle, Guild, Quests, Vault, Oracle, Temple, Settings

### Quick Commands

```bash
python3 -m training doctor      # Health check
python3 -m training start-all   # Start all services
python3 -m training stop-all    # Stop all services
python3 -m training status      # Show current status
```

---

## Key URLs

| Location | URL | Purpose |
|----------|-----|---------|
| **Tavern** | http://localhost:8888 | Main game UI |
| **Quests** | http://localhost:8888/quests | Quest board |
| **Jobs** | http://localhost:8888/jobs | Distributed job queue |
| **Oracle** | http://localhost:8888/oracle | Chat with DIO |
| **Vault** | http://localhost:8888/vault | Checkpoint browser |
| **Guild** | http://localhost:8888/guild | Skill progression |
| **Temple** | http://localhost:8888/temple | Validation rituals |
| **Settings** | http://localhost:8888/settings | Config, VRAM calc |

---

## Technical Architecture

*(For those who prefer their manifolds un-surjected)*

### Core Services

| Service | Port | Purpose |
|---------|------|---------|
| **Tavern** | 8888 | Web UI |
| **VaultKeeper** | 8767 | Checkpoint/asset registry |
| **RealmState** | 8866 | Real-time state (SSE) |
| **Skill SY** | 8080 | Syllacrostic curriculum |
| **Skill BIN** | 8090 | Binary arithmetic curriculum |
| **Oracle** | 8765 | Inference (remote 3090) |

### Key Components

**Training Pipeline:**
- `core/train.py` - HuggingFace Trainer with custom features
- `arena/hero_loop.py` - Continuous training orchestrator
- `core/training_queue.py` - Priority-based queue system
- `temple/cleric.py` - Data validation

**Guild (Skills & Progression):**
- `guild/skills/` - Skill engine and curriculum
- `guild/metrics/strain.py` - Strain/Effort tracking
- `guild/campaigns/` - Campaign management

**Infrastructure:**
- `vault/keeper.py` - Checkpoint ledger
- `realm/server.py` - Real-time state service
- `core/groundskeeper.py` - Resource cleanup daemon

### Memory Optimization Options

For 24GB VRAM:

| Mode | VRAM | Config |
|------|------|--------|
| **QLoRA** | ~6GB | `training_mode: "qlora"`, `load_in_4bit: true` |
| **LoRA** | ~12GB | `training_mode: "lora"` |
| **GaLore 8-bit** | ~17GB | `optimizer.type: "galore_8bit"` |
| **DeepSpeed ZeRO-2** | ~17GB | `deepspeed_config: "configs/ds_zero2_offload.json"` |
| **Full Fine-tune** | ~22GB | `training_mode: "full"`, `optimizer.type: "adamw_8bit"` |

---

## The Temple & Blessing System

Training effort must be **blessed** to become experience:

```
Effort → Temple Rituals → Quality Factor → Experience
```

The **Nine Orders** validate different aspects:

| Order | Domain | Critical? |
|-------|--------|-----------|
| **Forge** | GPU/hardware | Yes |
| **Champion** | Model/checkpoint | Yes |
| **Oracle** | Inference server | Yes |
| Quick | Fast sanity checks | No |
| API | HTTP validation | No |
| Weaver | Daemons/processes | No |
| Guild | Skills/curriculum | No |
| Scribe | Evaluation/logging | No |
| Deep | Comprehensive/meta | No |

**Blessing Quality:**
- **Blessed** (≥0.8): Full experience awarded
- **Partial** (0.3-0.8): Reduced experience
- **Cursed** (0): No experience (bad data, failed evals)

This prevents the gamification of model degradation—you can't just grind on garbage data.

---

## Data Flow

1. Drop `.jsonl` training file into `inbox/`
2. **Cleric** validates format, tokens, quality
3. File moved to priority queue (`queue/high/`, `queue/normal/`, `queue/low/`)
4. Training processes one file at a time
5. Strain tracked, zones calculated
6. Checkpoints saved to `models/current_model/`
7. VaultKeeper registers to ledger
8. Temple blesses effort → experience

---

## Quest Modules & Primitives

### Quest Modules (Shareable Content)

Quest Modules are shareable packages that extend the hero's curriculum:

```
quests/modules/pattern-master/
├── manifest.yaml    # Metadata, dependencies, curriculum
├── data/            # Training JSONL files by level
├── eval/            # Evaluation sets
└── README.md
```

**The vision:** Like "mods" for training. Download, create, share.

### Unified Primitives

**Primitives** are atomic cognitive operations that underlie all skills:

| Category | Examples |
|----------|----------|
| **Sequence** | `seq_continue`, `seq_transform`, `seq_reverse` |
| **Logic** | `logic_deduce`, `logic_chain`, `logic_contrapose` |
| **Memory** | `mem_recall`, `mem_context`, `mem_compose` |
| **Format** | `fmt_json`, `fmt_code`, `fmt_table` |
| **Attention** | `attn_select`, `attn_count`, `attn_compare` |
| **Transform** | `xfm_encode`, `xfm_map`, `xfm_reduce` |

**The insight:** Skills are *composed* of primitives.
- Primitives = Base stats (STR, DEX, INT)
- Skills = Abilities that use combinations
- Transfer learning happens at the primitive level
- Measuring primitives reveals *why* a skill is weak

See `CLAUDE.md` for full Quest Module manifest format and primitive definitions

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [`LORE.md`](LORE.md) | Complete RPG vocabulary & metaphors |
| [`QUICKSTART.md`](QUICKSTART.md) | Full setup guide |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System deep dive |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Common issues |
| [`REMOTE_INFERENCE.md`](REMOTE_INFERENCE.md) | 3090 inference setup |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Contributing |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |
| [`CLAUDE.md`](CLAUDE.md) | Full game design document |

---

## The Canonical Story

> A **Hero** follows a **Path** through a **Domain**, under a chosen **Physics** and **Technique**.
> Each step produces **Strain**; accumulated **Effort**, when **Blessed** by the **Temple's**
> **Cleric** and the **Nine Orders**, becomes **Experience**, which is recorded in the **Ledger**.
>
> The Hero learns through **Training Schools**: the **Scribe** teaches imitation, the **Mirror**
> teaches self-correction, the **Oracle** focuses attention on uncertainty.
>
> Jobs are processed by Workers trained in **Job Schools**: **Inference** tests the Hero,
> the **Forge** crafts data, the **Vault** preserves checkpoints.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*"The system being taught here is the research method itself—how to think about model training as a game of campaigns, skills, and level caps."*
