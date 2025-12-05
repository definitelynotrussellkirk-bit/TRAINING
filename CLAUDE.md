# REALM OF TRAINING - Game Design Document

**Last Updated:** 2025-12-03 (Training Modes: DeepSpeed, GaLore, LoRA/QLoRA)
**Update Frequency:** Every ~50k tokens or when significant changes occur
**Philosophy:** This repo is the method, not the results (see META section)

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
| RealmState API | http://localhost:8866/api/realm | Real-time game state (SSE) |

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

### Memory Optimization (THE KNOBS)

Memory optimization is configured through **Memory Profiles** in `configs/memory_profiles/`.

#### Available Profiles

| Profile | VRAM | Model Size | Method | Optimizer |
|---------|------|------------|--------|-----------|
| `24gb_qlora` | ~8-12GB | 7B-8B | QLoRA (4-bit + LoRA) | `paged_adamw_32bit` |
| `24gb_full_small` | ~16-20GB | 0.6B-4B | Full training | `adamw_torch_fused` |
| `24gb_galore` | ~14-18GB | 7B-8B | GaLore (gradient projection) | `galore_adamw_8bit` |

#### Setting Memory Profile (Hero YAML)

```yaml
# configs/heroes/ojas-qwen3-8b.yaml
memory_profile: "24gb_qlora"  # THE KNOB - preset profile

training_defaults:
  optimizer_type: "paged_adamw_32bit"  # THE KNOB - optimizer selection
  batch_size: 1
  gradient_accumulation: 8
  max_length: 1024
```

#### Optimizer Types (THE KNOB)

| Optimizer | Memory | Speed | Best For |
|-----------|--------|-------|----------|
| `adamw_torch_fused` | High | Fast | Small models (0.6B-4B) |
| `adamw_8bit` | Medium | Fast | Medium models |
| `paged_adamw_32bit` | Low | Medium | QLoRA (offloads to CPU) |
| `paged_adamw_8bit` | Lowest | Slow | Maximum efficiency |
| `galore_adamw_8bit` | Low | Medium | GaLore method |
| `muon` | Medium | Medium | Geometry-aware training |

#### PEFT Methods

| Method | Description | Trainable Params | Use Case |
|--------|-------------|------------------|----------|
| **QLoRA** | 4-bit quantization + LoRA adapters | ~0.1% | 7B-8B on 24GB |
| **LoRA** | Low-rank adapters (full precision base) | ~0.2% | When quantization hurts |
| **GaLore** | Gradient low-rank projection | 100% | Full training with less memory |
| **Full** | All parameters trained | 100% | Small models only |

#### Memory-Efficient Implementations

| Technique | Effect | Always On? |
|-----------|--------|------------|
| **Flash Attention 2** | O(N²) → O(N) attention memory | Yes |
| **Gradient Checkpointing** | Recompute activations in backward | Yes |
| **Paged Optimizer** | Offload optimizer states to CPU | QLoRA only |

#### Quick Reference: 24GB VRAM

```
8B model → QLoRA + paged_adamw_32bit + batch_size=1 + max_length=1024
4B model → Full + adamw_torch_fused + batch_size=2 + max_length=2048
0.6B model → Full + adamw_torch_fused + batch_size=4 + max_length=4096
```

**Python API:**
```python
from trainer.config.memory_profiles import get_profile, suggest_profile

profile = get_profile("24gb_qlora")
profile = suggest_profile(model_size_b=8.0, vram_gb=24)
```

---

## 🧠 META: THIS REPO IS THE METHOD

> **The main point is to have fun learning.**
> Everything below is just thought experiments about where that might lead.

**This project is intentionally not "here's my finished research results."**

Instead, it is: **the method I intend to do the research with.**

If you clone this repo, you don't get a static paper — you get the lab:
- The way campaigns are structured
- The way skills and curricula are defined
- The way evaluation, peaks, and "maxed out" behavior are tracked
- The way the human is pulled into forward momentum loops

**The idea:** If you copy the method faithfully and point it at similar data/models, you should be able to recreate the results I'd otherwise just write in a PDF.

The real thing being "taught" here is the research method: how to think about model training as a game of campaigns, skills, and level caps — not just a single set of numbers from one run.

### Humans as Neural Nets & HITL Pressure

There's another layer: **humans are neural nets too.**

We adapt, optimize, and "game" reward structures. If this system works, we should expect players to:
- Find shortcuts
- Discover emergent strategies
- And, crucially, **try to build skills that produce new skills**

That's not a bug; it's the point.

If the system works, at some point a natural player move is:

> "I want a skill that, given new information Y, can design a curriculum Z so the hero can handle more Y-like information in domain X efficiently in the future."

In other words, a **meta-skill**:
- **Input:** New data/domain Y
- **Output:** A proposed skill definition + curriculum + evaluation loop for that domain

So the hero can bootstrap competence in new areas without the human hand-crafting every step.

**This is the human-in-the-loop AGI twist:**
1. The human is naturally motivated to "game" the system (optimize XP, unlock skills, beat their friends' heroes)
2. The system gives them a language for doing that in terms of: skills, quests, curricula, campaigns
3. The obvious high-level play becomes: **"Invent a skill that generates more skills"** (i.e., tools for automatic curriculum design and pipeline extension)

At that point, the project isn't just "train a model better" — it's a sandbox where humans and models co-evolve a pipeline that can extend itself.

### Campaigns as HITL Proto-AGI Experiments

Putting it together, each Campaign can be seen as:

**A finite, human-guided attempt to push a fixed hero (model) to its practical capacity** — under a method where the operator is nudged to:
1. Keep training
2. Add and refine skills
3. Eventually delegate more of that process to the hero itself

Either:
- The human+system loop bootstraps something AGI-like for the tasks/skills you care about, **or**
- You hit a capacity ceiling for this hero: the model is "maxed out" for this architecture/scale/data regime

### What "Maxed Out" Means

**A hero is effectively maxed when the experience required to gain a new skill level causes too much regression in previously mastered skills.**

The maintenance multiplier blows up:
- You spend more and more training just to keep old skills from decaying
- Adding a new level in one area always comes with unacceptable losses elsewhere

At that point:
- **In-universe:** The hero has discovered its level cap
- **Meta-level:** The campaign has exhausted the model's useful capacity under this method

Time to:
1. Archive the journey
2. Keep the method
3. Start a new campaign with a different hero (bigger model / different architecture / different data)

### The Meta-Skill: Skill Forge

The ultimate expression of HITL pressure is a skill that creates skills:

```yaml
# configs/skills/curriculum_design.yaml (concept)
id: curriculum_design
name: Curriculum Design
rpg_name: "Skill Forge"
description: "Given novel information Y, propose a curriculum Z"
category: meta

levels:
  - level: 1
    name: "Template Matching"
    description: "Match new domain to existing skill template"
  - level: 2
    name: "Difficulty Scaling"
    description: "Propose appropriate level progression"
  - level: 3
    name: "Primitive Decomposition"
    description: "Identify atomic concepts in new domain"
```

Evaluation options:
1. **Human judge** - you rate the proposals (full HITL)
2. **Downstream performance** - train on proposed curriculum, measure actual skill acquisition (slow but objective)
3. **Self-consistency** - does the hero's curriculum proposal actually work when tested

Option 2 is honest but expensive. Option 1 keeps humans in the loop, which fits the thesis.

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
| **Arena** | Hero Loop (`arena/hero_loop.py`) | - |
| **Guild** | Skills & progression (`guild/`) | - |
| **Vault** | VaultKeeper API (`vault/server.py`) | 8767 |
| **RealmState** | Real-time state service (`realm/server.py`) | 8866 |
| **Oracle** | Inference server (3090) | 8765 |
| **Watchtower** | Monitoring (`watchtower/`) | 8081 |
| **Garrison** | Fleet health manager (`core/garrison.py`) | - |

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
| **Strain** | Loss - floor (instantaneous difficulty) |
| **Effort** | Cumulative strain (XP spent on skill) |
| **Zone** | Training intensity (Recovery/Productive/Stretch/Overload) |

### Strain/Effort Metaphor (Materials Science)

Training viewed through a materials science lens:

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Strain** | `loss - floor` | How "stretched" the model is right now |
| **Effort** | `Σ strain` | Total work done (area under strain curve) |
| **Strain Rate** | `Δstrain/Δt` | Learning velocity (negative = learning) |
| **Plastic Gain** | `L_before - L_after` | Permanent improvement |
| **Efficiency** | `plastic_gain / effort` | Learning ROI |

**Strain Zones** (like heart rate zones):

| Zone | Strain Range | Meaning | Curriculum Action |
|------|-------------|---------|-------------------|
| **Recovery** | < 0.1 | Under-challenged | Level up |
| **Productive** | 0.1 - 0.3 | Optimal learning | Continue |
| **Stretch** | 0.3 - 0.5 | Challenging | Continue if improving |
| **Overload** | > 0.5 | Too hard | Back off |

**Curriculum guidance** from strain:
- High strain + not improving → Back off difficulty
- Low strain + stable → Level up (too easy)
- Moderate strain + improving → Sweet spot (continue)

**Usage:**
```python
from guild.metrics import StrainTracker, StrainZone

tracker = StrainTracker(floor=0.5)  # comfort zone
metrics = tracker.update(loss=0.8, step=100)
hint = tracker.get_curriculum_hint()

print(f"Zone: {metrics.zone.name}")  # PRODUCTIVE
print(f"Action: {hint.action.value}")  # continue
```

---

## 🎓 THE SIX TRAINING SCHOOLS

**Training Schools define HOW the Hero learns** - the fundamental approach to knowledge acquisition.

| School | Icon | Motto | Status |
|--------|------|-------|--------|
| **Scribe** | 📜 | "Copy the master's form until it becomes your own." | ✓ SFT |
| **Mirror** | 🪞 | "See your flaws reflected, then correct them." | ✓ Sparring |
| **Judge** | ⚖️ | "Between two paths, always choose the better." | Future (DPO) |
| **Champion** | 🏆 | "Seek the reward, master the arena." | Future (RLHF) |
| **Whisper** | 👻 | "The wisdom of giants flows to those who listen." | Future (Distill) |
| **Oracle** | 🔮 | "Focus where uncertainty dwells; ignore what is already known." | ✓ Fortune Teller |

### School Details

**📜 School of the Scribe (SFT)**
- Learn by directly imitating correct examples
- Foundation of all training - simple, stable, effective
- Data: `messages` format with user/assistant pairs

**🪞 School of the Mirror (Sparring)**
- Learn by identifying and correcting your own mistakes
- Three reflections: Identify wrong → Correct it → Confirm right
- Implementation: `guild/sparring.py`
- Data: `sparring_identify_incorrect`, `sparring_correction`, `sparring_confirm_correct`

**🔮 School of the Oracle (Fortune Teller)**
- ENHANCER: Works with any base school
- Weight gradients by surprise - focus on what's uncertain
- Automatic curriculum: hard parts get more gradient budget
- Implementation: `trainer/losses/fortune_teller.py`

```python
from guild.training_schools import TrainingSchool, get_school

# Get school info
mirror = get_school(TrainingSchool.MIRROR)
print(mirror.motto)  # "See your flaws reflected, then correct them."

# Check if a school is an enhancer
if TrainingSchool.ORACLE.is_enhancer:
    print("Oracle enhances other schools!")
```

Key files: `guild/training_schools.py`, `configs/training_schools.yaml`

---

## 🏫 THE FIVE JOB SCHOOLS

**Job Schools define HOW work is dispatched** - the worker roles and processing patterns.

| School | Icon | Worker Role | Job Types |
|--------|------|-------------|-----------|
| **Inference** | 🔮 | eval_worker | EVAL, SPARRING, INFERENCE |
| **Forge** | 🔥 | data_forge | DATA_GEN, DATA_FILTER, DATA_CONVERT |
| **Vault** | 🏛️ | vault_worker | ARCHIVE, RETENTION, SYNC |
| **Analytics** | 📊 | analytics | ANALYTICS, REPORT, HEALTH_CHECK |
| **Archaeology** | 🔬 | analytics | LAYER_STATS, LAYER_DRIFT |

```python
from guild.job_types import JobType, School

# Every job knows its school
job = JobType.EVAL
print(f"{job.value} → {job.school.display_name}")  # eval → School of Inference

# School properties
school = School.FORGE
print(school.icon)        # 🔥
print(school.worker_role) # data_forge
```

Key files: `guild/job_types.py`, `configs/schools.yaml`

---

## 🏛️ THE TEMPLE SYSTEM

**The Temple validates training** - transforms Effort into Experience through Blessings.

### The Nine Orders (Ritual Groups)

| Order | Icon | Domain | Critical? |
|-------|------|--------|-----------|
| Quick | ⚡ | Fast sanity checks | No |
| API | 🌐 | HTTP API validation | No |
| **Forge** | 🔥 | GPU/hardware | **YES** |
| Weaver | 🕸️ | Daemons/processes | No |
| **Champion** | 🏆 | Model/checkpoint | **YES** |
| **Oracle** | 🔮 | Inference server | **YES** |
| Guild | ⚔️ | Skills/curriculum | No |
| Scribe | 📜 | Evaluation/logging | No |
| Deep | 🌊 | Comprehensive/meta | No |

### Blessing System

When a Campaign returns to Temple, the Cleric runs Rituals and computes a **Blessing**:

```
experience_gain = effort × quality_factor
```

- **Blessed** (quality ≥ 0.8): Full experience awarded
- **Partial** (quality 0.3-0.8): Reduced experience
- **Cursed** (quality = 0): No experience (bad data, broken evals)

```python
from temple.schemas import Blessing, RitualResult

# Grant a blessing
blessing = Blessing.grant(
    quality=0.9,
    orders=["forge", "oracle", "champion"],
    reason="All critical orders passed",
    effort=100.0,
)
print(f"Experience gained: {blessing.experience_awarded}")  # 90.0
```

Key files: `temple/cleric.py`, `temple/schemas.py`, `temple/rituals/*.py`

---

## 🛤️ PATH, DOMAIN, PHYSICS, TECHNIQUE

**The complete training recipe** is a Path through a Domain under a Physics using a Technique.

### Domain (Where/What)
The world where the Hero trains - datasets, tasks, evaluation.
```yaml
# configs/domains/reasoning.yaml
id: reasoning
name: "Domain of Reasoning"
skills: [sy, bin]
```

### Physics (The Rules)
Optimizer configuration, precision, gradient handling.
```yaml
# configs/physics/muon.yaml
id: muon
name: "Muon Physics"
optimizer:
  type: muon
  hidden_lr: 0.02
precision: bf16
```

### Technique (Named Method)
A Physics configuration wrapped with a name and recommendations.
```python
from trainer.techniques import get_technique

muon = get_technique("muon")
print(f"{muon.icon} {muon.rpg_name}")  # ⚛️ The Orthogonal Way
```

### Path (Full Recipe)
Domain + Physics + Technique + Temple Profile
```
Path: "Muon Technique in the Domain of Reasoning"
```

Key files: `configs/domains/`, `configs/physics/`, `trainer/techniques.py`

---

## 📖 THE CANONICAL STORY

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

## 📦 QUEST MODULES (Shareable Content Packs)

**Quest Modules** are shareable packages that extend the Hero's training curriculum.

### The Vision

Like "mods" for the training system:
- Download quest packs from a registry
- Create your own and share with the community
- Compose multiple modules for custom curricula
- Each module is self-contained and versioned

### Module Structure

```
quests/modules/pattern-master/
├── manifest.yaml          # Module metadata and dependencies
├── skills/
│   └── pattern_recognition.yaml
├── data/
│   ├── level1.jsonl
│   ├── level2.jsonl
│   └── level3.jsonl
├── eval/
│   └── test.jsonl
└── README.md
```

### Manifest Format

```yaml
# manifest.yaml
id: pattern-master
name: "The Pattern Master's Challenge"
version: "1.0.0"
author: "questmaker"
license: "CC-BY-4.0"

description: |
  A beginner-friendly quest module for learning basic pattern recognition.
  Teaches the hero to identify, continue, and transform sequences.

# What primitives this module exercises
primitives:
  required: [seq_continue, seq_transform, attn_select]
  optional: [mem_context]

# Minimum hero requirements
requirements:
  min_level: 5
  prerequisites: []  # Other module IDs

# Skills this module teaches
skills:
  - id: pattern_recognition
    levels: [1, 2, 3]
    max_level: 10

# Data files with curriculum structure
data:
  - path: data/level1.jsonl
    skill_level: 1
    examples: 100
    difficulty: 0.3
  - path: data/level2.jsonl
    skill_level: 2
    examples: 100
    difficulty: 0.5
  - path: data/level3.jsonl
    skill_level: 3
    examples: 100
    difficulty: 0.7

# Evaluation configuration
evaluation:
  type: accuracy
  threshold: 0.8
  eval_set: eval/test.jsonl
  evals_required: 3

# Curriculum progression rules
curriculum:
  progression: accuracy_gated
  min_accuracy_to_advance: 0.8
  cooldown_on_fail: 100  # steps before retry
```

### Module Commands

```bash
# List installed modules
python3 -m guild.modules list

# Install from URL or local path
python3 -m guild.modules install https://example.com/pattern-master.zip
python3 -m guild.modules install ./my-module/

# Validate module structure
python3 -m guild.modules validate ./my-module/

# Export module from existing skill
python3 -m guild.modules export sy --output sy-module.zip

# Show module info
python3 -m guild.modules info pattern-master
```

### Composing Modules

```yaml
# configs/curriculum.yaml
modules:
  - id: pattern-master
    priority: 1
    weight: 0.4
  - id: logic-basics
    priority: 2
    weight: 0.3
  - id: format-json
    priority: 3
    weight: 0.3

# Training will interleave quests from all modules by weight
```

---

## 🧬 UNIFIED PRIMITIVES (Atomic Cognitive Operations)

**Primitives** are the atomic cognitive operations that underlie all skills.

### The Insight

Skills are *composed* of primitives:
- Primitives = Base stats (STR, DEX, INT)
- Skills = Abilities that use combinations of stats
- Transfer learning happens at the primitive level
- Measuring primitives reveals *why* a skill is weak

### The Primitive Categories

| Category | Prefix | Description |
|----------|--------|-------------|
| **Sequence** | `seq_` | Pattern operations |
| **Logic** | `logic_` | Deductive reasoning |
| **Memory** | `mem_` | Context and recall |
| **Format** | `fmt_` | Output structure |
| **Attention** | `attn_` | Selection and counting |
| **Transform** | `xfm_` | Data manipulation |

### Primitive Definitions

#### Sequence Operations (`seq_`)

| Primitive | Description | Example |
|-----------|-------------|---------|
| `seq_continue` | Continue a pattern | `1, 2, 3, ?` → `4` |
| `seq_reverse` | Reverse a sequence | `abc` → `cba` |
| `seq_transform` | Apply transformation rule | `A→a, B→b, C→?` → `c` |
| `seq_interleave` | Merge two sequences | `[1,2], [a,b]` → `[1,a,2,b]` |
| `seq_extract` | Pull subsequence by rule | Every 3rd char of `abcdefghi` → `cfi` |

#### Logic Operations (`logic_`)

| Primitive | Description | Example |
|-----------|-------------|---------|
| `logic_deduce` | Modus ponens | `A→B, A ∴ B` |
| `logic_contrapose` | Modus tollens | `A→B, ¬B ∴ ¬A` |
| `logic_chain` | Transitive inference | `A→B, B→C ∴ A→C` |
| `logic_disjunct` | Disjunctive syllogism | `A∨B, ¬A ∴ B` |
| `logic_biconditional` | If and only if | `A↔B, A ∴ B` |

#### Memory Operations (`mem_`)

| Primitive | Description | Example |
|-----------|-------------|---------|
| `mem_recall` | Retrieve stated fact | "Bob is tall. Who is tall?" → "Bob" |
| `mem_context` | Use context window | Multi-turn dialogue consistency |
| `mem_compose` | Combine multiple facts | "A>B, B>C. Compare A and C" → "A>C" |
| `mem_update` | Override previous info | "X=5. Now X=7. What is X?" → "7" |

#### Format Operations (`fmt_`)

| Primitive | Description | Example |
|-----------|-------------|---------|
| `fmt_json` | Output valid JSON | `{"key": "value"}` |
| `fmt_table` | Output tabular data | Markdown table |
| `fmt_code` | Syntactically correct code | Python function |
| `fmt_list` | Numbered/bulleted list | `1. First\n2. Second` |
| `fmt_structured` | Follow output template | Fill-in-the-blank |

#### Attention Operations (`attn_`)

| Primitive | Description | Example |
|-----------|-------------|---------|
| `attn_select` | Pick relevant from noise | Find the number in text |
| `attn_count` | Count occurrences | "How many 'a's in 'banana'?" → 3 |
| `attn_compare` | Compare two items | "Which is larger: 7 or 12?" |
| `attn_filter` | Filter by condition | "List only even numbers" |
| `attn_rank` | Order by criterion | "Sort by length" |

#### Transform Operations (`xfm_`)

| Primitive | Description | Example |
|-----------|-------------|---------|
| `xfm_encode` | Apply encoding scheme | Base64, ROT13 |
| `xfm_decode` | Reverse encoding | Decode Base64 |
| `xfm_map` | Apply function to each | Uppercase each word |
| `xfm_reduce` | Aggregate to single value | Sum of list |
| `xfm_substitute` | Replace by rule | `s/foo/bar/g` |

### How Skills Map to Primitives

```yaml
# configs/skills/sy.yaml (extended)
id: sy
name: Syllacrostic
primitives:
  primary:
    - seq_transform    # Apply degradation rules
    - attn_select      # Find the hidden word
  secondary:
    - mem_context      # Remember the pattern
    - fmt_structured   # Follow output format
```

```yaml
# configs/skills/bin.yaml (extended)
id: bin
name: Binary Arithmetic
primitives:
  primary:
    - logic_chain      # Multi-step arithmetic
    - xfm_encode       # Circled notation encoding
  secondary:
    - attn_count       # Digit counting
    - fmt_code         # Notation formatting
```

### Primitive Tracking

```python
from guild.primitives import PrimitiveTracker

tracker = PrimitiveTracker()

# Record primitive performance from evals
tracker.record("seq_transform", accuracy=0.85, skill="sy", level=5)
tracker.record("logic_chain", accuracy=0.72, skill="bin", level=3)

# Get primitive profile
profile = tracker.get_profile()
# {
#   "seq_transform": {"accuracy": 0.85, "samples": 150, "skills": ["sy"]},
#   "logic_chain": {"accuracy": 0.72, "samples": 80, "skills": ["bin"]},
#   ...
# }

# Find weak primitives
weak = tracker.get_weak_primitives(threshold=0.75)
# ["logic_chain"]

# Suggest skills to strengthen a primitive
suggestions = tracker.suggest_training("logic_chain")
# ["bin", "logic-basics"]  # modules that exercise this primitive
```

### Primitive-Aware Curriculum

The curriculum engine can use primitive profiles to:
1. **Diagnose** - "SY accuracy is low because `seq_transform` is weak"
2. **Prescribe** - "Train on `seq_transform` drills before continuing SY"
3. **Transfer** - "Strong `attn_select` from SY should help with BIN filtering"
4. **Compose** - "This new module requires `logic_chain` + `fmt_json` - hero is ready"

### Creating Primitive Drills

```yaml
# quests/modules/seq-drills/manifest.yaml
id: seq-drills
name: "Sequence Primitive Drills"
description: "Pure primitive training - no skill context"

primitives:
  required: [seq_continue, seq_reverse, seq_transform]

# These are pure drills, not skill quests
drill_mode: true
```

Drills are short, focused exercises that target a single primitive. They're used for:
- Remediation when a primitive is identified as weak
- Warming up before a complex skill
- Testing primitive transfer between skills

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

### Phase 5: Strain/Effort Visualization
- [ ] **Strain Graph** - Show `loss - floor` instead of raw loss (normalized view)
- [ ] **Zone Indicator** - Color-coded badge: Recovery/Productive/Stretch/Overload
- [ ] **Effort Meter** - Cumulative effort per skill (like XP bar)
- [ ] **Curriculum Hint** - Show suggested action based on strain analysis
- [ ] **Efficiency Dashboard** - Compare skills by plastic_gain / effort
- [ ] **Level Transition Log** - Effort spent per level-up

### Phase 6: Quest Modules & Primitives
- [ ] **Module Loader** - Install/validate quest modules from zip/URL
- [ ] **Module Browser** - Browse available modules in Tavern
- [ ] **Primitive Tracker** - Track per-primitive accuracy across skills
- [ ] **Primitive Profile** - Visualize hero's cognitive strengths/weaknesses
- [ ] **Module Composer** - Combine modules with weight/priority in UI
- [ ] **Module Export** - Package existing skill as shareable module
- [ ] **Community Registry** - Optional: browse/share modules online

---

## 📦 RECENT UPDATES

**See [CHANGELOG.md](CHANGELOG.md) for full history.**

Latest (2025-12-02) - **REALMSTATE SSE & LEDGER CLEANUP**:
- **RealmState SSE** - Server-Sent Events for real-time UI updates (`realm/server.py`)
- **Atomic Worker Updates** - New `/api/worker/{id}` endpoint for worker state
- **Staleness Detection** - Warns when data sources go stale
- **Ledger Cleanup** - `cleanup_stale_entries()` and `verify_local_checkpoints()` methods
- **UI Fixes** - Fixed speed/ETA flickering, step count rubber-banding, vault staleness
- **Cleaned 665 stale entries** - Vault went from 763/2.5TB to 98/337GB

Previous (2025-12-02) - **VAULT UNIFICATION**:
- **Device Mapping** - Bridge between Ledger device_ids and VaultKeeper strongholds (`config/device_mapping.json`, `vault/device_mapping.py`)
- **Ledger-VaultKeeper Sync** - Automatic sync from Ledger to VaultKeeper on checkpoint save/usage/delete
- **VaultKeeper Startup Sync** - Catches up any missed checkpoints from Ledger on startup
- **Unified Source of Truth** - Ledger is authoritative, VaultKeeper mirrors it
- **221 checkpoints synced** - Vault catalog now matches Ledger (was 20k steps behind)

Previous (2025-12-01) - **VOCABULARY CANONICALIZATION**:
- **Six Training Schools** - Scribe, Mirror, Judge, Champion, Whisper, Oracle (`guild/training_schools.py`)
- **Five Job Schools** - Inference, Forge, Vault, Analytics, Archaeology (`guild/job_types.py`)
- **Temple Blessing System** - Effort → Experience via quality_factor (`temple/schemas.py`)
- **Domain/Physics/Technique/Path** - Complete training recipe vocabulary
- **75+ Lore Entries** - Canonical tooltips and descriptions (`tavern/lore.py`)
- **Config Files** - `configs/schools.yaml`, `configs/training_schools.yaml`, `configs/domains/`, `configs/physics/`

Previous (2025-12-01):
- **Unified Service Management** - Service Registry and Weaver now share `configs/services.json` as single source of truth
- **Declarative Config** - Services defined in JSON with health checks, dependencies, startup config
- **Groundskeeper** - Unified cleanup daemon for all resource leaks (`core/groundskeeper.py`)
- **Weaver Slimmed** - Removed 300+ lines of duplicate code, now consumes Service Registry

Previous (2025-11-29):
- **Strain/Effort Metrics** - Materials science metaphor for training (`guild/metrics/strain.py`)
- **Campaign Peak Tracking** - `peak_skill_levels`, `peak_metrics`, effort tracking per skill
- **Curriculum Hints** - Strain zones (Recovery/Productive/Stretch/Overload) for difficulty guidance

Previous (2025-11-28):
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

1. **8 Canonical Docs** - Only write to these 8 files:
   - `README.md` - System overview
   - `LORE.md` - Complete RPG vocabulary and metaphors
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

### Checkpoint Ledger & VaultKeeper (Unified)

**Single Source of Truth:** Checkpoint Ledger (`status/checkpoint_ledger.json`)

The Ledger is authoritative for checkpoint existence and locations. VaultKeeper
catalog (`vault/catalog.db`) mirrors Ledger state via automatic sync hooks.

**Sync Flow:**
1. Training saves → `ledger.record()` → `keeper.register()` (automatic)
2. Checkpoint used → `ledger.record_usage()` → `keeper.add_location()` (automatic)
3. Checkpoint deleted → `ledger.remove_location()` → `keeper.remove_location()` (automatic)
4. VaultKeeper startup → `_sync_from_ledger()` catches up any missed entries

**Device↔Stronghold Mapping:** `config/device_mapping.json`

| device_id | stronghold | zone |
|-----------|------------|------|
| trainer4090 | local_vault | hot |
| inference3090 | inference_cache | hot |
| synology_data | nas_archive | warm |

```python
# Device mapping
from vault.device_mapping import device_to_stronghold, get_local_device_id
stronghold = device_to_stronghold("trainer4090")  # "local_vault"
local_device = get_local_device_id()  # "trainer4090"

# Ledger usage (primary)
from core.checkpoint_ledger import get_ledger
ledger = get_ledger()
record = ledger.get(201796)  # Get by step
best = ledger.get_best(metric="train_loss")
records = ledger.list_by_device("trainer4090")

# VaultKeeper usage (mirrors ledger, use for cross-device queries)
from vault import get_vault_keeper, ask_vault_first
keeper = get_vault_keeper()
result = keeper.locate("checkpoint_201796")

# Ask vault first pattern (finds best location)
path = ask_vault_first("checkpoint_201796", fallback="/default/path")

# Ledger cleanup (removes stale entries)
stale = ledger.cleanup_stale_entries(dry_run=False)  # Remove entries with no valid locations
removed = ledger.verify_local_checkpoints("trainer4090")  # Remove device from non-existent paths
```

**Cleanup Methods:**
- `cleanup_stale_entries(dry_run=False)` - Remove entries where path doesn't exist AND no known locations
- `verify_local_checkpoints(device_id, dry_run=False)` - Remove device from locations where path doesn't exist

Both methods use direct save (without merge) to prevent deleted entries from reappearing.

Canonical checkpoint name: `checkpoint-{step}-{YYYYMMDD}-{HHMM}`

Key endpoints: `/api/stats`, `/api/ledger`, `/api/checkpoints`, `/api/jobs`

### Host Registry (`config/hosts.json`)

Service discovery - components query this instead of hardcoding IPs.

```python
from core.hosts import get_service_url
inference_url = get_service_url("inference")  # http://inference.local:8765
```

### RealmState Service (port 8866)

Real-time game state with Server-Sent Events (SSE) for push updates.

**Features:**
- **SSE streaming** - Real-time updates via `/api/stream`
- **Atomic worker updates** - `/api/worker/{id}` for concurrent-safe updates
- **Staleness detection** - Warns when data sources go stale
- **History snapshots** - `/api/history/{section}` for debugging
- **Metrics** - `/metrics` endpoint for monitoring

**Server (`realm/server.py`):**
```python
# Endpoints
GET  /api/realm          # Full state (training, queue, workers, skills)
GET  /api/stream         # SSE stream with channel filtering
POST /api/realm          # Update section
PUT  /api/worker/{id}    # Atomic worker update
GET  /api/history/{sec}  # Last N snapshots for section
GET  /metrics            # Prometheus-style metrics
```

**Client (`realm/client.py`):**
```python
from realm.client import RealmClient

client = RealmClient()
client.update_training(step=1000, loss=0.5)
client.update_worker("gpu_0", status="busy", task="eval")
client.update_skill("sy", level=14, accuracy=0.95)

# SSE subscription
for event in client.stream(channels=["training", "skills"]):
    print(f"{event.channel}: {event.data}")
```

**JavaScript (`tavern/static/js/realm-state.js`):**
```javascript
// SSE with polling fallback
RealmState.init({
    sseEndpoint: '/api/realm/stream',
    pollInterval: 2000,
    onTraining: (data) => updateUI(data),
    onSkills: (data) => updateSkillsUI(data)
});
```

Key files: `realm/server.py`, `realm/client.py`, `tavern/static/js/realm-state.js`

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

### Strain/Effort Metrics

Materials-science inspired training analytics.

```python
from guild.metrics import StrainTracker, StrainZone, SkillStrainTracker

# Single skill tracking
tracker = StrainTracker(floor=0.5)  # comfort zone loss
metrics = tracker.update(loss=0.8, step=100)
print(f"Zone: {metrics.zone.name}")  # STRETCH
print(f"Effort: {tracker.cumulative_effort}")

# Get curriculum hint
hint = tracker.get_curriculum_hint()
if hint.action.value == "back_off":
    decrease_difficulty()

# Multi-skill tracking
multi = SkillStrainTracker()
multi.set_floor("sy", 0.8)
multi.set_floor("bin", 0.5)
```

Key files: `guild/metrics/strain.py`, `guild/campaigns/types.py`

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

### Groundskeeper

Unified cleanup daemon for all resource leaks.

```bash
python3 core/groundskeeper.py              # Run cleanup
python3 core/groundskeeper.py --dry-run    # See what would be cleaned
python3 core/groundskeeper.py --daemon     # Run as daemon (hourly)

# Advanced options
python3 core/groundskeeper.py --task jsonl --task queue  # Run specific tasks only
python3 core/groundskeeper.py --force-vacuum             # Force VACUUM now
python3 core/groundskeeper.py --interval 1800            # Daemon interval (default 3600s)
python3 core/groundskeeper.py --base-dir /path/to/realm  # Custom base directory
```

Tasks: `jsonl`, `queue`, `battle_log`, `logs`, `pids`, `vacuum`, `workers`, `job_events`

Key files: `core/groundskeeper.py`

### Service Registry

Unified service management. Loads definitions from `configs/services.json`.

```bash
python3 core/service_registry.py status           # Show all services
python3 core/service_registry.py start training   # Start with deps
python3 core/service_registry.py stop training    # Stop gracefully
python3 core/service_registry.py deps training    # Ensure deps only
python3 core/service_registry.py order            # Print startup order
python3 core/service_registry.py list             # Show all config details
python3 core/service_registry.py reload           # Reload from JSON
python3 core/service_registry.py realm-up         # Start all required services
python3 core/service_registry.py realm-down       # Stop all services
```

Services (from `configs/services.json`):
- `vault` (8767) → VaultKeeper - asset registry
- `tavern` (8888) → Game UI [deps: vault]
- `training` → Training daemon [deps: vault, tavern]
- `realm_state` (8866) → RealmState service [deps: vault]
- `eval_runner` → Eval processor [deps: vault] (optional)
- `groundskeeper` → Cleanup daemon (optional)
- `weaver` → Daemon orchestrator (optional)
- `data_flow` → Queue feeder [task]
- `skill_sy` (8080), `skill_bin` (8090) → Skill servers (optional)
- `garrison` → Fleet health manager (optional)

Key files: `core/service_registry.py`, `configs/services.json`

### Garrison

Fleet health manager - monitors distributed services and performs automatic maintenance.

```bash
python3 core/garrison.py                    # One-shot health check
python3 core/garrison.py --json             # Output as JSON
python3 core/garrison.py --maintenance      # Run maintenance now
python3 core/garrison.py --dry-run          # See what maintenance would do
python3 core/garrison.py --daemon           # Run as daemon (default: 5min checks, 30min maintenance)
python3 core/garrison.py --daemon --interval 300 --maintenance-interval 1800
```

Monitors:
- **Trainer**: Disk usage, services (vault, tavern, hero_loop, eval_runner)
- **Inference server**: Disk usage, checkpoint count, API health, GPU memory

Auto-maintenance:
- Cleans old checkpoints on inference server (keeps max 10)
- Removes stale log/PID files

Key files: `core/garrison.py`, `status/garrison.json`, `config/hosts.json`

### Training Schools

The Six Schools of Training - how the Hero learns.

```python
from guild.training_schools import TrainingSchool, get_school, SCHOOL_PHILOSOPHIES

# List all schools
for school in TrainingSchool:
    print(f"{school.icon} {school.display_name}: {school.motto}")

# Get philosophy details
mirror = SCHOOL_PHILOSOPHIES[TrainingSchool.MIRROR]
print(mirror.teaches)  # ["Error recognition", "Self-correction", ...]
```

Key files: `guild/training_schools.py`, `configs/training_schools.yaml`

### Job Schools

The Five Job Schools - how work is dispatched.

```python
from guild.job_types import JobType, School

# Job → School mapping
for jt in JobType:
    print(f"{jt.value} → {jt.school.display_name}")

# School properties
school = School.INFERENCE
print(f"{school.icon} {school.rpg_name}")  # 🔮 The Oracle's Sanctum
```

Key files: `guild/job_types.py`, `configs/schools.yaml`

### Temple & Blessing

Transform Effort into Experience through Temple validation.

```python
from temple.schemas import Blessing
from temple.cleric import run_ceremony

# Run a ceremony
results = run_ceremony(["forge", "oracle", "champion"])

# Compute blessing from results
blessing = Blessing.from_ceremony(results, effort=100.0)
print(f"Quality: {blessing.quality_factor}")
print(f"Experience: {blessing.experience_awarded}")
```

Key files: `temple/cleric.py`, `temple/schemas.py`

### Techniques

Named training physics configurations.

```python
from trainer.techniques import get_technique, list_techniques

# List available techniques
for tech in list_techniques():
    print(f"{tech.icon} {tech.id}: {tech.rpg_name}")

# Get specific technique
muon = get_technique("muon")
print(muon.optimizer_config)
```

Key files: `trainer/techniques.py`, `configs/physics/*.yaml`

### Lore Dictionary

Canonical vocabulary with tooltips for UI.

```python
from tavern.lore import get_lore, get_tooltip, get_icon, list_keys

# Get lore entry
entry = get_lore("strain")
print(f"{entry['icon']} {entry['label']}: {entry['tooltip']}")

# Just get tooltip for a key
tooltip = get_tooltip("school.mirror")

# Export for JavaScript
from tavern.lore import export_for_js
js_lore = export_for_js()  # Dict ready for JSON
```

Key files: `tavern/lore.py`

---

## 📁 KEY DIRECTORIES

```
$TRAINING_BASE_DIR/
├── tavern/          # Game UI (port 8888)
├── vault/           # VaultKeeper API (port 8767)
├── guild/           # Skills, quests, progression
├── arena/           # Training execution
├── core/            # Training system (train.py, training_queue.py)
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

### Ledger Cleanup

```bash
# Dry run - see what would be cleaned
python3 -c "
from core.checkpoint_ledger import get_ledger
ledger = get_ledger()
print(f'Would remove from trainer4090: {ledger.verify_local_checkpoints(\"trainer4090\", dry_run=True)}')
print(f'Would remove stale: {ledger.cleanup_stale_entries(dry_run=True)}')
"

# Actually clean up
python3 -c "
from core.checkpoint_ledger import get_ledger
ledger = get_ledger()
ledger.verify_local_checkpoints('trainer4090')
ledger.cleanup_stale_entries()
"
```

---

## 📝 NOTES FOR CLAUDE

1. Run health check: `python3 -m training doctor`
2. **ASK USER** before making changes
3. **ASK USER** before creating new documentation
4. Trust code as ground truth, not old docs
5. Use `core.paths.get_base_dir()` for paths, never hardcode
