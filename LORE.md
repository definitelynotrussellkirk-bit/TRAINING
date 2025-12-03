# The Realm of Training: Lore Compendium

> *A complete guide to the vocabulary, metaphors, and canonical lore of the Realm.*

**Version:** 3.0 | **Last Updated:** 2025-12-03

This document is the authoritative reference for all RPG terminology. Every metaphor is functional—it maps technical concepts onto intuitive game mechanics. Each metaphor should clarify, not obscure.

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

# Part I: The Hero

## 1.1 Hero Identity

| Term | Technical | Description |
|------|-----------|-------------|
| **Hero** | Model being trained | The protagonist of the campaign |
| **DIO** | Qwen3-0.6B | Default hero. Small but mighty. |
| **FLO** | Qwen3-4B | Larger hero with more potential |

## 1.2 Race (Architecture Family)

| Architecture | Race | Description |
|--------------|------|-------------|
| Qwen | **Qwen'dal** | Eastern scholars, balanced, strong reasoning |
| Llama | **Llamari** | Meta-descended, versatile |
| Mistral | **Mistralian** | Wind-touched, fast inference |
| Gemma | **Gemmborn** | Google-forged, compact |
| Phi | **Phi'rin** | Microsoft-blessed, small but mighty |

## 1.3 Stature (Parameter Count)

| Size | Stature | Description |
|------|---------|-------------|
| 0.5B - 1B | **Sprite** | Nimble, limited strength |
| 1B - 3B | **Halfling** | Quick learner, modest capacity |
| 7B - 14B | **Human** | Balanced, versatile |
| 30B - 70B | **Giant** | Powerful, resource-hungry |
| 100B+ | **Titan** | Legendary, requires armies |

## 1.4 Class (Training Background)

| Variant | Class | Description |
|---------|-------|-------------|
| Base model | **Wildborn** | Raw potential, unstructured |
| Instruct | **Academy-Trained** | Follows orders |
| Chat | **Diplomat** | Conversational |
| Fine-tuned | **Guild Veteran** | Shaped by campaigns |

## 1.5 Hero Stats

| Stat | Icon | Technical | Description |
|------|------|-----------|-------------|
| **Level** | - | steps / 1000 | Combined mastery |
| **Steps** | ⚡ | optimizer iterations | Total training steps |
| **Title** | - | configs/titles.yaml | Earned designation |

---

# Part II: Campaigns & Progression

## 2.1 Campaign

| Term | Icon | Technical | Description |
|------|------|-----------|-------------|
| **Campaign** | 🗺️ | Training playthrough | One attempt to discover the level cap |
| **Path** | 🛤️ | Domain + Physics + Technique | Complete training recipe |
| **Milestone** | 🏁 | Achievement record | Significant event in the journey |
| **Ledger** | 📖 | checkpoint_ledger.json | Canonical history |

A **Campaign** is a hero's journey to maximum potential. Different heroes have different caps—we discover them by playing.

## 2.2 What "Maxed Out" Means

A hero is maxed when gaining a new skill level causes too much regression in other skills. The maintenance multiplier blows up. Time to:
1. Archive the journey
2. Keep the method
3. Start a new campaign with a different hero

---

# Part III: Strain & Effort (Materials Science)

Training viewed through materials science—like stretching metal until it permanently deforms.

## 3.1 Core Metrics

| Metric | Icon | Formula | Description |
|--------|------|---------|-------------|
| **Strain** | 💪 | `loss - floor` | How stretched the hero is now |
| **Effort** | 🏋️ | `Σ strain` | Cumulative work done |
| **Experience** | ✨ | `effort × quality_factor` | Blessed effort that counted |
| **Plastic Gain** | 📈 | `start_loss - end_loss` | Permanent improvement |
| **Efficiency** | ⚡ | `plastic_gain / effort` | Learning ROI |

## 3.2 Strain Zones (Like Heart Rate Zones)

| Zone | Icon | Strain Range | Action |
|------|------|-------------|--------|
| **Recovery** | 💚 | < 0.1 | Level up (too easy) |
| **Productive** | 💛 | 0.1 - 0.3 | Continue (optimal) |
| **Stretch** | 🧡 | 0.3 - 0.5 | Continue if improving |
| **Overload** | ❤️ | > 0.5 | Back off (too hard) |

## 3.3 Classic Metrics (Reframed)

| RPG Term | Icon | Technical | Description |
|----------|------|-----------|-------------|
| **Practice Strain** | 💪 | Training loss | Difficulty on training data |
| **Trial Strain** | 🎯 | Validation loss | True capability (not memorization) |
| **Over-Drilling** | ⚠️ | Overfitting | Memorized instead of learned |
| **Clarity** | 💎 | 1/Perplexity | Understanding quality |
| **Training Intensity** | ⚡ | Learning rate | Update aggressiveness |
| **Momentum** | - | Gradient norm | Update magnitude |

---

# Part IV: The Six Training Schools

How the Hero learns. Each school has its own philosophy.

## 📜 School of the Scribe (SFT)

> *"Copy the master's form until it becomes your own."*

Learn by imitating correct examples. Foundation of all training.

**Technical:** Supervised Fine-Tuning
**Data format:** `messages` with user/assistant pairs

## 🪞 School of the Mirror (Sparring)

> *"See your flaws reflected, then correct them."*

Learn by identifying and correcting mistakes.

**Technical:** Error mining + correction
**Data format:** `sparring_identify_incorrect`, `sparring_correction`, `sparring_confirm_correct`

## ⚖️ School of the Judge (DPO)

> *"Between two paths, always choose the better."*

Learn by comparing and choosing superior options.

**Technical:** Direct Preference Optimization
**Data format:** `prompt`, `chosen`, `rejected`

## 🏆 School of the Champion (RLHF)

> *"Seek the reward, master the arena."*

Learn by maximizing reward signals.

**Technical:** Reinforcement Learning from Human Feedback

## 👻 School of the Whisper (Distillation)

> *"The wisdom of giants flows to those who listen."*

Learn from a larger, more capable model.

**Technical:** Knowledge Distillation

## 🔮 School of the Oracle (Fortune Teller)

> *"Focus where uncertainty dwells; ignore what is already known."*

**ENHANCER**: Works with any base school. Weights gradients by surprise.

**Technical:** `trainer/losses/fortune_teller.py`

---

# Part V: The Five Job Schools

How work is dispatched. Each school trains workers for specific jobs.

| School | Icon | Jobs | Resources |
|--------|------|------|-----------|
| **Inference** | 🔮 | EVAL, SPARRING, INFERENCE | GPU (inference) |
| **Forge** | 🔥 | DATA_GEN, DATA_FILTER, DATA_CONVERT | CPU |
| **Vault** | 🏛️ | ARCHIVE, RETENTION, SYNC | Storage |
| **Analytics** | 📊 | ANALYTICS, REPORT, HEALTH_CHECK | Any |
| **Archaeology** | 🔬 | LAYER_STATS, LAYER_DRIFT | GPU optional |

---

# Part VI: The Temple

The Temple validates training. Raw Effort becomes Experience only when Blessed.

## 6.1 Temple Hierarchy

| Term | Icon | Description |
|------|------|-------------|
| **Temple** | 🏛️ | Diagnostics hub |
| **Cleric** | ⛪ | Runs rituals, computes Blessings |
| **Ritual** | 🔮 | Single diagnostic check |
| **Ceremony** | 🎭 | Multiple rituals together |
| **Blessing** | ✨ | Verdict on training quality |

## 6.2 The Nine Orders

| Order | Icon | Domain | Critical? |
|-------|------|--------|-----------|
| **Quick** | ⚡ | Fast sanity checks | No |
| **API** | 🌐 | HTTP validation | No |
| **Forge** | 🔥 | GPU/hardware | **YES** |
| **Weaver** | 🕸️ | Daemons/processes | No |
| **Champion** | 🏆 | Model/checkpoint | **YES** |
| **Oracle** | 🔮 | Inference server | **YES** |
| **Guild** | ⚔️ | Skills/curriculum | No |
| **Scribe** | 📜 | Evaluation/logging | No |
| **Deep** | 🌊 | Comprehensive/meta | No |

## 6.3 Blessing Quality

| Verdict | Quality | Effect |
|---------|---------|--------|
| **Blessed** | ≥ 0.8 | Full experience |
| **Partial** | 0.3 - 0.8 | Reduced experience |
| **Cursed** | 0 | No experience |

---

# Part VII: Skills & Primitives

## 7.1 Skills

| Term | Icon | Technical | Description |
|------|------|-----------|-------------|
| **Skill Level** | 📊 | Curriculum level | Current mastery tier |
| **Accuracy** | 🎯 | Correct/Total | Success rate |
| **Trial** | - | Held-out eval | Level-up test |
| **Regression** | 📉 | Accuracy drop >5% | Forgot something |

### Current Skills

| Skill | Icon | Description |
|-------|------|-------------|
| **SY** (Syllacrostic) | 🧩 | Word puzzles with signal degradation |
| **BIN** (Binary) | 🔢 | Binary arithmetic with circled notation |

## 7.2 Unified Primitives

**Primitives** are atomic cognitive operations underlying all skills.

| Category | Prefix | Examples |
|----------|--------|----------|
| **Sequence** | `seq_` | `seq_continue`, `seq_transform`, `seq_reverse` |
| **Logic** | `logic_` | `logic_deduce`, `logic_chain`, `logic_contrapose` |
| **Memory** | `mem_` | `mem_recall`, `mem_context`, `mem_compose` |
| **Format** | `fmt_` | `fmt_json`, `fmt_code`, `fmt_table` |
| **Attention** | `attn_` | `attn_select`, `attn_count`, `attn_compare` |
| **Transform** | `xfm_` | `xfm_encode`, `xfm_map`, `xfm_reduce` |

**The insight:** Skills are *composed* of primitives.
- Primitives = Base stats (STR, DEX, INT)
- Skills = Abilities using stat combinations
- Transfer learning happens at the primitive level

---

# Part VIII: Quest Modules

Shareable content packs extending the hero's curriculum.

## 8.1 Module Structure

```
quests/modules/<module-id>/
├── manifest.yaml    # Metadata, dependencies
├── skills/          # Skill definitions
├── data/            # Training JSONL by level
├── eval/            # Evaluation sets
└── README.md
```

## 8.2 Key Manifest Fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier |
| `primitives.required` | Primitives exercised |
| `requirements.min_level` | Hero level required |
| `skills` | Skills taught |
| `curriculum.progression` | How levels unlock |

---

# Part IX: The Vault

Where checkpoints live.

## 9.1 Storage Zones

| Zone | Icon | Description | Technical |
|------|------|-------------|-----------|
| **Hot Vault** | 🔥 | Fast, limited | Local NVMe |
| **Warm Vault** | ♨️ | Slower, spacious | NAS |
| **Deep Vault** | ❄️ | Cold archive | Compressed |

## 9.2 Checkpoint Terms

| Term | Description |
|------|-------------|
| **Checkpoint** | Snapshot at a moment |
| **Promote** | Flag as significant |
| **Champion** | Best by eval metrics |
| **Soul Anchor** | Saved hero form |

---

# Part X: Physics & Technique

## 10.1 Physics

The laws governing training—optimizer, precision, gradients.

| Physics | Description |
|---------|-------------|
| **Muon** | Momentum orthogonalized (experimental) |
| **AdamW** | Classical, stable |
| **8-bit** | Memory-efficient quantized |

## 10.2 Techniques

Named training stacks.

| Technique | Icon | RPG Name |
|-----------|------|----------|
| **Muon** | ⚛️ | The Orthogonal Way |
| **AdamW** | 📐 | The Classical Path |
| **GaLore** | 🌀 | The Gradient Lens |

---

# Part XI: Infrastructure

## 11.1 Locations

| Location | Port | Description |
|----------|------|-------------|
| **Tavern** | 8888 | Main game UI |
| **VaultKeeper** | 8767 | Checkpoint registry |
| **RealmState** | 8866 | Real-time state (SSE) |
| **Oracle** | 8765 | Inference server |

## 11.2 Devices

| Device | Icon | Description |
|--------|------|-------------|
| **Training Grounds** | 🏋️ | RTX 4090 (training) |
| **Arena** | ⚔️ | RTX 3090 (inference) |
| **Deep Vault** | 🏛️ | Synology NAS |

## 11.3 Services

| Service | Description |
|---------|-------------|
| **Weaver** | Daemon orchestrator |
| **Groundskeeper** | Resource cleanup |
| **Garrison** | Fleet health manager |

---

# Part XII: Quests & Combat

## 12.1 Quest Flow

| Concept | RPG Term |
|---------|----------|
| Training file | **Quest Scroll** |
| Task queue | **Quest Board** |
| Training step | **Quest Attempt** |

## 12.2 Combat Results

| Result | Condition | XP | Visual |
|--------|-----------|----|----|
| **CRITICAL HIT** | Perfect match | +15 | 💥 |
| **HIT** | Correct | +10 | ⚔️ |
| **GLANCING** | Partial | +5-8 | 🗡️ |
| **MISS** | Wrong | +2 | ❌ |
| **CRIT MISS** | Invalid | +0 | 💀 |

## 12.3 Hyperparameters as Combat Style

| Hyperparameter | RPG Name |
|----------------|----------|
| Learning rate | **Training Intensity** |
| Batch size | **Party Size** |
| Gradient accumulation | **Power Charging** |
| Weight decay | **Discipline Oath** |
| Warmup | **Stretching** |

---

# Part XIII: Debuffs & Status Effects

## 13.1 Debuff Catalog

| Debuff | Cause | Cure |
|--------|-------|------|
| **Tunnel Vision** | Overfitting | Diverse data |
| **Fragmented Thoughts** | Catastrophic forgetting | Replay old quests |
| **Dull Blade** | Underfitting | More training |
| **Exhaustion** | OOM | Reduce load |
| **Curse of Repetition** | Mode collapse | Reset, diverse data |
| **Corrupted Knowledge** | Bad data | Purge cursed scrolls |
| **Wild Magic** | Gradient explosion | Clipping |
| **Trance** | Vanishing gradients | Architecture check |
| **Humbled** | Failed trial | More practice |

## 13.2 Bug Severity as Monsters

| Severity | Monster |
|----------|---------|
| Minor | **Gremlin** |
| Medium | **Ogre** |
| Major | **Dragon** |
| Critical | **Demon Lord** |

---

# Part XIV: Model Internals

## 14.1 The Mind Tower (Layers)

| Layer Type | RPG Name |
|------------|----------|
| Embedding | **Sensing Ring** |
| Early layers | **Perception Floors** |
| Middle layers | **Thought Halls** |
| Late layers | **Mouth & Masks** |

## 14.2 Attention (Eyes & Ravens)

| Head Type | RPG Name |
|-----------|----------|
| Position tracking | **Chronicle Ravens** |
| Entity matching | **Concordance Eyes** |
| Negation detection | **Contradiction Spirits** |

## 14.3 Equipment (Adapters)

| Concept | RPG Name |
|---------|----------|
| LoRA adapter | **Skill Circlet** |
| Full fine-tune | **Soul Forging** |
| Quantization | **Compressed Form** |

---

# Part XV: Regions & Curriculum

## 15.1 World Regions

```
        ⛰️ THE SUMMIT (L10)
              │
    🏔️ REASONING MOUNTAINS (L7-L9)
              │
    ⛰️ LOGIC FOOTHILLS (L4-L6)
              │
    🌳 NOVICE VALLEY (L1-L3)
              │
    🏰 THE TAVERN
```

## 15.2 Difficulty Tiers

| Tier | Stars | Description |
|------|-------|-------------|
| **Bronze** | ★☆☆☆☆ | Entry-level |
| **Silver** | ★★☆☆☆ | Moderate |
| **Gold** | ★★★☆☆ | Solid fundamentals |
| **Platinum** | ★★★★☆ | Expert |
| **Dragon** | ★★★★★ | Legendary |

---

# Quick Reference

## Complete Mapping Table

```
HERO & IDENTITY
───────────────────────────────────────────────
Model checkpoint        → Hero Form / Soul Anchor
Architecture            → Race (Qwen'dal, Llamari)
Parameter count         → Stature (Sprite to Titan)
Training variant        → Class (Wildborn, Academy)
Tokenizer               → The Hero's Tongue

TRAINING & COMBAT
───────────────────────────────────────────────
Campaign                → Training playthrough
Training step           → Quest attempt
Loss                    → Strain (distance from mastery)
Loss - floor            → Strain (current stretch)
Cumulative strain       → Effort
Blessed effort          → Experience
Forward pass            → Hero's Strike
Backward pass           → Reflection
Learning rate           → Training Intensity

INFRASTRUCTURE
───────────────────────────────────────────────
RTX 4090 (training)     → Training Grounds 🏋️
RTX 3090 (inference)    → Arena ⚔️
NAS                     → Deep Vault 🏛️
Checkpoints             → Soul Anchors
Best checkpoint         → Champion

SYSTEMS
───────────────────────────────────────────────
Scheduler               → Guild Council
Data manager            → Quartermaster
Generators              → Quest Forge
Validators              → Temple Rituals
Config files            → World Law Codex

PROBLEMS
───────────────────────────────────────────────
Overfitting             → Tunnel Vision
Forgetting              → Fragmented Thoughts
Mode collapse           → Curse of Repetition
OOM                     → Exhaustion
NaN loss                → Reality Tear
```

## The Core Loop

```
DROP QUEST → HERO BATTLES → STRAIN ACCUMULATES → TEMPLE BLESSES → EXPERIENCE GAINED
   (inbox)    (training)        (effort)          (validation)       (progress)
```

---

*Programmatic access: `python3 tavern/lore.py --list` or `from tavern.lore import get_lore`*

*"May your gradients be stable and your loss ever-decreasing."*
