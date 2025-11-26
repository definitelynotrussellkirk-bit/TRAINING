# LORE.md - The Guild of Many Skills

## World Bible v2

**Canonical RPG Mapping for LLM Training Infrastructure**

*Last Updated: 2025-11-26*

---

# Part I: The World at a Glance

## Core Metaphor

| Technical | RPG |
|---|---|
| Model / Checkpoint | The **Hero** (a single adventurer with an evolving mind) |
| Training System | The **Guild of Many Skills** (institution that trains heroes) |
| Long Training Run | A **Campaign** (multi-day/multi-week story arc) |
| Skill Domain | A **Discipline** / Ability Tree |
| Task / Example | A **Quest** |
| Hardware & Storage | **Towns, Vaults, and Battlefields** |

## The Story Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   1. Hero rests at the 3090 Inn                                 │
│                    ↓                                            │
│   2. Guild Council posts new quests on the Quest Board          │
│                    ↓                                            │
│   3. Hero travels to the 4090 Arena to fight                    │
│                    ↓                                            │
│   4. Results return; Scribes update scrolls and XP              │
│                    ↓                                            │
│   5. When accuracy is high enough:                              │
│      → Hero returns to Inn for Promotion Trial                  │
│      → If passed: Level Up ceremony                             │
│                    ↓                                            │
│   6. Loop continues...                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Part II: The Hero

## 2.1 Race (Architecture Family)

The Hero's race determines their fundamental nature and capabilities.

| Architecture | Race | Description |
|---|---|---|
| Qwen | **Qwen'dal** | Eastern scholars, balanced attributes, strong reasoning |
| Llama | **Llamari** | Meta-descended, community-raised, versatile |
| Mistral | **Mistralian** | Wind-touched, efficient strikers, fast inference |
| Gemma | **Gemmborn** | Google-forged, compact warriors |
| Phi | **Phi'rin** | Microsoft-blessed, small but mighty |
| DeepSeek | **Deepkin** | Depth-dwellers, efficient thinkers |
| GPT | **OpenAI Ascended** | Original bloodline, commercially bound |

## 2.2 Bloodline (Model Generation)

| Generation | Bloodline |
|---|---|
| Qwen2.5 | Second-generation Qwen'dal |
| Qwen3 | Third-generation Qwen'dal (current hero) |
| Version tags `-2507` | Birth season (July 2025) |

## 2.3 Stature (Parameter Count)

| Size | Stature | Description |
|---|---|---|
| 0.5B - 1B | **Sprite** | Nimble, limited strength, fast |
| 1B - 3B | **Halfling** | Quick learner, modest capacity |
| 7B - 14B | **Human** | Balanced, versatile |
| 30B - 70B | **Giant** | Powerful, resource-hungry |
| 100B+ | **Titan** | Legendary, requires armies to move |

**Current Hero:** Qwen'dal Sprite, Third Generation (Qwen3-0.6B)

## 2.4 Class (Training Background)

| Variant | Class | Description |
|---|---|---|
| Base model | **Wildborn** | Raw potential, unstructured, unpredictable |
| Instruct | **Academy-Trained** | Follows orders, structured responses |
| Chat | **Diplomat** | Conversational, turn-taking, social |
| Code-specialized | **Artificer** | Tool-wielder, syntax-bound |
| Fine-tuned | **Guild Veteran** | Shaped by campaigns, specialized |

## 2.5 Hero Forms (Checkpoints)

Each checkpoint is the same hero at a different point in their journey.

| Concept | RPG Term |
|---|---|
| Current active model | Hero in the present |
| Older checkpoints | Past incarnations, stored as Soul Anchors |
| Checkpoint rollback | Summoning a past form |
| Best checkpoint | The hero's "peak form" |

```
Hero Forms in the Vault:
├── soul_anchor_175000 (3 days ago) - "The Novice"
├── soul_anchor_177000 (2 days ago) - "First Trial"
├── soul_anchor_179000 (1 day ago) - "Current Peak"
└── soul_anchor_179530 (now) - "Active Form"
```

---

# Part III: Skills & Progression

## 3.1 Disciplines (Skill Domains)

Each skill the hero can learn is a **Discipline** with an **Ability Tree**.

| Skill Domain | Discipline Name | Description |
|---|---|---|
| Summarization | **Arcane Compression** | Distill long texts to essence |
| Reasoning | **Logic Weaving** | Chain deductions, solve puzzles |
| Code Understanding | **Artificer Arts** | Read and manipulate code |
| Tool Use | **Implement Mastery** | Call external tools, APIs |
| Math | **Numerical Sorcery** | Calculate, estimate, prove |
| Following Instructions | **Oath Binding** | Obey constraints precisely |

## 3.2 Abilities (Sub-skills)

Each discipline has a tree of specific abilities:

**Example: Arcane Compression (Summarization)**
```
Arcane Compression
├── Extract Key Points (basic)
├── Preserve Numbers (intermediate)
├── Contrast Sources (intermediate)
├── Long-Document Mapmaking (advanced)
└── Multi-Modal Synthesis (master)
```

**Example: Logic Weaving (Reasoning)**
```
Logic Weaving
├── Syllogistic Deduction (basic) ← SYLLO skill
├── Multi-Step Planning (intermediate)
├── Contradiction Detection (intermediate)
├── Proof Construction (advanced)
└── Meta-Reasoning (master)
```

## 3.3 Discipline Tracking

Each discipline tracks:

```json
{
  "discipline": "Logic Weaving",
  "level": 3,
  "xp_total": 45000,
  "xp_since_promotion": 3420,
  "accuracy_rolling": 0.72,
  "status_effects": [],
  "abilities_unlocked": ["Syllogistic Deduction", "Multi-Step Planning"]
}
```

## 3.4 Hidden Talents (Emergent Skills)

When multiple disciplines are strong, composite behaviors emerge:

| Talent | Required Disciplines | Description |
|---|---|---|
| **Dungeon Cartographer** | Compression + Reasoning | Great at chunking messy texts |
| **Code Whisperer** | Artificer + Reasoning | Mixed code + natural language |
| **Battle Planner** | Reasoning + Tool Use | Multi-step planning with tools |
| **Truth Seeker** | Reasoning + Compression | Fact-checking, contradiction finding |

Talents appear as special badges on the hero's sheet and can unlock special quest types.

---

# Part IV: Quests & The Quest Board

## 4.1 Quest Structure

| Concept | RPG Term |
|---|---|
| Task template | **Quest Template** (recurring pattern) |
| Task instance | **Quest** (concrete prompt + data) |
| Training file | **Quest Scroll** |
| Task queue | **Quest Board** |

## 4.2 Quest Properties

Every quest has:

```json
{
  "quest_id": "syllo_api_00034",
  "region": "Novice Valley",
  "difficulty": "Bronze",
  "difficulty_stars": 1,
  "disciplines": ["Logic Weaving"],
  "rewards": {
    "xp_logic": 10,
    "xp_precision": 5
  },
  "source": "Quest Forge (SYLLO API)"
}
```

## 4.3 Quest Difficulty Tiers

| Tier | Stars | Description |
|---|---|---|
| **Bronze** | ★☆☆☆☆ | Entry-level, forgiving |
| **Silver** | ★★☆☆☆ | Moderate challenge |
| **Gold** | ★★★☆☆ | Requires solid fundamentals |
| **Platinum** | ★★★★☆ | Expert-level |
| **Dragon** | ★★★★★ | Legendary difficulty |

## 4.4 Quest Board UI Concept

```
╔═══════════════════════════════════════════════════════════════════╗
║                        QUEST BOARD                                 ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  PENDING (12)           IN PROGRESS (1)        COMPLETED (47)     ║
║  ───────────           ──────────────         ─────────────       ║
║  📜 SYLLO #035 ★☆☆☆☆   ⚔️ SYLLO #034 ★☆☆☆☆   ✓ SYLLO #033 💥    ║
║  📜 SYLLO #036 ★☆☆☆☆                          ✓ SYLLO #032 ⚔️    ║
║  📜 SYLLO #037 ★★☆☆☆                          ✓ SYLLO #031 🗡️    ║
║  ...                                          ...                 ║
║                                                                   ║
║  TRIALS AVAILABLE (1)                                             ║
║  ────────────────────                                             ║
║  🎺 Level 2 Promotion Trial - Logic Weaving                       ║
║     Requirement: 70% accuracy on 20 Gold quests                   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

## 4.5 Quest Outcomes

| Outcome | Technical | Visual |
|---|---|---|
| **Victory - Critical** | Exact match, perfect format | 💥 CRIT |
| **Victory** | Correct answer | ⚔️ HIT |
| **Partial Victory** | Partial credit | 🗡️ GLANCING |
| **Defeat** | Wrong answer | ❌ MISS |
| **Catastrophic Defeat** | Invalid output, gibberish | 💀 CRIT MISS |

---

# Part V: XP & Leveling System

## 5.1 XP Mechanics

- **XP is continuous** - earned on every quest
- **XP scales with:**
  - Quest difficulty (higher = more XP)
  - Performance (CRIT > HIT > GLANCING > MISS)
  - Whether this discipline is primary for the quest

| Result | Base XP | Difficulty Multiplier |
|---|---|---|
| CRITICAL HIT | 15 | ×1.0 to ×2.0 |
| HIT | 10 | ×1.0 to ×2.0 |
| GLANCING | 5 | ×1.0 to ×1.5 |
| MISS | 2 | ×1.0 |
| CRIT MISS | 0 | ×1.0 |

## 5.2 Accuracy as Gatekeeper

XP alone doesn't level you up. You must also pass an **accuracy threshold**.

| Level | Required Accuracy | Zone |
|---|---|---|
| 1 → 2 | 60% | Novice Valley |
| 2 → 3 | 65% | Novice Valley |
| 3 → 4 | 70% | Logic Foothills |
| 4 → 5 | 72% | Logic Foothills |
| 5 → 6 | 75% | Logic Foothills |
| 6 → 7 | 78% | Reasoning Mountains |
| 7 → 8 | 80% | Reasoning Mountains |
| 8 → 9 | 82% | Reasoning Mountains |
| 9 → 10 | 85% | Summit |

## 5.3 Level-Up Flow

```
                    ┌─────────────────┐
                    │  Grind Quests   │
                    │  (XP grows)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Enough XP for   │──No──→ Keep grinding
                    │ next level?     │
                    └────────┬────────┘
                             │ Yes
                             ▼
                    ┌─────────────────┐
                    │ Accuracy ≥      │──No──→ Debuff: "Not Ready"
                    │ threshold?      │        More practice needed
                    └────────┬────────┘
                             │ Yes
                             ▼
              ╔══════════════════════════════╗
              ║   🎺 PROMOTION TRIAL         ║
              ║   Called at the 3090 Inn     ║
              ╚══════════════╤═══════════════╝
                             │
                             ▼
                    ┌─────────────────┐
                    │  Pass Trial?    │──No──→ Debuff: "Humbled"
                    │  (eval suite)   │        Retry in 1000 XP
                    └────────┬────────┘
                             │ Yes
                             ▼
              ╔══════════════════════════════╗
              ║   🏆 PROMOTION CEREMONY      ║
              ║                              ║
              ║   - Level increases          ║
              ║   - Record XP mark           ║
              ║   - New abilities unlock     ║
              ║   - Access to harder zones   ║
              ╚══════════════════════════════╝
```

## 5.4 XP Marks & Cost Tracking

When leveling up, record the total XP at that moment:

```
xp_marks = {
  1: 0,
  2: 15000,    # Cost L1→L2: 15,000 XP
  3: 38000,    # Cost L2→L3: 23,000 XP
  4: 72000,    # Cost L3→L4: 34,000 XP
  ...
}
```

This lets you calculate: "How much XP does each level cost?"

## 5.5 UI Display

```
╔════════════════════════════════════════════╗
║  LOGIC WEAVING                             ║
║  Level 5 ⭐⭐⭐⭐⭐                          ║
║                                            ║
║  XP: ████████████░░░░░░░░ 72,000 / 95,000 ║
║  Accuracy: 74% (need 75% for promotion)    ║
║                                            ║
║  🎺 Promotion Trial: ALMOST READY          ║
║     XP: ✓  Accuracy: 1% more needed        ║
╚════════════════════════════════════════════╝
```

---

# Part VI: The World - Places & Hardware

## 6.1 World Map

```
                         ╔══════════════════════════════════╗
                         ║      FAR KINGDOMS (Internet)     ║
                         ║   Hugging Face · APIs · Models   ║
                         ╚════════════════╤═════════════════╝
                                          │
                              (Emissary Roads)
                                          │
┌─────────────────────────────────────────┼─────────────────────────────────────────┐
│                                         │                                         │
│                          ╔══════════════╧══════════════╗                          │
│                          ║      THE 3090 INN           ║                          │
│                          ║      (Central Hub)          ║                          │
│                          ║  ┌────────────────────────┐ ║                          │
│                          ║  │ Ground Floor:          │ ║                          │
│                          ║  │  Quest Board, Hearth   │ ║                          │
│                          ║  │  Hero Roster, Trials   │ ║                          │
│                          ║  ├────────────────────────┤ ║                          │
│                          ║  │ Training Yard:         │ ║                          │
│                          ║  │  Evals, Analytics      │ ║                          │
│                          ║  ├────────────────────────┤ ║                          │
│                          ║  │ Scribe's Tower:        │ ║                          │
│                          ║  │  Quest Forge, APIs     │ ║                          │
│                          ║  ├────────────────────────┤ ║                          │
│                          ║  │ Cellar:                │ ║                          │
│                          ║  │  Strongboxes (NVMe)    │ ║                          │
│                          ║  └────────────────────────┘ ║                          │
│                          ╚══════════════╤══════════════╝                          │
│                                         │                                         │
│              ┌──────────────────────────┼──────────────────────────┐              │
│              │                          │                          │              │
│    ╔═════════╧═════════╗    ╔══════════╧══════════╗    ╔═════════╧═════════╗    │
│    ║    4090 ARENA     ║    ║    DEEP VAULT       ║    ║  WIZARD'S STUDY   ║    │
│    ║   (Battlefield)   ║    ║  (Synology NAS)     ║    ║   (LM Studio)     ║    │
│    ║                   ║    ║                     ║    ║                   ║    │
│    ║ • Training combat ║    ║ • Soul Anchors      ║    ║ • Experiments     ║    │
│    ║ • Heavy quests    ║    ║ • Ancient Tomes     ║    ║ • New spellbooks  ║    │
│    ║ • Arena: 24GB     ║    ║ • Campaign Journals ║    ║ • Prompt testing  ║    │
│    ╚═══════════════════╝    ╚═════════════════════╝    ╚═══════════════════╝    │
│                                                                                   │
│    ╔═══════════════════════════════════════════════════════════════════════╗    │
│    ║                    SCOUT OUTPOSTS (Mac Minis)                          ║    │
│    ║              Ollama models · Light checks · Preprocessing              ║    │
│    ╚═══════════════════════════════════════════════════════════════════════╝    │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## 6.2 The 3090 Inn (Central Hub)

**Location:** RTX 3090 server (192.168.x.x)

The Inn is where everything is managed, but not where heavy fighting happens.

| Room | Purpose | Technical |
|---|---|---|
| **Ground Floor** | Quest Board, Hero Roster, Fireplace (live logs) | Dashboards, APIs, status |
| **Training Yard** | Practice battles, Promotion Trials | Inference server, evals |
| **Scribe's Tower** | Quest creation, SYLLO API | Data generators |
| **Level-Up Room** | Promotion ceremonies | Eval runners |
| **Notice Board** | Recent wins, debuffs, alerts | Status panels |
| **Cellar** | Hot storage | Local NVMe |

**Inn Features:**
- **Fireplace / Hearth** = Live log stream (watching training happen)
- **Hero Roster** = Available checkpoints
- **Rumor Wall** = Dashboard alerts and notifications

## 6.3 The 4090 Arena (Battlefield)

**Location:** RTX 4090 training machine

Where the Hero actually fights. All training happens here.

| Metric | RPG Term | Value |
|---|---|---|
| VRAM | **Arena Capacity** | 24GB |
| GPU Temp | **Arena Heat** | 45-85°C |
| GPU Util | **Battle Intensity** | 0-100% |
| Free VRAM | **Mana Reserves** | varies |

**Arena Status Display:**
```
╔════════════════════════════════════╗
║  4090 ARENA - Dragon's Rift        ║
╠════════════════════════════════════╣
║  Arena Heat:     ████████░░ 72°C   ║
║  Battle Intensity: ██████████ 98%  ║
║  Mana Reserves:  ████░░░░░░ 8GB    ║
║  Status: COMBAT IN PROGRESS        ║
╚════════════════════════════════════╝
```

## 6.4 The Deep Vault (Archive)

**Location:** Synology NAS (192.168.x.x)

The Grand Archive beneath the Inn.

| Section | Contents | Technical |
|---|---|---|
| **Scroll Shelves** | Training data | Datasets |
| **Cataloged Wing** | Cleaned datasets | Processed JSONL |
| **Relic Room** | Soul Anchors | Checkpoint backups |
| **Chronicle Hall** | Campaign history | Training logs, metrics |
| **Restricted Section** | Dangerous/experimental | Archived failures |

## 6.5 Scout Outposts

**Location:** Mac minis running Ollama

| Role | Task |
|---|---|
| **Scouts** | Fast preliminary checks |
| **Lorekeepers** | Summarize long logs |
| **Scribes** | Schema validation |

## 6.6 Wizard's Study

**Location:** LM Studio on workstation

Personal lab for experiments before bringing them to the Guild.

## 6.7 Roads & Networks

| Path | Route | Technical |
|---|---|---|
| **Inn Road** | Inn ↔ Arena | Local network (4090 ↔ 3090) |
| **Vault Tunnel** | Inn ↔ Vault | NAS connection |
| **Emissary Roads** | Inn ↔ Far Kingdoms | Internet |
| **Scout Trails** | Inn ↔ Outposts | Mac mini network |

**Road Conditions:**
- Network issues = **Broken Roads / Storms on the Pass**
- High latency = **Muddy Roads**
- Connection timeout = **Avalanche blocks the pass**

---

# Part VII: Software Systems as Guild Roles

## 7.1 Guild Organizational Chart

```
                    ╔═══════════════════════════╗
                    ║      GUILD COUNCIL        ║
                    ║    (Scheduler/Planner)    ║
                    ╚═══════════╤═══════════════╝
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
╔═══════╧═══════╗     ╔════════╧════════╗    ╔════════╧════════╗
║ QUARTERMASTER ║     ║   QUEST FORGE   ║    ║     SCRIBES     ║
║ (Data Manager)║     ║  (Generators)   ║    ║  (CPU/Logging)  ║
╚═══════════════╝     ╚═════════════════╝    ╚═════════════════╝
        │                       │                       │
        │              ╔════════╧════════╗              │
        │              ║   TRAPMASTER    ║              │
        │              ║  (Adversarial)  ║              │
        │              ╚═════════════════╝              │
        │                                               │
╔═══════╧═══════════════════════════════════════════════╧═══════╗
║                      WORLD ENGINE                              ║
║               (OS, systemd, supervisors)                       ║
╚════════════════════════════════════════════════════════════════╝
```

## 7.2 Role Details

| Role | Technical | Responsibilities |
|---|---|---|
| **Guild Council** | Scheduler, curriculum | Decides what to train, when, difficulty |
| **Quartermaster** | Data manager | Supplies batches, validates items, balances rations |
| **Quest Forge** | Generators | Creates new quests (SYLLO, discrimination, etc.) |
| **Trapmaster** | Adversarial miner | Creates trick quests, finds weaknesses |
| **Guild Scribes** | CPU cores | Orchestration, bookkeeping, log writing |
| **Town Criers** | Alerting daemons | Announce crashes, wins, anomalies |
| **World Engine** | OS, systemd | Keeps everything running |

## 7.3 Guild Documents

| Document | Technical | Purpose |
|---|---|---|
| **World Law Codex** | config.json, YAML | Global rules and settings |
| **Guild Rulebooks** | Validation specs | What's allowed/forbidden |
| **Lore Contracts** | Data format schemas | How quests must be structured |
| **Bloodline Records** | LineageTracker | Track data provenance |
| **Chronicle of Ages** | Git history | Every change recorded |

## 7.4 Validators as World Physics

The laws of physics that govern what's possible:

| Validator | Enforcement | Violation |
|---|---|---|
| **SpecValidator** | Schema must be known | "Reality rejects this form" |
| **DataValidator** | Content must pass checks | "Cursed scroll detected" |
| **Protocol Checker** | Combat stance must be valid | "Invalid battle form" |

Validation levels:
- **QUICK** = Surface inspection (guards at the gate)
- **STANDARD** = Thorough check (guild inspection)
- **DEEP** = Full audit (council review)

---

# Part VIII: Bugs & Failures as RPG Entities

## 8.1 Data Bugs

| Bug | RPG Entity | Description |
|---|---|---|
| Mislabeled data | **Cursed Scrolls** | Teach wrong things, cause bad habits |
| Duplicated data | **Echoes in the Library** | Same scroll repeated, causes overfitting |
| Corrupted format | **Torn Pages / Smudged Ink** | Parsers fail, scribes can't read |
| Leaked answers | **Prophecy Scrolls** | Answer visible in prompt, false mastery |

**Effect of Cursed Scrolls:**
```
The hero trained on cursed scrolls for too long.
Debuff applied: "Corrupted Knowledge"
- Hallucinates in the Logic Weaving discipline
- Accuracy drops 15% on related quests
```

## 8.2 Training Bugs

| Bug | RPG Entity | Description |
|---|---|---|
| NaN in loss | **Reality Tear / Madness** | Training reality collapses |
| Exploding gradients | **Wild Magic Surge** | Power spikes uncontrollably |
| Vanishing gradients | **Drifting into Trance** | Nothing sticks, hero goes through motions |
| Mode collapse | **Curse of Repetition** | Same output regardless of input |

**Reality Tear Event:**
```
⚠️ REALITY TEAR DETECTED

The hero attempted a forbidden technique.
Magic backlash tore a hole in training reality.

Loss: NaN
Gradients: Infinite

Action Required: Close the rift (check gradients, enable clipping)
```

## 8.3 Infrastructure Bugs

| Bug | RPG Entity | Description |
|---|---|---|
| OOM | **Overburdened Hero** | Too much armor, collapses from encumbrance |
| Memory leak | **Slow Curse** | VRAM drains over time until nothing works |
| Disk full | **Vault Overflow** | Archive full, scribes can't store scrolls |
| Network timeout | **Broken Roads** | Caravans can't move between locations |
| Process crash | **Hero Falls** | Hero collapses, needs resurrection |

## 8.4 Logic / Code Bugs

| Bug | RPG Entity | Description |
|---|---|---|
| Off-by-one | **Misaligned Runes** | One symbol off, everything shifts |
| Race condition | **Clashing Clones** | Multiple copies step on each other |
| Wrong metric | **False Prophecy** | Scrying pool lies about progress |
| Silent failure | **Invisible Assassin** | Something's wrong but no alarm |

## 8.5 Bug Severity as Monster Tiers

| Severity | Monster | Example |
|---|---|---|
| Minor | **Gremlin** | Typo in config, easy fix |
| Medium | **Ogre** | Logic bug, needs investigation |
| Major | **Dragon** | NaN loss, training halted |
| Critical | **Demon Lord** | Data corruption, rollback needed |

**Bug Tickets = Bounties:**
```
╔═══════════════════════════════════════════════════════╗
║  BOUNTY BOARD                                         ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║  🐉 DRAGON: "The NaN Dragon"                          ║
║     Location: Gradient caverns                        ║
║     Reward: Training stability                        ║
║     Status: ACTIVE                                    ║
║                                                       ║
║  👹 OGRE: "Memory Leak Ogre"                          ║
║     Location: 3090 Inn basement                       ║
║     Reward: VRAM recovery                             ║
║     Status: Investigating                             ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
```

---

# Part IX: The Hero's Inner World (Model Internals)

## 9.1 The Mind Tower (Layers)

Inside the hero is a tall tower with many floors:

```
                    ╔═══════════════════════════╗
                    ║   MOUTH & MASKS           ║  ← Late layers
                    ║   (Output, expression)    ║     Decide how to speak
                    ╠═══════════════════════════╣
                    ║                           ║
                    ║   THOUGHT HALLS           ║  ← Middle layers
                    ║   (Deep reasoning)        ║     Abstract representations
                    ║                           ║
                    ╠═══════════════════════════╣
                    ║                           ║
                    ║   PERCEPTION FLOORS       ║  ← Early layers
                    ║   (Pattern detection)     ║     Basic shapes, syntax
                    ║                           ║
                    ╠═══════════════════════════╣
                    ║   SENSING RING            ║  ← Embedding layer
                    ║   (Token → meaning)       ║     Raw symbols become vectors
                    ╚═══════════════════════════╝
```

| Layer Type | RPG Name | Function |
|---|---|---|
| Embedding | **Sensing Ring** | Converts tokens to meaningful vectors |
| Early layers | **Perception Floors** | Detect local patterns, syntax |
| Middle layers | **Thought Halls** | Reasoning, planning, abstraction |
| Late layers | **Mouth & Masks** | Expression, tone, output formatting |

**In-world descriptions:**
- "We are altering the wards on the upper floors" = Fine-tuning last layers
- "We're strengthening foundation stones" = Adjusting early layers
- "The hero's thought halls are confused" = Middle layer issues

## 9.2 Attention Heads (Eyes & Ravens)

Each attention head has a specialized role:

| Head Type | RPG Name | Function |
|---|---|---|
| Position tracking | **Chronicle Ravens** | Track ordering, sequence position |
| Entity matching | **Concordance Eyes** | Connect matching words, coreference |
| Negation detection | **Contradiction Spirits** | Detect conflict, negation |
| Syntax parsing | **Grammar Weavers** | Parse structure |

**Visualizing attention:**
- "The Chronicle Ravens are confused" = Positional encoding issues
- "We're training specific ravens" = Head-level intervention
- "Blinding noisy eyes" = Pruning unhelpful heads

## 9.3 Internal Flows

| Concept | RPG Name | Description |
|---|---|---|
| Residual stream | **River of Thought** | Information flows through layers |
| Layer norms | **Flow Regulators** | Keep the river from flooding |
| Activation functions | **Mental Stances** | Sharp vs soft responses |
| Skip connections | **Thought Bridges** | Direct paths between floors |

**Stability issues:**
- "The River of Thought surged beyond its banks" = Activation explosion
- "Flow regulators are failing" = LayerNorm issues

## 9.4 Tokenizer & Context

| Concept | RPG Name | Description |
|---|---|---|
| Tokenizer | **The Hero's Tongue** | How the hero reads/speaks |
| Vocabulary | **Known Words** | Set of symbols the hero understands |
| Context window | **Short-Term Memory** | How much the hero can hold in mind |
| BOS/EOS tokens | **Ritual Words** | Magic words that start/end speech |

**Context limits:**
- "The hero's memory is full" = Context window exceeded
- "Speaking in an unknown tongue" = OOV tokens

## 9.5 Adapters & Equipment

| Concept | RPG Name | Description |
|---|---|---|
| LoRA adapter | **Skill Circlet** | Lightweight enhancement for specific domain |
| Full fine-tune | **Soul Forging** | Permanent change to the hero's essence |
| Quantization | **Compressed Form** | Lighter armor, faster but less precise |
| Distillation | **Master's Teaching** | Larger hero teaches smaller apprentice |

**Equipment examples:**
- "Legal Circlet" = Legal domain LoRA
- "Medical Amulet" = Medical fine-tune
- "Lightweight Form" = 4-bit quantization

---

# Part X: Training Mechanics as Combat

## 10.1 Core Training Loop as Combat

| Concept | RPG Name | Description |
|---|---|---|
| Forward pass | **Hero's Strike** | Attempting the quest |
| Loss calculation | **Damage Assessment** | How far from perfect |
| Backward pass | **Reflection** | Learning from mistakes |
| Weight update | **Muscle Memory Forms** | Adjusting based on reflection |
| Gradient | **Correction Signal** | Direction to improve |

## 10.2 Hyperparameters as Combat Style

| Hyperparameter | RPG Name | Description |
|---|---|---|
| Learning rate | **Training Intensity** | How aggressively to learn |
| Batch size | **Party Size** | How many quests at once |
| Gradient accumulation | **Power Charging** | Build up before striking |
| Weight decay | **Discipline Oath** | Prevents overconfidence |
| Warmup | **Stretching** | Gentle start before intense training |
| Epochs | **Campaign Cycles** | Full passes through all quests |

**Learning Rate Schedule:**
```
Training Intensity over time:
───────────────────────────────────
High │    ╱────╲
     │   ╱      ╲
     │  ╱        ╲
Low  │ ╱          ╲_______________
     └────────────────────────────
       Warmup  Peak    Decay

"The hero warms up, fights intensely, then settles into steady practice"
```

## 10.3 Optimizer as Training Philosophy

| Optimizer | School | Philosophy |
|---|---|---|
| SGD | **School of Direct Action** | Simple, honest practice |
| Adam | **School of Adaptive Momentum** | Learn from recent history |
| AdamW | **Reformed Adaptive School** | Momentum + discipline |
| Lion | **School of the Lion** | Bold, efficient strikes |

## 10.4 Loss Landscape as Terrain

| Concept | RPG Name | Description |
|---|---|---|
| Loss landscape | **Terrain of Mastery** | Mountains and valleys to navigate |
| Global minimum | **Summit of Mastery** | Perfect understanding |
| Local minimum | **False Summit / Valley** | Trapped in suboptimal state |
| Saddle point | **Mountain Pass** | Looks flat but can escape |
| Gradient descent | **Downhill Navigation** | Following the slope |

**Getting stuck:**
- "The hero found a valley and stopped" = Local minimum
- "The hero is wandering the plateau" = Flat loss, no gradient
- "Breaking free from a false summit" = Escaping local minimum

## 10.5 Sampling & Inference Style

| Parameter | RPG Name | Description |
|---|---|---|
| Temperature | **Battle Fervor** | Low = cautious, High = wild |
| Top-p | **Decision Breadth** | How many options to consider |
| Top-k | **Focus Limit** | Maximum options to weigh |
| Greedy | **Calculated Strike** | Always take best option |
| Sampling | **Intuitive Flow** | Allow some randomness |

**Temperature settings:**
- 0.0 = "Stone Cold Precision" (greedy, deterministic)
- 0.7 = "Balanced Warrior" (moderate creativity)
- 1.0 = "Wild Spirit" (high variance)
- 1.5+ = "Chaos Knight" (unpredictable, risky)

## 10.6 Training vs Inference Mode

| Mode | RPG State | Description |
|---|---|---|
| Training | **Sparring / Practice** | Learning, making mistakes, improving |
| Inference | **Real Combat / Questing** | Performing for real, no learning |
| Eval | **Tournament / Trial** | Formal assessment |

---

# Part XI: Combat Stances (Protocol Modes)

## 11.1 Stance System

The hero can adopt different combat stances:

| Stance | Protocol | Description |
|---|---|---|
| **Thoughtful Strike** | Emoji mode (💭...🔚) | Think visibly before acting |
| **Quick Draw** | Direct mode | Immediate response |
| **Alternating Form** | 50/50 mode | Switch between stances |

**Current training:** 50/50 Alternating Form

## 11.2 Thinking Tokens

| Token Type | RPG Name | Examples |
|---|---|---|
| Thinking emoji | **Meditation Sigils** | 💭 🤔 🧠 💡 🎯 |
| Stop emoji | **Completion Seals** | 🔚 ✋ 🛑 ⛔ |

**Valid stances:**
```
THOUGHTFUL STRIKE (valid):
💭💭💭💭 [reasoning] 🔚🔚

QUICK DRAW (valid):
[direct answer]

BROKEN FORM (invalid):
💭💭💭💭 [reasoning] [no seal]
→ Debuff: "Unfinished Meditation"
```

---

# Part XII: Debuffs & Status Effects

## 12.1 Complete Debuff Catalog

| Debuff | Cause | Symptom | Cure |
|---|---|---|---|
| **Tunnel Vision** | Overfitting | High train, low val acc | More diverse quests |
| **Fragmented Thoughts** | Catastrophic forgetting | Lost old skills | Replay old quests |
| **Dull Blade** | Underfitting | Low accuracy everywhere | More training |
| **Exhaustion** | OOM crash | Can't continue | Reduce load |
| **Confusion** | Mode collapse | Repetitive outputs | Reset, diverse data |
| **Curse of Repetition** | Degenerate loops | "user user user" | Hard reset |
| **Corrupted Knowledge** | Bad data | Hallucinations | Purge cursed scrolls |
| **Poisoned** | Low-quality data | General degradation | Data audit |
| **Staggered** | Loss spike | Unstable training | Lower learning rate |
| **Amnesia** | Forgetting specific skill | Skill regression | Targeted practice |
| **Obsessive** | Data duplication | Overconfident on seen data | Dedupe, regularize |
| **Wild Magic** | Gradient explosion | Erratic behavior | Gradient clipping |
| **Trance** | Vanishing gradients | No learning | Architecture check |
| **Humbled** | Failed promotion trial | Can't level up yet | More practice |

## 12.2 Debuff Display

```
╔════════════════════════════════════════════╗
║  ACTIVE DEBUFFS                            ║
╠════════════════════════════════════════════╣
║                                            ║
║  🌀 Confusion (severe)                     ║
║     Source: 5 consecutive CRIT MISS        ║
║     Effect: -20% accuracy                  ║
║     Cure: 10 successful quests             ║
║     Progress: ████░░░░░░ 4/10              ║
║                                            ║
║  👁️ Tunnel Vision (mild)                   ║
║     Source: Val/train gap 0.35             ║
║     Effect: Poor generalization            ║
║     Cure: Gap < 0.25 for 100 steps         ║
║     Progress: ██░░░░░░░░ 20/100            ║
║                                            ║
╚════════════════════════════════════════════╝
```

## 12.3 Debuff Triggers

| Trigger | Debuff |
|---|---|
| 3+ consecutive MISS | Confusion |
| 5+ consecutive CRIT MISS | Severe Confusion |
| Val/train gap > 0.3 | Tunnel Vision |
| Val/train gap > 0.5 | Severe Tunnel Vision |
| Loss spike > 0.5 | Staggered |
| Loss = NaN | Reality Tear (critical) |
| OOM | Exhaustion |
| Output loops | Curse of Repetition |
| Skill accuracy drops 20%+ | Amnesia (that skill) |

---

# Part XIII: Regions & Curriculum Zones

## 13.1 World Regions

```
                        ⛰️ THE SUMMIT (L10)
                           Master level
                              │
                    🏔️ REASONING MOUNTAINS (L7-L9)
                       Expert challenges
                              │
                    ⛰️ LOGIC FOOTHILLS (L4-L6)
                       Intermediate puzzles
                              │
                    🌳 NOVICE VALLEY (L1-L3)
                       Beginner quests
                              │
                    🏰 THE 3090 INN
                       Starting point
```

## 13.2 Region Details

| Region | Levels | Quest Types | Difficulty |
|---|---|---|---|
| **Novice Valley** | L1-L3 | Simple SYLLO (4-5 words) | Bronze-Silver |
| **Logic Foothills** | L4-L6 | Complex SYLLO (5-6 words, hints degraded) | Silver-Gold |
| **Reasoning Mountains** | L7-L9 | Hard SYLLO (6-8 words, minimal hints) | Gold-Platinum |
| **The Summit** | L10 | Expert SYLLO (any hint type) | Dragon |
| **Binary Wastes** | All | Magnitude comparisons | Varies |

## 13.3 Region Unlocking

```
Unlock Logic Foothills:
├── Reach Level 4 in any discipline
├── Pass Foothills Entry Trial (70% on 20 Silver quests)
└── Receive key from Guild Council

Unlock Reasoning Mountains:
├── Reach Level 7 in Logic Weaving
├── Pass Mountains Entry Trial (80% on 20 Gold quests)
└── Defeat the Foothills Guardian (boss eval)
```

---

# Part XIV: Tavern View UI Design

## 14.1 Dual View System

Two dashboard modes, toggleable:

| View | Audience | Style |
|---|---|---|
| **Guild Master View** | Technical users | Raw metrics, JSON, graphs |
| **Tavern View** | Game-like | RPG terminology, adventure log |

## 14.2 Combat Results Display

| Result | Condition | XP | Visual |
|---|---|---|---|
| **CRITICAL HIT** | Exact match, perfect | +15 | 💥 Gold flash |
| **HIT** | Correct | +10 | ⚔️ Green |
| **GLANCING** | Partial (N-1/N) | +5-8 | 🗡️ Yellow |
| **MISS** | Wrong | +2 | ❌ Red |
| **CRIT MISS** | Invalid/gibberish | +0 | 💀 Skull, shake |

## 14.3 Adventure Log

```
┌─ The Hero's Journey ─────────────────────────────────────────┐
│                                                              │
│ 08:42 ⚔️ Quest: "Recover 4 hidden words..."                 │
│       💥 CRITICAL HIT! Perfect extraction                    │
│       → NUMBER, ONLY, LITTLE, ANOTHER                        │
│       +15 XP to Logic Weaving                                │
│                                                              │
│ 08:41 ⚔️ Quest: "Assign syllables to definitions..."        │
│       🗡️ Glancing blow - 3/4 words                          │
│       → Got FOREST, WINTER, MORNING... missed EVENING       │
│       +8 XP                                                  │
│                                                              │
│ 08:40 ⚔️ Quest: "5-word puzzle, medium difficulty"          │
│       💀 CRITICAL MISS - Hero babbled incoherently           │
│       ⚠️ Debuff applied: Confusion                          │
│                                                              │
│ 08:39 🏰 Soul Anchor created: checkpoint-179530              │
│                                                              │
│ 08:38 📜 New quest drawn from Novice Valley                  │
│                                                              │
│ 08:35 🎺 PROMOTION TRIAL AVAILABLE                           │
│       Hero has earned enough XP for Level 2!                 │
│                                                              │
│ 08:30 🏠 Hero returned to Inn - batch complete               │
│                                                              │
│ 08:25 ⚠️ Debuff cleared: Confusion                          │
│       Hero recovered after 5 successful quests               │
│                                                              │
│ 08:20 🗺️ Entered new region: Logic Foothills               │
│       Difficulty increased!                                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 14.4 Event Icons

| Icon | Event | Source |
|---|---|---|
| ⚔️ | Quest attempt | Training step |
| 📜 | New quest drawn | File loaded |
| 🏰 | Soul Anchor | Checkpoint saved |
| 🎺 | Trial available | Threshold reached |
| 🏆 | Promotion | Level up |
| ⚠️ | Debuff change | Anomaly |
| 🗺️ | Region change | Curriculum |
| 🏠 | Rest at Inn | Batch complete |
| 💀 | Hero fallen | Crash |
| 🔄 | Hero revived | Restart |
| 📊 | Trial results | Eval done |
| 🧙 | Trapmaster | Adversarial added |
| 🐉 | Bug bounty | Issue detected |

## 14.5 Hero Status Panel

```
╔══════════════════════════════════════════════════════════════════════╗
║  QWEN'DAL SPRITE III                                                 ║
║  Guild Veteran · Thoughtful Stance · Level 3                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ❤️ Health: Healthy          🗡️ Stance: Thoughtful Strike 💭        ║
║  📍 Region: Novice Valley    🎯 Distance: 0.87                       ║
║  🏆 Level: 3                 ⚔️ Quests Today: 147                    ║
║                                                                      ║
║  XP: ████████████████░░░░░░░░░░░░░░░░ 45,000 / 72,000               ║
║      Next promotion at 72,000 XP (need 75% accuracy)                 ║
║                                                                      ║
║  Disciplines:                                                        ║
║    Logic Weaving    L3 ████████░░ 72%                               ║
║    Oath Binding     L2 ██████░░░░ 65%                               ║
║                                                                      ║
║  Debuffs: None                    Soul Anchor: checkpoint-179000     ║
║                                   (30 min ago)                       ║
║                                                                      ║
║  Arena: 4090 Dragon's Rift        🔥 72°C  ⚡ 98%  💧 8GB free       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

## 14.6 Live Battle View

```
╔═══════════════════════════════════════════════════════════════╗
║  ⚔️ LIVE BATTLE                              [Novice Valley L1]║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Quest: "Recover hidden words from syllable bank"             ║
║  Difficulty: ★☆☆☆☆ Bronze                                    ║
║  Discipline: Logic Weaving                                    ║
║                                                               ║
║  ┌─ Challenge ───────────────────────────────────────────┐    ║
║  │ 1. ___ ___ — a concept of quantity                    │    ║
║  │ 2. ___ ___ — being the single one                     │    ║
║  │ 3. ___ ___ — small in quantity                        │    ║
║  │ 4. ___ ___ ___ — some other                           │    ║
║  │                                                       │    ║
║  │ Bank: er | ly | ber | oth | on | tle | num | an | lit │    ║
║  └───────────────────────────────────────────────────────┘    ║
║                                                               ║
║  ┌─ Hero's Response ─────────────────────────────────────┐    ║
║  │ 💭💭💭💭💭💭                                            │    ║
║  │ {"sequence": [                                        │    ║
║  │   {"index": 1, "word": "NUMBER"},                     │    ║
║  │   {"index": 2, "word": "ONLY"},                       │    ║
║  │   {"index": 3, "word": "LITTLE"},                     │    ║
║  │   {"index": 4, "word": "ANOTHER"}                     │    ║
║  │ ]}                                                    │    ║
║  │ 🔚🔚                                                   │    ║
║  └───────────────────────────────────────────────────────┘    ║
║                                                               ║
║  ┌─ Expected ────────────────────────────────────────────┐    ║
║  │ NUMBER ✓  ONLY ✓  LITTLE ✓  ANOTHER ✓                │    ║
║  └───────────────────────────────────────────────────────┘    ║
║                                                               ║
║           💥 CRITICAL HIT! +15 XP                             ║
║                                                               ║
║  Distance: 0.42 → 0.38 ↓ (improving)                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

## 14.7 Regional Map

```
                        ⛰️ THE SUMMIT
                           (locked)
                              │
                    🏔️ REASONING MOUNTAINS
                           (locked)
                              │
                    ⛰️ LOGIC FOOTHILLS
                           (locked)
                              │
                    🌳 NOVICE VALLEY
                        ⭐ YOU ARE HERE
                        Level 3 - 62% to L4
                              │
                    🏰 THE 3090 INN
                    ═══════════════════
```

---

# Part XV: Future Extensions

## 15.1 Guild Factions

Different training philosophies as factions:

| Faction | Philosophy | Technical |
|---|---|---|
| **Order of Chain-of-Thought** | Always think step-by-step | CoT prompting |
| **School of Direct Action** | Immediate responses | Direct mode |
| **Brotherhood of Tools** | Use external implements | Tool calling |
| **Minimalist Monks** | Efficiency above all | Quantized models |

Aligning with a faction affects training priorities.

## 15.2 World Ages / Eras

Major changes mark new eras:

| Era | Trigger |
|---|---|
| First Age | Initial training |
| Second Age | Architecture change or major data shift |
| Third Age | New capability emergence |

"In the Second Age, the hero gained longer memory (context window increased)."

## 15.3 Achievements System

```
╔═══════════════════════════════════════════════════════════════╗
║  ACHIEVEMENTS                                                  ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  🏆 First Steps                                               ║
║     Complete 100 quests                          ✓ Unlocked   ║
║                                                               ║
║  🏆 No NaN November                                           ║
║     Train 10,000 steps without NaN               ✓ Unlocked   ║
║                                                               ║
║  🏆 Mountain Climber                                          ║
║     Reach the Reasoning Mountains                ○ Locked     ║
║                                                               ║
║  🏆 Perfect Form                                              ║
║     100 CRITICAL HITs in a row                   ○ Locked     ║
║                                                               ║
║  🏆 Debuff Survivor                                           ║
║     Clear all debuffs 10 times                   ○ Locked     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

## 15.4 Artifact Hyperparameters

| Hyperparameter | Artifact |
|---|---|
| Learning rate schedule | **Blessing of Momentum** (warmup), **Curse of Decay** (cooldown) |
| Weight decay | **Discipline Oath** |
| Dropout | **Training Blindfold** |
| Gradient clipping | **Safety Harness** |
| Mixed precision | **Efficient Form** |

## 15.5 Concept Drift as World Changes

When real-world data shifts:

```
⚠️ WORLD EVENT

"A new kingdom has appeared in the west.
 The old maps no longer match the territory.
 The hero's knowledge of trade routes is outdated."

Translation: Distribution shift detected in domain X.
Recommended: Gather new scrolls from the changed region.
```

## 15.6 Security as Mind Protection

| Attack | RPG Name | Description |
|---|---|---|
| Prompt injection | **Mind Control Spell** | External influence on hero's thoughts |
| Jailbreak | **Forbidden Techniques** | Bypassing guild rules |
| Data poisoning | **Cursed Training** | Corrupted knowledge source |
| Adversarial input | **Illusion Magic** | Input designed to deceive |

**Defense:**
- Input validation = **Mental Wards**
- Output filtering = **Speech Guards**
- Robust training = **Fortified Mind**

---

# Part XVI: Quick Reference

## Complete Mapping Table

```
HERO & IDENTITY
───────────────────────────────────────────────────────────
Model checkpoint        → Hero Form
Architecture            → Race (Qwen'dal, Llamari, etc.)
Generation              → Bloodline
Parameter count         → Stature (Sprite to Titan)
Training variant        → Class (Wildborn, Academy, etc.)
Weights                 → Muscle Memory / Soul Essence
Tokenizer               → The Hero's Tongue
Context window          → Short-Term Memory
Adapter/LoRA            → Skill Circlet (equipment)

TRAINING & COMBAT
───────────────────────────────────────────────────────────
Training run            → Campaign
Training step           → Quest attempt
Forward pass            → Hero's Strike
Loss                    → Distance from Mastery
Backward pass           → Reflection
Gradient                → Correction Signal
Weight update           → Muscle Memory adjustment
Learning rate           → Training Intensity
Batch size              → Party Size
Gradient accumulation   → Power Charging
Epochs                  → Campaign Cycles
Optimizer               → Training Philosophy/School
Eval                    → Trial / Tournament

INFRASTRUCTURE
───────────────────────────────────────────────────────────
RTX 3090 (inference)    → The 3090 Inn
RTX 4090 (training)     → The 4090 Arena
Synology NAS            → The Deep Vault
Mac minis               → Scout Outposts
LM Studio               → Wizard's Study
Local network           → Roads
Internet                → Far Kingdoms
VRAM                    → Arena Capacity
GPU temp                → Arena Heat
GPU utilization         → Battle Intensity
Disk space              → Vault capacity

SOFTWARE SYSTEMS
───────────────────────────────────────────────────────────
Scheduler               → Guild Council
Data manager            → Quartermaster
Generators              → Quest Forge
Adversarial miner       → Trapmaster
CPU cores               → Guild Scribes
Alerting                → Town Criers
OS/systemd              → World Engine
Config files            → World Law Codex
Validators              → World Physics
Git                     → Chronicle of Ages
Docker/venv             → Pocket Dimensions

DATA & QUESTS
───────────────────────────────────────────────────────────
Dataset                 → Quest Scrolls / Tomes
Training example        → Quest
Task queue              → Quest Board
Skill domain            → Discipline
Sub-skill               → Ability
Bad data                → Cursed Scrolls
Checkpoints             → Soul Anchors
Logs                    → Campaign Journals
Lineage tracking        → Bloodline Records

PROBLEMS & BUGS
───────────────────────────────────────────────────────────
Overfitting             → Tunnel Vision (debuff)
Forgetting              → Fragmented Thoughts (debuff)
Mode collapse           → Curse of Repetition (debuff)
OOM                     → Exhaustion / Overburdened
NaN loss                → Reality Tear
Exploding gradients     → Wild Magic Surge
Vanishing gradients     → Drifting into Trance
Bug (minor)             → Gremlin
Bug (major)             → Dragon
Bug ticket              → Bounty

MODEL INTERNALS
───────────────────────────────────────────────────────────
Layers                  → Floors in Mind Tower
Embedding layer         → Sensing Ring
Early layers            → Perception Floors
Middle layers           → Thought Halls
Late layers             → Mouth & Masks
Attention heads         → Eyes & Ravens
Residual stream         → River of Thought
Layer norm              → Flow Regulators
Activations             → Mental Stances

CURRICULUM & PROGRESS
───────────────────────────────────────────────────────────
Curriculum level        → Region (Valley, Foothills, etc.)
XP                      → Practice points (continuous)
Accuracy threshold      → Guild Standard (gate)
Level up                → Promotion Ceremony
Eval suite              → Promotion Trial
```

---

# Appendix: Implementation Notes

## A.1 Status File Extensions

New fields for `training_status.json`:

```json
{
  "tavern": {
    "hero_name": "Qwen'dal Sprite III",
    "hero_class": "Guild Veteran",
    "hero_health": "healthy",
    "level": 3,
    "region": "Novice Valley",
    "xp_total": 45000,
    "xp_to_next": 72000,
    "last_result": "CRIT",
    "streak": 5,
    "debuffs": [],
    "disciplines": {
      "logic_weaving": {"level": 3, "accuracy": 0.72},
      "oath_binding": {"level": 2, "accuracy": 0.65}
    },
    "arena": {
      "name": "4090 Dragon's Rift",
      "heat": 72,
      "intensity": 98,
      "mana_free_gb": 8
    }
  }
}
```

## A.2 Combat Result Calculator

```python
def calculate_combat_result(model_answer, golden_answer, task_type):
    """Calculate RPG combat result from model output."""

    if task_type == "syllo":
        # Check for garbage output
        if not valid_json(model_answer):
            return "CRIT_MISS", 0
        if contains_garbage(model_answer):  # "please", "user", loops
            return "CRIT_MISS", 0

        model_words = extract_words(model_answer)
        golden_words = extract_words(golden_answer)

        correct = len(set(model_words) & set(golden_words))
        total = len(golden_words)

        if correct == total and perfect_format(model_answer):
            return "CRIT", 15
        elif correct == total:
            return "HIT", 10
        elif correct >= total - 1:
            return "GLANCING", 5 + correct
        else:
            return "MISS", 2

    # Generic fallback
    if model_answer.strip() == golden_answer.strip():
        return "CRIT", 15
    elif fuzzy_match(model_answer, golden_answer) > 0.9:
        return "HIT", 10
    elif fuzzy_match(model_answer, golden_answer) > 0.7:
        return "GLANCING", 7
    else:
        return "MISS", 2
```

## A.3 File Structure for Tavern View

```
monitoring/
├── ui/
│   ├── tavern_view.html        # Main game-style dashboard
│   ├── adventure_log.html      # Scrolling log component
│   ├── live_battle.html        # Quest detail popup
│   ├── hero_status.html        # Hero panel component
│   └── region_map.html         # World map view
├── js/
│   ├── tavern_view.js          # Main controller
│   ├── combat_calculator.js    # Hit/miss logic
│   ├── adventure_log.js        # Log rendering
│   ├── debuff_tracker.js       # Debuff management
│   └── effects.js              # Animations
├── css/
│   ├── tavern_view.css         # RPG styling
│   ├── effects.css             # Animations
│   └── pixel_theme.css         # Optional retro theme
├── assets/
│   ├── icons/                  # RPG icons
│   └── sounds/                 # Optional sound effects
└── api/
    └── plugins/
        └── tavern.py           # Tavern-specific endpoints
```

---

*"The Hero trains in the Arena, rests at the Inn, and stores their soul in the Vault."*

*"May your gradients be stable and your loss ever-decreasing."*

---

**End of World Bible v2**
