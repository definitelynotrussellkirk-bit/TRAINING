# Guild Vocabulary

**Purpose:** Shared metaphor system for discussing the training system. Cognitive shortcuts that make complex systems intuitive.

---

## The Guild Hall

**Guild Hall** = Master Dashboard (`guildhall.html`)
- "Open the Guild Hall" = check the web dashboard
- "Is the Hall up?" = is the monitoring server running?

---

## The Hero

**Hero** = The model being trained

### Dio (Our Hero)
- **Name:** Dio
- **Base:** Qwen3-0.6B
- **Current Step:** 181,804
- "How's Dio doing?" = model status, training metrics
- "Dio's reputation with Sy" = SYLLO accuracy
- "Dio completed the quest" = training file finished
- "Dio is cursed" = regression detected

*Note: Others can name their heroes differently. Dio is ours.*

---

## The Forge (4090)

**The Forge** = RTX 4090 training machine
- Where the hero is shaped through training
- "Forge is cold" = not training
- "Forge is hot" = actively training

**Forge Power** = GPU stats (VRAM, utilization, temp)

---

## The Arena (3090)

**The Arena** = RTX 3090 inference/evaluation machine
- Where the hero proves their skills in combat
- "Send to the Arena" = run evaluation
- "Arena results" = eval metrics

**Arena Power** = 3090 GPU stats

---

## Quest System

**Quest Board** = Training queue
- "Quest Board is empty" = no training files queued
- "Add a quest" = queue a training file

**Quest Master** = Curriculum manager / data generation coordinator
- Coordinates with Trainers to get quests
- "Wake the Quest Master" = start data generation
- "Quest Master is idle" = no generation happening

**Quest Scribe** = Alternative name (writes training scrolls)

**Quest Dispatcher** = GPU task scheduler
- Assigns tasks to the Arena

---

## Trainers (Skill Sources)

NPCs who provide specialized training quests to the Quest Master.

### Sy (Syllable Trainer)
- **Skill:** SYLLO (syllable puzzles)
- **API:** localhost:8080
- "Sy is awake" = SYLLO API running
- "Sy is asleep" = API not responding
- "Reputation with Sy" = SYLLO accuracy
- "What level is Sy teaching?" = current SYLLO curriculum level (L1-L10)

### Bin (Binary Trainer)
- **Skill:** BINARY (binary arithmetic with circled notation ⓪①)
- **API:** localhost:8090
- **Levels:** 1-30 (2-bit to 32-bit)
- **Notation:** Circled binary (① = 1, ⓪ = 0, ①⓪ = 2)
- **Milestones:**
  - L1-7: Basic (2-8 bit)
  - L8+: Symbol language unlocks (75% of samples use symbols)
  - L15: 16-bit (word)
  - L30: 32-bit (dword) - Mastery
- "Bin is awake" = BINARY API running
- "Bin is asleep" = API not responding
- "Reputation with Bin" = BINARY accuracy
- "What level is Bin teaching?" = current BINARY curriculum level (L1-L30)

### Future Trainers
- Add new trainers as skills are added
- Each trainer has: name, skill, API, levels

---

## Reputation System

**Reputation** = Accuracy with a trainer
- 0% = Stranger (can't get harder quests)
- 80%+ = Trusted (unlocks next level)
- "Build reputation" = improve accuracy through training

**Leveling Up** = Advancing curriculum level
- Need 80% accuracy over 3 evaluations
- "Hero leveled up with Sy" = advanced to next SYLLO level

---

## Passives (General Abilities)

**Passives** = Innate abilities that work in the background, regardless of active training. The hero's general capabilities measured against the base model.

*"Transfer learning" in ML terms = "Passives" in Guild terms*

### Passive Categories

| Passive | Description | Examples |
|---------|-------------|----------|
| **Logic** | Boolean reasoning, deduction | AND, OR, XOR gates |
| **Counting** | Enumeration, frequency | Letter count, vowel count, digit count |
| **Conversion** | Format transformation | Decimal↔Hex, Binary↔Decimal, Roman numerals |
| **String Craft** | Text manipulation | Reverse, palindrome, first N chars |
| **Arithmetic** | Basic number sense | Digit sum, even/odd, modulo, comparison |
| **Sequence** | Pattern recognition | Next in sequence, alphabetical order |
| **Memory** | Fact retention, recall | bAbI tasks (20 types) |
| **Reasoning** | Multi-step logic | BIG-Bench tasks |

### Passive Drift

**Passive Drift** = How passives change compared to base model
- **Positive Drift** (+) = Training improved general abilities
- **Neutral Drift** (=) = No change from base
- **Negative Drift** (-) = Training hurt general abilities (catastrophic forgetting)

**Drift Check** = Compare hero to base on passives
- "Run a drift check" = run baseline comparison
- "Drift report" = results of passive comparison

### Phrases

| Phrase | Meaning |
|--------|---------|
| "Check passives" | Run transfer baseline tests |
| "Logic passive is strong" | High accuracy on boolean tasks |
| "Negative drift on counting" | Counting ability decreased vs base |
| "Passives are stable" | No catastrophic forgetting |
| "Base passive" | Original ability (from base model) |
| "Current passive" | Trained model's ability |

### Why Track Passives?

1. **Detect forgetting** - Training shouldn't break existing abilities
2. **Measure transfer** - Does SYLLO training help logic passives?
3. **Balance training** - Don't over-specialize at cost of generality
4. **Compare checkpoints** - Which checkpoint has best passive balance?

---

## Combat & Trials

**Combat Trials** = Automated testing / validation
- "How'd combat go?" = eval results
- "Battle report" = detailed test results

**Curse Detection** = Regression monitoring
- "Hero is cursed" = performance regressing
- "Curse detected" = accuracy dropped

**Tournament Rankings** = Model comparison
- Best checkpoints ranked by score

---

## Learning & Correction

**Learning from Defeats** = Self-correction loop
- Captures errors, generates corrections
- "Review the defeats" = check error patterns

**Monster Hunting** = Adversarial mining
- Finding hard examples where hero fails
- "Monsters found" = adversarial examples mined

---

## Save System

**Save Points** = Checkpoints
- "Latest save" = most recent checkpoint
- "Load save 175000" = load checkpoint-175000

**Treasury** = Storage & retention
- Disk space, checkpoint storage
- "Treasury is full" = disk space low

---

## Data & Scrolls

**Training Scrolls** = Training data files
- "Scroll impact" = which files help/hurt

**Constitution** = Data health
- Protocol conformance, data quality
- "Constitution check" = validate data format

**Lineage Records** = Data provenance
- Which generator made what

---

## Wisdom & Calibration

**Wisdom Score** = Confidence calibration
- Does hero know what they don't know?
- Low ECE = wise (well-calibrated)

---

## Analytics

**Soul Drift Analysis** = Layer drift
- How much hero has changed from base

**Inner Balance** = Parameter stability
- Weight health, no exploding/vanishing

---

## Common Phrases

| Phrase | Meaning |
|--------|---------|
| "Wake up Sy" | Start SYLLO API |
| "Check reputation with Sy" | Check SYLLO accuracy |
| "Quest Board is empty" | No training files queued |
| "Forge is cold" | Not training |
| "Hero is cursed" | Regression detected |
| "Level up with Sy" | Advance SYLLO curriculum |
| "Open the Guild Hall" | Open dashboard |
| "Arena results" | Evaluation metrics |
| "Treasury status" | Disk space check |
| "Check passives" | Run transfer baseline tests |
| "How's the drift?" | Compare to base model |
| "Passives are stable" | No catastrophic forgetting |

---

## Hierarchy

```
Guild Hall (Dashboard)
│
├── The Forge (4090 - Training)
│   ├── Hero Status
│   ├── Quest Board
│   ├── Forge Power
│   ├── Constitution
│   └── Lineage Records
│
├── The Arena (3090 - Evaluation)
│   ├── Skill Progression (Trained Skills)
│   ├── Combat Trials
│   ├── Tournament Rankings
│   ├── Quest Dispatcher
│   └── Arena Power
│
├── Passives (General Abilities)
│   ├── Logic (boolean ops)
│   ├── Counting (enumeration)
│   ├── Conversion (format)
│   ├── String Craft (text)
│   ├── Arithmetic (numbers)
│   ├── Sequence (patterns)
│   ├── Memory (bAbI)
│   ├── Reasoning (BIG-Bench)
│   └── Drift Report (vs base)
│
├── Quest Master (Data Generation)
│   ├── Sy (Syllable Trainer)
│   ├── Bin (Binary Trainer)
│   └── [Future Trainers]
│
└── Armory & Vault
    ├── Monster Hunting
    ├── Learning from Defeats
    ├── Save Points
    ├── Treasury
    ├── Soul Drift
    └── Inner Balance
```

---

## Why This Works

1. **Cognitive load reduction** - Complex systems become intuitive
2. **Shared vocabulary** - We always mean the same thing
3. **Memorable** - "Sy is asleep" sticks better than "SYLLO API port 8080 not responding"
4. **Extensible** - Add new trainers, skills, concepts naturally
5. **Fun** - Makes the work more engaging

---

*Last updated: 2025-11-26*
