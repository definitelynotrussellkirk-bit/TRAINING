# Trainer Contract

**Version:** 1.0.0
**Last Updated:** 2025-11-26

Every Trainer (skill source) that provides quests to the Quest Master must conform to this contract.

---

## Required: API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Returns `{"status": "ok"}` when alive |
| `/info` | GET | Server metadata (skill name, features, operations) |
| `/levels` | GET | List all difficulty levels with configs |
| `/generate` | POST | Generate training samples |

### `/generate` Request

```json
{
  "level": 1,           // Required: difficulty level (1-N)
  "count": 100,         // Required: number of samples
  "seed": 12345         // Optional: for reproducibility
}
```

### `/generate` Response

```json
{
  "count": 100,
  "level": 1,
  "level_info": {
    "level": 1,
    "bits": 2,          // Trainer-specific metadata
    "description": "...",
    "threshold": 0.80
  },
  "samples": [...]      // Array of samples
}
```

---

## Required: Sample Format

Each sample in the `samples[]` array MUST have:

```json
{
  "id": "sample-0001",              // Unique ID within batch
  "user_prompt": "The question...", // What to ask Dio
  "assistant_response": "Answer",   // Expected correct answer
  "tags": ["easy", "level-1", ...], // Searchable tags
  "rubric": "Brief description"     // What this tests
}
```

### Conversion to Training Format

The Quest Master converts samples to `messages[]` format:

```json
{
  "messages": [
    {"role": "user", "content": "<user_prompt>"},
    {"role": "assistant", "content": "<assistant_response>"}
  ],
  "metadata": {
    "source": "<trainer_id>_api",
    "generator_id": "<trainer_id>_api",
    "generator_version": "1.0.0",
    "skill": "<skill_name>",
    "level": 1,
    "tags": [...],
    "difficulty": "easy"
  }
}
```

---

## Required: Level System

Every Trainer MUST have a progressive level system:

| Property | Description |
|----------|-------------|
| `level` | Integer (1-N), increasing difficulty |
| `name` | Human-readable name |
| `threshold` | Accuracy required to advance (0.80 = 80%) |
| `description` | What this level tests |

### Advancement Rules

- Dio advances when: **accuracy >= threshold over 3 consecutive evals**
- Final level has `threshold: null` (mastery level)
- Levels should be designed so L1 is learnable in ~1000 examples

---

## Required: Metadata Fields

Every sample MUST include these for lineage tracking:

| Field | Example | Purpose |
|-------|---------|---------|
| `generator_id` | `"bin_api"` | Identifies the data source |
| `generator_version` | `"1.0.0"` | Tracks generator changes |
| `source` | `"bin_api"` | Short source identifier |
| `skill` | `"binary"` | Skill category |
| `level` | `1` | The ONLY difficulty scale |
| `tags` | `["add", "chain"]` | What operations (not difficulty) |

### Tags Requirements

Tags describe WHAT the sample tests (not difficulty):
- Operation type: `add`, `subtract`, `multiply`, `chain`
- Format hints: `verify`, `trace`, `show-work`
- Content type: `signed`, `unsigned`, `comparison`
- Level tag: `level-1`, `level-2`, etc. (for filtering)

---

## Required: Evaluation Support

For the curriculum eval loop to test Dio, each Trainer MUST define:

### 1. Expected Answer Format

Document what format the `assistant_response` uses:
- Plain text answer?
- JSON structure?
- Multiple parts?

### 2. Answer Extraction

How to extract the "answer" from Dio's response for grading:
- Regex pattern?
- JSON field?
- Exact match?

### 3. Correctness Check

How to determine if Dio's answer matches expected:
- Exact string match?
- Set comparison (order doesn't matter)?
- Numeric tolerance?

---

## Design Principle: Level = Unlocks

**Level is the ONLY scale.** Each level UNLOCKS new capabilities:

```
L1: add, subtract (tiny numbers)
L2: + multiply
L3: + larger numbers
L4: + division
L5: + chained operations
L6: + signed numbers
L7: + comparisons
L8: + symbol language (new notation!)
...
L30: everything unlocked, max complexity
```

### What Each Level Unlocks

| Level | Unlocks |
|-------|---------|
| 1 | Core operations, smallest inputs |
| N | New operation OR larger input range OR new format |
| Max | Everything available, mastery challenge |

### Unlock Types

1. **New Operations** - `multiply` unlocked at L2
2. **Input Range** - L3 handles bigger numbers than L2
3. **Output Complexity** - L5 shows work, L3 just shows answer
4. **Special Features** - Symbol language at L8, signed numbers at L6

### Why Unlocks?

1. **RPG Feel** - "Level up" means gaining new abilities
2. **Natural Progression** - Master basics before advanced
3. **Clear Milestones** - "At L8 you get symbols"
4. **No Confusion** - L5 problem is ALWAYS L5 difficulty

### Tags Still Useful For

Tags describe WHAT, not difficulty:
- Operation type: `add`, `subtract`, `multiply`
- Format: `verify`, `trace`, `chain`
- Content: `signed`, `unsigned`

But NOT: `easy`, `medium`, `hard`, `expert` (these are now implicit in level)

---

## Step-by-Step Integration Guide

When adding a new Trainer, follow these steps in order:

---

### Step 1: Create API Server (in singleSKILL)

Location: `/path/to/skills/skill_{name}/api_server.py`

```python
# Minimum viable API server
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/info')
def info():
    return jsonify({
        "skill": "myskill",
        "name": "My Skill Trainer",
        "version": "1.0.0",
        "levels": 10,
        "features": ["operation1", "operation2"]
    })

@app.route('/levels')
def levels():
    return jsonify({
        "levels": [
            {"level": 1, "name": "Beginner", "threshold": 0.80, "description": "..."},
            # ... more levels
        ]
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    level = data.get('level', 1)
    count = data.get('count', 10)
    samples = generate_samples(level, count)  # Your implementation
    return jsonify({
        "count": len(samples),
        "level": level,
        "level_info": get_level_info(level),
        "samples": samples
    })

if __name__ == '__main__':
    app.run(port=80XX)
```

- [ ] `/health` returns `{"status": "ok"}`
- [ ] `/info` returns skill metadata
- [ ] `/levels` returns level configs with thresholds
- [ ] `/generate` returns samples array

---

### Step 2: Add to Quest Master (data_manager/skill_api_client.py)

Location: `data_manager/skill_api_client.py` line ~31

```python
SKILL_APIS = {
    # ... existing skills ...
    "myskill": {
        "name": "My Skill",
        "base_url": "http://127.0.0.1:80XX",
        "levels": 10,
        "server_script": "/path/to/skills/skill_myskill/api_server.py",
    }
}
```

Add conversion function (same file, ~line 150):

```python
# Generator ID for lineage tracking
GENERATOR_ID = "myskill_api"
GENERATOR_VERSION = "1.0.0"

def myskill_to_training_format(sample: Dict[str, Any], level_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convert myskill sample to training format."""
    return {
        "messages": [
            {"role": "user", "content": sample["user_prompt"]},
            {"role": "assistant", "content": sample["assistant_response"]}
        ],
        "metadata": {
            "source": "myskill_api",
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "skill": "myskill",
            "sample_id": sample.get("id"),
            "tags": sample.get("tags", []),
            "level": level_info.get("level") if level_info else None,
        }
    }
```

Update `_convert_to_training_format()` method to call your converter.

- [ ] Added to `SKILL_APIS` dict
- [ ] Created `{skill}_to_training_format()` function
- [ ] Converter called in `_convert_to_training_format()`

---

### Step 3: Add to Curriculum Manager (data_manager/curriculum_manager.py)

Location: `data_manager/curriculum_manager.py` line ~27

```python
SKILL_LEVELS = {
    # ... existing skills ...
    "myskill": {
        "name": "My Skill",
        "total_levels": 10,
        "api_port": 80XX,
        "levels": [
            {"level": 1, "name": "Beginner", "threshold": 0.80, "my_param": "..."},
            {"level": 2, "name": "Easy", "threshold": 0.80, "my_param": "..."},
            # ... up to max level
            {"level": 10, "name": "Master", "threshold": None},  # Mastery level
        ]
    }
}
```

- [ ] Added skill to `SKILL_LEVELS` dict
- [ ] All levels defined with thresholds
- [ ] Final level has `threshold: None`

---

### Step 4: Add to Lineage Registry (core/lineage.py)

Location: `core/lineage.py` line ~20

```python
GENERATOR_REGISTRY = {
    # ... existing generators ...
    "myskill_api": {
        "version": "1.0.0",
        "source": "singleSKILL myskill API"
    },
}
```

- [ ] Added generator to `GENERATOR_REGISTRY`

---

### Step 5: Add Arena Evaluator (monitoring/skill_evaluators.py)

Location: `monitoring/skill_evaluators.py`

```python
class MySkillEvaluator(SkillEvaluator):
    """Evaluator for MySkill."""

    skill_name = "myskill"

    def generate_problems(self, level: int, count: int) -> List[Dict]:
        """Generate problems from MySkill API."""
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={"level": level, "count": count},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("samples", [])
        except Exception as e:
            logger.error(f"Failed to generate myskill problems: {e}")
            return []

    def get_prompt(self, problem: Dict) -> str:
        """Get prompt from sample."""
        return problem.get("user_prompt", "")

    def get_expected(self, problem: Dict) -> Any:
        """Get expected answer."""
        return problem.get("assistant_response", "")

    def extract_answer(self, response: str) -> Any:
        """Extract answer from model's response."""
        # Clean up response, extract the relevant part
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return response.strip()

    def check_correct(self, expected: Any, actual: Any) -> Tuple[bool, float]:
        """Check if answer is correct."""
        is_correct = expected == actual  # Customize for your skill
        partial = 1.0 if is_correct else 0.0
        return is_correct, partial
```

Add to registry (same file, line ~256):

```python
SKILL_EVALUATORS = {
    # ... existing evaluators ...
    "myskill": {
        "class": MySkillEvaluator,
        "default_url": "http://localhost:80XX",
        "description": "My Skill training",
    },
}
```

- [ ] Created `MySkillEvaluator` class
- [ ] Implemented all 5 abstract methods
- [ ] Added to `SKILL_EVALUATORS` registry

---

### Step 6: Create Guild Skill Config (configs/skills/{skill}.yaml)

Location: `configs/skills/myskill.yaml`

```yaml
id: myskill
name: My Skill
description: >
  Description of what this skill teaches Dio.
category: reasoning  # or: math, code, compression, etc.

tags:
  - tag1
  - tag2

metrics:
  - accuracy
primary_metric: accuracy

accuracy_thresholds:
  1: 0.60
  2: 0.65
  3: 0.70
  # ... up to max level

xp_multiplier: 1.0

rpg_name: My Skill Name
rpg_description: >
  RPG-flavored description for the Guild Hall.
```

- [ ] Created `configs/skills/{skill}.yaml`
- [ ] All levels have accuracy thresholds

---

### Step 7: Update Documentation

- [ ] Add Trainer to "Registered Trainers" section below
- [ ] Update `GUILD_VOCABULARY.md` if needed

---

### Quick Verification

After integration, verify:

```bash
# 1. API is running
curl http://localhost:80XX/health

# 2. Can generate samples
curl -X POST http://localhost:80XX/generate \
  -H "Content-Type: application/json" \
  -d '{"level": 1, "count": 5}'

# 3. Quest Master recognizes skill
python3 -c "from data_manager.skill_api_client import SKILL_APIS; print('myskill' in SKILL_APIS)"

# 4. Evaluator loads
python3 -c "from monitoring.skill_evaluators import get_evaluator; e = get_evaluator('myskill'); print(e.skill_name)"

# 5. Guild config loads
python3 -c "from guild.skills.registry import get_skill; s = get_skill('myskill'); print(s.name)"
```

---

## Registration Checklist (Summary)

| Step | File | What to Add |
|------|------|-------------|
| 1 | `singleSKILL/skill_{name}/api_server.py` | API server with 4 endpoints |
| 2 | `data_manager/skill_api_client.py` | `SKILL_APIS` entry + converter |
| 3 | `data_manager/curriculum_manager.py` | `SKILL_LEVELS` entry |
| 4 | `core/lineage.py` | `GENERATOR_REGISTRY` entry |
| 5 | `monitoring/skill_evaluators.py` | Evaluator class + registry entry |
| 6 | `configs/skills/{skill}.yaml` | Guild skill config |
| 7 | `TRAINER_CONTRACT.md` | Registered Trainers entry |

---

## Registered Trainers

### Sy (Syllable Trainer)
- **Skill:** `syllo`
- **Port:** 8080
- **Levels:** 1-10 (word count progression)
- **Answer Format:** JSON with solutions
- **Status:** ⚠️ Evaluation format mismatch (needs fix)

### Bin (Binary Trainer)
- **Skill:** `binary`
- **Port:** 8090
- **Levels:** 1-30 (bit-width progression)
- **Answer Format:** Circled binary notation (⓪①)
- **Status:** ✅ Fully integrated

---

## Future Trainers (Template)

### [Name] ([Skill] Trainer)
- **Skill:** `skill_name`
- **Port:** 80XX
- **Levels:** 1-N
- **Answer Format:** ...
- **Status:** Planning/Development/Integrated

---

*This contract ensures all Trainers work seamlessly with the Quest Master and maintain proper data lineage.*
