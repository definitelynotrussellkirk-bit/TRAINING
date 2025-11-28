# Skill Engine Implementation Plan

**Created:** 2025-11-28
**Status:** Draft - Pending Approval
**Scope:** Scalable skill system (40+ skills, extensible to unbounded)

---

## 1. Executive Summary

### Problem
The current system has four separate domains that don't share a unified runtime object:
- **Skill configs** (`configs/skills/*.yaml`) - Metadata and API config
- **Generators** (singleSKILL APIs) - Training data generation
- **Guild** (task_master, sparring) - Quest dispatch and skill tracking
- **Passives** (`guild/passives/`) - Eval problem generation and answer checking

### Solution
Introduce a **Skill Engine** with a unified `Skill` abstraction that can:
1. Generate training examples at level L
2. Generate eval tasks at level L
3. Score eval results
4. Decide XP/leveling

Everything else (Guild, DataManager, Monitoring, UI) talks to `Skill` objects.

### Key Benefits
- **Single interface** for all skill operations
- **Primitive-level tracking** for ultra-specific concept mastery
- **Pluggable adapters** connect existing generators + passives
- **Scalable** - add skills via YAML + API + passive, no code changes to orchestration

---

## 2. Current State Analysis

### Existing Components

| Component | Location | What It Does |
|-----------|----------|--------------|
| `SkillConfig` | `guild/skills/types.py:67-138` | Metadata from YAML (id, name, api, eval, levels, thresholds) |
| `SkillState` | `guild/skills/types.py:140-218` | Runtime state (level, xp, rolling accuracy, trial state) |
| `SkillRegistry` | `guild/skills/registry.py` | Lookup skills by ID/category/tag |
| `SkillClient` | `guild/skills/contract.py:150-305` | HTTP client for skill APIs (`/generate`, `/info`, `/levels`) |
| `PassiveModule` | `guild/passives/base.py` | ABC for eval modules (`generate_problems`, `check_answer`) |
| Concrete passives | `guild/passives/{arithmetic,logic,counting,string_craft}.py` | Eval implementations |

### What's Missing

1. **Unified Skill object** - No single class that owns training + eval + scoring
2. **Primitive tracking** - No concept of atomic testable units within skills
3. **Scoring integration** - `check_answer` and XP/leveling are disconnected
4. **Adapter layer** - No clean bridge between APIs and passives

### Current Flow (Fragmented)
```
[YAML Config] → SkillRegistry → SkillConfig
                                    ↓
                          SkillClient (for training via API)

[PassiveModule] → generate_problems → check_answer (disconnected from above)
```

### Target Flow (Unified)
```
[YAML Config] → SkillRegistry → Skill object
                                    ↓
                     ┌──────────────┴──────────────┐
                     ↓                             ↓
          generate_training_batch          generate_eval_batch
          (via GeneratorAdapter)           (via PassiveAdapter)
                                                   ↓
                                            score_eval
                                                   ↓
                                        update_state_from_eval
```

---

## 3. Domain Model

### 3.1 PrimitiveId (NEW)

**Purpose:** Atomic testable concept - the "ONE idea" eval.

```python
# guild/skills/primitives.py

@dataclass(frozen=True)
class PrimitiveId:
    """Atomic testable concept inside a skill."""
    name: str        # "add_single_digit_no_carry", "modus_ponens"
    track: str       # "arithmetic", "logic", "binary"
    version: str     # "v1" - bump when definition changes

    def __str__(self) -> str:
        return f"{self.track}/{self.name}@{self.version}"
```

### 3.2 Primitive Metadata (NEW)

```python
@dataclass
class PrimitiveMeta:
    """Metadata for a primitive."""
    id: PrimitiveId
    display_name: str          # "Single-Digit Addition (No Carry)"
    description: str           # One sentence explaining what this tests
    difficulty: int            # 1-10 scale
    prerequisites: list[str]   # Other primitive names that should be mastered first
    tags: list[str]            # ["foundation", "arithmetic", "add"]
```

### 3.3 EvalBatch and EvalResult (NEW)

```python
# guild/skills/eval_types.py

@dataclass
class EvalProblem:
    """Single evaluation problem."""
    prompt: str
    expected: str
    primitive_id: str | None = None  # Which primitive this tests
    metadata: dict = field(default_factory=dict)

@dataclass
class EvalBatch:
    """Collection of problems for evaluation."""
    skill_id: str
    level: int
    problems: list[EvalProblem]
    metadata: dict = field(default_factory=dict)

@dataclass
class EvalResultItem:
    """Result for a single problem."""
    problem: EvalProblem
    model_answer: str
    is_correct: bool
    primitive_id: str | None

@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    accuracy: float
    per_primitive_accuracy: dict[str, float]  # primitive_name -> accuracy
    num_examples: int
    items: list[EvalResultItem]

    @property
    def passed(self) -> bool:
        """Did this eval pass the threshold? (caller must provide threshold)"""
        return False  # Determined by caller with skill config
```

### 3.4 Skill ABC (NEW)

**The core abstraction:**

```python
# guild/skills/skill.py

from abc import ABC, abstractmethod

class Skill(ABC):
    """
    Runtime object that knows how to train & evaluate a skill.

    This is THE interface for all skill operations:
    - Training data generation
    - Eval problem generation
    - Answer scoring
    - State updates (XP, leveling)
    """

    def __init__(self, config: SkillConfig):
        self.config = config

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def primitives(self) -> list[PrimitiveId]:
        """List of primitives this skill covers."""
        return []  # Override in subclasses

    @abstractmethod
    def generate_training_batch(self, *, level: int, count: int, seed: int | None = None) -> list[dict]:
        """
        Generate training examples at given level.

        Returns list of dicts in inbox format:
        [{"messages": [...], "metadata": {...}}, ...]
        """
        ...

    @abstractmethod
    def generate_eval_batch(self, *, level: int, count: int, seed: int | None = None) -> EvalBatch:
        """Generate evaluation problems at given level."""
        ...

    @abstractmethod
    def score_eval(self, batch: EvalBatch, model_answers: list[str]) -> EvalResult:
        """
        Score model answers against expected.

        Args:
            batch: The eval batch that was given to model
            model_answers: Model's responses (same order as batch.problems)

        Returns:
            EvalResult with accuracy and per-primitive breakdown
        """
        ...

    def update_state_from_eval(self, state: SkillState, result: EvalResult) -> SkillState:
        """
        Update skill state based on eval results.

        Default policy:
        - Add XP based on accuracy
        - Level up if threshold met
        - Track per-primitive accuracy in state

        Can be overridden for skill-specific leveling policies.
        """
        # Update stats
        state.total_evals += 1
        state.total_samples_seen += result.num_examples
        state.last_eval_accuracy = result.accuracy

        # XP gain: xp_per_eval * accuracy
        xp_per_eval = self.config.xp_multiplier * 10  # Base 10 XP per eval
        gained = int(xp_per_eval * result.accuracy)
        state.xp_total += gained

        # Record result for rolling accuracy
        passed = result.accuracy >= self.config.get_threshold(state.level)
        state.record_result(passed)

        # Level up check
        if passed and state.accuracy >= self.config.get_threshold(state.level):
            if state.level < self.config.max_level:
                state.record_level_up()

        # Store per-primitive accuracy
        state.primitive_accuracy = result.per_primitive_accuracy

        return state
```

### 3.5 Extended SkillState

Add to existing `SkillState` in `guild/skills/types.py`:

```python
@dataclass
class SkillState(SerializableMixin):
    # ... existing fields ...

    # NEW: Per-primitive tracking
    primitive_accuracy: dict[str, float] = field(default_factory=dict)
    primitive_history: dict[str, list[bool]] = field(default_factory=dict)

    # NEW: Eval tracking
    total_evals: int = 0
    total_samples_seen: int = 0
    last_eval_accuracy: float | None = None
    last_eval_timestamp: float | None = None
```

---

## 4. Adapters

### 4.1 GeneratorAdapter

Connects existing skill APIs (singleSKILL) to the Skill interface.

```python
# guild/skills/adapters/generator.py

class GeneratorAdapter:
    """
    Adapts skill API (singleSKILL) to training generation interface.

    Uses SkillClient internally to call /generate endpoint.
    """

    def __init__(self, skill_id: str, api_url: str):
        self.skill_id = skill_id
        self.client = SkillClient(skill_id, api_url)

    def generate_training_batch(
        self,
        level: int,
        count: int,
        seed: int | None = None
    ) -> list[dict]:
        """Generate training samples via API."""
        batch = self.client.sample(level=level, count=count, seed=seed)

        # Convert to inbox format
        return [
            {
                "messages": sample.messages,
                "metadata": sample.metadata,
            }
            for sample in batch.samples
        ]

    def health(self) -> bool:
        """Check if API is available."""
        return self.client.health()
```

### 4.2 PassiveAdapter

Connects existing passives to the Skill interface.

```python
# guild/skills/adapters/passive.py

class PassiveAdapter:
    """
    Adapts PassiveModule to eval generation/scoring interface.
    """

    def __init__(self, passive: PassiveModule):
        self.passive = passive

    def generate_eval_batch(
        self,
        skill_id: str,
        level: int,
        count: int,
        seed: int | None = None
    ) -> EvalBatch:
        """Generate eval problems via passive."""
        problems_raw = self.passive.generate_problems(count=count, seed=seed)

        problems = [
            EvalProblem(
                prompt=p["prompt"],
                expected=p["expected"],
                primitive_id=p.get("type") or p.get("primitive_id"),
                metadata={k: v for k, v in p.items()
                         if k not in ("prompt", "expected", "type", "primitive_id")}
            )
            for p in problems_raw
        ]

        return EvalBatch(
            skill_id=skill_id,
            level=level,
            problems=problems,
            metadata={"passive_id": self.passive.id, "version": self.passive.version}
        )

    def score_eval(self, batch: EvalBatch, model_answers: list[str]) -> EvalResult:
        """Score model answers using passive.check_answer."""
        items = []
        per_primitive: dict[str, list[bool]] = {}

        for problem, answer in zip(batch.problems, model_answers):
            is_correct = self.passive.check_answer(problem.expected, answer)

            items.append(EvalResultItem(
                problem=problem,
                model_answer=answer,
                is_correct=is_correct,
                primitive_id=problem.primitive_id,
            ))

            # Track per-primitive
            prim = problem.primitive_id or "unknown"
            per_primitive.setdefault(prim, []).append(is_correct)

        # Calculate accuracies
        total_correct = sum(1 for item in items if item.is_correct)
        accuracy = total_correct / len(items) if items else 0.0

        per_primitive_accuracy = {
            prim: sum(vals) / len(vals)
            for prim, vals in per_primitive.items()
        }

        return EvalResult(
            accuracy=accuracy,
            per_primitive_accuracy=per_primitive_accuracy,
            num_examples=len(items),
            items=items,
        )
```

### 4.3 CompositeSkill

The concrete implementation that combines generator + passive:

```python
# guild/skills/composite.py

class CompositeSkill(Skill):
    """
    Skill implementation using Generator (for training) + Passive (for eval).

    This is the standard skill type - most skills will use this.
    """

    def __init__(
        self,
        config: SkillConfig,
        generator: GeneratorAdapter,
        passive: PassiveAdapter,
        primitives: list[PrimitiveId] | None = None,
    ):
        super().__init__(config)
        self.generator = generator
        self.passive = passive
        self._primitives = primitives or []

    @property
    def primitives(self) -> list[PrimitiveId]:
        return self._primitives

    def generate_training_batch(
        self, *,
        level: int,
        count: int,
        seed: int | None = None
    ) -> list[dict]:
        return self.generator.generate_training_batch(level, count, seed)

    def generate_eval_batch(
        self, *,
        level: int,
        count: int,
        seed: int | None = None
    ) -> EvalBatch:
        return self.passive.generate_eval_batch(
            skill_id=self.id,
            level=level,
            count=count,
            seed=seed,
        )

    def score_eval(self, batch: EvalBatch, model_answers: list[str]) -> EvalResult:
        return self.passive.score_eval(batch, model_answers)
```

---

## 5. Registry and Loading

### 5.1 SkillEngine (NEW)

Central manager for skill lifecycle:

```python
# guild/skills/engine.py

class SkillEngine:
    """
    Central Skill Engine - manages skill lifecycle.

    Responsibilities:
    - Load skills from YAML + wire adapters
    - Provide unified access to skills
    - Track skill states
    - Coordinate eval/leveling
    """

    def __init__(
        self,
        config_dir: Path | None = None,
        state_file: Path | None = None,
    ):
        self.config_dir = config_dir or get_config_dir()
        self.state_file = state_file or Path("status/skill_states.json")

        self._skills: dict[str, Skill] = {}
        self._states: dict[str, SkillState] = {}
        self._passive_registry: dict[str, PassiveModule] = {}

        self._load_passives()
        self._load_states()

    def _load_passives(self):
        """Auto-discover and register passives."""
        from guild.passives import get_all_passives
        for passive in get_all_passives():
            self._passive_registry[passive.id] = passive

    def _load_states(self):
        """Load persisted skill states."""
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            for skill_id, state_dict in data.items():
                self._states[skill_id] = SkillState.from_dict(state_dict)

    def _save_states(self):
        """Persist skill states."""
        data = {
            skill_id: state.to_dict()
            for skill_id, state in self._states.items()
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(data, indent=2))

    def get(self, skill_id: str) -> Skill:
        """Get a skill, loading if needed."""
        if skill_id not in self._skills:
            self._load_skill(skill_id)
        return self._skills[skill_id]

    def _load_skill(self, skill_id: str):
        """Load and wire up a skill."""
        # Load config from YAML
        config = load_skill_config(skill_id, self.config_dir)

        # Find matching passive
        passive_id = config.eval.passive_id if hasattr(config.eval, 'passive_id') else skill_id
        passive = self._passive_registry.get(passive_id)

        if not passive:
            # Try to find by category match
            for p in self._passive_registry.values():
                if p.category == config.category.value:
                    passive = p
                    break

        # Create adapters
        generator = None
        if config.api_url:
            generator = GeneratorAdapter(skill_id, config.api_url)

        passive_adapter = PassiveAdapter(passive) if passive else None

        # Create skill
        if generator and passive_adapter:
            skill = CompositeSkill(config, generator, passive_adapter)
        elif generator:
            skill = GeneratorOnlySkill(config, generator)
        elif passive_adapter:
            skill = PassiveOnlySkill(config, passive_adapter)
        else:
            raise ValueError(f"Skill {skill_id} has no generator or passive")

        self._skills[skill_id] = skill

    def get_state(self, skill_id: str) -> SkillState:
        """Get current state for a skill."""
        if skill_id not in self._states:
            self._states[skill_id] = SkillState(skill_id=skill_id)
        return self._states[skill_id]

    def run_eval(
        self,
        skill_id: str,
        model_answers: list[str],
        level: int | None = None,
        count: int = 5,
    ) -> tuple[EvalResult, SkillState]:
        """
        Run a full eval cycle: generate problems, score, update state.

        Returns (result, updated_state).
        """
        skill = self.get(skill_id)
        state = self.get_state(skill_id)

        if level is None:
            level = state.level

        # Generate and score
        batch = skill.generate_eval_batch(level=level, count=count)
        result = skill.score_eval(batch, model_answers)

        # Update state
        state = skill.update_state_from_eval(state, result)
        self._states[skill_id] = state
        self._save_states()

        return result, state

    def list_skills(self) -> list[str]:
        """List all available skill IDs."""
        return discover_skills(self.config_dir)

    def all_skills(self) -> dict[str, Skill]:
        """Load and return all skills."""
        for skill_id in self.list_skills():
            if skill_id not in self._skills:
                try:
                    self._load_skill(skill_id)
                except Exception as e:
                    logger.warning(f"Failed to load skill {skill_id}: {e}")
        return self._skills.copy()

# Global engine
_engine: SkillEngine | None = None

def get_engine() -> SkillEngine:
    """Get global skill engine."""
    global _engine
    if _engine is None:
        _engine = SkillEngine()
    return _engine
```

---

## 6. YAML Schema Updates

### 6.1 Add Primitives to Skill Config

Update `configs/skills/*.yaml` to include primitives:

```yaml
# configs/skills/bin.yaml (additions)

# NEW: Passive mapping (which passive to use for evals)
eval:
  passive_id: "binary_arithmetic"  # References guild/passives/binary_arithmetic.py
  # ... existing eval config ...

# NEW: Primitives this skill covers
primitives:
  - name: binary_add_no_carry
    track: binary
    version: v1
    display_name: "Binary Addition (No Carry)"
    description: "Add two binary numbers where no column requires carrying"
    difficulty: 1

  - name: binary_add_with_carry
    track: binary
    version: v1
    display_name: "Binary Addition (With Carry)"
    description: "Add two binary numbers with carrying between columns"
    difficulty: 2
    prerequisites: [binary_add_no_carry]

  - name: binary_sub_no_borrow
    track: binary
    version: v1
    display_name: "Binary Subtraction (No Borrow)"
    difficulty: 2

  - name: binary_sub_with_borrow
    track: binary
    version: v1
    display_name: "Binary Subtraction (With Borrow)"
    difficulty: 3
    prerequisites: [binary_sub_no_borrow]

  - name: binary_compare_magnitude
    track: binary
    version: v1
    display_name: "Compare Binary Numbers"
    difficulty: 2

  - name: bitwise_and
    track: binary
    version: v1
    display_name: "Bitwise AND"
    difficulty: 2

  - name: bitwise_or
    track: binary
    version: v1
    display_name: "Bitwise OR"
    difficulty: 2

  - name: bitwise_xor
    track: binary
    version: v1
    display_name: "Bitwise XOR"
    difficulty: 3
```

### 6.2 Template Update

Update `configs/skills/_template.yaml`:

```yaml
# NEW SECTION: Evaluation passive
eval:
  passive_id: "${id}"  # Usually matches skill id
  samples_per_level: 5
  endpoint: "/eval"
  local_cache: "data/validation/${id}/"

# NEW SECTION: Primitives (atomic testable concepts)
primitives:
  - name: primitive_name_here
    track: category_name
    version: v1
    display_name: "Human Readable Name"
    description: "One sentence explaining what this tests"
    difficulty: 1  # 1-10
    prerequisites: []  # Other primitive names
    tags: []
```

---

## 7. Passive Updates

### 7.1 Add Primitive Labels to Passives

Update passives to label which primitive each problem tests:

```python
# guild/passives/arithmetic.py (updated)

def _digit_sum(self) -> Dict[str, Any]:
    num = random.randint(10, 9999)
    answer = sum(int(d) for d in str(num))
    return {
        "prompt": f"What is the sum of the digits in {num}?",
        "expected": str(answer),
        "primitive_id": "digit_sum",  # NEW: primitive label
        "type": "digit_sum",
    }

def _basic_ops(self) -> Dict[str, Any]:
    a = random.randint(1, 50)
    b = random.randint(1, 50)
    op = random.choice(['+', '-', '*'])

    # NEW: Specific primitive based on operation
    primitive_map = {
        '+': 'add_two_digit',
        '-': 'sub_two_digit',
        '*': 'mul_single_digit',
    }

    if op == '+':
        answer = a + b
    elif op == '-':
        answer = a - b
    else:
        answer = a * b

    return {
        "prompt": f"What is {a} {op} {b}?",
        "expected": str(answer),
        "primitive_id": primitive_map[op],  # NEW
        "type": "basic_ops",
    }
```

### 7.2 New Binary Arithmetic Passive

Create dedicated passive for BIN skill:

```python
# guild/passives/binary_arithmetic.py

class BinaryArithmeticPassive(PassiveModule):
    """Binary arithmetic eval - matches BIN skill."""

    id = "binary_arithmetic"
    name = "Binary Arithmetic"
    category = "math"
    description = "Binary addition, subtraction, bitwise operations"
    version = "1.0.0"

    def generate_problems(self, count: int, seed: int | None = None) -> list[dict]:
        if seed is not None:
            random.seed(seed)

        problems = []
        generators = [
            self._binary_add_no_carry,
            self._binary_add_with_carry,
            self._binary_sub,
            self._bitwise_and,
            self._bitwise_or,
            self._binary_compare,
        ]

        for i in range(count):
            gen = generators[i % len(generators)]
            problems.append(gen())

        return problems

    def _binary_add_no_carry(self) -> dict:
        # Generate numbers that don't require carry
        a = random.randint(1, 5)   # Small, sparse bits
        b = random.randint(1, 5)
        # Ensure no carry by checking bit overlap
        while a & b != 0:  # If any bits overlap, there'd be carry
            b = random.randint(1, 5)

        result = a + b
        return {
            "prompt": f"What is {bin(a)} + {bin(b)} in binary?",
            "expected": bin(result),
            "primitive_id": "binary_add_no_carry",
        }

    # ... more generators ...

    def check_answer(self, expected: str, got: str) -> bool:
        # Normalize binary representations
        expected_norm = expected.lower().replace('0b', '').lstrip('0') or '0'
        got_norm = got.lower().replace('0b', '').lstrip('0') or '0'

        # Also check if decimal equivalent appears
        try:
            expected_int = int(expected_norm, 2)
            if str(expected_int) in got:
                return True
        except ValueError:
            pass

        return expected_norm in got_norm or expected_norm == got_norm
```

---

## 8. Primitives Catalog

### 8.1 Arithmetic & Numeracy (15 primitives)

| ID | Display Name | Difficulty | Description |
|----|--------------|------------|-------------|
| `add_single_digit_no_carry` | Single Digit Add (No Carry) | 1 | 3 + 5 |
| `add_single_digit_with_carry` | Single Digit Add (Carry) | 2 | 7 + 8 |
| `add_two_digit_no_carry` | Two Digit Add (No Carry) | 2 | 12 + 24 |
| `add_two_digit_with_carry` | Two Digit Add (Carry) | 3 | 27 + 58 |
| `sub_single_digit_no_borrow` | Single Digit Sub (No Borrow) | 1 | 9 - 3 |
| `sub_single_digit_with_borrow` | Single Digit Sub (Borrow) | 2 | 12 - 9 |
| `sub_two_digit` | Two Digit Subtraction | 3 | 45 - 28 |
| `mul_single_digit` | Single Digit Multiply | 2 | 7 x 8 |
| `mul_by_10_100` | Multiply by 10/100 | 2 | 34 x 10 |
| `div_exact_single` | Exact Division | 2 | 56 / 7 |
| `compare_integers` | Compare Two Integers | 1 | Which is larger: 47 or 83? |
| `round_to_integer` | Round to Nearest | 2 | round(3.6) |
| `fraction_simplify` | Simplify Fraction | 3 | 6/8 -> 3/4 |
| `percent_of_number` | Percentage | 3 | 25% of 80 |
| `parity_even_odd` | Even or Odd | 1 | Is 37 odd or even? |

### 8.2 Binary & Bitwise (13 primitives)

| ID | Display Name | Difficulty | Description |
|----|--------------|------------|-------------|
| `binary_add_no_carry` | Binary Add (No Carry) | 2 | 0101 + 0010 |
| `binary_add_with_carry` | Binary Add (Carry) | 3 | 0111 + 0001 |
| `binary_sub_no_borrow` | Binary Sub (No Borrow) | 2 | 1010 - 0010 |
| `binary_sub_with_borrow` | Binary Sub (Borrow) | 3 | 1000 - 0011 |
| `bitwise_and` | Bitwise AND | 2 | 1010 AND 1100 |
| `bitwise_or` | Bitwise OR | 2 | 1010 OR 1100 |
| `bitwise_xor` | Bitwise XOR | 3 | 1010 XOR 1100 |
| `bitwise_not` | Bitwise NOT | 2 | NOT 1010 |
| `binary_left_shift` | Left Shift | 3 | 0011 << 1 |
| `binary_right_shift` | Right Shift | 3 | 1100 >> 1 |
| `binary_compare` | Compare Binary | 2 | Which is greater: 0101 or 0110? |
| `binary_to_decimal` | Binary to Decimal | 2 | 1010 = ? |
| `decimal_to_binary` | Decimal to Binary | 3 | 10 = ? in binary |

### 8.3 Logic (8 primitives)

| ID | Display Name | Difficulty | Description |
|----|--------------|------------|-------------|
| `truth_table_basic` | Truth Table (AND/OR/NOT) | 1 | P AND Q given P=T, Q=F |
| `implication_eval` | Implication Evaluation | 2 | P -> Q given values |
| `modus_ponens` | Modus Ponens | 2 | If P then Q. P is true. Conclude? |
| `modus_tollens` | Modus Tollens | 3 | If P then Q. Q is false. Conclude? |
| `syllogism_all` | Universal Syllogism | 2 | All A are B. X is A. What is X? |
| `syllogism_some` | Existential Syllogism | 3 | Some A are B. What can be concluded? |
| `de_morgan` | De Morgan's Laws | 4 | NOT(P AND Q) = ? |
| `xor_vs_or` | XOR vs OR | 3 | Distinguish exclusive/inclusive or |

### 8.4 String & Pattern (6 primitives)

| ID | Display Name | Difficulty | Description |
|----|--------------|------------|-------------|
| `reverse_string` | Reverse String | 1 | Reverse "cat" |
| `count_char` | Count Character | 1 | How many 'a' in "banana"? |
| `substring_check` | Substring Check | 2 | Is "ana" in "banana"? |
| `sort_letters` | Sort Letters | 2 | Sort letters of "cab" |
| `palindrome_check` | Palindrome Check | 2 | Is "level" a palindrome? |
| `string_length` | String Length | 1 | Length of "hello" |

### 8.5 Code Trace (5 primitives)

| ID | Display Name | Difficulty | Description |
|----|--------------|------------|-------------|
| `trace_assignment` | Variable Assignment | 1 | x = 3; y = x + 2; y = ? |
| `trace_if_else` | If-Else Trace | 2 | Single branch, no nesting |
| `trace_loop_fixed` | Fixed Loop Trace | 3 | for i in [1,2,3]: sum += i |
| `eval_bool_expr` | Boolean Expression | 2 | (True and False) or True |
| `function_call_trace` | Function Call | 3 | def f(x): return x+1; f(2) = ? |

### 8.6 Language (6 primitives)

| ID | Display Name | Difficulty | Description |
|----|--------------|------------|-------------|
| `synonym_easy` | Simple Synonym | 1 | Synonym of "big" |
| `antonym_easy` | Simple Antonym | 1 | Antonym of "hot" |
| `pluralize_regular` | Regular Plural | 1 | Plural of "cat" |
| `identify_subject` | Identify Subject | 2 | Subject of "The dog runs" |
| `pronoun_resolution` | Pronoun Resolution | 3 | Who does "she" refer to? |
| `sentence_type` | Sentence Classification | 2 | Question or statement? |

### 8.7 Counting & Combinatorics (5 primitives)

| ID | Display Name | Difficulty | Description |
|----|--------------|------------|-------------|
| `count_objects` | Count Objects | 1 | How many apples? |
| `count_even_in_range` | Count Even Numbers | 2 | Even numbers 1-10? |
| `permutation_tiny` | Simple Permutation | 3 | Ways to arrange ABC? |
| `combination_tiny` | Simple Combination | 3 | 2-element subsets of {A,B,C}? |
| `divisibility` | Divisibility Check | 2 | Is 35 divisible by 5? |

### 8.8 Meta / Calibration (2 primitives)

| ID | Display Name | Difficulty | Description |
|----|--------------|------------|-------------|
| `confidence_easy` | Confidence Check | 2 | Answer + confidence for easy Q |
| `refusal_insufficient` | Appropriate Refusal | 3 | Refuse when info insufficient |

---

## 9. Implementation Phases

### Phase 1: Core Types (1-2 days)

**Files to create:**
- `guild/skills/primitives.py` - PrimitiveId, PrimitiveMeta
- `guild/skills/eval_types.py` - EvalProblem, EvalBatch, EvalResult
- `guild/skills/skill.py` - Skill ABC

**Files to update:**
- `guild/skills/types.py` - Add primitive_accuracy to SkillState

**Deliverable:** Core type definitions that compile and can be imported.

### Phase 2: Adapters (1-2 days)

**Files to create:**
- `guild/skills/adapters/__init__.py`
- `guild/skills/adapters/generator.py` - GeneratorAdapter
- `guild/skills/adapters/passive.py` - PassiveAdapter
- `guild/skills/composite.py` - CompositeSkill

**Deliverable:** Adapters that can wrap existing SkillClient and PassiveModule.

### Phase 3: Engine (2-3 days)

**Files to create:**
- `guild/skills/engine.py` - SkillEngine

**Files to update:**
- `guild/skills/__init__.py` - Export engine
- `configs/skills/*.yaml` - Add passive_id to eval section

**Deliverable:** Working SkillEngine that can load skills and run evals.

### Phase 4: Primitive Labels (1-2 days)

**Files to update:**
- `guild/passives/arithmetic.py` - Add primitive_id to problems
- `guild/passives/logic.py` - Add primitive_id to problems
- `guild/passives/counting.py` - Add primitive_id to problems
- `guild/passives/string_craft.py` - Add primitive_id to problems

**Files to create:**
- `guild/passives/binary_arithmetic.py` - New passive for BIN

**Deliverable:** All passives label problems with primitive IDs.

### Phase 5: YAML Primitives (1 day)

**Files to update:**
- `configs/skills/bin.yaml` - Add primitives section
- `configs/skills/sy.yaml` - Add primitives section
- `configs/skills/_template.yaml` - Document primitives

**Deliverable:** YAML configs include primitive definitions.

### Phase 6: Integration (2-3 days)

**Files to update:**
- `guild/task_master.py` - Use SkillEngine for evals
- `tavern/server.py` - Add /api/skills/primitives endpoint
- `monitoring/` - Add per-primitive accuracy display

**Deliverable:** Full integration with existing systems.

### Phase 7: Expansion (ongoing)

**New passives to create (one per skill category):**
- `guild/passives/code_trace.py`
- `guild/passives/language_basics.py`
- `guild/passives/combinatorics.py`

**New skill YAMLs:**
- As skills are added in singleSKILL

---

## 10. Testing Strategy

### Unit Tests

```python
# tests/test_skill_engine.py

def test_primitive_id_equality():
    """PrimitiveId is value-equal and hashable."""
    p1 = PrimitiveId("add_basic", "arithmetic", "v1")
    p2 = PrimitiveId("add_basic", "arithmetic", "v1")
    assert p1 == p2
    assert hash(p1) == hash(p2)

def test_eval_result_accuracy():
    """EvalResult computes accuracy correctly."""
    items = [
        EvalResultItem(..., is_correct=True, primitive_id="a"),
        EvalResultItem(..., is_correct=True, primitive_id="a"),
        EvalResultItem(..., is_correct=False, primitive_id="b"),
    ]
    result = EvalResult(
        accuracy=2/3,
        per_primitive_accuracy={"a": 1.0, "b": 0.0},
        num_examples=3,
        items=items,
    )
    assert result.accuracy == pytest.approx(0.667, rel=0.01)
    assert result.per_primitive_accuracy["a"] == 1.0

def test_composite_skill_generates_training():
    """CompositeSkill uses generator for training."""
    # Mock generator
    mock_gen = Mock()
    mock_gen.generate_training_batch.return_value = [{"messages": [...]}]

    skill = CompositeSkill(config, mock_gen, mock_passive)
    batch = skill.generate_training_batch(level=1, count=10)

    mock_gen.generate_training_batch.assert_called_once_with(1, 10, None)

def test_skill_engine_loads_from_yaml():
    """SkillEngine loads skills from YAML configs."""
    engine = SkillEngine(config_dir=test_config_dir)
    skill = engine.get("bin")

    assert skill.id == "bin"
    assert skill.config.max_level == 30

def test_passive_adapter_scores_correctly():
    """PassiveAdapter scores using passive.check_answer."""
    passive = ArithmeticPassive()
    adapter = PassiveAdapter(passive)

    batch = EvalBatch(
        skill_id="test",
        level=1,
        problems=[EvalProblem(prompt="2+2?", expected="4", primitive_id="add")],
    )

    result = adapter.score_eval(batch, ["4"])
    assert result.accuracy == 1.0
```

### Integration Tests

```python
def test_full_eval_cycle():
    """Test complete eval: generate -> score -> update state."""
    engine = SkillEngine()

    # Get initial state
    state_before = engine.get_state("bin")
    level_before = state_before.level

    # Run eval with mock answers
    skill = engine.get("bin")
    batch = skill.generate_eval_batch(level=1, count=5)

    # Assume perfect answers
    answers = [p.expected for p in batch.problems]
    result, state_after = engine.run_eval("bin", answers, level=1, count=5)

    assert result.accuracy == 1.0
    assert state_after.total_evals == state_before.total_evals + 1
```

---

## 11. Migration Path

### Backward Compatibility

1. **Existing SkillRegistry** - Remains unchanged, loads SkillConfig
2. **Existing SkillClient** - Wrapped by GeneratorAdapter, not replaced
3. **Existing PassiveModule** - Wrapped by PassiveAdapter, not replaced
4. **Existing Guild code** - Can gradually migrate to SkillEngine

### Gradual Adoption

```python
# Old way (still works)
from guild.skills import get_skill, get_trainer
config = get_skill("bin")
trainer = get_trainer("bin")
batch = trainer.sample(level=5, count=100)

# New way (recommended)
from guild.skills import get_engine
engine = get_engine()
skill = engine.get("bin")
training_batch = skill.generate_training_batch(level=5, count=100)
eval_batch = skill.generate_eval_batch(level=5, count=5)
```

---

## 12. API Endpoints

### New Endpoints for Tavern

```python
# tavern/server.py additions

@app.get("/api/skills/primitives")
def list_all_primitives():
    """List all primitives across all skills."""
    engine = get_engine()
    primitives = {}
    for skill in engine.all_skills().values():
        for prim in skill.primitives:
            primitives[str(prim)] = {
                "skill_id": skill.id,
                "name": prim.name,
                "track": prim.track,
            }
    return {"primitives": primitives}

@app.get("/api/skills/{skill_id}/primitives")
def skill_primitives(skill_id: str):
    """Get primitives for a specific skill."""
    engine = get_engine()
    skill = engine.get(skill_id)
    return {"primitives": [vars(p) for p in skill.primitives]}

@app.get("/api/skills/{skill_id}/primitive-accuracy")
def primitive_accuracy(skill_id: str):
    """Get per-primitive accuracy for a skill."""
    engine = get_engine()
    state = engine.get_state(skill_id)
    return {
        "skill_id": skill_id,
        "primitive_accuracy": state.primitive_accuracy,
    }

@app.post("/api/skills/{skill_id}/eval")
def run_skill_eval(skill_id: str, request: EvalRequest):
    """Run an eval for a skill."""
    engine = get_engine()
    result, state = engine.run_eval(
        skill_id,
        request.model_answers,
        level=request.level,
        count=request.count,
    )
    return {
        "accuracy": result.accuracy,
        "per_primitive": result.per_primitive_accuracy,
        "level": state.level,
        "xp": state.xp_total,
    }
```

---

## 13. UI Updates

### Skill Radar Enhancement

Add per-primitive breakdown to skill cards:

```
┌─────────────────────────────────────────────┐
│ BIN - Binary Alchemy          Lv.5  ████░░  │
│ Overall: 78%                                │
│                                             │
│ Primitives:                                 │
│   binary_add_no_carry    ████████░░  95%   │
│   binary_add_with_carry  ██████░░░░  72%   │
│   binary_sub            █████░░░░░  60%    │
│   bitwise_and           ███████░░░  85%    │
│                                             │
│ Weakest: binary_sub (needs practice)        │
└─────────────────────────────────────────────┘
```

### Primitive Drilldown

New view showing all primitives across skills:

```
┌─────────────────────────────────────────────┐
│ PRIMITIVE MASTERY                           │
├─────────────────────────────────────────────┤
│ Track: Arithmetic                           │
│   add_single_digit        █████████░  95%  │
│   sub_single_digit        ████████░░  85%  │
│   mul_single_digit        ██████░░░░  72%  │
│                                             │
│ Track: Binary                               │
│   binary_add_no_carry     ████████░░  90%  │
│   binary_compare          █████░░░░░  60%  │
│                                             │
│ Track: Logic                                │
│   modus_ponens            ███████░░░  80%  │
│   syllogism_all           ████░░░░░░  55%  │
└─────────────────────────────────────────────┘
```

---

## 14. Open Questions

1. **Primitive prerequisite enforcement** - Should we block testing higher primitives until prerequisites pass?

2. **Cross-skill primitives** - Some primitives (like `compare_integers`) appear in multiple skills. Share or duplicate?

3. **Dynamic difficulty** - Should primitive difficulty affect which problems are generated?

4. **Primitive versioning** - When a primitive definition changes, how do we handle historical data?

5. **Passive-per-skill vs shared** - Should each skill have its own passive, or share passives by category?

---

## 15. Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Skills loadable | 40+ | `len(engine.list_skills())` |
| Primitive coverage | 60+ defined | Count in YAML |
| Eval latency | <500ms per batch | Time `run_eval()` |
| State persistence | 100% reliable | Test restart recovery |
| Backward compat | 0 breaking changes | Existing tests pass |

---

## 16. Summary

This plan introduces a unified **Skill Engine** that:

1. **Abstracts** training generation + eval generation + scoring behind one `Skill` interface
2. **Adapts** existing generators (singleSKILL APIs) and passives to the new interface
3. **Tracks** per-primitive accuracy for fine-grained skill analysis
4. **Scales** to 40+ skills through YAML-driven configuration
5. **Integrates** with existing Guild, Tavern, and monitoring systems

The implementation is split into 7 phases over ~10-14 days, with each phase delivering working, testable code.
