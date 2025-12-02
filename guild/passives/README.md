# Passives - Transfer Learning Evaluations

**Passives** measure general abilities independent of active training.
See `GUILD_VOCABULARY.md` for RPG terminology.

## Contract (v2.0)

Every passive MUST comply with this contract:

```python
from guild.passives.base import PassiveModule

class MyPassive(PassiveModule):
    # REQUIRED class attributes
    id = "my_passive"           # Unique ID (snake_case)
    name = "My Passive"         # Display name
    category = "reasoning"      # Category from GUILD_VOCABULARY.md
    description = "What it tests"
    version = "1.0.0"           # BUMP when changing logic!

    # OPTIONAL class attributes
    tier = PassiveTier.EXTENDED  # "core" or "extended" (default: extended)
    priority = 50                # Lower runs first (0-100)
    lite_count = 5               # Problems for LITE mode
    full_count = 30              # Problems for FULL mode
    enabled = True               # Whether active

    # REQUIRED method - MUST accept all three parameters!
    def generate_problems(self, count: int, seed: Optional[int] = None, level: int = 1):
        """
        Args:
            count: Number of problems
            seed: Optional random seed for reproducibility
            level: Skill level 1-30 for difficulty scaling (can be ignored)

        Returns:
            List of dicts with REQUIRED keys:
            - prompt: str       - Question for the model
            - expected: str     - Correct answer
            - primitive_id: str - For per-primitive tracking
        """
        return [{
            "prompt": "What is 2+2?",
            "expected": "4",
            "primitive_id": "addition",  # REQUIRED!
        }]

    # REQUIRED method
    def check_answer(self, expected: str, got: str) -> bool:
        """Return True if answer is correct."""
        return expected.lower() in got.lower()
```

## Validate Contract

```bash
# Quick validation
python3 tests/test_passive_contract.py

# Full pytest
pytest tests/test_passive_contract.py -v
```

## Quick Start

```python
from guild.passives import list_passives, get_passive

# List available
passives = list_passives()  # ['arithmetic', 'logic', 'counting', ...]

# Get and use
passive = get_passive('arithmetic')
problems = passive.generate_problems(count=5, seed=42, level=1)
is_correct = passive.check_answer(expected="42", got="The answer is 42")
```

## Current Passives (11)

| ID | Category | Description | Tier |
|----|----------|-------------|------|
| `arithmetic` | arithmetic | Digit sum, even/odd, comparison, modulo | core |
| `logic` | logic | Boolean AND, OR, XOR, NOT | core |
| `binary_arithmetic` | math | Binary add/sub, bitwise ops, conversions | core |
| `word_puzzles` | reasoning | Syllable assembly, word clues | core |
| `counting` | counting | Letter/vowel/word count | extended |
| `string_craft` | string_craft | Reverse, palindrome, first/last N | extended |
| `sequence` | reasoning | Number sequences, pattern recognition | extended |
| `memory` | memory | Simple facts, ordinal knowledge | extended |
| `code_trace` | code | Variable tracing, conditionals, loops | extended |
| `combinatorics` | math | Permutations, combinations, factorials | extended |
| `language_basics` | reasoning | Synonyms, antonyms, plurals | extended |

## Tier System

| Tier | When | Purpose |
|------|------|---------|
| **core** | Every checkpoint save | Sentinel passives - catch catastrophic forgetting |
| **extended** | On-demand | Comprehensive capability assessment |

## Versioning

**CRITICAL**: Results are only comparable within the same version.

When you change:
- Problem generation logic
- Answer checking logic
- Problem difficulty

**You MUST bump the version number!**

The eval system records version with every result, so old and new results
can be distinguished.

## Categories (from GUILD_VOCABULARY.md)

- **logic**: Boolean reasoning, deduction
- **counting**: Enumeration, frequency
- **conversion**: Format transformation
- **string_craft**: Text manipulation
- **arithmetic**: Basic number sense
- **math**: Advanced mathematics
- **sequence**: Pattern recognition
- **memory**: Fact retention
- **reasoning**: Multi-step logic
- **code**: Code tracing, execution

## Files

```
guild/passives/
├── __init__.py           # Auto-discovery, registry, tier functions
├── base.py               # PassiveModule base class + CONTRACT
├── arithmetic.py         # Arithmetic (core)
├── logic.py              # Logic gates (core)
├── binary_arithmetic.py  # Binary ops (core)
├── word_puzzles.py       # Syllable puzzles (core)
├── counting.py           # Counting (extended)
├── string_craft.py       # String manipulation (extended)
├── sequence.py           # Sequences (extended)
├── memory.py             # Memory/recall (extended)
├── code_trace.py         # Code tracing (extended)
├── combinatorics.py      # Combinatorics (extended)
├── language_basics.py    # Language (extended)
└── README.md             # This file
```

## API

```python
# Discovery
from guild.passives import (
    list_passives,           # All IDs
    get_passive,             # Get by ID
    get_all_passives,        # All instances
    get_passive_configs,     # All configs
    get_passives_by_category,# Filter by category
)

# Tier functions
from guild.passives import (
    get_core_passives,       # Core tier (limited)
    get_extended_passives,   # Extended tier
    get_passives_by_tier,    # Get by tier name
    list_core_passive_ids,   # Core IDs
    list_extended_passive_ids, # Extended IDs
    get_tier_summary,        # Summary stats
)
```

## API Endpoints (on VaultKeeper :8767)

```
GET /api/passives               # List all results
GET /api/passives/summary       # Summary stats
GET /api/passives/queue         # Pending queue
GET /api/passives/checkpoint/N  # Results for checkpoint
```
