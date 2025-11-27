# Passives - Transfer Learning Evaluations

**Passives** measure general abilities independent of active training.
See `GUILD_VOCABULARY.md` for RPG terminology.

## Quick Start

```python
from guild.passives import list_passives, get_passive

# List available
passives = list_passives()  # ['arithmetic', 'logic', 'counting', 'string_craft']

# Get and use
passive = get_passive('arithmetic')
problems = passive.generate_problems(count=5, seed=42)
is_correct = passive.check_answer(expected="42", got="The answer is 42")
```

## Current Passives

| ID | Category | Description | Version |
|----|----------|-------------|---------|
| `arithmetic` | arithmetic | Digit sum, even/odd, comparison, modulo | 1.0.0 |
| `logic` | logic | Boolean AND, OR, XOR, NOT | 1.0.0 |
| `counting` | counting | Letter/vowel/word count | 1.0.0 |
| `string_craft` | string_craft | Reverse, palindrome, first/last N | 1.0.0 |

## Adding a New Passive

1. Create `guild/passives/my_passive.py`:

```python
from guild.passives.base import PassiveModule

class MyPassive(PassiveModule):
    id = "my_passive"
    name = "My Passive"
    category = "reasoning"  # See GUILD_VOCABULARY.md
    description = "What it tests"
    version = "1.0.0"  # BUMP when changing logic!

    def generate_problems(self, count, seed=None):
        # Return list of {"prompt": ..., "expected": ...}
        ...

    def check_answer(self, expected, got):
        # Return True if correct
        ...
```

2. That's it! Auto-discovered on next import.

## Versioning

**CRITICAL**: Results are only comparable within the same version.

When you change:
- Problem generation logic
- Answer checking logic
- Problem difficulty

**You MUST bump the version number!**

```python
version = "1.1.0"  # Was "1.0.0"
```

The eval system records version with every result, so old and new results
can be distinguished.

## Evaluation Modes

| Mode | Problems | When | Purpose |
|------|----------|------|---------|
| LITE | 5 | Auto on checkpoint save | Quick health check |
| FULL | 30 | Manual queue | Detailed analysis |

## Categories (from GUILD_VOCABULARY.md)

- **logic**: Boolean reasoning, deduction
- **counting**: Enumeration, frequency
- **conversion**: Format transformation
- **string_craft**: Text manipulation
- **arithmetic**: Basic number sense
- **sequence**: Pattern recognition
- **memory**: Fact retention
- **reasoning**: Multi-step logic

## Files

```
guild/passives/
├── __init__.py     # Auto-discovery, registry
├── base.py         # PassiveModule base class
├── arithmetic.py   # Arithmetic passive
├── logic.py        # Logic gates passive
├── counting.py     # Counting passive
├── string_craft.py # String manipulation passive
└── README.md       # This file
```

## API Endpoints (on VaultKeeper :8767)

```
GET /api/passives               # List all results
GET /api/passives/summary       # Summary stats
GET /api/passives/queue         # Pending queue
GET /api/passives/checkpoint/N  # Results for checkpoint
```
