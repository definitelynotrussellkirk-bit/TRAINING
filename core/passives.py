"""
Passives - Transfer learning evaluations.

See GUILD_VOCABULARY.md for RPG terminology and full documentation.

================================================================================
WHAT ARE PASSIVES?
================================================================================

Passives = Innate abilities that work regardless of active training.
They measure general capabilities vs the base model.

"Transfer learning" in ML terms = "Passives" in Guild terms

KEY CONCEPT: Passive Drift
- Positive Drift (+) = Training improved general abilities
- Neutral Drift (=) = No change from base
- Negative Drift (-) = Catastrophic forgetting (BAD!)

================================================================================
PASSIVE CATEGORIES (from GUILD_VOCABULARY.md)
================================================================================

| Category     | Description                  | Examples                        |
|--------------|------------------------------|---------------------------------|
| Logic        | Boolean reasoning, deduction | AND, OR, XOR gates              |
| Counting     | Enumeration, frequency       | Letter count, vowel count       |
| Conversion   | Format transformation        | Decimal↔Hex, Roman numerals     |
| String Craft | Text manipulation            | Reverse, palindrome, first N    |
| Arithmetic   | Basic number sense           | Digit sum, even/odd, comparison |
| Sequence     | Pattern recognition          | Next in sequence, alphabetical  |
| Memory       | Fact retention, recall       | bAbI tasks (20 types)           |
| Reasoning    | Multi-step logic             | BIG-Bench tasks                 |

================================================================================
EVALUATION MODES
================================================================================

LITE Mode (auto on checkpoint save):
- 5 problems per passive
- Fast (~30 seconds total)
- Priority: runs before FULL
- Purpose: Quick health check, detect major regressions

FULL Mode (manually queued):
- 30-50 problems per passive
- Comprehensive (~5 minutes total)
- Purpose: Detailed analysis, publishable results

================================================================================
HOW TO ADD NEW PASSIVES
================================================================================

1. Define the passive in BUILTIN_PASSIVES below (or configs/passives/*.yaml)
2. Add problem generator in _generate_passive_problems() in eval_runner.py
3. Add answer checker in _check_passive_answer() in eval_runner.py
4. (Optional) Create static validation set in data/validation/passives/

Future: Load from configs/passives/*.yaml for extensibility

================================================================================
STORAGE
================================================================================

- status/passives_ledger.json - Master results index (append-only)
- status/passive_queue.json - Pending evaluations
- Query via API: /api/passives, /api/passives/checkpoint/{step}

================================================================================
USAGE
================================================================================

    from core.passives import get_passives_ledger, queue_passive_lite

    # Queue LITE passives (auto on checkpoint save)
    queue_passive_lite(checkpoint_step=190000)

    # Queue FULL passives (manual)
    queue_passive_full(checkpoint_step=190000)

    # Query results
    ledger = get_passives_ledger()
    results = ledger.get_by_checkpoint(190000)
    summary = ledger.get_checkpoint_summary(190000)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class PassiveMode(Enum):
    """Evaluation mode for passives."""
    LITE = "lite"  # Quick checks (5-10 problems per passive)
    FULL = "full"  # Comprehensive (50+ problems per passive)


@dataclass
class PassiveDefinition:
    """Definition of a passive evaluation."""
    id: str  # e.g., "decimal_math", "reading_comp"
    name: str  # Display name
    description: str
    category: str  # "reasoning", "math", "comprehension", etc.
    lite_count: int = 5  # Problems for LITE mode
    full_count: int = 50  # Problems for FULL mode
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PassiveDefinition':
        return cls(**data)


@dataclass
class PassiveResult:
    """
    Result of a passive evaluation.

    IMPORTANT: version tracks which passive definition was used.
    Results are only comparable within the same version.
    """
    checkpoint_step: int
    passive_id: str
    mode: str  # "lite" or "full"
    accuracy: float
    correct: int
    total: int
    timestamp: str
    version: str = "1.0.0"  # Passive definition version when evaluated
    config_hash: str = ""   # Hash of full config for verification
    problems: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PassiveResult':
        return cls(
            checkpoint_step=data["checkpoint_step"],
            passive_id=data["passive_id"],
            mode=data["mode"],
            accuracy=data["accuracy"],
            correct=data["correct"],
            total=data["total"],
            timestamp=data["timestamp"],
            version=data.get("version", "1.0.0"),
            config_hash=data.get("config_hash", ""),
            problems=data.get("problems", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        """Unique key for this result (includes version for comparability)."""
        return f"{self.checkpoint_step}:{self.passive_id}:{self.mode}:{self.version}"

    def is_comparable_to(self, other: 'PassiveResult') -> bool:
        """Check if two results can be meaningfully compared."""
        return (self.passive_id == other.passive_id and
                self.mode == other.mode and
                self.version == other.version)


# ================================================================================
# PASSIVE MODULE INTEGRATION
# ================================================================================
# Passives are now defined as modular plugins in guild/passives/
# Each passive is a separate Python file that auto-registers.
# See guild/passives/base.py for how to create new passives.
# ================================================================================

def _get_passive_modules():
    """Load passive modules from guild/passives/."""
    try:
        from guild.passives import get_all_passives, get_passive_configs
        return get_all_passives(), get_passive_configs()
    except ImportError:
        logger.warning("guild.passives not available, using empty passives list")
        return [], []


def _convert_config_to_definition(config) -> PassiveDefinition:
    """Convert a PassiveConfig to PassiveDefinition for compatibility."""
    return PassiveDefinition(
        id=config.id,
        name=config.name,
        description=config.description,
        category=config.category,
        lite_count=config.lite_count,
        full_count=config.full_count,
        enabled=config.enabled,
    )


class PassivesLedger:
    """Storage for passive evaluation results."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.ledger_file = self.base_dir / "status" / "passives_ledger.json"
        self._lock = Lock()
        self._cache: Dict[str, PassiveResult] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            if self.ledger_file.exists():
                try:
                    with open(self.ledger_file) as f:
                        data = json.load(f)
                    for result_data in data.get("results", []):
                        result = PassiveResult.from_dict(result_data)
                        self._cache[result.key] = result
                    logger.info(f"Loaded {len(self._cache)} passive results")
                except Exception as e:
                    logger.error(f"Failed to load passives ledger: {e}")

            self._loaded = True

    def _save(self):
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "count": len(self._cache),
            "results": [r.to_dict() for r in sorted(
                self._cache.values(),
                key=lambda x: (x.checkpoint_step, x.passive_id, x.mode)
            )]
        }

        with open(self.ledger_file, "w") as f:
            json.dump(data, f, indent=2)

    def record(self, result: PassiveResult) -> bool:
        """Record a passive result. Returns True if new."""
        self._ensure_loaded()

        with self._lock:
            if result.key in self._cache:
                logger.info(f"Passive {result.passive_id} ({result.mode}) already recorded for checkpoint {result.checkpoint_step}")
                return False

            self._cache[result.key] = result
            self._save()
            logger.info(f"Recorded passive: checkpoint-{result.checkpoint_step} {result.passive_id} ({result.mode}) "
                       f"accuracy={result.accuracy:.1%}")
            return True

    def has_result(self, checkpoint_step: int, passive_id: str, mode: str) -> bool:
        self._ensure_loaded()
        key = f"{checkpoint_step}:{passive_id}:{mode}"
        return key in self._cache

    def get(self, checkpoint_step: int, passive_id: str, mode: str) -> Optional[PassiveResult]:
        self._ensure_loaded()
        key = f"{checkpoint_step}:{passive_id}:{mode}"
        return self._cache.get(key)

    def get_by_checkpoint(self, checkpoint_step: int, mode: Optional[str] = None) -> List[PassiveResult]:
        """Get all passive results for a checkpoint."""
        self._ensure_loaded()
        results = [r for r in self._cache.values() if r.checkpoint_step == checkpoint_step]
        if mode:
            results = [r for r in results if r.mode == mode]
        return sorted(results, key=lambda x: x.passive_id)

    def get_by_passive(self, passive_id: str, mode: Optional[str] = None) -> List[PassiveResult]:
        """Get history of a specific passive across checkpoints."""
        self._ensure_loaded()
        results = [r for r in self._cache.values() if r.passive_id == passive_id]
        if mode:
            results = [r for r in results if r.mode == mode]
        return sorted(results, key=lambda x: x.checkpoint_step)

    def get_checkpoint_summary(self, checkpoint_step: int) -> Dict[str, Any]:
        """Get summary of all passives for a checkpoint."""
        results = self.get_by_checkpoint(checkpoint_step)

        lite_results = [r for r in results if r.mode == "lite"]
        full_results = [r for r in results if r.mode == "full"]

        def avg_accuracy(results: List[PassiveResult]) -> Optional[float]:
            if not results:
                return None
            return sum(r.accuracy for r in results) / len(results)

        return {
            "checkpoint_step": checkpoint_step,
            "lite": {
                "count": len(lite_results),
                "avg_accuracy": avg_accuracy(lite_results),
                "results": {r.passive_id: r.accuracy for r in lite_results},
            },
            "full": {
                "count": len(full_results),
                "avg_accuracy": avg_accuracy(full_results),
                "results": {r.passive_id: r.accuracy for r in full_results},
            },
        }

    def summary(self) -> Dict[str, Any]:
        """Get overall summary statistics."""
        self._ensure_loaded()

        by_passive: Dict[str, Dict[str, Any]] = {}
        for result in self._cache.values():
            if result.passive_id not in by_passive:
                by_passive[result.passive_id] = {
                    "lite_count": 0,
                    "full_count": 0,
                    "best_lite": 0,
                    "best_full": 0,
                }
            p = by_passive[result.passive_id]
            if result.mode == "lite":
                p["lite_count"] += 1
                p["best_lite"] = max(p["best_lite"], result.accuracy)
            else:
                p["full_count"] += 1
                p["best_full"] = max(p["best_full"], result.accuracy)

        return {
            "total_results": len(self._cache),
            "by_passive": by_passive,
        }


# Singleton
_passives_ledger: Optional[PassivesLedger] = None
_passives_ledger_lock = Lock()


def get_passives_ledger(base_dir: Optional[Path] = None) -> PassivesLedger:
    """Get singleton passives ledger."""
    global _passives_ledger

    with _passives_ledger_lock:
        if _passives_ledger is None:
            _passives_ledger = PassivesLedger(base_dir)
        return _passives_ledger


def get_passive_definitions() -> List[PassiveDefinition]:
    """Get all passive definitions from guild/passives/ modules."""
    _, configs = _get_passive_modules()
    return [_convert_config_to_definition(c) for c in configs if c.enabled]


def get_passive(passive_id: str) -> Optional[PassiveDefinition]:
    """Get a specific passive definition."""
    for p in get_passive_definitions():
        if p.id == passive_id:
            return p
    return None


def get_passive_module(passive_id: str):
    """Get the actual passive module instance (for running evaluations)."""
    try:
        from guild.passives import get_passive as get_guild_passive
        return get_guild_passive(passive_id)
    except ImportError:
        return None


# Queue for passive evaluations
_passive_queue: List[Dict[str, Any]] = []
_passive_queue_lock = Lock()


def queue_passive_lite(checkpoint_step: int):
    """Queue LITE passive evaluations for a checkpoint (all passives)."""
    ledger = get_passives_ledger()

    for passive in get_passive_definitions():
        if not ledger.has_result(checkpoint_step, passive.id, "lite"):
            with _passive_queue_lock:
                _passive_queue.append({
                    "checkpoint_step": checkpoint_step,
                    "passive_id": passive.id,
                    "mode": "lite",
                    "queued_at": datetime.now().isoformat(),
                })

    _save_passive_queue()
    logger.info(f"Queued LITE passives for checkpoint-{checkpoint_step}")


def queue_passive_full(checkpoint_step: int):
    """Queue FULL passive evaluations for a checkpoint (all passives, manual)."""
    ledger = get_passives_ledger()

    for passive in get_passive_definitions():
        if not ledger.has_result(checkpoint_step, passive.id, "full"):
            with _passive_queue_lock:
                _passive_queue.append({
                    "checkpoint_step": checkpoint_step,
                    "passive_id": passive.id,
                    "mode": "full",
                    "queued_at": datetime.now().isoformat(),
                })

    _save_passive_queue()
    logger.info(f"Queued FULL passives for checkpoint-{checkpoint_step}")


# =============================================================================
# TIERED QUEUE FUNCTIONS (NEW)
# =============================================================================


def queue_core_passives(checkpoint_step: int, mode: str = "lite"):
    """
    Queue only CORE (sentinel) passives for a checkpoint.

    Use this for automatic checkpoint evaluation - runs the minimal set
    of passives needed to detect catastrophic forgetting.

    Args:
        checkpoint_step: Checkpoint step number
        mode: "lite" (5 problems) or "full" (30 problems)
    """
    from guild.passives import get_core_passives

    ledger = get_passives_ledger()
    core_passives = get_core_passives()
    queued = 0

    for passive in core_passives:
        if not ledger.has_result(checkpoint_step, passive.id, mode):
            with _passive_queue_lock:
                _passive_queue.append({
                    "checkpoint_step": checkpoint_step,
                    "passive_id": passive.id,
                    "mode": mode,
                    "tier": "core",
                    "priority": passive.priority,
                    "queued_at": datetime.now().isoformat(),
                })
                queued += 1

    _save_passive_queue()
    if queued > 0:
        logger.info(f"Queued {queued} CORE {mode.upper()} passives for checkpoint-{checkpoint_step}")


def queue_extended_passives(checkpoint_step: int, mode: str = "lite"):
    """
    Queue only EXTENDED passives for a checkpoint.

    Use this for comprehensive on-demand evaluation.

    Args:
        checkpoint_step: Checkpoint step number
        mode: "lite" (5 problems) or "full" (30 problems)
    """
    from guild.passives import get_extended_passives

    ledger = get_passives_ledger()
    extended_passives = get_extended_passives()
    queued = 0

    for passive in extended_passives:
        if not ledger.has_result(checkpoint_step, passive.id, mode):
            with _passive_queue_lock:
                _passive_queue.append({
                    "checkpoint_step": checkpoint_step,
                    "passive_id": passive.id,
                    "mode": mode,
                    "tier": "extended",
                    "priority": passive.priority,
                    "queued_at": datetime.now().isoformat(),
                })
                queued += 1

    _save_passive_queue()
    if queued > 0:
        logger.info(f"Queued {queued} EXTENDED {mode.upper()} passives for checkpoint-{checkpoint_step}")


def queue_all_passives(checkpoint_step: int, mode: str = "lite"):
    """
    Queue ALL passives (core + extended) for a checkpoint.

    Use this for comprehensive evaluation.

    Args:
        checkpoint_step: Checkpoint step number
        mode: "lite" (5 problems) or "full" (30 problems)
    """
    queue_core_passives(checkpoint_step, mode)
    queue_extended_passives(checkpoint_step, mode)


def get_pending_passives() -> List[Dict[str, Any]]:
    """Get pending passive evaluations."""
    _load_passive_queue()
    with _passive_queue_lock:
        return list(_passive_queue)


def pop_passive() -> Optional[Dict[str, Any]]:
    """
    Pop next passive from queue, skipping already-recorded items.

    Sort order:
    1. Mode: LITE before FULL
    2. Tier: CORE before EXTENDED
    3. Priority: Lower priority numbers first
    4. Queue time: Earlier first
    """
    _load_passive_queue()
    ledger = get_passives_ledger()

    with _passive_queue_lock:
        if not _passive_queue:
            return None

        # Sort: LITE before FULL, CORE before EXTENDED, then by priority, then by queue time
        def sort_key(x):
            mode_order = 0 if x["mode"] == "lite" else 1
            tier_order = 0 if x.get("tier") == "core" else 1
            priority = x.get("priority", 50)
            queued_at = x.get("queued_at", "")
            return (mode_order, tier_order, priority, queued_at)

        _passive_queue.sort(key=sort_key)

        # Find first item that isn't already recorded
        items_removed = 0
        while _passive_queue:
            item = _passive_queue[0]
            # Check if already recorded
            if ledger.has_result(item["checkpoint_step"], item["passive_id"], item["mode"]):
                _passive_queue.pop(0)
                items_removed += 1
                continue
            # Found an unrecorded item
            _passive_queue.pop(0)
            if items_removed > 0:
                logger.info(f"Cleaned {items_removed} already-recorded items from passive queue")
            _save_passive_queue()
            return item

        # All items were already recorded
        if items_removed > 0:
            logger.info(f"Cleaned {items_removed} already-recorded items from passive queue (queue now empty)")
        _save_passive_queue()
        return None


def _get_passive_queue_file() -> Path:
    base_dir = Path(__file__).parent.parent
    return base_dir / "status" / "passive_queue.json"


def _save_passive_queue():
    queue_file = _get_passive_queue_file()
    queue_file.parent.mkdir(parents=True, exist_ok=True)

    with open(queue_file, "w") as f:
        json.dump({
            "queue": _passive_queue,
            "updated_at": datetime.now().isoformat(),
        }, f, indent=2)


def _load_passive_queue():
    global _passive_queue

    queue_file = _get_passive_queue_file()
    if not queue_file.exists():
        return

    with _passive_queue_lock:
        try:
            with open(queue_file) as f:
                data = json.load(f)
            _passive_queue = data.get("queue", [])
        except Exception as e:
            logger.error(f"Failed to load passive queue: {e}")
