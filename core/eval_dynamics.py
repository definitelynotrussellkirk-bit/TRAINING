"""
Eval Dynamics - Dynamic Generation & Context Drift Detection
=============================================================

Gap 5: Dynamic Eval Generation
- Generate fresh eval problems per checkpoint (not static files)
- Use checkpoint step as seed for reproducibility
- Allows evals to adapt to curriculum level

Gap 6: Context Drift Detection
- Detect when model context has changed significantly
- Auto-queue re-evaluation when drift exceeds threshold
- Track context hash history

Usage:
    from core.eval_dynamics import (
        generate_dynamic_eval,
        compute_context_hash,
        check_drift,
        should_reeval,
    )

    # Generate dynamic eval problems
    problems = generate_dynamic_eval(
        skill="bin",
        level=5,
        checkpoint_step=1000,  # Used as seed
        count=10,
    )

    # Check context drift
    drift = check_drift(old_hash, new_hash)
    if should_reeval(drift):
        queue_reeval(checkpoint_step)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DYNAMIC EVAL GENERATION
# =============================================================================

@dataclass
class DynamicEvalConfig:
    """Configuration for dynamic eval generation."""
    skill: str
    level: int
    count: int = 10
    seed: Optional[int] = None  # If None, uses checkpoint_step
    include_primitive_tags: bool = True
    difficulty_jitter: float = 0.1  # Random variation in difficulty


def generate_dynamic_eval(
    skill: str,
    level: int,
    checkpoint_step: int,
    count: int = 10,
    config: Optional[DynamicEvalConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Generate dynamic eval problems for a skill/level.

    Uses checkpoint_step as seed for reproducibility - same checkpoint
    always gets same problems.

    Args:
        skill: Skill ID (bin, sy)
        level: Skill level
        checkpoint_step: Used as random seed
        count: Number of problems to generate
        config: Optional detailed configuration

    Returns:
        List of problem dicts with messages format
    """
    if config is None:
        config = DynamicEvalConfig(skill=skill, level=level, count=count)

    # Use checkpoint as seed for reproducibility
    seed = config.seed if config.seed is not None else checkpoint_step
    rng = random.Random(seed)

    if skill in ("bin", "binary"):
        return _generate_bin_problems(level, count, rng, config)
    elif skill in ("sy", "syllo"):
        return _generate_sy_problems(level, count, rng, config)
    else:
        logger.warning(f"No dynamic generator for skill {skill}, using fallback")
        return _generate_fallback_problems(skill, level, count, rng)


def _generate_bin_problems(
    level: int,
    count: int,
    rng: random.Random,
    config: DynamicEvalConfig,
) -> List[Dict[str, Any]]:
    """Generate binary arithmetic problems."""
    problems = []

    # Level determines bit width: level N = (N+1) bits
    bits = min(level + 1, 32)
    max_val = (1 << bits) - 1

    # Operations available at different levels
    operations = ["increment"]
    if level >= 2:
        operations.append("decrement")
    if level >= 4:
        operations.append("add")
    if level >= 6:
        operations.append("subtract")

    # Circled digit mapping
    circled = {
        "0": "⓪", "1": "①", "2": "②", "3": "③", "4": "④",
        "5": "⑤", "6": "⑥", "7": "⑦", "8": "⑧", "9": "⑨",
    }

    def to_circled(n: int) -> str:
        return "".join(circled[d] for d in str(n))

    for i in range(count):
        op = rng.choice(operations)

        if op == "increment":
            n = rng.randint(0, max_val - 1)
            result = n + 1
            prompt = f"increment({to_circled(n)}) = ?"
            expected = to_circled(result)
            primitives = ["logic_chain", "xfm_encode"]

        elif op == "decrement":
            n = rng.randint(1, max_val)
            result = n - 1
            prompt = f"decrement({to_circled(n)}) = ?"
            expected = to_circled(result)
            primitives = ["logic_chain", "xfm_encode"]

        elif op == "add":
            a = rng.randint(0, max_val // 2)
            b = rng.randint(0, max_val // 2)
            result = a + b
            prompt = f"add({to_circled(a)}, {to_circled(b)}) = ?"
            expected = to_circled(result)
            primitives = ["logic_chain", "xfm_encode", "mem_compose"]

        elif op == "subtract":
            a = rng.randint(1, max_val)
            b = rng.randint(0, a)
            result = a - b
            prompt = f"subtract({to_circled(a)}, {to_circled(b)}) = ?"
            expected = to_circled(result)
            primitives = ["logic_chain", "xfm_encode", "mem_compose"]

        else:
            continue

        problem = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps({"answer": expected})},
            ],
            "metadata": {
                "skill": "bin",
                "level": level,
                "operation": op,
                "dynamic": True,
            },
        }

        if config.include_primitive_tags:
            problem["primitive_ids"] = primitives

        problems.append(problem)

    return problems


def _generate_sy_problems(
    level: int,
    count: int,
    rng: random.Random,
    config: DynamicEvalConfig,
) -> List[Dict[str, Any]]:
    """Generate syllacrostic problems (simplified version)."""
    problems = []

    # Word list (simplified - real impl would use skill API)
    word_lists = {
        1: ["cat", "dog", "sun", "run", "big"],
        2: ["hello", "world", "happy", "funny", "quiet"],
        3: ["python", "coding", "summer", "winter", "garden"],
        4: ["elephant", "mountain", "computer", "beautiful", "wonderful"],
        5: ["programming", "development", "engineering", "mathematics", "philosophy"],
    }

    level_key = min(level, max(word_lists.keys()))
    words = word_lists.get(level_key, word_lists[1])

    for i in range(count):
        word = rng.choice(words)

        # Simple degradation: shuffle middle letters
        if len(word) > 3:
            middle = list(word[1:-1])
            rng.shuffle(middle)
            degraded = word[0] + "".join(middle) + word[-1]
        else:
            degraded = word

        prompt = f"Unscramble this word: {degraded}"
        expected = word

        problem = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps({"word": expected})},
            ],
            "metadata": {
                "skill": "sy",
                "level": level,
                "original": word,
                "dynamic": True,
            },
        }

        if config.include_primitive_tags:
            problem["primitive_ids"] = ["seq_transform", "attn_select", "mem_context"]

        problems.append(problem)

    return problems


def _generate_fallback_problems(
    skill: str,
    level: int,
    count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Fallback generator for unknown skills."""
    return [
        {
            "messages": [
                {"role": "user", "content": f"[{skill} L{level}] Problem {i+1}"},
                {"role": "assistant", "content": f"Solution {i+1}"},
            ],
            "metadata": {"skill": skill, "level": level, "dynamic": True, "fallback": True},
        }
        for i in range(count)
    ]


# =============================================================================
# CONTEXT DRIFT DETECTION
# =============================================================================

@dataclass
class ContextSnapshot:
    """Snapshot of evaluation context for drift detection."""
    checkpoint_step: int
    context_hash: str
    timestamp: str
    components: Dict[str, str] = field(default_factory=dict)  # Component -> hash


@dataclass
class DriftResult:
    """Result of drift detection."""
    old_hash: str
    new_hash: str
    drift_detected: bool
    changed_components: List[str]
    severity: str  # "none", "minor", "major", "complete"

    @property
    def should_reeval(self) -> bool:
        """Whether this drift warrants re-evaluation."""
        return self.severity in ("major", "complete")


def compute_context_hash(
    checkpoint_step: int,
    skill: Optional[str] = None,
    level: Optional[int] = None,
    include_config: bool = True,
    include_curriculum: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """
    Compute hash of evaluation context.

    Context includes:
    - Skill configuration (levels, thresholds)
    - Curriculum state (current level)
    - Validation set metadata
    - Inference config

    Returns:
        (combined_hash, component_hashes)
    """
    components: Dict[str, str] = {}

    # Skill config hash
    if skill:
        try:
            from guild.skills import get_engine
            engine = get_engine()
            skill_info = engine.get(skill)
            if skill_info:
                skill_hash = hashlib.md5(
                    json.dumps(skill_info.to_dict(), sort_keys=True).encode()
                ).hexdigest()[:8]
                components["skill_config"] = skill_hash
        except Exception as e:
            logger.debug(f"Could not hash skill config: {e}")

    # Curriculum state hash
    if include_curriculum:
        try:
            from data_manager.curriculum_manager import CurriculumManager
            from core.paths import get_base_dir
            cm = CurriculumManager(get_base_dir(), {})
            state = cm.state
            curr_hash = hashlib.md5(
                json.dumps(state, sort_keys=True).encode()
            ).hexdigest()[:8]
            components["curriculum"] = curr_hash
        except Exception as e:
            logger.debug(f"Could not hash curriculum: {e}")

    # Validation set hash (if using static)
    if skill and level:
        try:
            from core.paths import get_base_dir
            val_file = get_base_dir() / "data" / "validation" / skill / f"level_{level:02d}.json"
            if val_file.exists():
                val_hash = hashlib.md5(val_file.read_bytes()).hexdigest()[:8]
                components["validation_set"] = val_hash
        except Exception as e:
            logger.debug(f"Could not hash validation set: {e}")

    # Combine all component hashes
    combined = hashlib.md5(
        json.dumps(components, sort_keys=True).encode()
    ).hexdigest()[:16]

    return combined, components


def check_drift(
    old_snapshot: ContextSnapshot,
    new_snapshot: ContextSnapshot,
) -> DriftResult:
    """
    Check for context drift between two snapshots.

    Args:
        old_snapshot: Previous context snapshot
        new_snapshot: Current context snapshot

    Returns:
        DriftResult with drift analysis
    """
    changed = []

    # Check each component
    all_components = set(old_snapshot.components.keys()) | set(new_snapshot.components.keys())

    for comp in all_components:
        old_val = old_snapshot.components.get(comp)
        new_val = new_snapshot.components.get(comp)

        if old_val != new_val:
            changed.append(comp)

    # Determine severity
    if not changed:
        severity = "none"
    elif len(changed) == 1 and changed[0] == "curriculum":
        severity = "minor"  # Just curriculum changed, probably normal
    elif len(changed) <= 2:
        severity = "major"  # Multiple changes
    else:
        severity = "complete"  # Everything changed

    return DriftResult(
        old_hash=old_snapshot.context_hash,
        new_hash=new_snapshot.context_hash,
        drift_detected=len(changed) > 0,
        changed_components=changed,
        severity=severity,
    )


def should_reeval(drift: DriftResult) -> bool:
    """Check if drift warrants re-evaluation."""
    return drift.should_reeval


# =============================================================================
# AUTO RE-EVAL
# =============================================================================

class DriftMonitor:
    """
    Monitors context drift and triggers re-evaluation when needed.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.history_file = self.base_dir / "status" / "context_drift_history.json"
        self._history: List[ContextSnapshot] = []
        self._load_history()

    def _load_history(self):
        """Load drift history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self._history = [
                    ContextSnapshot(
                        checkpoint_step=h["checkpoint_step"],
                        context_hash=h["context_hash"],
                        timestamp=h["timestamp"],
                        components=h.get("components", {}),
                    )
                    for h in data.get("history", [])
                ]
            except Exception as e:
                logger.warning(f"Failed to load drift history: {e}")

    def _save_history(self):
        """Save drift history to disk."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "w") as f:
            json.dump({
                "updated_at": datetime.now().isoformat(),
                "history": [
                    {
                        "checkpoint_step": h.checkpoint_step,
                        "context_hash": h.context_hash,
                        "timestamp": h.timestamp,
                        "components": h.components,
                    }
                    for h in self._history[-100:]  # Keep last 100
                ],
            }, f, indent=2)

    def record_snapshot(
        self,
        checkpoint_step: int,
        skill: Optional[str] = None,
        level: Optional[int] = None,
    ) -> ContextSnapshot:
        """Record a context snapshot."""
        context_hash, components = compute_context_hash(
            checkpoint_step, skill, level
        )

        snapshot = ContextSnapshot(
            checkpoint_step=checkpoint_step,
            context_hash=context_hash,
            timestamp=datetime.now().isoformat(),
            components=components,
        )

        self._history.append(snapshot)
        self._save_history()

        return snapshot

    def get_last_snapshot(self) -> Optional[ContextSnapshot]:
        """Get the most recent snapshot."""
        if self._history:
            return self._history[-1]
        return None

    def check_and_queue_reeval(
        self,
        checkpoint_step: int,
        skill: str,
        level: int,
    ) -> Optional[DriftResult]:
        """
        Check for drift and queue re-eval if needed.

        Args:
            checkpoint_step: Checkpoint to potentially re-eval
            skill: Skill to check
            level: Level to check

        Returns:
            DriftResult if drift detected, None otherwise
        """
        # Get previous snapshot
        old_snapshot = self.get_last_snapshot()

        # Record new snapshot
        new_snapshot = self.record_snapshot(checkpoint_step, skill, level)

        if old_snapshot is None:
            logger.debug("No previous snapshot, skipping drift check")
            return None

        # Check drift
        drift = check_drift(old_snapshot, new_snapshot)

        if drift.drift_detected:
            logger.info(
                f"Context drift detected: {drift.severity} "
                f"(changed: {', '.join(drift.changed_components)})"
            )

            if drift.should_reeval:
                # Queue re-evaluation
                try:
                    from core.evaluation_ledger import queue_evaluation

                    # Delete old eval and re-queue
                    from core.evaluation_ledger import get_eval_ledger
                    ledger = get_eval_ledger()

                    # Remove old eval if exists
                    ledger.delete_for_checkpoint_and_skill(checkpoint_step, skill, level)

                    # Queue fresh eval
                    queue_evaluation(
                        checkpoint_step=checkpoint_step,
                        skill=skill,
                        level=level,
                        priority=8,  # High priority for re-evals
                    )

                    logger.info(
                        f"Queued re-eval for checkpoint {checkpoint_step} "
                        f"{skill} L{level} due to drift"
                    )

                except Exception as e:
                    logger.error(f"Failed to queue re-eval: {e}")

        return drift if drift.drift_detected else None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_drift_monitor: Optional[DriftMonitor] = None


def get_drift_monitor() -> DriftMonitor:
    """Get singleton drift monitor."""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor


def check_and_reeval_if_needed(
    checkpoint_step: int,
    skill: str,
    level: int,
) -> Optional[DriftResult]:
    """Check for drift and queue re-eval if needed."""
    return get_drift_monitor().check_and_queue_reeval(checkpoint_step, skill, level)


# =============================================================================
# HERO CONTEXT - Multi-Hero Support
# =============================================================================

@dataclass
class HeroContext:
    """Active hero and campaign context."""
    hero_id: str
    campaign_id: str
    campaign_path: Path
    checkpoint_dir: Path
    hero_config: Optional[Dict[str, Any]] = None

    @property
    def base_model(self) -> Optional[str]:
        """Get base model from hero config."""
        if self.hero_config and "model" in self.hero_config:
            return self.hero_config["model"].get("hf_name")
        return None

    @property
    def is_peft(self) -> bool:
        """Check if hero uses PEFT (QLoRA/LoRA)."""
        if self.hero_config:
            qlora = self.hero_config.get("qlora", {})
            return qlora.get("enabled", False)
        return False


def get_base_dir() -> Path:
    """Get base directory for the training project."""
    try:
        from core.paths import get_base_dir as _get_base_dir
        return _get_base_dir()
    except ImportError:
        return Path(__file__).parent.parent


def get_active_hero_context(base_dir: Optional[Path] = None) -> Optional[HeroContext]:
    """
    Get the currently active hero and campaign context.

    Reads from:
    - control/state.json (active_hero)
    - control/active_campaign.json (campaign details)
    - configs/heroes/{hero_id}.yaml (hero config)

    Returns:
        HeroContext with all relevant info, or None if no active hero
    """
    base = Path(base_dir) if base_dir else get_base_dir()

    # Read active hero from state.json
    state_file = base / "control" / "state.json"
    if not state_file.exists():
        logger.warning("No control/state.json found")
        return None

    try:
        with open(state_file) as f:
            state = json.load(f)
        hero_id = state.get("active_hero")
        if not hero_id:
            logger.warning("No active_hero in state.json")
            return None
    except Exception as e:
        logger.error(f"Failed to read state.json: {e}")
        return None

    # Read campaign details
    campaign_file = base / "control" / "active_campaign.json"
    campaign_id = "campaign-001"  # Default
    campaign_path = base / "campaigns" / hero_id / campaign_id

    if campaign_file.exists():
        try:
            with open(campaign_file) as f:
                campaign = json.load(f)
            campaign_id = campaign.get("campaign_id", campaign_id)
            campaign_path_str = campaign.get("campaign_path")
            if campaign_path_str:
                campaign_path = base / campaign_path_str
        except Exception as e:
            logger.warning(f"Failed to read active_campaign.json: {e}")

    # Determine checkpoint directory
    checkpoint_dir = campaign_path / "checkpoints"

    # Load hero config if available
    hero_config = None
    hero_config_file = base / "configs" / "heroes" / f"{hero_id}.yaml"
    if hero_config_file.exists():
        try:
            import yaml
            with open(hero_config_file) as f:
                hero_config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load hero config: {e}")

    return HeroContext(
        hero_id=hero_id,
        campaign_id=campaign_id,
        campaign_path=campaign_path,
        checkpoint_dir=checkpoint_dir,
        hero_config=hero_config,
    )


def infer_hero_from_path(path: str) -> Optional[str]:
    """
    Infer hero_id from a checkpoint path.

    Patterns detected:
    - campaigns/{hero_id}/{campaign_id}/checkpoints/...
    - models/current_model/... (legacy DIO)

    Args:
        path: Checkpoint path (absolute or relative)

    Returns:
        hero_id string or None if cannot infer
    """
    import re
    path_str = str(path)

    # Pattern 1: campaigns/{hero_id}/{campaign_id}/...
    match = re.search(r'campaigns/([^/]+)/([^/]+)', path_str)
    if match:
        return match.group(1)

    # Pattern 2: models/current_model (legacy - assume default hero)
    if "models/current_model" in path_str:
        return None

    return None


def get_checkpoints_for_hero(
    hero_id: str,
    base_dir: Optional[Path] = None,
    campaign_id: Optional[str] = None,
) -> List[Path]:
    """
    Get all checkpoint directories for a specific hero.

    Args:
        hero_id: The hero ID to search for
        base_dir: Base directory (auto-detected if None)
        campaign_id: Specific campaign (all campaigns if None)

    Returns:
        List of checkpoint directory paths
    """
    import re
    base = Path(base_dir) if base_dir else get_base_dir()
    checkpoints = []

    hero_campaigns_dir = base / "campaigns" / hero_id
    if not hero_campaigns_dir.exists():
        return checkpoints

    # Find campaigns
    if campaign_id:
        campaigns = [hero_campaigns_dir / campaign_id]
    else:
        campaigns = [d for d in hero_campaigns_dir.iterdir() if d.is_dir()]

    # Find checkpoints in each campaign
    for campaign_dir in campaigns:
        ckpt_dir = campaign_dir / "checkpoints"
        if ckpt_dir.exists():
            for item in ckpt_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    checkpoints.append(item)

    def extract_step(name: str) -> int:
        match = re.match(r'checkpoint-(\d+)', name)
        return int(match.group(1)) if match else 0

    return sorted(checkpoints, key=lambda p: extract_step(p.name), reverse=True)


def get_latest_checkpoint_for_hero(
    hero_id: str,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Get the most recent checkpoint for a hero.

    Args:
        hero_id: The hero ID
        base_dir: Base directory

    Returns:
        Path to latest checkpoint directory, or None
    """
    checkpoints = get_checkpoints_for_hero(hero_id, base_dir)
    return checkpoints[0] if checkpoints else None


def get_hero_model_info(hero_id: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get model info needed for inference server.

    Returns:
        {
            "base_model": "models/Qwen3-8B",
            "is_peft": True,
            "load_in_4bit": True,
            "checkpoint_dir": Path(...),
        }
    """
    import yaml
    base = Path(base_dir) if base_dir else get_base_dir()

    # Load hero config
    hero_config_file = base / "configs" / "heroes" / f"{hero_id}.yaml"
    if not hero_config_file.exists():
        return {}

    try:
        with open(hero_config_file) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load hero config: {e}")
        return {}

    # Extract relevant info
    model_info = config.get("model", {})
    qlora_info = config.get("qlora", {})
    training_defaults = config.get("training_defaults", {})

    return {
        "base_model": model_info.get("hf_name"),
        "is_peft": qlora_info.get("enabled", False),
        "load_in_4bit": training_defaults.get("load_in_4bit", False),
        "hero_id": hero_id,
    }


def detect_hero_change(
    previous_hero_id: Optional[str],
    base_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Detect if the active hero has changed.

    Args:
        previous_hero_id: The last known hero ID
        base_dir: Base directory

    Returns:
        Tuple of (changed: bool, new_hero_id: Optional[str])
    """
    ctx = get_active_hero_context(base_dir)
    if ctx is None:
        return False, previous_hero_id

    current_hero = ctx.hero_id
    changed = current_hero != previous_hero_id

    if changed:
        logger.info(f"Hero change detected: {previous_hero_id} -> {current_hero}")

    return changed, current_hero
