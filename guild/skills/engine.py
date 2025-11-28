"""
SkillEngine - Central manager for skill lifecycle.

The SkillEngine is THE entry point for all skill operations. It:
- Loads skills from YAML configs
- Wires up generators and passives via adapters
- Manages skill states (persisted to disk)
- Provides unified access to all skill operations

Usage:
    from guild.skills import get_engine

    engine = get_engine()

    # Get a skill
    skill = engine.get("binary")

    # Generate training data
    training = skill.generate_training_batch(level=5, count=100)

    # Run eval
    result, state = engine.run_eval("binary", model_answers, level=5, count=5)
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING
import json
import logging
import time

from guild.skills.types import SkillConfig, SkillState
from guild.skills.skill import Skill, GeneratorOnlySkill, PassiveOnlySkill
from guild.skills.composite import CompositeSkill, LocalSkill
from guild.skills.primitives import PrimitiveId, PrimitiveMeta
from guild.skills.eval_types import EvalBatch, EvalResult
from guild.skills.loader import load_skill_config, discover_skills, get_config_dir
from guild.skills.adapters.generator import GeneratorAdapter
from guild.skills.adapters.passive import PassiveAdapter

if TYPE_CHECKING:
    from guild.passives.base import PassiveModule

logger = logging.getLogger(__name__)


class SkillEngine:
    """
    Central Skill Engine - manages skill lifecycle.

    Responsibilities:
    - Load skills from YAML + wire adapters
    - Provide unified access to skills
    - Track skill states
    - Coordinate eval/leveling

    Example:
        engine = SkillEngine()

        # Get a skill
        skill = engine.get("binary")

        # Generate training
        training = skill.generate_training_batch(level=5, count=100)

        # Run eval and update state
        result, state = engine.run_eval("binary", model_answers, level=5)

        # List all skills
        all_skills = engine.list_skills()
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        state_file: Optional[Path] = None,
    ):
        """
        Initialize skill engine.

        Args:
            config_dir: Directory containing skill YAML configs
                       Defaults to configs/ relative to project root
            state_file: Path to persist skill states
                       Defaults to status/skill_states.json
        """
        self.config_dir = Path(config_dir) if config_dir else get_config_dir()

        # Determine state file location
        if state_file:
            self.state_file = Path(state_file)
        else:
            # Default to status/skill_states.json relative to base_dir
            try:
                from core.paths import get_base_dir
                base_dir = get_base_dir()
            except ImportError:
                base_dir = Path.cwd()
            self.state_file = base_dir / "status" / "skill_states.json"

        # Caches
        self._skills: dict[str, Skill] = {}
        self._states: dict[str, SkillState] = {}
        self._passive_registry: dict[str, "PassiveModule"] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization - load passives and states on first access."""
        if self._initialized:
            return

        self._load_passives()
        self._load_states()
        self._initialized = True

    def _load_passives(self):
        """Auto-discover and register passives."""
        try:
            from guild.passives import get_all_passives
            for passive in get_all_passives():
                self._passive_registry[passive.id] = passive
                logger.debug(f"Registered passive: {passive.id}")
        except ImportError as e:
            logger.warning(f"Could not load passives: {e}")

    def _load_states(self):
        """Load persisted skill states."""
        if not self.state_file.exists():
            logger.debug(f"No state file at {self.state_file}, starting fresh")
            return

        try:
            data = json.loads(self.state_file.read_text())
            for skill_id, state_dict in data.items():
                self._states[skill_id] = SkillState.from_dict(state_dict)
            logger.info(f"Loaded {len(self._states)} skill states from {self.state_file}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load skill states: {e}")

    def _save_states(self):
        """Persist skill states."""
        data = {
            skill_id: state.to_dict()
            for skill_id, state in self._states.items()
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved {len(self._states)} skill states to {self.state_file}")

    def get(self, skill_id: str) -> Skill:
        """
        Get a skill by ID, loading if needed.

        Args:
            skill_id: Skill identifier (e.g., "binary", "sy")

        Returns:
            Skill instance

        Raises:
            KeyError: If skill config not found
        """
        self._ensure_initialized()

        if skill_id not in self._skills:
            self._load_skill(skill_id)

        return self._skills[skill_id]

    def _load_skill(self, skill_id: str):
        """Load and wire up a skill."""
        logger.debug(f"Loading skill: {skill_id}")

        # Load config from YAML
        config = load_skill_config(skill_id, self.config_dir)

        # Find matching passive
        passive = self._find_passive_for_skill(skill_id, config)

        # Create generator adapter if API configured
        generator = None
        if config.api_url:
            generator = GeneratorAdapter(skill_id, config.api_url)
            logger.debug(f"Created generator for {skill_id} at {config.api_url}")

        # Create passive adapter if passive found
        passive_adapter = None
        if passive:
            passive_adapter = PassiveAdapter(passive)
            logger.debug(f"Created passive adapter for {skill_id} using {passive.id}")

        # Load primitives from config if available
        primitives = self._load_primitives_for_skill(skill_id, config)

        # Create appropriate skill type
        if generator and passive_adapter:
            skill = CompositeSkill(config, generator, passive_adapter, primitives)
        elif generator:
            skill = GeneratorOnlySkill(config, generator)
        elif passive_adapter:
            skill = LocalSkill(config, passive_adapter, primitives)
        else:
            raise ValueError(
                f"Skill '{skill_id}' has no generator or passive configured. "
                f"Add API config or matching passive."
            )

        self._skills[skill_id] = skill
        logger.info(f"Loaded skill: {skill}")

    def _find_passive_for_skill(
        self,
        skill_id: str,
        config: SkillConfig
    ) -> Optional["PassiveModule"]:
        """
        Find a passive module for a skill.

        Looks for passive by:
        1. Explicit passive_id from config (highest priority)
        2. Exact skill_id match
        3. Category match

        Args:
            skill_id: Skill identifier
            config: Skill configuration

        Returns:
            PassiveModule if found, None otherwise
        """
        # Try explicit passive_id from config (highest priority)
        if config.passive_id:
            if config.passive_id in self._passive_registry:
                logger.debug(f"Found passive via config.passive_id: {config.passive_id}")
                return self._passive_registry[config.passive_id]
            else:
                logger.warning(
                    f"Skill {skill_id} specifies passive_id={config.passive_id} "
                    f"but passive not found. Available: {list(self._passive_registry.keys())}"
                )

        # Try exact skill_id match
        if skill_id in self._passive_registry:
            return self._passive_registry[skill_id]

        # Try category match
        category_value = config.category.value
        for passive in self._passive_registry.values():
            if passive.category == category_value:
                return passive

        return None

    def _load_primitives_for_skill(
        self,
        skill_id: str,
        config: SkillConfig
    ) -> list[PrimitiveId]:
        """
        Load primitives defined in skill config.

        Primitives are optional - returns empty list if none defined.
        """
        if not config.primitives:
            return []

        primitives = []
        for prim_data in config.primitives:
            try:
                prim_id = PrimitiveId(
                    name=prim_data.get("name", ""),
                    track=prim_data.get("track", skill_id),
                    version=prim_data.get("version", "v1"),
                )
                primitives.append(prim_id)
                logger.debug(f"Loaded primitive: {prim_id}")
            except Exception as e:
                logger.warning(f"Failed to parse primitive in {skill_id}: {e}")

        logger.info(f"Loaded {len(primitives)} primitives for skill {skill_id}")
        return primitives

    def get_state(self, skill_id: str) -> SkillState:
        """
        Get current state for a skill.

        Creates new state if not exists.

        Args:
            skill_id: Skill identifier

        Returns:
            SkillState for the skill
        """
        self._ensure_initialized()

        if skill_id not in self._states:
            self._states[skill_id] = SkillState(skill_id=skill_id)

        return self._states[skill_id]

    def run_eval(
        self,
        skill_id: str,
        model_answers: list[str],
        level: Optional[int] = None,
        count: int = 5,
        seed: Optional[int] = None,
    ) -> tuple[EvalResult, SkillState]:
        """
        Run a full eval cycle: generate problems, score, update state.

        This is the main entry point for running evaluations.

        Args:
            skill_id: Skill to evaluate
            model_answers: Model's answers (must match count)
            level: Skill level (defaults to current state level)
            count: Number of problems (default 5)
            seed: Optional random seed

        Returns:
            Tuple of (EvalResult, updated SkillState)

        Raises:
            ValueError: If answer count doesn't match count
        """
        if len(model_answers) != count:
            raise ValueError(
                f"Expected {count} answers, got {len(model_answers)}"
            )

        skill = self.get(skill_id)
        state = self.get_state(skill_id)

        if level is None:
            level = state.level

        logger.info(f"Running eval for {skill_id} at level {level} with {count} problems")

        # Generate eval batch
        batch = skill.generate_eval_batch(level=level, count=count, seed=seed)

        # Score answers
        result = skill.score_eval(batch, model_answers)

        # Update state
        state = skill.update_state_from_eval(state, result)
        self._states[skill_id] = state
        self._save_states()

        logger.info(
            f"Eval complete: {result.accuracy:.1%} accuracy, "
            f"level {state.level}, {state.xp_total:.0f} XP"
        )

        # Battle Log - eval result event
        try:
            from core.battle_log import log_eval
            acc_pct = result.accuracy * 100
            severity = "success" if result.accuracy >= 0.8 else ("warning" if result.accuracy >= 0.5 else "error")
            log_eval(
                f"Eval {skill_id} L{level}: {acc_pct:.1f}% ({result.correct}/{result.total})",
                severity=severity,
                source="guild.skills.engine",
                hero_id="DIO",
                details={
                    "skill_id": skill_id,
                    "level": level,
                    "accuracy": result.accuracy,
                    "correct": result.correct,
                    "total": result.total,
                    "per_primitive": result.per_primitive_accuracy,
                },
            )
        except Exception:
            pass  # Don't let battle log errors affect eval

        return result, state

    def generate_eval_batch(
        self,
        skill_id: str,
        level: Optional[int] = None,
        count: int = 5,
        seed: Optional[int] = None,
    ) -> EvalBatch:
        """
        Generate an eval batch without scoring.

        Useful when you need to run the model separately.

        Args:
            skill_id: Skill to generate for
            level: Skill level (defaults to current state level)
            count: Number of problems
            seed: Optional random seed

        Returns:
            EvalBatch with problems
        """
        skill = self.get(skill_id)

        if level is None:
            state = self.get_state(skill_id)
            level = state.level

        return skill.generate_eval_batch(level=level, count=count, seed=seed)

    def score_eval(
        self,
        skill_id: str,
        batch: EvalBatch,
        model_answers: list[str],
        update_state: bool = True,
    ) -> tuple[EvalResult, SkillState]:
        """
        Score an eval batch and optionally update state.

        Args:
            skill_id: Skill to score for
            batch: The eval batch
            model_answers: Model's answers
            update_state: Whether to update and persist state

        Returns:
            Tuple of (EvalResult, SkillState)
        """
        skill = self.get(skill_id)
        state = self.get_state(skill_id)

        # Score
        result = skill.score_eval(batch, model_answers)

        # Update state if requested
        if update_state:
            state = skill.update_state_from_eval(state, result)
            self._states[skill_id] = state
            self._save_states()

        return result, state

    def list_skills(self) -> list[str]:
        """
        List all available skill IDs.

        Returns:
            List of skill IDs from config directory
        """
        return discover_skills(self.config_dir)

    def all_skills(self) -> dict[str, Skill]:
        """
        Load and return all skills.

        Returns:
            Dict mapping skill_id to Skill
        """
        self._ensure_initialized()

        for skill_id in self.list_skills():
            if skill_id not in self._skills:
                try:
                    self._load_skill(skill_id)
                except Exception as e:
                    logger.warning(f"Failed to load skill {skill_id}: {e}")

        return self._skills.copy()

    def all_states(self) -> dict[str, SkillState]:
        """
        Get all skill states.

        Returns:
            Dict mapping skill_id to SkillState
        """
        self._ensure_initialized()
        return self._states.copy()

    def reset_state(self, skill_id: str):
        """Reset state for a skill to initial values."""
        self._states[skill_id] = SkillState(skill_id=skill_id)
        self._save_states()

    def reset_all_states(self):
        """Reset all skill states."""
        self._states.clear()
        self._save_states()

    def refresh(self):
        """Refresh engine - clear caches and reload."""
        self._skills.clear()
        self._initialized = False

    def get_primitive_summary(self) -> dict[str, dict[str, float]]:
        """
        Get per-primitive accuracy across all skills.

        Returns:
            Dict mapping primitive_id to accuracy across skills
        """
        summary: dict[str, list[float]] = {}

        for state in self._states.values():
            for prim, acc in state.primitive_accuracy.items():
                summary.setdefault(prim, []).append(acc)

        return {
            prim: sum(accs) / len(accs) if accs else 0.0
            for prim, accs in summary.items()
        }

    def health_check(self) -> dict:
        """
        Check health of all loaded skills.

        Returns:
            Dict with health status
        """
        self._ensure_initialized()

        healthy_count = 0
        unhealthy = []

        for skill_id, skill in self._skills.items():
            if isinstance(skill, CompositeSkill):
                if skill.generator.health():
                    healthy_count += 1
                else:
                    unhealthy.append(skill_id)
            else:
                healthy_count += 1  # Local skills are always healthy

        return {
            "total_skills": len(self._skills),
            "healthy": healthy_count,
            "unhealthy": unhealthy,
            "states_loaded": len(self._states),
            "passives_registered": len(self._passive_registry),
        }

    def __repr__(self) -> str:
        return (
            f"SkillEngine(config_dir={self.config_dir}, "
            f"skills={len(self._skills)}, states={len(self._states)})"
        )


# =============================================================================
# Global engine access
# =============================================================================

_engine: Optional[SkillEngine] = None


def get_engine(
    config_dir: Optional[Path] = None,
    state_file: Optional[Path] = None,
) -> SkillEngine:
    """
    Get the global skill engine, initializing if needed.

    Args:
        config_dir: Optional config directory (only used on first call)
        state_file: Optional state file path (only used on first call)

    Returns:
        Global SkillEngine instance
    """
    global _engine
    if _engine is None:
        _engine = SkillEngine(config_dir=config_dir, state_file=state_file)
    return _engine


def reset_engine():
    """Reset the global skill engine (useful for testing)."""
    global _engine
    _engine = None


def init_engine(
    config_dir: Optional[Path] = None,
    state_file: Optional[Path] = None,
) -> SkillEngine:
    """
    Initialize a new global skill engine.

    Replaces any existing engine.

    Args:
        config_dir: Config directory
        state_file: State file path

    Returns:
        New SkillEngine instance
    """
    global _engine
    _engine = SkillEngine(config_dir=config_dir, state_file=state_file)
    return _engine
