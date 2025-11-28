"""
Hero Titles - Threshold-based labels for capability milestones.

Titles provide quick visual feedback on training progress and skill mastery.
They're not cosmetic - they represent meaningful capability thresholds.

Usage:
    from guild.titles import get_titles, get_primary_title

    # Get all earned titles
    titles = get_titles(hero_state, skill_states)

    # Get the primary (highest priority) title
    title = get_primary_title(hero_state, skill_states)
    print(f"DIO - {title.name}")  # "DIO - Binary Arithmetician"
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class Title:
    """A title that can be earned."""
    id: str
    name: str
    description: str = ""
    icon: str = ""
    priority: int = 0
    category: str = "general"  # global, skill, primitive, warning, achievement
    skill_id: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TitleResult:
    """Result of title evaluation."""
    primary: Optional[Title] = None
    skill_titles: Dict[str, Title] = field(default_factory=dict)  # skill_id -> title
    warnings: List[Title] = field(default_factory=list)
    achievements: List[Title] = field(default_factory=list)
    all_titles: List[Title] = field(default_factory=list)


class TitleEngine:
    """Evaluates and awards titles based on hero/skill state."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "titles.yaml"
        self.config_path = config_path
        self._config: Optional[Dict] = None

    @property
    def config(self) -> Dict:
        """Load config lazily."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> Dict:
        """Load titles configuration from YAML."""
        if not self.config_path.exists():
            return {"global_titles": [], "skill_titles": {}, "primitive_titles": [], "warnings": [], "achievements": []}

        with open(self.config_path) as f:
            return yaml.safe_load(f) or {}

    def reload(self):
        """Reload configuration from disk."""
        self._config = None

    def evaluate(
        self,
        total_steps: int = 0,
        total_level: int = 0,
        skill_states: Optional[Dict[str, Any]] = None,
    ) -> TitleResult:
        """
        Evaluate all titles and return earned ones.

        Args:
            total_steps: Total training steps completed
            total_level: Sum of all skill levels
            skill_states: Dict of skill_id -> SkillState (or dict with level, accuracy, primitive_accuracy)

        Returns:
            TitleResult with all earned titles categorized
        """
        skill_states = skill_states or {}
        result = TitleResult()

        # Evaluate global titles
        global_title = self._evaluate_global(total_steps, total_level, skill_states)
        if global_title:
            result.all_titles.append(global_title)
            result.primary = global_title

        # Evaluate skill-specific titles
        for skill_id, state in skill_states.items():
            skill_title = self._evaluate_skill(skill_id, state)
            if skill_title:
                result.skill_titles[skill_id] = skill_title
                result.all_titles.append(skill_title)
                # Skill title can override global if higher priority
                if result.primary is None or skill_title.priority > result.primary.priority:
                    result.primary = skill_title

        # Evaluate primitive mastery titles
        for title in self._evaluate_primitives(skill_states):
            result.all_titles.append(title)
            if result.primary is None or title.priority > result.primary.priority:
                result.primary = title

        # Evaluate warnings
        result.warnings = self._evaluate_warnings(skill_states)

        # Evaluate achievements
        result.achievements = self._evaluate_achievements(total_steps, total_level, skill_states)
        result.all_titles.extend(result.achievements)

        return result

    def _evaluate_global(
        self,
        total_steps: int,
        total_level: int,
        skill_states: Dict[str, Any],
    ) -> Optional[Title]:
        """Evaluate global titles, return highest priority match."""
        global_titles = self.config.get("global_titles", [])
        matched = None

        for t in global_titles:
            conditions = t.get("conditions", {})
            if self._check_condition(conditions.get("total_steps"), total_steps):
                title = Title(
                    id=t["id"],
                    name=t["name"],
                    description=t.get("description", ""),
                    priority=t.get("priority", 0),
                    category="global",
                    conditions=conditions,
                )
                if matched is None or title.priority > matched.priority:
                    matched = title

        return matched

    def _evaluate_skill(self, skill_id: str, state: Any) -> Optional[Title]:
        """Evaluate skill-specific titles for a single skill."""
        skill_titles = self.config.get("skill_titles", {}).get(skill_id, [])

        # Extract level and accuracy from state (supports both SkillState and dict)
        if hasattr(state, "level"):
            level = state.level
            accuracy = state.accuracy
        else:
            level = state.get("level", 1)
            accuracy = state.get("accuracy", 0.0)

        matched = None
        for t in skill_titles:
            conditions = t.get("conditions", {})

            # Check level condition
            if not self._check_condition(conditions.get("level"), level):
                continue

            # Check accuracy condition
            if not self._check_condition(conditions.get("accuracy"), accuracy):
                continue

            title = Title(
                id=t["id"],
                name=t["name"],
                description=t.get("description", ""),
                priority=t.get("priority", 0),
                category="skill",
                skill_id=skill_id,
                conditions=conditions,
            )
            if matched is None or title.priority > matched.priority:
                matched = title

        return matched

    def _evaluate_primitives(self, skill_states: Dict[str, Any]) -> List[Title]:
        """Evaluate primitive mastery titles."""
        primitive_titles = self.config.get("primitive_titles", [])
        earned = []

        for t in primitive_titles:
            skill_id = t.get("skill")
            primitive = t.get("primitive")
            conditions = t.get("conditions", {})

            if skill_id not in skill_states:
                continue

            state = skill_states[skill_id]

            # Get primitive accuracy
            if hasattr(state, "primitive_accuracy"):
                prim_acc = state.primitive_accuracy.get(primitive, 0.0)
            else:
                prim_acc = state.get("primitive_accuracy", {}).get(primitive, 0.0)

            if self._check_condition(conditions.get("accuracy"), prim_acc):
                earned.append(Title(
                    id=t["id"],
                    name=t["name"],
                    description=t.get("description", ""),
                    priority=t.get("priority", 0),
                    category="primitive",
                    skill_id=skill_id,
                    conditions=conditions,
                ))

        return earned

    def _evaluate_warnings(self, skill_states: Dict[str, Any]) -> List[Title]:
        """Evaluate warning conditions."""
        warnings_config = self.config.get("warnings", [])
        warnings = []

        for w in warnings_config:
            conditions = w.get("conditions", {})

            # Check any_primitive_below
            threshold = conditions.get("any_primitive_below")
            if threshold is not None:
                for skill_id, state in skill_states.items():
                    if hasattr(state, "primitive_accuracy"):
                        prim_acc = state.primitive_accuracy
                    else:
                        prim_acc = state.get("primitive_accuracy", {})

                    for prim, acc in prim_acc.items():
                        if acc < threshold:
                            warnings.append(Title(
                                id=w["id"],
                                name=w["name"],
                                description=f"{prim}: {acc:.0%}",
                                icon=w.get("icon", ""),
                                category="warning",
                                skill_id=skill_id,
                            ))
                            break  # One warning per skill

        return warnings

    def _evaluate_achievements(
        self,
        total_steps: int,
        total_level: int,
        skill_states: Dict[str, Any],
    ) -> List[Title]:
        """Evaluate achievement titles."""
        achievements_config = self.config.get("achievements", [])
        earned = []

        for a in achievements_config:
            conditions = a.get("conditions", {})

            # Skip trigger-based achievements (need event system)
            if "trigger" in a:
                continue

            # Check total_steps
            if not self._check_condition(conditions.get("total_steps"), total_steps):
                continue

            # Check skills_at_level_10
            level_cond = conditions.get("skills_at_level_10")
            if level_cond is not None:
                count = sum(
                    1 for s in skill_states.values()
                    if (s.level if hasattr(s, "level") else s.get("level", 0)) >= 10
                )
                if not self._check_condition(level_cond, count):
                    continue

            earned.append(Title(
                id=a["id"],
                name=a["name"],
                description=a.get("description", ""),
                icon=a.get("icon", ""),
                priority=a.get("priority", 0),
                category="achievement",
            ))

        return earned

    def _check_condition(self, condition: Any, value: float) -> bool:
        """
        Check a condition against a value.

        Condition can be:
            - None: Always passes
            - {gte: X}: value >= X
            - {gt: X}: value > X
            - {lte: X}: value <= X
            - {lt: X}: value < X
            - {gte: X, lt: Y}: X <= value < Y
        """
        if condition is None:
            return True

        if not isinstance(condition, dict):
            return value == condition

        if "gte" in condition and value < condition["gte"]:
            return False
        if "gt" in condition and value <= condition["gt"]:
            return False
        if "lte" in condition and value > condition["lte"]:
            return False
        if "lt" in condition and value >= condition["lt"]:
            return False

        return True


# Module-level singleton
_engine: Optional[TitleEngine] = None


def get_engine() -> TitleEngine:
    """Get the title engine singleton."""
    global _engine
    if _engine is None:
        _engine = TitleEngine()
    return _engine


def get_titles(
    total_steps: int = 0,
    total_level: int = 0,
    skill_states: Optional[Dict[str, Any]] = None,
) -> TitleResult:
    """
    Get all earned titles for the current state.

    Convenience wrapper around TitleEngine.evaluate().
    """
    return get_engine().evaluate(total_steps, total_level, skill_states)


def get_primary_title(
    total_steps: int = 0,
    total_level: int = 0,
    skill_states: Optional[Dict[str, Any]] = None,
) -> Optional[Title]:
    """
    Get the primary (highest priority) title.

    Convenience wrapper for quick title display.
    """
    result = get_titles(total_steps, total_level, skill_states)
    return result.primary


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Title Engine CLI")
    parser.add_argument("--steps", type=int, default=180000, help="Total training steps")
    parser.add_argument("--skill", type=str, default="bin", help="Skill ID to test")
    parser.add_argument("--level", type=int, default=5, help="Skill level")
    parser.add_argument("--accuracy", type=float, default=0.85, help="Skill accuracy")
    args = parser.parse_args()

    # Build test state
    skill_states = {
        args.skill: {
            "level": args.level,
            "accuracy": args.accuracy,
            "primitive_accuracy": {
                "binary_add_no_carry": 0.95,
                "binary_add_with_carry": 0.80,
                "bitwise_and": 0.90,
            }
        }
    }

    total_level = args.level

    print(f"Testing with steps={args.steps}, skill={args.skill} L{args.level} @ {args.accuracy:.0%}")
    print("=" * 60)

    result = get_titles(args.steps, total_level, skill_states)

    print(f"\nPrimary Title: {result.primary.name if result.primary else 'None'}")
    if result.primary:
        print(f"  - {result.primary.description}")

    print(f"\nSkill Titles:")
    for skill_id, title in result.skill_titles.items():
        print(f"  {skill_id}: {title.name} - {title.description}")

    print(f"\nWarnings: {len(result.warnings)}")
    for w in result.warnings:
        print(f"  {w.icon} {w.name}: {w.description}")

    print(f"\nAchievements: {len(result.achievements)}")
    for a in result.achievements:
        print(f"  {a.icon} {a.name}")

    print(f"\nAll Titles: {len(result.all_titles)}")
