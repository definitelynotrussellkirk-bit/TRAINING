"""Quest template loading from YAML files."""

from pathlib import Path
from typing import Optional

from guild.config.loader import (
    get_config_dir,
    load_yaml,
)
from guild.quests.types import QuestTemplate, QuestDifficulty


def load_quest_template(
    quest_id: str,
    category: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> QuestTemplate:
    """
    Load a quest template from YAML.

    Quest templates are stored in configs/quests/{category}/{id}.yaml
    or configs/quests/{id}.yaml if no category.

    Args:
        quest_id: Quest identifier
        category: Optional category subdirectory
        config_dir: Optional config directory

    Returns:
        QuestTemplate instance

    Raises:
        FileNotFoundError: If template not found
        ValueError: If config is invalid
    """
    if config_dir is None:
        config_dir = get_config_dir()

    quests_dir = config_dir / "quests"

    # Try category path first, then root
    if category:
        quest_path = quests_dir / category / f"{quest_id}.yaml"
    else:
        quest_path = quests_dir / f"{quest_id}.yaml"

    if not quest_path.exists():
        # Try finding in any category
        for subdir in quests_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / f"{quest_id}.yaml"
                if candidate.exists():
                    quest_path = candidate
                    break

    if not quest_path.exists():
        raise FileNotFoundError(f"Quest template not found: {quest_id}")

    data = load_yaml(quest_path)
    return _dict_to_quest_template(data, quest_id)


def _dict_to_quest_template(data: dict, default_id: str = "") -> QuestTemplate:
    """Convert dict to QuestTemplate with validation."""
    # Required fields
    quest_id = data.get("id", default_id)
    if not quest_id:
        raise ValueError("Quest template missing 'id' field")

    name = data.get("name", quest_id)
    description = data.get("description", "")

    # Skills and regions
    skills = data.get("skills", [])
    if isinstance(skills, str):
        skills = [skills]
    if not skills:
        raise ValueError(f"Quest '{quest_id}' must have at least one skill")

    regions = data.get("regions", [])
    if isinstance(regions, str):
        regions = [regions]

    # Difficulty handling
    difficulty_level = data.get("difficulty_level", 1)
    difficulty = data.get("difficulty")
    if difficulty is None:
        difficulty = QuestDifficulty.from_level(difficulty_level)
    elif isinstance(difficulty, str):
        difficulty = QuestDifficulty[difficulty.upper()]
    elif isinstance(difficulty, int):
        difficulty = QuestDifficulty(difficulty)

    # Generator and evaluator
    generator_id = data.get("generator_id", "default")
    evaluator_id = data.get("evaluator_id", "exact_match")
    generator_params = data.get("generator_params", {})
    evaluator_params = data.get("evaluator_params", {})

    # XP rewards
    base_xp = data.get("base_xp", {})
    if not base_xp:
        # Default: 10 XP per skill
        base_xp = {skill: 10 for skill in skills}

    # Optional
    tags = data.get("tags", [])
    primitives = data.get("primitives", [])
    if isinstance(primitives, str):
        primitives = [primitives]
    module_id = data.get("module_id")
    enabled = data.get("enabled", True)

    return QuestTemplate(
        id=quest_id,
        name=name,
        description=description,
        skills=skills,
        regions=regions,
        difficulty=difficulty,
        difficulty_level=difficulty_level,
        generator_id=generator_id,
        evaluator_id=evaluator_id,
        generator_params=generator_params,
        evaluator_params=evaluator_params,
        base_xp=base_xp,
        tags=tags,
        primitives=primitives,
        module_id=module_id,
        enabled=enabled,
    )


def discover_quest_templates(config_dir: Optional[Path] = None) -> list[tuple[str, str]]:
    """
    Discover all quest template IDs and their categories.

    Returns:
        List of (quest_id, category) tuples. Category is empty string for root templates.
    """
    if config_dir is None:
        config_dir = get_config_dir()

    quests_dir = config_dir / "quests"
    if not quests_dir.exists():
        return []

    templates = []

    # Root-level templates
    for path in quests_dir.glob("*.yaml"):
        if not path.name.startswith("_"):
            templates.append((path.stem, ""))

    # Category subdirectories
    for subdir in quests_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("_"):
            for path in subdir.glob("*.yaml"):
                if not path.name.startswith("_"):
                    templates.append((path.stem, subdir.name))

    return sorted(templates)


def load_all_quest_templates(
    config_dir: Optional[Path] = None,
) -> dict[str, QuestTemplate]:
    """
    Load all quest templates.

    Returns:
        Dict mapping quest_id to QuestTemplate
    """
    if config_dir is None:
        config_dir = get_config_dir()

    templates = {}
    for quest_id, category in discover_quest_templates(config_dir):
        try:
            templates[quest_id] = load_quest_template(quest_id, category, config_dir)
        except Exception as e:
            import logging
            logging.warning(f"Failed to load quest '{quest_id}': {e}")

    return templates


class QuestLoader:
    """
    Cached quest template loader.

    Provides caching to avoid re-reading YAML files.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else get_config_dir()
        self._cache: dict[str, QuestTemplate] = {}
        self._discovered: Optional[list[tuple[str, str]]] = None

    def load(
        self,
        quest_id: str,
        category: Optional[str] = None,
        use_cache: bool = True,
    ) -> QuestTemplate:
        """Load a quest template with optional caching."""
        cache_key = f"{category or ''}:{quest_id}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        template = load_quest_template(quest_id, category, self.config_dir)

        if use_cache:
            self._cache[cache_key] = template

        return template

    def load_all(self, use_cache: bool = True) -> dict[str, QuestTemplate]:
        """Load all quest templates with optional caching."""
        if use_cache and self._cache:
            return self._cache.copy()

        templates = load_all_quest_templates(self.config_dir)

        if use_cache:
            self._cache = {f":{k}": v for k, v in templates.items()}

        return templates

    def discover(self, use_cache: bool = True) -> list[tuple[str, str]]:
        """Discover quest template IDs with optional caching."""
        if use_cache and self._discovered is not None:
            return self._discovered.copy()

        discovered = discover_quest_templates(self.config_dir)

        if use_cache:
            self._discovered = discovered.copy()

        return discovered

    def exists(self, quest_id: str) -> bool:
        """Check if a quest template exists."""
        return any(qid == quest_id for qid, _ in self.discover())

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._discovered = None

    def invalidate(self, quest_id: str):
        """Invalidate cache for a specific quest."""
        keys_to_remove = [k for k in self._cache if k.endswith(f":{quest_id}")]
        for key in keys_to_remove:
            del self._cache[key]
