"""Quest template registry - central access point for quest definitions."""

from pathlib import Path
from typing import Optional, Iterator

from guild.quests.types import QuestTemplate, QuestDifficulty
from guild.quests.loader import QuestLoader


class QuestRegistry:
    """
    Central registry for quest templates.

    Provides:
    - Lazy loading of templates from YAML
    - Lookup by ID, skill, difficulty, or tags
    - Iteration over all templates
    - Singleton-style global access via module functions

    Example:
        registry = QuestRegistry()
        quest = registry.get("syllo_puzzle")
        logic_quests = registry.by_skill("logic_weaving")
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self._loader = QuestLoader(config_dir)
        self._loaded = False

    def _ensure_loaded(self):
        """Ensure all templates are loaded."""
        if not self._loaded:
            self._loader.load_all()
            self._loaded = True

    def get(self, quest_id: str) -> QuestTemplate:
        """
        Get a quest template by ID.

        Args:
            quest_id: Quest identifier

        Returns:
            QuestTemplate

        Raises:
            KeyError: If template not found
        """
        if not self._loader.exists(quest_id):
            raise KeyError(f"Unknown quest: {quest_id}")
        return self._loader.load(quest_id)

    def get_or_none(self, quest_id: str) -> Optional[QuestTemplate]:
        """Get a quest template by ID, returning None if not found."""
        try:
            return self.get(quest_id)
        except (KeyError, FileNotFoundError):
            return None

    def exists(self, quest_id: str) -> bool:
        """Check if a quest template exists."""
        return self._loader.exists(quest_id)

    def list_ids(self) -> list[str]:
        """List all quest template IDs."""
        return [qid for qid, _ in self._loader.discover()]

    def all(self) -> dict[str, QuestTemplate]:
        """Get all templates as a dict."""
        self._ensure_loaded()
        return self._loader.load_all()

    def __iter__(self) -> Iterator[QuestTemplate]:
        """Iterate over all quest templates."""
        self._ensure_loaded()
        for template in self._loader.load_all().values():
            yield template

    def __len__(self) -> int:
        """Number of registered quest templates."""
        return len(self._loader.discover())

    def __contains__(self, quest_id: str) -> bool:
        """Check if quest exists."""
        return self.exists(quest_id)

    def by_skill(self, skill_id: str) -> list[QuestTemplate]:
        """Get all quests that train a specific skill."""
        self._ensure_loaded()
        return [
            quest for quest in self._loader.load_all().values()
            if skill_id in quest.skills
        ]

    def by_difficulty(self, difficulty: QuestDifficulty) -> list[QuestTemplate]:
        """Get all quests at a specific difficulty tier."""
        self._ensure_loaded()
        return [
            quest for quest in self._loader.load_all().values()
            if quest.difficulty == difficulty
        ]

    def by_difficulty_level(
        self,
        level: int,
        tolerance: int = 0,
    ) -> list[QuestTemplate]:
        """
        Get all quests at a specific difficulty level.

        Args:
            level: Target difficulty level (1-10)
            tolerance: How many levels above/below to include

        Returns:
            List of matching templates
        """
        self._ensure_loaded()
        min_level = max(1, level - tolerance)
        max_level = min(10, level + tolerance)

        return [
            quest for quest in self._loader.load_all().values()
            if min_level <= quest.difficulty_level <= max_level
        ]

    def by_tag(self, tag: str) -> list[QuestTemplate]:
        """Get all quests with a specific tag."""
        self._ensure_loaded()
        return [
            quest for quest in self._loader.load_all().values()
            if tag in quest.tags
        ]

    def by_region(self, region: str) -> list[QuestTemplate]:
        """Get all quests in a specific region."""
        self._ensure_loaded()
        return [
            quest for quest in self._loader.load_all().values()
            if region in quest.regions
        ]

    def enabled_only(self) -> list[QuestTemplate]:
        """Get only enabled quest templates."""
        self._ensure_loaded()
        return [
            quest for quest in self._loader.load_all().values()
            if quest.enabled
        ]

    def search(
        self,
        skill: Optional[str] = None,
        difficulty: Optional[QuestDifficulty] = None,
        difficulty_level: Optional[int] = None,
        tags: Optional[list[str]] = None,
        region: Optional[str] = None,
        enabled_only: bool = True,
    ) -> list[QuestTemplate]:
        """
        Search quests by multiple criteria.

        Args:
            skill: Filter by skill
            difficulty: Filter by difficulty tier
            difficulty_level: Filter by difficulty level
            tags: Filter by tags (must have ALL)
            region: Filter by region
            enabled_only: Only return enabled quests

        Returns:
            List of matching templates
        """
        self._ensure_loaded()
        results = list(self._loader.load_all().values())

        if enabled_only:
            results = [q for q in results if q.enabled]

        if skill is not None:
            results = [q for q in results if skill in q.skills]

        if difficulty is not None:
            results = [q for q in results if q.difficulty == difficulty]

        if difficulty_level is not None:
            results = [q for q in results if q.difficulty_level == difficulty_level]

        if tags:
            results = [q for q in results if all(t in q.tags for t in tags)]

        if region is not None:
            results = [q for q in results if region in q.regions]

        return results

    def refresh(self):
        """Refresh the registry by clearing caches."""
        self._loader.clear_cache()
        self._loaded = False

    def invalidate(self, quest_id: str):
        """Invalidate a specific quest's cache."""
        self._loader.invalidate(quest_id)


# Global registry instance
_registry: Optional[QuestRegistry] = None


def init_quest_registry(config_dir: Optional[Path] = None) -> QuestRegistry:
    """Initialize the global quest registry."""
    global _registry
    _registry = QuestRegistry(config_dir)
    return _registry


def get_quest_registry() -> QuestRegistry:
    """Get the global quest registry, initializing if needed."""
    global _registry
    if _registry is None:
        _registry = QuestRegistry()
    return _registry


def reset_quest_registry():
    """Reset the global quest registry (useful for testing)."""
    global _registry
    _registry = None


# Convenience functions using global registry

def get_quest(quest_id: str) -> QuestTemplate:
    """Get a quest template by ID from the global registry."""
    return get_quest_registry().get(quest_id)


def list_quests() -> list[str]:
    """List all quest IDs from the global registry."""
    return get_quest_registry().list_ids()


def quests_by_skill(skill_id: str) -> list[QuestTemplate]:
    """Get quests by skill from the global registry."""
    return get_quest_registry().by_skill(skill_id)


def quests_by_difficulty(level: int) -> list[QuestTemplate]:
    """Get quests by difficulty level from the global registry."""
    return get_quest_registry().by_difficulty_level(level)
