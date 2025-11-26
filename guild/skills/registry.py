"""Skill registry - central access point for skill definitions."""

from pathlib import Path
from typing import Optional, Iterator

from guild.skills.types import SkillConfig, SkillCategory
from guild.skills.loader import SkillLoader


class SkillRegistry:
    """
    Central registry for skill configurations.

    Provides:
    - Lazy loading of skill configs from YAML
    - Lookup by ID, category, or tag
    - Iteration over all skills
    - Singleton-style global access via module functions

    Example:
        registry = SkillRegistry()
        skill = registry.get("logic_weaving")
        reasoning_skills = registry.by_category(SkillCategory.REASONING)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self._loader = SkillLoader(config_dir)
        self._loaded = False

    def _ensure_loaded(self):
        """Ensure all skills are loaded."""
        if not self._loaded:
            self._loader.load_all()
            self._loaded = True

    def get(self, skill_id: str) -> SkillConfig:
        """
        Get a skill by ID.

        Args:
            skill_id: Skill identifier

        Returns:
            SkillConfig

        Raises:
            KeyError: If skill not found
        """
        if not self._loader.exists(skill_id):
            raise KeyError(f"Unknown skill: {skill_id}")
        return self._loader.load(skill_id)

    def get_or_none(self, skill_id: str) -> Optional[SkillConfig]:
        """Get a skill by ID, returning None if not found."""
        try:
            return self.get(skill_id)
        except (KeyError, FileNotFoundError):
            return None

    def exists(self, skill_id: str) -> bool:
        """Check if a skill exists."""
        return self._loader.exists(skill_id)

    def list_ids(self) -> list[str]:
        """List all skill IDs."""
        return self._loader.discover()

    def all(self) -> dict[str, SkillConfig]:
        """Get all skills as a dict."""
        self._ensure_loaded()
        return self._loader.load_all()

    def __iter__(self) -> Iterator[SkillConfig]:
        """Iterate over all skill configs."""
        self._ensure_loaded()
        for config in self._loader.load_all().values():
            yield config

    def __len__(self) -> int:
        """Number of registered skills."""
        return len(self._loader.discover())

    def __contains__(self, skill_id: str) -> bool:
        """Check if skill exists."""
        return self.exists(skill_id)

    def by_category(self, category: SkillCategory) -> list[SkillConfig]:
        """Get all skills in a category."""
        self._ensure_loaded()
        return [
            skill for skill in self._loader.load_all().values()
            if skill.category == category
        ]

    def by_tag(self, tag: str) -> list[SkillConfig]:
        """Get all skills with a specific tag."""
        self._ensure_loaded()
        return [
            skill for skill in self._loader.load_all().values()
            if tag in skill.tags
        ]

    def search(
        self,
        category: Optional[SkillCategory] = None,
        tags: Optional[list[str]] = None,
        name_contains: Optional[str] = None,
    ) -> list[SkillConfig]:
        """
        Search skills by multiple criteria.

        Args:
            category: Filter by category
            tags: Filter by tags (skill must have ALL tags)
            name_contains: Filter by name substring (case-insensitive)

        Returns:
            List of matching skills
        """
        self._ensure_loaded()
        results = list(self._loader.load_all().values())

        if category is not None:
            results = [s for s in results if s.category == category]

        if tags:
            results = [s for s in results if all(t in s.tags for t in tags)]

        if name_contains:
            needle = name_contains.lower()
            results = [s for s in results if needle in s.name.lower()]

        return results

    def refresh(self):
        """Refresh the registry by clearing caches."""
        self._loader.clear_cache()
        self._loaded = False

    def invalidate(self, skill_id: str):
        """Invalidate a specific skill's cache."""
        self._loader.invalidate(skill_id)


# Global registry instance
_registry: Optional[SkillRegistry] = None


def init_registry(config_dir: Optional[Path] = None) -> SkillRegistry:
    """Initialize the global skill registry."""
    global _registry
    _registry = SkillRegistry(config_dir)
    return _registry


def get_registry() -> SkillRegistry:
    """Get the global skill registry, initializing if needed."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def reset_registry():
    """Reset the global skill registry (useful for testing)."""
    global _registry
    _registry = None


# Convenience functions using global registry

def get_skill(skill_id: str) -> SkillConfig:
    """Get a skill by ID from the global registry."""
    return get_registry().get(skill_id)


def list_skills() -> list[str]:
    """List all skill IDs from the global registry."""
    return get_registry().list_ids()


def skills_by_category(category: SkillCategory) -> list[SkillConfig]:
    """Get skills by category from the global registry."""
    return get_registry().by_category(category)


def skills_by_tag(tag: str) -> list[SkillConfig]:
    """Get skills by tag from the global registry."""
    return get_registry().by_tag(tag)
