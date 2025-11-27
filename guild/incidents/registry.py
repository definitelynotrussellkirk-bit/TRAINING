"""Incident rule registry - central access point for incident rule definitions."""

from pathlib import Path
from typing import Optional, Iterator

from guild.incidents.types import IncidentRule, IncidentCategory
from guild.incidents.loader import IncidentRuleLoader
from guild.types import Severity


class IncidentRuleRegistry:
    """
    Central registry for incident rule configurations.

    Provides:
    - Lazy loading of incident rules from YAML
    - Lookup by ID, category, or severity
    - Iteration over all rules
    - Singleton-style global access via module functions

    Example:
        registry = IncidentRuleRegistry()
        rule = registry.get("oom_crash")
        training_rules = registry.by_category(IncidentCategory.TRAINING)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self._loader = IncidentRuleLoader(config_dir)
        self._loaded = False

    def _ensure_loaded(self):
        """Ensure all rules are loaded."""
        if not self._loaded:
            self._loader.load_all()
            self._loaded = True

    def get(self, rule_id: str) -> IncidentRule:
        """
        Get a rule by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            IncidentRule

        Raises:
            KeyError: If rule not found
        """
        if not self._loader.exists(rule_id):
            raise KeyError(f"Unknown incident rule: {rule_id}")
        return self._loader.load(rule_id)

    def get_or_none(self, rule_id: str) -> Optional[IncidentRule]:
        """Get a rule by ID, returning None if not found."""
        try:
            return self.get(rule_id)
        except (KeyError, FileNotFoundError):
            return None

    def exists(self, rule_id: str) -> bool:
        """Check if a rule exists."""
        return self._loader.exists(rule_id)

    def list_ids(self) -> list[str]:
        """List all rule IDs."""
        return self._loader.discover()

    def all(self) -> dict[str, IncidentRule]:
        """Get all rules as a dict."""
        self._ensure_loaded()
        return self._loader.load_all()

    def __iter__(self) -> Iterator[IncidentRule]:
        """Iterate over all rule configs."""
        self._ensure_loaded()
        for rule in self._loader.load_all().values():
            yield rule

    def __len__(self) -> int:
        """Number of registered rules."""
        return len(self._loader.discover())

    def __contains__(self, rule_id: str) -> bool:
        """Check if rule exists."""
        return self.exists(rule_id)

    def by_category(self, category: IncidentCategory) -> list[IncidentRule]:
        """Get all rules of a specific category."""
        self._ensure_loaded()
        return [
            rule for rule in self._loader.load_all().values()
            if rule.category == category
        ]

    def by_severity(self, severity: Severity) -> list[IncidentRule]:
        """Get all rules with a specific severity."""
        self._ensure_loaded()
        return [
            rule for rule in self._loader.load_all().values()
            if rule.severity == severity
        ]

    def by_detector_type(self, detector_type: str) -> list[IncidentRule]:
        """Get all rules using a specific detector type."""
        self._ensure_loaded()
        return [
            rule for rule in self._loader.load_all().values()
            if rule.detector_type == detector_type
        ]

    def search(
        self,
        category: Optional[IncidentCategory] = None,
        severity: Optional[Severity] = None,
        detector_type: Optional[str] = None,
        name_contains: Optional[str] = None,
    ) -> list[IncidentRule]:
        """
        Search rules by multiple criteria.

        Args:
            category: Filter by category
            severity: Filter by severity
            detector_type: Filter by detector type
            name_contains: Filter by name substring (case-insensitive)

        Returns:
            List of matching rules
        """
        self._ensure_loaded()
        results = list(self._loader.load_all().values())

        if category is not None:
            results = [r for r in results if r.category == category]

        if severity is not None:
            results = [r for r in results if r.severity == severity]

        if detector_type is not None:
            results = [r for r in results if r.detector_type == detector_type]

        if name_contains:
            needle = name_contains.lower()
            results = [r for r in results if needle in r.name.lower()]

        return results

    def refresh(self):
        """Refresh the registry by clearing caches."""
        self._loader.clear_cache()
        self._loaded = False

    def invalidate(self, rule_id: str):
        """Invalidate a specific rule's cache."""
        self._loader.invalidate(rule_id)


# Global registry instance
_registry: Optional[IncidentRuleRegistry] = None


def init_incident_rule_registry(config_dir: Optional[Path] = None) -> IncidentRuleRegistry:
    """Initialize the global incident rule registry."""
    global _registry
    _registry = IncidentRuleRegistry(config_dir)
    return _registry


def get_incident_rule_registry() -> IncidentRuleRegistry:
    """Get the global incident rule registry, initializing if needed."""
    global _registry
    if _registry is None:
        _registry = IncidentRuleRegistry()
    return _registry


def reset_incident_rule_registry():
    """Reset the global incident rule registry (useful for testing)."""
    global _registry
    _registry = None


# Convenience functions using global registry

def get_incident_rule(rule_id: str) -> IncidentRule:
    """Get an incident rule by ID from the global registry."""
    return get_incident_rule_registry().get(rule_id)


def list_incident_rules() -> list[str]:
    """List all incident rule IDs from the global registry."""
    return get_incident_rule_registry().list_ids()


def incident_rules_by_category(category: IncidentCategory) -> list[IncidentRule]:
    """Get incident rules by category from the global registry."""
    return get_incident_rule_registry().by_category(category)


def incident_rules_by_severity(severity: Severity) -> list[IncidentRule]:
    """Get incident rules by severity from the global registry."""
    return get_incident_rule_registry().by_severity(severity)
