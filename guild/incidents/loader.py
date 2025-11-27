"""Incident rule loading from YAML files."""

from pathlib import Path
from typing import Optional

from guild.config.loader import get_config_dir, load_yaml
from guild.incidents.types import IncidentRule, IncidentCategory
from guild.types import Severity


def load_incident_rule(rule_id: str, config_dir: Optional[Path] = None) -> IncidentRule:
    """
    Load an incident rule configuration from YAML.

    Args:
        rule_id: Rule identifier (e.g., "oom_crash")
        config_dir: Optional config directory (defaults to GUILD_CONFIG_DIR)

    Returns:
        IncidentRule instance

    Raises:
        FileNotFoundError: If rule config file doesn't exist
        ValueError: If config is invalid
    """
    if config_dir is None:
        config_dir = get_config_dir()

    rule_path = config_dir / "incidents" / f"{rule_id}.yaml"

    if not rule_path.exists():
        raise FileNotFoundError(f"Incident rule config not found: {rule_path}")

    data = load_yaml(rule_path)
    return _dict_to_incident_rule(data)


def _dict_to_incident_rule(data: dict) -> IncidentRule:
    """Convert a dict to IncidentRule, handling enum conversion."""
    rule_id = data.get("id")
    if not rule_id:
        raise ValueError("Incident rule config missing 'id' field")

    name = data.get("name", rule_id)

    # Category enum
    category_str = data.get("category", "logic")
    try:
        category = IncidentCategory(category_str)
    except ValueError:
        valid = [c.value for c in IncidentCategory]
        raise ValueError(f"Invalid incident category: {category_str}. Valid: {valid}")

    # Severity enum
    severity_str = data.get("severity", "medium")
    try:
        severity = Severity(severity_str)
    except ValueError:
        valid = [s.value for s in Severity]
        raise ValueError(f"Invalid severity: {severity_str}. Valid: {valid}")

    # Detector config
    detector_type = data.get("detector_type", "pattern")
    detector_config = data.get("detector_config", {})

    return IncidentRule(
        id=rule_id,
        name=name,
        category=category,
        severity=severity,
        detector_type=detector_type,
        detector_config=detector_config,
        title_template=data.get("title_template", ""),
        description_template=data.get("description_template", ""),
        rpg_name_template=data.get("rpg_name_template"),
    )


def discover_incident_rules(config_dir: Optional[Path] = None) -> list[str]:
    """
    Discover all incident rule IDs from config files.

    Returns:
        List of rule IDs (file stems from configs/incidents/*.yaml)
    """
    if config_dir is None:
        config_dir = get_config_dir()

    incidents_dir = config_dir / "incidents"
    if not incidents_dir.exists():
        return []

    rule_ids = []
    for path in incidents_dir.glob("*.yaml"):
        if not path.name.startswith("_"):
            rule_ids.append(path.stem)

    return sorted(rule_ids)


def load_all_incident_rules(config_dir: Optional[Path] = None) -> dict[str, IncidentRule]:
    """
    Load all incident rule configurations.

    Returns:
        Dict mapping rule_id to IncidentRule
    """
    if config_dir is None:
        config_dir = get_config_dir()

    rules = {}
    for rule_id in discover_incident_rules(config_dir):
        try:
            rules[rule_id] = load_incident_rule(rule_id, config_dir)
        except Exception as e:
            import logging
            logging.warning(f"Failed to load incident rule '{rule_id}': {e}")

    return rules


class IncidentRuleLoader:
    """
    Cached incident rule configuration loader.

    Provides caching to avoid re-reading YAML files.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else get_config_dir()
        self._cache: dict[str, IncidentRule] = {}
        self._discovered: Optional[list[str]] = None

    def load(self, rule_id: str, use_cache: bool = True) -> IncidentRule:
        """Load an incident rule with optional caching."""
        if use_cache and rule_id in self._cache:
            return self._cache[rule_id]

        rule = load_incident_rule(rule_id, self.config_dir)

        if use_cache:
            self._cache[rule_id] = rule

        return rule

    def load_all(self, use_cache: bool = True) -> dict[str, IncidentRule]:
        """Load all incident rules with optional caching."""
        if use_cache and self._cache:
            return self._cache.copy()

        rules = load_all_incident_rules(self.config_dir)

        if use_cache:
            self._cache = rules.copy()

        return rules

    def discover(self, use_cache: bool = True) -> list[str]:
        """Discover rule IDs with optional caching."""
        if use_cache and self._discovered is not None:
            return self._discovered.copy()

        discovered = discover_incident_rules(self.config_dir)

        if use_cache:
            self._discovered = discovered.copy()

        return discovered

    def exists(self, rule_id: str) -> bool:
        """Check if an incident rule config exists."""
        return rule_id in self.discover()

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._discovered = None

    def invalidate(self, rule_id: str):
        """Invalidate cache for a specific rule."""
        self._cache.pop(rule_id, None)
