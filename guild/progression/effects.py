"""Status effect loading, registry, and rule evaluation."""

import logging
from pathlib import Path
from typing import Optional, Any

from guild.config.loader import get_config_dir, load_yaml
from guild.types import Severity
from guild.progression.types import (
    EffectType,
    EffectDefinition,
    EffectRuleConfig,
    EffectRuleState,
    StatusEffect,
    HeroState,
)
from guild.quests.types import CombatResult


logger = logging.getLogger(__name__)


def load_effect_definition(data: dict) -> EffectDefinition:
    """Load an effect definition from dict."""
    return EffectDefinition(
        id=data["id"],
        name=data["name"],
        description=data.get("description", ""),
        type=EffectType(data.get("type", "debuff")),
        severity=Severity(data.get("severity", "low")),
        default_duration_steps=data.get("default_duration_steps"),
        cure_condition=data.get("cure_condition"),
        effects=data.get("effects", {}),
        rpg_name=data.get("rpg_name"),
        rpg_description=data.get("rpg_description"),
    )


def load_effect_rule(data: dict) -> EffectRuleConfig:
    """Load an effect rule from dict."""
    return EffectRuleConfig(
        id=data["id"],
        effect_id=data["effect_id"],
        trigger_type=data.get("trigger_type", "event"),
        trigger_config=data.get("trigger_config", {}),
        cooldown_steps=data.get("cooldown_steps", 100),
        skill_id=data.get("skill_id"),
    )


class EffectRegistry:
    """
    Registry for effect definitions and rules.

    Loads from configs/progression/effects.yaml
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or get_config_dir()
        self._effects: dict[str, EffectDefinition] = {}
        self._rules: list[EffectRuleConfig] = []
        self._loaded = False

    def _load(self):
        """Load effect definitions and rules."""
        if self._loaded:
            return

        effects_file = self.config_dir / "progression" / "effects.yaml"
        if not effects_file.exists():
            logger.warning(f"Effects config not found: {effects_file}")
            self._loaded = True
            return

        data = load_yaml(effects_file)

        # Load effect definitions
        for effect_id, effect_data in data.get("effects", {}).items():
            effect_data["id"] = effect_id
            self._effects[effect_id] = load_effect_definition(effect_data)

        # Load rules
        for rule_data in data.get("rules", []):
            self._rules.append(load_effect_rule(rule_data))

        self._loaded = True
        logger.debug(f"Loaded {len(self._effects)} effects, {len(self._rules)} rules")

    def get_effect(self, effect_id: str) -> Optional[EffectDefinition]:
        """Get an effect definition by ID."""
        self._load()
        return self._effects.get(effect_id)

    def get_all_effects(self) -> dict[str, EffectDefinition]:
        """Get all effect definitions."""
        self._load()
        return self._effects.copy()

    def get_rules(self) -> list[EffectRuleConfig]:
        """Get all effect rules."""
        self._load()
        return self._rules.copy()

    def get_rules_for_effect(self, effect_id: str) -> list[EffectRuleConfig]:
        """Get rules that trigger a specific effect."""
        self._load()
        return [r for r in self._rules if r.effect_id == effect_id]

    def create_effect(
        self,
        effect_id: str,
        step: int,
        cause: Optional[dict] = None,
    ) -> Optional[StatusEffect]:
        """Create an effect instance from definition."""
        definition = self.get_effect(effect_id)
        if definition is None:
            return None
        return definition.create_instance(step, cause)

    def refresh(self):
        """Reload effect definitions."""
        self._loaded = False
        self._effects.clear()
        self._rules.clear()


class EffectEvaluator:
    """
    Evaluates effect rules against hero state.

    Determines when to apply/remove effects based on:
    - Metric thresholds
    - Consecutive results
    - Events
    """

    def __init__(self, registry: Optional[EffectRegistry] = None):
        self.registry = registry or EffectRegistry()
        self._rule_states: dict[str, EffectRuleState] = {}

    def get_rule_state(self, rule_id: str) -> EffectRuleState:
        """Get or create rule state."""
        if rule_id not in self._rule_states:
            self._rule_states[rule_id] = EffectRuleState(rule_id=rule_id)
        return self._rule_states[rule_id]

    def check_cooldown(self, rule: EffectRuleConfig, current_step: int) -> bool:
        """Check if rule is on cooldown."""
        state = self.get_rule_state(rule.id)
        # Never triggered = not on cooldown
        if state.trigger_count == 0:
            return False
        return (current_step - state.last_triggered_step) < rule.cooldown_steps

    def evaluate_metric_threshold(
        self,
        rule: EffectRuleConfig,
        metrics: dict[str, float],
    ) -> bool:
        """Evaluate a metric threshold trigger."""
        config = rule.trigger_config
        metric_name = config.get("metric")
        op = config.get("op", "gt")
        value = config.get("value", 0)

        if metric_name not in metrics:
            return False

        metric_value = metrics[metric_name]

        if op == "is_nan":
            return metric_value != metric_value  # NaN check
        elif op == "gt":
            return metric_value > value
        elif op == "lt":
            return metric_value < value
        elif op == "gte":
            return metric_value >= value
        elif op == "lte":
            return metric_value <= value
        elif op == "eq":
            return metric_value == value

        return False

    def evaluate_consecutive(
        self,
        rule: EffectRuleConfig,
        recent_results: list[CombatResult],
    ) -> bool:
        """Evaluate consecutive result triggers."""
        config = rule.trigger_config
        count = config.get("count", 5)
        result_type = config.get("result", "miss")

        if len(recent_results) < count:
            return False

        recent = recent_results[-count:]

        if result_type == "miss":
            return all(r == CombatResult.MISS for r in recent)
        elif result_type == "crit_miss":
            return all(r == CombatResult.CRITICAL_MISS for r in recent)
        elif result_type == "hit_or_crit":
            return all(r in [CombatResult.HIT, CombatResult.CRITICAL_HIT] for r in recent)
        elif result_type == "hit":
            return all(r == CombatResult.HIT for r in recent)
        elif result_type == "crit":
            return all(r == CombatResult.CRITICAL_HIT for r in recent)

        return False

    def evaluate_rules(
        self,
        hero_state: HeroState,
        metrics: Optional[dict[str, float]] = None,
        recent_results: Optional[list[CombatResult]] = None,
    ) -> list[StatusEffect]:
        """
        Evaluate all rules and return effects to apply.

        Args:
            hero_state: Current hero state
            metrics: Current metrics (loss, accuracy, etc.)
            recent_results: Recent combat results

        Returns:
            List of StatusEffect instances to apply
        """
        effects_to_apply = []
        current_step = hero_state.current_step

        for rule in self.registry.get_rules():
            # Skip if on cooldown
            if self.check_cooldown(rule, current_step):
                continue

            # Skip if effect already active
            if any(e.id == rule.effect_id for e in hero_state.active_effects):
                continue

            triggered = False

            if rule.trigger_type == "metric_threshold" and metrics:
                triggered = self.evaluate_metric_threshold(rule, metrics)

            elif rule.trigger_type in ["consecutive_failures", "consecutive_successes"]:
                if recent_results:
                    triggered = self.evaluate_consecutive(rule, recent_results)

            if triggered:
                effect = self.registry.create_effect(
                    rule.effect_id,
                    current_step,
                    cause={"rule_id": rule.id, "trigger_type": rule.trigger_type},
                )
                if effect:
                    effects_to_apply.append(effect)
                    # Update rule state
                    state = self.get_rule_state(rule.id)
                    state.last_triggered_step = current_step
                    state.trigger_count += 1
                    logger.info(
                        f"Effect '{effect.id}' triggered by rule '{rule.id}' "
                        f"at step {current_step}"
                    )

        return effects_to_apply

    def get_xp_multiplier(self, hero_state: HeroState) -> float:
        """Calculate combined XP multiplier from active effects."""
        multiplier = 1.0
        for effect in hero_state.active_effects:
            effect_mult = effect.effects.get("xp_multiplier", 1.0)
            multiplier *= effect_mult
        return multiplier


class EffectManager:
    """
    High-level effect management.

    Combines registry and evaluator for complete effect lifecycle.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.registry = EffectRegistry(config_dir)
        self.evaluator = EffectEvaluator(self.registry)

    def apply_effect(
        self,
        hero_state: HeroState,
        effect_id: str,
        cause: Optional[dict] = None,
    ) -> Optional[StatusEffect]:
        """Apply an effect to hero."""
        effect = self.registry.create_effect(
            effect_id, hero_state.current_step, cause
        )
        if effect:
            hero_state.add_effect(effect)
            logger.info(f"Applied effect '{effect_id}' to hero {hero_state.hero_id}")
        return effect

    def remove_effect(self, hero_state: HeroState, effect_id: str):
        """Remove an effect from hero."""
        hero_state.remove_effect(effect_id)
        logger.info(f"Removed effect '{effect_id}' from hero {hero_state.hero_id}")

    def update_effects(
        self,
        hero_state: HeroState,
        metrics: Optional[dict[str, float]] = None,
        recent_results: Optional[list[CombatResult]] = None,
    ) -> list[StatusEffect]:
        """
        Update hero effects based on current state.

        1. Clear expired effects
        2. Evaluate rules for new effects
        3. Apply triggered effects

        Returns:
            List of newly applied effects
        """
        # Clear expired
        hero_state.clear_expired_effects(hero_state.current_step)

        # Evaluate rules
        new_effects = self.evaluator.evaluate_rules(
            hero_state, metrics, recent_results
        )

        # Apply new effects
        for effect in new_effects:
            hero_state.add_effect(effect)

        return new_effects

    def get_xp_multiplier(self, hero_state: HeroState) -> float:
        """Get combined XP multiplier from effects."""
        return self.evaluator.get_xp_multiplier(hero_state)

    def has_blocking_effect(self, hero_state: HeroState) -> bool:
        """Check if hero has any training-blocking effects."""
        for effect in hero_state.active_effects:
            if effect.effects.get("training_blocked", False):
                return True
        return False

    def requires_attention(self, hero_state: HeroState) -> bool:
        """Check if hero has effects requiring attention."""
        for effect in hero_state.active_effects:
            if effect.effects.get("requires_attention", False):
                return True
            if effect.effects.get("requires_immediate_attention", False):
                return True
        return False


# Global manager
_manager: Optional[EffectManager] = None


def get_effect_manager() -> EffectManager:
    """Get the global effect manager."""
    global _manager
    if _manager is None:
        _manager = EffectManager()
    return _manager


def reset_effect_manager():
    """Reset the global manager (for testing)."""
    global _manager
    _manager = None


def init_effect_manager(config_dir: Optional[Path] = None) -> EffectManager:
    """Initialize the global effect manager."""
    global _manager
    _manager = EffectManager(config_dir)
    return _manager


# Convenience functions

def apply_effect(
    hero_state: HeroState,
    effect_id: str,
    cause: Optional[dict] = None,
) -> Optional[StatusEffect]:
    """Apply an effect using the global manager."""
    return get_effect_manager().apply_effect(hero_state, effect_id, cause)


def update_effects(
    hero_state: HeroState,
    metrics: Optional[dict[str, float]] = None,
    recent_results: Optional[list[CombatResult]] = None,
) -> list[StatusEffect]:
    """Update effects using the global manager."""
    return get_effect_manager().update_effects(hero_state, metrics, recent_results)


def get_xp_multiplier(hero_state: HeroState) -> float:
    """Get XP multiplier using the global manager."""
    return get_effect_manager().get_xp_multiplier(hero_state)
