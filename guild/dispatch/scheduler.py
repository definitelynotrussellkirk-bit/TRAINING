#!/usr/bin/env python3
"""
Curriculum Scheduler - Decides which skill/level to train next.

Reads from configs/schedule.yaml and current curriculum state to determine
the optimal next batch of training data to generate.
"""

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent


@dataclass
class SkillState:
    """Current state of a skill."""
    skill_id: str
    current_level: int
    max_level: int
    enabled: bool
    priority: int
    weight: float
    recent_accuracy: float  # Average of last N evals
    examples_generated: int  # Total examples generated for this skill
    last_trained: Optional[str]  # ISO timestamp


@dataclass
class ScheduleDecision:
    """Result of scheduler deciding what to train next."""
    skill_id: str
    level: int
    count: int
    reason: str


class CurriculumScheduler:
    """
    Decides which skill/level to generate training data for.

    Strategies:
    - equal: Round-robin between skills
    - focus: Complete one skill before next
    - weighted: Proportional to weights
    - lowest_first: Always train lowest-level skill
    """

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or BASE_DIR / "configs" / "schedule.yaml"
        self.state_path = BASE_DIR / "status" / "scheduler_state.json"
        self.curriculum_state_path = BASE_DIR / "data_manager" / "curriculum_state.json"

        self.config = self._load_config()
        self.state = self._load_state()
        self._eval_ledger = None  # Lazy load

    def _get_eval_ledger(self):
        """Get evaluation ledger (lazy loaded)."""
        if self._eval_ledger is None:
            try:
                from core.evaluation_ledger import get_eval_ledger
                self._eval_ledger = get_eval_ledger()
            except Exception as e:
                logger.warning(f"Failed to load eval ledger: {e}")
        return self._eval_ledger

    def _get_recent_accuracy(self, skill_id: str, level: int, n_evals: int = 3) -> float:
        """Get average accuracy from last N evals for skill at current level."""
        ledger = self._get_eval_ledger()
        if not ledger:
            return 0.0

        try:
            evals = ledger.get_by_skill(skill_id, level=level)
            if not evals:
                return 0.0

            # Take last N evals
            recent = evals[-n_evals:]
            if not recent:
                return 0.0

            return sum(e.accuracy for e in recent) / len(recent)
        except Exception as e:
            logger.warning(f"Failed to get accuracy for {skill_id} L{level}: {e}")
            return 0.0

    def _load_config(self) -> dict:
        """Load schedule configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {
            "strategy": "equal",
            "settings": {"batch_size": 100},
            "skills": {}
        }

    def _load_state(self) -> dict:
        """Load scheduler state."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return {
            "last_skill": None,
            "skill_counts": {},
            "round_robin_index": 0,
        }

    def _save_state(self):
        """Save scheduler state."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def _get_curriculum_state(self) -> Dict[str, int]:
        """Get current curriculum levels from curriculum_state.json."""
        if self.curriculum_state_path.exists():
            with open(self.curriculum_state_path) as f:
                data = json.load(f)
            return data.get("levels", {})
        return {}

    def get_skill_states(self) -> List[SkillState]:
        """Get current state of all configured skills."""
        curriculum_levels = self._get_curriculum_state()
        skills_config = self.config.get("skills", {})

        states = []
        for skill_id, config in skills_config.items():
            # Get max level - priority: schedule.yaml > skill YAML > default
            max_level = config.get("max_level", 50)

            # Try to read from skill YAML (authoritative source)
            skill_yaml = BASE_DIR / "configs" / "skills" / f"{skill_id}.yaml"
            if skill_yaml.exists():
                with open(skill_yaml) as f:
                    skill_data = yaml.safe_load(f)
                # Root-level 'levels' is the authoritative max level
                max_level = skill_data.get("levels", max_level)

            current_level = curriculum_levels.get(skill_id, 1)
            states.append(SkillState(
                skill_id=skill_id,
                current_level=current_level,
                max_level=max_level,
                enabled=config.get("enabled", True),
                priority=config.get("priority", 99),
                weight=config.get("weight", 1.0),
                recent_accuracy=self._get_recent_accuracy(skill_id, current_level),
                examples_generated=self.state.get("skill_counts", {}).get(skill_id, 0),
                last_trained=None,
            ))

        return states

    def decide_next(self, count: int = None) -> ScheduleDecision:
        """
        Decide which skill/level to train next.

        Args:
            count: Number of examples to generate (default from config)

        Returns:
            ScheduleDecision with skill_id, level, count, and reason
        """
        count = count or self.config.get("settings", {}).get("batch_size", 100)
        strategy = self.config.get("strategy", "equal")

        # Get enabled skills
        skill_states = [s for s in self.get_skill_states() if s.enabled]

        if not skill_states:
            raise ValueError("No skills enabled in schedule config")

        # Filter out mastered skills
        active_skills = [s for s in skill_states if s.current_level < s.max_level]
        if not active_skills:
            # All skills mastered - pick random enabled one for maintenance
            skill = random.choice(skill_states)
            return ScheduleDecision(
                skill_id=skill.skill_id,
                level=skill.max_level,
                count=count,
                reason="All skills mastered - maintenance mode"
            )

        # Apply strategy
        if strategy == "equal" or strategy == "round_robin":
            return self._decide_round_robin(active_skills, count)
        elif strategy == "focus":
            return self._decide_focus(active_skills, count)
        elif strategy == "weighted":
            return self._decide_weighted(active_skills, count)
        elif strategy == "lowest_first":
            return self._decide_lowest_first(active_skills, count)
        else:
            logger.warning(f"Unknown strategy '{strategy}', using equal")
            return self._decide_round_robin(active_skills, count)

    def _decide_round_robin(self, skills: List[SkillState], count: int) -> ScheduleDecision:
        """Equal alternation between skills."""
        idx = self.state.get("round_robin_index", 0) % len(skills)
        skill = skills[idx]

        # Update state
        self.state["round_robin_index"] = (idx + 1) % len(skills)
        self.state["last_skill"] = skill.skill_id
        self._save_state()

        return ScheduleDecision(
            skill_id=skill.skill_id,
            level=skill.current_level,
            count=count,
            reason=f"Round robin (turn {idx + 1}/{len(skills)})"
        )

    def _decide_focus(self, skills: List[SkillState], count: int) -> ScheduleDecision:
        """Focus on one skill until mastered."""
        # Sort by priority (lower = higher priority)
        sorted_skills = sorted(skills, key=lambda s: s.priority)
        skill = sorted_skills[0]

        return ScheduleDecision(
            skill_id=skill.skill_id,
            level=skill.current_level,
            count=count,
            reason=f"Focus mode (priority {skill.priority})"
        )

    def _decide_weighted(self, skills: List[SkillState], count: int) -> ScheduleDecision:
        """Choose skill proportional to weights."""
        total_weight = sum(s.weight for s in skills)
        if total_weight == 0:
            return self._decide_round_robin(skills, count)

        # Weighted random selection
        r = random.random() * total_weight
        cumulative = 0
        for skill in skills:
            cumulative += skill.weight
            if r <= cumulative:
                return ScheduleDecision(
                    skill_id=skill.skill_id,
                    level=skill.current_level,
                    count=count,
                    reason=f"Weighted selection (weight={skill.weight:.1f})"
                )

        # Fallback
        return ScheduleDecision(
            skill_id=skills[-1].skill_id,
            level=skills[-1].current_level,
            count=count,
            reason="Weighted selection (fallback)"
        )

    def _decide_lowest_first(self, skills: List[SkillState], count: int) -> ScheduleDecision:
        """Focus on the skill with lowest current level."""
        sorted_skills = sorted(skills, key=lambda s: s.current_level)
        skill = sorted_skills[0]

        return ScheduleDecision(
            skill_id=skill.skill_id,
            level=skill.current_level,
            count=count,
            reason=f"Lowest first (L{skill.current_level})"
        )

    def apply_preset(self, preset_name: str):
        """Apply a preset configuration."""
        presets = self.config.get("presets", {})
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        preset = presets[preset_name]

        # Update strategy
        if "strategy" in preset:
            self.config["strategy"] = preset["strategy"]

        # Update skill configs
        if "skills" in preset:
            for skill_id, skill_config in preset["skills"].items():
                if skill_id in self.config["skills"]:
                    self.config["skills"][skill_id].update(skill_config)

        # Save config
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"Applied preset: {preset_name}")

    def get_status(self) -> dict:
        """Get current scheduler status for API/UI."""
        skill_states = self.get_skill_states()

        return {
            "strategy": self.config.get("strategy", "equal"),
            "settings": self.config.get("settings", {}),
            "skills": [{
                "id": s.skill_id,
                "enabled": s.enabled,
                "current_level": s.current_level,
                "max_level": s.max_level,
                "priority": s.priority,
                "weight": s.weight,
                "examples_generated": s.examples_generated,
            } for s in skill_states],
            "presets": list(self.config.get("presets", {}).keys()),
            "last_skill": self.state.get("last_skill"),
            "next_decision": self.decide_next().__dict__ if skill_states else None,
        }


# Singleton
_scheduler = None


def get_scheduler() -> CurriculumScheduler:
    """Get singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = CurriculumScheduler()
    return _scheduler


if __name__ == "__main__":
    # Test
    scheduler = get_scheduler()
    print("Current status:")
    print(json.dumps(scheduler.get_status(), indent=2))
    print()
    print("Next decision:")
    decision = scheduler.decide_next()
    print(f"  {decision.skill_id} L{decision.level} x{decision.count}")
    print(f"  Reason: {decision.reason}")
