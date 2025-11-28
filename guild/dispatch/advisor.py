"""
Progression Advisor - Counsels on hero skill advancement.

The Progression Advisor:
- Tracks hero's current level in each skill
- Records trial results (accuracy)
- Advises when hero is ready to advance
- Recommends quest difficulty based on progression

RPG Flavor:
    The Progression Advisor is a wise mentor who watches the hero's
    performance in trials. When the hero demonstrates mastery
    (80% accuracy over 3 trials), the Advisor recommends advancing
    to more challenging quests.

Progression Rules:
    - Hero starts at Level 1 for each skill
    - Must achieve threshold accuracy (default 80%) over N trials
    - Trials must be at current level
    - Advancement unlocks harder quests with more XP

Level System (Binary):
    Level 1-7: Basic (2-bit to 8-bit)
    Level 8-15: Intermediate (9-bit to 16-bit with symbols)
    Level 16-30: Advanced (17-bit to 32-bit mastery)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from guild.dispatch.types import ProgressionStatus


logger = logging.getLogger(__name__)


# =============================================================================
# SKILL CONFIGURATIONS - What the hero can learn
# =============================================================================

SKILL_CURRICULA: dict[str, dict[str, Any]] = {
    "binary": {
        "name": "Binary Arithmetic",
        "total_levels": 30,
        "api_port": 8090,
        # Level = bit-width: level N = (N+1) bits
        "levels": [
            {"level": 1, "name": "Tiny (2-bit)", "bits": 2, "max": 3, "threshold": 0.80},
            {"level": 2, "name": "3-bit", "bits": 3, "max": 7, "threshold": 0.80},
            {"level": 3, "name": "4-bit", "bits": 4, "max": 15, "threshold": 0.80},
            {"level": 4, "name": "5-bit", "bits": 5, "max": 31, "threshold": 0.80},
            {"level": 5, "name": "6-bit", "bits": 6, "max": 63, "threshold": 0.80},
            {"level": 6, "name": "7-bit", "bits": 7, "max": 127, "threshold": 0.80},
            {"level": 7, "name": "Byte (8-bit)", "bits": 8, "max": 255, "threshold": 0.80},
            {"level": 8, "name": "9-bit + Symbols", "bits": 9, "max": 511, "threshold": 0.80, "symbols": True},
            {"level": 9, "name": "10-bit", "bits": 10, "max": 1023, "threshold": 0.80, "symbols": True},
            {"level": 10, "name": "11-bit", "bits": 11, "max": 2047, "threshold": 0.80, "symbols": True},
            {"level": 11, "name": "12-bit", "bits": 12, "max": 4095, "threshold": 0.80, "symbols": True},
            {"level": 12, "name": "13-bit", "bits": 13, "max": 8191, "threshold": 0.80, "symbols": True},
            {"level": 13, "name": "14-bit", "bits": 14, "max": 16383, "threshold": 0.80, "symbols": True},
            {"level": 14, "name": "15-bit", "bits": 15, "max": 32767, "threshold": 0.80, "symbols": True},
            {"level": 15, "name": "Word (16-bit)", "bits": 16, "max": 65535, "threshold": 0.80, "symbols": True},
            {"level": 16, "name": "17-bit", "bits": 17, "max": 131071, "threshold": 0.80, "symbols": True},
            {"level": 17, "name": "18-bit", "bits": 18, "max": 262143, "threshold": 0.80, "symbols": True},
            {"level": 18, "name": "19-bit", "bits": 19, "max": 524287, "threshold": 0.80, "symbols": True},
            {"level": 19, "name": "20-bit", "bits": 20, "max": 1048575, "threshold": 0.80, "symbols": True},
            {"level": 20, "name": "21-bit (~2M)", "bits": 21, "max": 2097151, "threshold": 0.80, "symbols": True},
            {"level": 21, "name": "22-bit", "bits": 22, "max": 4194303, "threshold": 0.80, "symbols": True},
            {"level": 22, "name": "23-bit", "bits": 23, "max": 8388607, "threshold": 0.80, "symbols": True},
            {"level": 23, "name": "24-bit", "bits": 24, "max": 16777215, "threshold": 0.80, "symbols": True},
            {"level": 24, "name": "25-bit", "bits": 25, "max": 33554431, "threshold": 0.80, "symbols": True},
            {"level": 25, "name": "26-bit (~67M)", "bits": 26, "max": 67108863, "threshold": 0.80, "symbols": True},
            {"level": 26, "name": "27-bit", "bits": 27, "max": 134217727, "threshold": 0.80, "symbols": True},
            {"level": 27, "name": "28-bit", "bits": 28, "max": 268435455, "threshold": 0.80, "symbols": True},
            {"level": 28, "name": "29-bit", "bits": 29, "max": 536870911, "threshold": 0.80, "symbols": True},
            {"level": 29, "name": "30-bit (~1B)", "bits": 30, "max": 1073741823, "threshold": 0.80, "symbols": True},
            {"level": 30, "name": "DWord (32-bit)", "bits": 32, "max": 4294967295, "threshold": None, "symbols": True},
        ]
    },
    # DISABLED: Re-enable when Sy trainer is ready
    # "syllo": { ... }
}


@dataclass
class TrialRecord:
    """Record of a hero's trial (evaluation) result."""
    step: int
    accuracy: float
    level: int
    timestamp: str
    metadata: dict = field(default_factory=dict)


class ProgressionAdvisor:
    """
    Counsels the hero on skill advancement.

    Tracks performance, records trial results, and advises when
    the hero is ready for harder challenges.

    Usage:
        advisor = ProgressionAdvisor(base_dir, config)

        # What level should the hero train at?
        params = advisor.recommend_quest_params("binary", count=100)

        # Record a trial result
        advisor.record_trial("binary", accuracy=0.85, step=10000)

        # Check if ready to advance
        if advisor.ready_to_advance("binary"):
            advisor.advance_level("binary")
    """

    def __init__(
        self,
        base_dir: Path | str,
        config: dict[str, Any] | None = None,
        min_trials_for_advancement: int = 3,
    ):
        """
        Initialize the Progression Advisor.

        Args:
            base_dir: Base training directory
            config: Optional config dict (uses config.json if None)
            min_trials_for_advancement: Trials needed before advancing
        """
        self.base_dir = Path(base_dir)
        self.config = config or {}

        # State persistence
        self.state_file = self.base_dir / "guild" / "dispatch" / "progression_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize state
        self._state = self._load_state()

        # Advisor settings
        self.min_trials = min_trials_for_advancement
        self.enabled = self.config.get("curriculum", {}).get("enabled", True)

    def _load_state(self) -> dict:
        """Load progression state from disk."""
        # Try new location first
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)

        # Fall back to legacy location
        legacy_file = self.base_dir / "data_manager" / "curriculum_state.json"
        if legacy_file.exists():
            with open(legacy_file) as f:
                state = json.load(f)
            # Migrate to new location
            self._save_state(state)
            logger.info(f"Migrated progression state to {self.state_file}")
            return state

        # Default state
        return {
            "skills": {
                skill_id: {
                    "current_level": 1,
                    "trial_history": [],
                    "advancement_history": [],
                }
                for skill_id in SKILL_CURRICULA.keys()
            },
            "active_skill": "binary",
            "started_at": datetime.now().isoformat(),
        }

    def _save_state(self, state: dict | None = None):
        """Save progression state to disk."""
        state = state or self._state
        state["last_updated"] = datetime.now().isoformat()

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        logger.debug("Progression state saved")

    def get_skill_curriculum(self, skill_id: str) -> dict[str, Any]:
        """Get curriculum configuration for a skill."""
        if skill_id not in SKILL_CURRICULA:
            raise ValueError(f"Unknown skill: {skill_id}. Available: {list(SKILL_CURRICULA.keys())}")
        return SKILL_CURRICULA[skill_id]

    def get_current_level(self, skill_id: str) -> dict[str, Any]:
        """
        Get current level configuration for a skill.

        Returns the level dict with name, threshold, etc.
        """
        curriculum = self.get_skill_curriculum(skill_id)
        skill_state = self._state["skills"].get(skill_id, {"current_level": 1})
        current_level_num = skill_state.get("current_level", 1)

        for lvl in curriculum["levels"]:
            if lvl["level"] == current_level_num:
                return lvl

        # Fallback to first level
        return curriculum["levels"][0]

    def recommend_quest_params(
        self,
        skill_id: str,
        count: int = 100,
    ) -> dict[str, Any]:
        """
        Recommend quest parameters based on hero's current level.

        Returns params suitable for passing to the skill trainer.

        Args:
            skill_id: Skill to train
            count: Number of quests to request

        Returns:
            Dict with skill, level, count
        """
        current_level = self.get_current_level(skill_id)

        return {
            "skill": skill_id,
            "level": current_level["level"],
            "count": count,
        }

    def record_trial(
        self,
        skill_id: str,
        accuracy: float,
        step: int,
        metadata: dict | None = None,
    ):
        """
        Record a trial (evaluation) result.

        Args:
            skill_id: Skill that was evaluated
            accuracy: Accuracy score (0-1)
            step: Training step number
            metadata: Optional additional info
        """
        if skill_id not in self._state["skills"]:
            self._state["skills"][skill_id] = {
                "current_level": 1,
                "trial_history": [],
                "advancement_history": [],
            }

        skill_state = self._state["skills"][skill_id]

        record = {
            "step": step,
            "accuracy": accuracy,
            "level": skill_state["current_level"],
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            record["metadata"] = metadata

        skill_state["trial_history"].append(record)

        # Keep only last 100 records
        if len(skill_state["trial_history"]) > 100:
            skill_state["trial_history"] = skill_state["trial_history"][-100:]

        self._save_state()
        logger.info(
            f"[{skill_id}] Recorded trial: {accuracy:.1%} at step {step} "
            f"(Level {skill_state['current_level']})"
        )

    def ready_to_advance(self, skill_id: str) -> tuple[bool, str]:
        """
        Check if hero is ready to advance to the next level.

        Returns:
            (ready, reason) - True if should advance, with explanation
        """
        if not self.enabled:
            return False, "Progression disabled"

        skill_state = self._state["skills"].get(skill_id, {})
        current_level_num = skill_state.get("current_level", 1)

        curriculum = self.get_skill_curriculum(skill_id)
        current_level = self.get_current_level(skill_id)

        # Check if at max level
        if current_level_num >= curriculum["total_levels"]:
            return False, f"Already at max level ({curriculum['total_levels']})"

        threshold = current_level.get("threshold")
        if threshold is None:
            return False, "Skill mastered (no threshold)"

        # Get trials at current level
        history = skill_state.get("trial_history", [])
        at_level = [r for r in history if r.get("level") == current_level_num]

        if len(at_level) < self.min_trials:
            return False, f"Need {self.min_trials} trials at level (have {len(at_level)})"

        # Check last N trials
        recent = at_level[-self.min_trials:]
        avg_accuracy = sum(r["accuracy"] for r in recent) / len(recent)

        if avg_accuracy >= threshold:
            return True, f"Avg accuracy {avg_accuracy:.1%} >= threshold {threshold:.0%}"

        return False, f"Avg accuracy {avg_accuracy:.1%} < threshold {threshold:.0%}"

    def advance_level(self, skill_id: str) -> dict[str, Any]:
        """
        Advance hero to the next level.

        Returns the new level configuration.
        """
        skill_state = self._state["skills"][skill_id]
        old_level = skill_state["current_level"]
        new_level = old_level + 1

        curriculum = self.get_skill_curriculum(skill_id)
        if new_level > curriculum["total_levels"]:
            new_level = curriculum["total_levels"]

        # Record advancement
        advancement = {
            "from_level": old_level,
            "to_level": new_level,
            "timestamp": datetime.now().isoformat(),
        }
        skill_state["advancement_history"].append(advancement)
        skill_state["current_level"] = new_level

        self._save_state()

        new_level_config = self.get_current_level(skill_id)
        logger.info(
            f"[{skill_id}] ADVANCEMENT: Level {old_level} -> {new_level} "
            f"({new_level_config['name']})"
        )

        return new_level_config

    def record_and_check(
        self,
        skill_id: str,
        accuracy: float,
        step: int,
        metadata: dict | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Record trial and check for advancement.

        Convenience method combining record_trial and advance_level.

        Returns:
            (advanced, new_level_config or None)
        """
        self.record_trial(skill_id, accuracy, step, metadata)

        ready, reason = self.ready_to_advance(skill_id)
        logger.info(f"[{skill_id}] Advancement check: {reason}")

        if ready:
            new_level = self.advance_level(skill_id)
            return True, new_level

        return False, None

    def get_status(self, skill_id: str | None = None) -> dict[str, Any] | ProgressionStatus:
        """
        Get progression status for one or all skills.

        Args:
            skill_id: Specific skill, or None for all

        Returns:
            Status dict or ProgressionStatus object
        """
        if skill_id:
            return self._get_skill_status(skill_id)

        return {
            "enabled": self.enabled,
            "active_skill": self._state.get("active_skill", "binary"),
            "skills": {
                s: self._get_skill_status(s).to_dict()
                for s in SKILL_CURRICULA.keys()
            }
        }

    def _get_skill_status(self, skill_id: str) -> ProgressionStatus:
        """Get status for a single skill."""
        skill_state = self._state["skills"].get(
            skill_id,
            {"current_level": 1, "trial_history": []}
        )
        curriculum = self.get_skill_curriculum(skill_id)
        current_level = self.get_current_level(skill_id)

        # Recent accuracy at current level
        history = skill_state.get("trial_history", [])
        at_level = [r for r in history if r.get("level") == skill_state["current_level"]]
        recent = at_level[-self.min_trials:] if at_level else []
        avg_accuracy = sum(r["accuracy"] for r in recent) / len(recent) if recent else None

        ready, reason = self.ready_to_advance(skill_id)

        return ProgressionStatus(
            skill_id=skill_id,
            skill_name=curriculum["name"],
            current_level=skill_state["current_level"],
            total_levels=curriculum["total_levels"],
            level_name=current_level["name"],
            accuracy_threshold=current_level.get("threshold"),
            evals_at_level=len(at_level),
            avg_accuracy=avg_accuracy,
            ready_to_advance=ready,
            reason=reason,
            progressions_completed=len(skill_state.get("advancement_history", [])),
        )

    def set_active_skill(self, skill_id: str):
        """Set the active skill for training."""
        if skill_id not in SKILL_CURRICULA:
            raise ValueError(f"Unknown skill: {skill_id}")
        self._state["active_skill"] = skill_id
        self._save_state()
        logger.info(f"Active skill set to: {skill_id}")

    def get_active_skill(self) -> str:
        """Get the currently active skill."""
        return self._state.get("active_skill", "binary")

    def reset_skill(self, skill_id: str):
        """Reset a skill to level 1."""
        self._state["skills"][skill_id] = {
            "current_level": 1,
            "trial_history": [],
            "advancement_history": [],
            "reset_at": datetime.now().isoformat(),
        }
        self._save_state()
        logger.info(f"[{skill_id}] Reset to Level 1")

    def set_level(self, skill_id: str, level: int):
        """Manually set skill level (for testing/debugging)."""
        curriculum = self.get_skill_curriculum(skill_id)
        if level < 1 or level > curriculum["total_levels"]:
            raise ValueError(f"Level must be 1-{curriculum['total_levels']}")

        self._state["skills"][skill_id]["current_level"] = level
        self._save_state()
        logger.info(f"[{skill_id}] Level set to {level}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for Progression Advisor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Progression Advisor - Hero skill advancement"
    )
    parser.add_argument('--base-dir', default=None, help="Base dir (default: from core.paths)")

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Status
    status_p = subparsers.add_parser('status', help='Show progression status')
    status_p.add_argument('--skill', choices=list(SKILL_CURRICULA.keys()))

    # Record trial
    record_p = subparsers.add_parser('record', help='Record trial result')
    record_p.add_argument('skill', choices=list(SKILL_CURRICULA.keys()))
    record_p.add_argument('accuracy', type=float, help='Accuracy (0-1)')
    record_p.add_argument('step', type=int, help='Training step')

    # Check advancement
    check_p = subparsers.add_parser('check', help='Record and check for advancement')
    check_p.add_argument('skill', choices=list(SKILL_CURRICULA.keys()))
    check_p.add_argument('accuracy', type=float)
    check_p.add_argument('step', type=int)

    # Force advance
    adv_p = subparsers.add_parser('advance', help='Force advancement')
    adv_p.add_argument('skill', choices=list(SKILL_CURRICULA.keys()))

    # Set level
    set_p = subparsers.add_parser('set-level', help='Set skill level')
    set_p.add_argument('skill', choices=list(SKILL_CURRICULA.keys()))
    set_p.add_argument('level', type=int)

    # Reset
    reset_p = subparsers.add_parser('reset', help='Reset skill to level 1')
    reset_p.add_argument('skill', choices=list(SKILL_CURRICULA.keys()))

    # Recommend
    rec_p = subparsers.add_parser('recommend', help='Get quest recommendation')
    rec_p.add_argument('skill', choices=list(SKILL_CURRICULA.keys()))
    rec_p.add_argument('--count', type=int, default=100)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    advisor = ProgressionAdvisor(Path(args.base_dir))

    if args.command == 'status':
        status = advisor.get_status(args.skill)
        if isinstance(status, ProgressionStatus):
            status = status.to_dict()
        print(json.dumps(status, indent=2))

    elif args.command == 'record':
        advisor.record_trial(args.skill, args.accuracy, args.step)
        print(f"Recorded: {args.skill} {args.accuracy:.1%} at step {args.step}")

    elif args.command == 'check':
        advanced, new_level = advisor.record_and_check(args.skill, args.accuracy, args.step)
        if advanced:
            print(f"ADVANCED to Level {new_level['level']}: {new_level['name']}")
        else:
            print("No advancement")
        status = advisor.get_status(args.skill)
        print(f"Current: Level {status.current_level} ({status.reason})")

    elif args.command == 'advance':
        new_level = advisor.advance_level(args.skill)
        print(f"Advanced to Level {new_level['level']}: {new_level['name']}")

    elif args.command == 'set-level':
        advisor.set_level(args.skill, args.level)
        print(f"Set {args.skill} to Level {args.level}")

    elif args.command == 'reset':
        advisor.reset_skill(args.skill)
        print(f"Reset {args.skill} to Level 1")

    elif args.command == 'recommend':
        params = advisor.recommend_quest_params(args.skill, args.count)
        print(json.dumps(params, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
