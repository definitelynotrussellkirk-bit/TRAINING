#!/usr/bin/env python3
"""
Curriculum Manager - Adaptive Difficulty Progression

Integrates with singleSKILL API servers to generate training data
at appropriate difficulty levels based on model performance.

Skills supported:
- SYLLO: 10 levels (signal degradation model)
- Binary: 7 levels (magnitude ranges)

Progression: Advance when accuracy >= threshold over N evaluations
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# Skill definitions matching API server levels
# All levels require 80% accuracy to advance
SKILL_LEVELS = {
    "sy": {
        "name": "Syllacrostic Puzzles",
        "total_levels": 50,
        "api_port": 8080,
        "levels": [
            {"level": 1, "name": "Beginner", "words": 4, "threshold": 0.80},
            {"level": 2, "name": "Easy", "words": 5, "threshold": 0.80},
            {"level": 3, "name": "Novice", "words": 5, "threshold": 0.80},
            {"level": 4, "name": "Learning", "words": 6, "threshold": 0.80},
            {"level": 5, "name": "Intermediate", "words": 6, "threshold": 0.80},
            # Levels 6-50 follow skill YAML progression
        ] + [{"level": i, "name": f"Level {i}", "threshold": 0.80} for i in range(6, 51)]
    },
    "bin": {
        "name": "Binary Arithmetic",
        "total_levels": 30,
        "api_port": 8090,
        # Bin uses bit-width progression: level N = (N+1) bits
        # Key milestones in 30-level system:
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
            {"level": 30, "name": "DWord (32-bit)", "bits": 32, "max": 4294967295, "threshold": None, "symbols": True},  # Mastered
        ]
    }
}


class CurriculumManager:
    """
    Manages adaptive curriculum based on model performance.

    Integrates with singleSKILL APIs via skill_api_client.
    """

    def __init__(self, base_dir: Path, config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.config = config

        # State file
        self.state_file = self.base_dir / "data_manager" / "curriculum_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize state
        self.state = self._load_state()

        # Curriculum config
        self.curriculum_config = config.get("curriculum", {})
        self.enabled = self.curriculum_config.get("enabled", True)
        self.min_evals = self.curriculum_config.get("min_evals_for_progression", 1)

    def _load_state(self) -> Dict:
        """Load curriculum state from disk"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)

            # Migrate old format (single skill) to new format (multi-skill)
            if "skills" not in state:
                # NOTE: Default to 0 (nothing mastered), not 1!
                old_level = state.get("current_level", 0)
                old_history = state.get("accuracy_history", [])
                old_progressions = state.get("progression_history", [])

                state = {
                    "skills": {
                        "syllo": {
                            "current_level": old_level,
                            "accuracy_history": old_history,
                            "progression_history": old_progressions
                        },
                        "bin": {"current_level": 0, "accuracy_history": [], "progression_history": []},
                    },
                    "active_skill": state.get("current_skill", "syllo"),
                    "started_at": state.get("started_at", datetime.now().isoformat()),
                    "migrated_at": datetime.now().isoformat()
                }
                # Save migrated state
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                logger.info("Migrated curriculum state to new multi-skill format")

            return state

        # Default state - start at level 0 (no mastery yet)
        # current_level = highest level MASTERED (0 = haven't proven any skill)
        # training_level = current_level + 1 (what you're working on)
        return {
            "skills": {
                "syllo": {"current_level": 0, "accuracy_history": [], "progression_history": []},
                "bin": {"current_level": 0, "accuracy_history": [], "progression_history": []},
            },
            "active_skill": "binary",  # Binary only for now
            "started_at": datetime.now().isoformat()
        }

    def _save_state(self):
        """Save curriculum state to disk"""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        logger.debug("Curriculum state saved")

    def get_skill_config(self, skill: str) -> Dict[str, Any]:
        """Get skill configuration"""
        if skill not in SKILL_LEVELS:
            raise ValueError(f"Unknown skill: {skill}")
        return SKILL_LEVELS[skill]

    def get_mastered_level(self, skill: str) -> int:
        """Get the highest level mastered for a skill (0 = none)."""
        return self.state["skills"][skill]["current_level"]

    def get_training_level(self, skill: str) -> int:
        """Get the level currently being trained on (mastered + 1)."""
        mastered = self.get_mastered_level(skill)
        skill_config = self.get_skill_config(skill)
        max_level = skill_config["total_levels"]
        return min(mastered + 1, max_level)

    def get_current_level(self, skill: str) -> Dict[str, Any]:
        """Get current TRAINING level configuration for a skill."""
        skill_config = self.get_skill_config(skill)
        training_level = self.get_training_level(skill)

        for lvl in skill_config["levels"]:
            if lvl["level"] == training_level:
                return lvl

        # Fallback to first level
        return skill_config["levels"][0]

    def get_generation_params(self, skill: str, count: int = 50) -> Dict[str, Any]:
        """
        Get parameters for skill_api_client.generate_skill_data()

        Returns params ready to pass to the API.
        Uses training_level (mastered + 1), not mastered level.
        """
        training_level = self.get_training_level(skill)

        return {
            "skill": skill,
            "level": training_level,
            "count": count,
        }

    def record_accuracy(
        self,
        skill: str,
        accuracy: float,
        step: int,
        metadata: Optional[Dict] = None
    ):
        """
        Record accuracy from evaluation.

        Args:
            skill: Skill name ("syllo" or "binary")
            accuracy: Accuracy score (0-1)
            step: Training step number
            metadata: Optional metadata
        """
        if skill not in self.state["skills"]:
            self.state["skills"][skill] = {
                "current_level": 0,  # 0 = nothing mastered yet, train on level 1
                "accuracy_history": [],
                "progression_history": []
            }

        skill_state = self.state["skills"][skill]
        training_level = self.get_training_level(skill)
        mastered_level = self.get_mastered_level(skill)

        record = {
            "step": step,
            "accuracy": accuracy,
            "training_level": training_level,  # Level being tested
            "mastered_level": mastered_level,  # Level already earned
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            record["metadata"] = metadata

        skill_state["accuracy_history"].append(record)

        # Keep only last 100 records
        if len(skill_state["accuracy_history"]) > 100:
            skill_state["accuracy_history"] = skill_state["accuracy_history"][-100:]

        self._save_state()
        logger.info(f"[{skill}] Recorded accuracy: {accuracy:.1%} at step {step} (Training L{training_level}, Mastered L{mastered_level})")

    def should_progress(self, skill: str) -> Tuple[bool, str]:
        """
        Check if should progress to next level (earn the training level).

        Returns:
            (should_progress, reason)
        """
        if not self.enabled:
            return False, "Curriculum disabled"

        skill_state = self.state["skills"].get(skill, {})
        mastered_level = skill_state.get("current_level", 0)
        training_level = self.get_training_level(skill)

        skill_config = self.get_skill_config(skill)
        current_level_config = self.get_current_level(skill)  # Training level config

        # Check if already mastered max level
        if mastered_level >= skill_config["total_levels"]:
            return False, f"Already mastered max level ({skill_config['total_levels']})"

        threshold = current_level_config.get("threshold")
        if threshold is None:
            return False, "Skill mastered (no threshold)"

        # Get recent evals at current TRAINING level
        history = skill_state.get("accuracy_history", [])
        # Support both old format (level) and new format (training_level)
        at_level = [r for r in history if r.get("training_level", r.get("level")) == training_level]

        if len(at_level) < self.min_evals:
            return False, f"Need {self.min_evals} evals at L{training_level} (have {len(at_level)})"

        # Check last N evals
        recent = at_level[-self.min_evals:]
        avg_accuracy = sum(r["accuracy"] for r in recent) / len(recent)

        if avg_accuracy >= threshold:
            return True, f"Avg accuracy {avg_accuracy:.1%} >= threshold {threshold:.0%}"

        return False, f"Avg accuracy {avg_accuracy:.1%} < threshold {threshold:.0%}"

    def progress_to_next_level(self, skill: str) -> Dict[str, Any]:
        """
        Earn the current training level (mastered level increases by 1).

        Example: If mastered=0, training=1, and you pass:
          -> mastered becomes 1, next training becomes 2

        Automatically queues evaluation for the new training level.
        """
        skill_state = self.state["skills"][skill]
        old_mastered = skill_state["current_level"]
        earned_level = self.get_training_level(skill)  # The level just passed
        new_mastered = earned_level  # Now mastered up to this level

        skill_config = self.get_skill_config(skill)
        if new_mastered > skill_config["total_levels"]:
            new_mastered = skill_config["total_levels"]

        # Record progression
        progression = {
            "earned_level": earned_level,
            "old_mastered": old_mastered,
            "new_mastered": new_mastered,
            "timestamp": datetime.now().isoformat()
        }
        skill_state["progression_history"].append(progression)
        skill_state["current_level"] = new_mastered

        self._save_state()

        # Get the NEW training level config (what they'll work on next)
        new_training_config = self.get_current_level(skill)
        new_training_level = self.get_training_level(skill)
        logger.info(f"[{skill}] LEVEL EARNED! Mastered L{old_mastered} -> L{new_mastered}. Now training on L{new_training_level}")

        # Queue evaluation for next level immediately
        try:
            from core.evaluation_ledger import queue_evaluation
            from core.checkpoint_ledger import get_ledger

            ledger = get_ledger()
            latest = ledger.get_latest()

            if latest:
                queue_evaluation(latest.step, skill, new_training_level)
                logger.info(f"[{skill}] Queued eval for L{new_training_level} at checkpoint {latest.step}")
        except Exception as e:
            logger.warning(f"[{skill}] Failed to queue next level eval: {e}")

        return new_training_config

    def check_and_progress(
        self,
        skill: str,
        accuracy: float,
        step: int,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Record accuracy and check if should progress.

        Returns:
            (progressed, new_level_config or None)
        """
        self.record_accuracy(skill, accuracy, step, metadata)

        should_prog, reason = self.should_progress(skill)
        logger.info(f"[{skill}] Progression check: {reason}")

        if should_prog:
            new_level = self.progress_to_next_level(skill)
            return True, new_level

        return False, None

    def get_status(self, skill: Optional[str] = None) -> Dict[str, Any]:
        """Get curriculum status for one or all skills."""
        if skill:
            return self._get_skill_status(skill)

        return {
            "enabled": self.enabled,
            "active_skill": self.state.get("active_skill", "syllo"),
            "skills": {
                s: self._get_skill_status(s) for s in SKILL_LEVELS.keys()
            }
        }

    def _get_skill_status(self, skill: str) -> Dict[str, Any]:
        """Get status for a single skill."""
        skill_state = self.state["skills"].get(skill, {"current_level": 0, "accuracy_history": []})
        skill_config = self.get_skill_config(skill)
        current_level = self.get_current_level(skill)

        # Recent accuracy at current level
        history = skill_state.get("accuracy_history", [])
        at_level = [r for r in history if r.get("level") == skill_state["current_level"]]
        recent = at_level[-self.min_evals:] if at_level else []
        avg_accuracy = sum(r["accuracy"] for r in recent) / len(recent) if recent else None

        should_prog, reason = self.should_progress(skill)

        return {
            "skill": skill,
            "skill_name": skill_config["name"],
            "current_level": skill_state["current_level"],
            "total_levels": skill_config["total_levels"],
            "level_name": current_level["name"],
            "level_details": current_level,
            "threshold": current_level.get("threshold"),
            "evals_at_level": len(at_level),
            "avg_accuracy": avg_accuracy,
            "should_progress": should_prog,
            "reason": reason,
            "progressions": len(skill_state.get("progression_history", []))
        }

    def set_active_skill(self, skill: str):
        """Set the active skill for training."""
        if skill not in SKILL_LEVELS:
            raise ValueError(f"Unknown skill: {skill}")
        self.state["active_skill"] = skill
        self._save_state()
        logger.info(f"Active skill set to: {skill}")

    def reset_skill(self, skill: str):
        """Reset a skill to level 0 (nothing mastered, train on level 1)."""
        self.state["skills"][skill] = {
            "current_level": 0,  # 0 = nothing mastered, training_level = 1
            "accuracy_history": [],
            "progression_history": [],
            "reset_at": datetime.now().isoformat()
        }
        self._save_state()
        logger.info(f"[{skill}] Reset to mastered=0 (training on level 1)")

    # =========================================================================
    # SKILL ROTATION - Prevents catastrophic forgetting by training all skills
    # =========================================================================

    def get_rotation_config(self) -> Dict[str, Any]:
        """Get skill rotation configuration."""
        return self.state.get("skill_rotation", {
            "enabled": False,
            "skills": ["sy", "bin"],
            "index": 0
        })

    def set_rotation_enabled(self, enabled: bool):
        """Enable or disable skill rotation."""
        if "skill_rotation" not in self.state:
            self.state["skill_rotation"] = {
                "enabled": enabled,
                "skills": list(self.state.get("skills", {}).keys()),
                "index": 0
            }
        else:
            self.state["skill_rotation"]["enabled"] = enabled
        self._save_state()
        logger.info(f"Skill rotation {'enabled' if enabled else 'disabled'}")

    def set_rotation_skills(self, skills: List[str]):
        """Set which skills to include in rotation."""
        # Validate skills
        for skill in skills:
            if skill not in SKILL_LEVELS:
                raise ValueError(f"Unknown skill: {skill}")

        if "skill_rotation" not in self.state:
            self.state["skill_rotation"] = {"enabled": True, "skills": skills, "index": 0}
        else:
            self.state["skill_rotation"]["skills"] = skills
            # Reset index if it's now out of bounds
            if self.state["skill_rotation"]["index"] >= len(skills):
                self.state["skill_rotation"]["index"] = 0
        self._save_state()
        logger.info(f"Rotation skills set to: {skills}")

    def get_next_rotation_skill(self) -> str:
        """
        Get the next skill in rotation and advance the index.

        If rotation is disabled, returns the current active_skill.
        If rotation is enabled, returns the next skill in the rotation list
        and advances the index for the next call.

        Returns:
            Skill ID to generate data for
        """
        rotation = self.get_rotation_config()

        if not rotation.get("enabled", False):
            return self.state.get("active_skill", "sy")

        skills = rotation.get("skills", ["sy", "bin"])
        if not skills:
            return self.state.get("active_skill", "sy")

        index = rotation.get("index", 0) % len(skills)
        skill = skills[index]

        # Advance index for next call
        self.state["skill_rotation"]["index"] = (index + 1) % len(skills)
        self._save_state()

        logger.info(f"🔄 Rotation: selected {skill} (index {index}/{len(skills)})")
        return skill

    def get_rotation_status(self) -> Dict[str, Any]:
        """Get current rotation status for display."""
        rotation = self.get_rotation_config()
        skills = rotation.get("skills", [])
        index = rotation.get("index", 0)

        return {
            "enabled": rotation.get("enabled", False),
            "skills": skills,
            "current_index": index,
            "next_skill": skills[index % len(skills)] if skills else None,
            "total_skills": len(skills)
        }


def main():
    """CLI for Curriculum Manager"""
    import argparse

    parser = argparse.ArgumentParser(description="Curriculum Manager - Adaptive Difficulty")
    parser.add_argument('--base-dir', default=None, help='Base directory (auto-detect from core.paths)')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Status
    status_p = subparsers.add_parser('status', help='Show curriculum status')
    status_p.add_argument('--skill', choices=['syllo', 'binary'], help='Specific skill')

    # Record accuracy
    record_p = subparsers.add_parser('record', help='Record accuracy')
    record_p.add_argument('skill', choices=['syllo', 'binary'])
    record_p.add_argument('accuracy', type=float, help='Accuracy (0-1)')
    record_p.add_argument('step', type=int, help='Training step')

    # Check & progress
    check_p = subparsers.add_parser('check', help='Record and check for progression')
    check_p.add_argument('skill', choices=['syllo', 'binary'])
    check_p.add_argument('accuracy', type=float)
    check_p.add_argument('step', type=int)

    # Force progress
    prog_p = subparsers.add_parser('progress', help='Force progress to next level')
    prog_p.add_argument('skill', choices=['syllo', 'binary'])

    # Set level
    set_p = subparsers.add_parser('set-level', help='Set skill level')
    set_p.add_argument('skill', choices=['syllo', 'binary'])
    set_p.add_argument('level', type=int)

    # Reset
    reset_p = subparsers.add_parser('reset', help='Reset skill to level 1')
    reset_p.add_argument('skill', choices=['syllo', 'binary'])

    # Generate params
    gen_p = subparsers.add_parser('gen-params', help='Get generation params for current level')
    gen_p.add_argument('skill', choices=['syllo', 'binary'])
    gen_p.add_argument('--count', type=int, default=50)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    if args.base_dir is None:
        try:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        except Exception:
            base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(args.base_dir)

    config_file = base_dir / "config.json"
    with open(config_file) as f:
        config = json.load(f)

    manager = CurriculumManager(base_dir, config)

    if args.command == 'status':
        status = manager.get_status(args.skill)
        print(json.dumps(status, indent=2))

    elif args.command == 'record':
        manager.record_accuracy(args.skill, args.accuracy, args.step)
        print(f"Recorded: {args.skill} {args.accuracy:.1%} at step {args.step}")

    elif args.command == 'check':
        progressed, new_level = manager.check_and_progress(args.skill, args.accuracy, args.step)
        if progressed:
            print(f"PROGRESSED to Level {new_level['level']}: {new_level['name']}")
        else:
            print("No progression")
        status = manager.get_status(args.skill)
        print(f"Current: Level {status['current_level']} ({status['reason']})")

    elif args.command == 'progress':
        new_level = manager.progress_to_next_level(args.skill)
        print(f"Progressed to Level {new_level['level']}: {new_level['name']}")

    elif args.command == 'set-level':
        manager.state["skills"][args.skill]["current_level"] = args.level
        manager._save_state()
        print(f"Set {args.skill} to Level {args.level}")

    elif args.command == 'reset':
        manager.reset_skill(args.skill)
        print(f"Reset {args.skill} to Level 1")

    elif args.command == 'gen-params':
        params = manager.get_generation_params(args.skill, args.count)
        print(json.dumps(params, indent=2))
        print(f"\nUse: python3 data_manager/skill_api_client.py {args.skill} --level {params['level']} --count {params['count']}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
