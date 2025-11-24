#!/usr/bin/env python3
"""
Curriculum Manager - Adaptive Difficulty Progression

Automatically increases training difficulty when model achieves target accuracy.

Progression for SYLLO skill:
1. Mixed (Easy + Medium + Hard)  (start - DEFAULT)
2. Medium + Hard                 (after 75% on mixed)
3. Hard + Ultra                  (after 75% on medium+hard)
4. Ultra only                    (after 75% on hard+ultra)
5. Next skill                    (after 75% on ultra)

Target: 75% accuracy for progression
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class CurriculumManager:
    """
    Manages adaptive curriculum based on model performance

    Tracks difficulty levels and automatically progresses when
    accuracy threshold is reached.
    """

    # Difficulty progression for SYLLO skill
    SYLLO_LEVELS = [
        {
            "level": 1,
            "name": "mixed_default",
            "description": "Mixed (Easy + Medium + Hard) - DEFAULT",
            "difficulties": ["easy", "medium", "hard"],
            "next_threshold": 0.75
        },
        {
            "level": 2,
            "name": "medium_hard",
            "description": "Medium + Hard",
            "difficulties": ["medium", "hard"],
            "next_threshold": 0.75
        },
        {
            "level": 3,
            "name": "hard_ultra",
            "description": "Hard + Ultra",
            "difficulties": ["hard", "ultra"],
            "next_threshold": 0.75
        },
        {
            "level": 4,
            "name": "ultra_only",
            "description": "Ultra only",
            "difficulties": ["ultra"],
            "next_threshold": 0.75
        },
        {
            "level": 5,
            "name": "mastered",
            "description": "SYLLO mastered - ready for next skill",
            "difficulties": ["ultra"],  # Keep generating ultra while transitioning
            "next_threshold": None  # No next level
        }
    ]

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
        self.target_accuracy = self.curriculum_config.get("target_accuracy", 0.75)
        self.enabled = self.curriculum_config.get("enabled", True)

    def _load_state(self) -> Dict:
        """Load curriculum state from disk"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)

        # Default state
        return {
            "current_skill": "syllo",
            "current_level": 1,
            "accuracy_history": [],
            "progression_history": [],
            "started_at": datetime.now().isoformat()
        }

    def _save_state(self):
        """Save curriculum state to disk"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

        logger.info(f"Curriculum state saved: Level {self.state['current_level']}")

    def get_current_level(self) -> Dict[str, Any]:
        """Get current difficulty level configuration"""
        level_idx = self.state["current_level"] - 1

        if self.state["current_skill"] == "syllo":
            if level_idx < len(self.SYLLO_LEVELS):
                return self.SYLLO_LEVELS[level_idx]

        # Fallback
        return self.SYLLO_LEVELS[0]

    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get configuration for data generation based on current level

        Returns dict with difficulty settings to pass to remote generator
        """
        level = self.get_current_level()

        return {
            "difficulties": level["difficulties"],
            "difficulty_weights": self._get_difficulty_weights(level["difficulties"]),
            "skill": self.state["current_skill"],
            "level": level["level"],
            "level_name": level["name"]
        }

    def _get_difficulty_weights(self, difficulties: List[str]) -> Dict[str, float]:
        """
        Get probability weights for each difficulty

        Strategy: Equal weight across all difficulties in current level
        """
        if not difficulties:
            return {"easy": 1.0}

        weight = 1.0 / len(difficulties)
        return {diff: weight for diff in difficulties}

    def record_accuracy(self, accuracy: float, step: int, metadata: Optional[Dict] = None):
        """
        Record accuracy from evaluation

        Args:
            accuracy: Accuracy score (0-1)
            step: Training step number
            metadata: Optional metadata (model_id, eval_job_id, etc.)
        """
        record = {
            "step": step,
            "accuracy": accuracy,
            "level": self.state["current_level"],
            "timestamp": datetime.now().isoformat()
        }

        if metadata:
            record["metadata"] = metadata

        self.state["accuracy_history"].append(record)

        # Keep only last 100 records
        if len(self.state["accuracy_history"]) > 100:
            self.state["accuracy_history"] = self.state["accuracy_history"][-100:]

        self._save_state()

        logger.info(f"Recorded accuracy: {accuracy:.2%} at step {step} (Level {self.state['current_level']})")

    def should_progress(self) -> Tuple[bool, str]:
        """
        Check if model should progress to next difficulty level

        Returns:
            (should_progress, reason)
        """
        if not self.enabled:
            return False, "Curriculum disabled"

        current_level = self.get_current_level()

        # Check if already at max level
        if current_level["level"] >= 6:
            return False, "Already at maximum level (SYLLO mastered)"

        # Need at least 3 recent evals
        recent_evals = [r for r in self.state["accuracy_history"]
                       if r["level"] == self.state["current_level"]]

        if len(recent_evals) < 3:
            return False, f"Need more evals (have {len(recent_evals)}, need 3)"

        # Check last 3 evals
        last_3 = recent_evals[-3:]
        avg_accuracy = sum(r["accuracy"] for r in last_3) / len(last_3)

        threshold = current_level["next_threshold"]

        if avg_accuracy >= threshold:
            return True, f"Average accuracy {avg_accuracy:.2%} ≥ {threshold:.0%}"

        return False, f"Average accuracy {avg_accuracy:.2%} < {threshold:.0%}"

    def progress_to_next_level(self) -> Dict[str, Any]:
        """
        Progress to next difficulty level

        Returns:
            New level configuration
        """
        old_level = self.state["current_level"]
        new_level = old_level + 1

        # Record progression
        progression = {
            "from_level": old_level,
            "to_level": new_level,
            "timestamp": datetime.now().isoformat(),
            "trigger_accuracy": self.state["accuracy_history"][-1]["accuracy"] if self.state["accuracy_history"] else None
        }

        self.state["progression_history"].append(progression)
        self.state["current_level"] = new_level

        self._save_state()

        new_level_config = self.get_current_level()

        logger.info(f"📈 CURRICULUM PROGRESSION!")
        logger.info(f"   Level {old_level} → Level {new_level}")
        logger.info(f"   {new_level_config['description']}")
        logger.info(f"   Difficulties: {', '.join(new_level_config['difficulties'])}")

        return new_level_config

    def check_and_progress(self, accuracy: float, step: int,
                          metadata: Optional[Dict] = None) -> Tuple[bool, Optional[Dict]]:
        """
        Record accuracy and check if should progress

        Args:
            accuracy: Accuracy from eval
            step: Training step
            metadata: Optional metadata

        Returns:
            (progressed, new_level_config or None)
        """
        # Record accuracy
        self.record_accuracy(accuracy, step, metadata)

        # Check if should progress
        should_prog, reason = self.should_progress()

        logger.info(f"Progression check: {reason}")

        if should_prog:
            new_level = self.progress_to_next_level()
            return True, new_level

        return False, None

    def get_status(self) -> Dict[str, Any]:
        """Get curriculum status"""
        level = self.get_current_level()

        # Recent accuracy
        recent = self.state["accuracy_history"][-5:] if self.state["accuracy_history"] else []
        recent_at_level = [r for r in self.state["accuracy_history"]
                          if r["level"] == self.state["current_level"]]

        avg_recent = sum(r["accuracy"] for r in recent_at_level[-3:]) / len(recent_at_level[-3:]) if len(recent_at_level) >= 3 else None

        should_prog, reason = self.should_progress()

        return {
            "enabled": self.enabled,
            "current_skill": self.state["current_skill"],
            "current_level": self.state["current_level"],
            "level_name": level["name"],
            "level_description": level["description"],
            "difficulties": level["difficulties"],
            "target_accuracy": level["next_threshold"],
            "recent_accuracy": [{"step": r["step"], "acc": r["accuracy"]} for r in recent],
            "avg_accuracy_current_level": avg_recent,
            "evals_at_current_level": len(recent_at_level),
            "should_progress": should_prog,
            "progression_reason": reason,
            "total_progressions": len(self.state["progression_history"])
        }

    def reset(self):
        """Reset curriculum to start (Level 1)"""
        self.state = {
            "current_skill": "syllo",
            "current_level": 1,
            "accuracy_history": [],
            "progression_history": [],
            "started_at": datetime.now().isoformat(),
            "reset_at": datetime.now().isoformat()
        }
        self._save_state()
        logger.info("Curriculum reset to Level 1")


def main():
    """CLI for Curriculum Manager"""
    import argparse

    parser = argparse.ArgumentParser(description="Curriculum Manager - Adaptive Difficulty")
    parser.add_argument('--base-dir', default='/path/to/training', help='Base directory')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Status
    subparsers.add_parser('status', help='Show curriculum status')

    # Record accuracy
    record_parser = subparsers.add_parser('record', help='Record accuracy')
    record_parser.add_argument('accuracy', type=float, help='Accuracy (0-1)')
    record_parser.add_argument('step', type=int, help='Training step')

    # Check progression
    check_parser = subparsers.add_parser('check', help='Check if should progress')
    check_parser.add_argument('accuracy', type=float, help='Accuracy (0-1)')
    check_parser.add_argument('step', type=int, help='Training step')

    # Force progress
    subparsers.add_parser('progress', help='Force progress to next level')

    # Reset
    subparsers.add_parser('reset', help='Reset to Level 1')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    base_dir = Path(args.base_dir)
    config_file = base_dir / "config.json"

    with open(config_file) as f:
        config = json.load(f)

    manager = CurriculumManager(base_dir, config)

    if args.command == 'status':
        status = manager.get_status()

        print("\n" + "="*80)
        print("CURRICULUM STATUS")
        print("="*80 + "\n")

        print(f"Skill: {status['current_skill'].upper()}")
        print(f"Level: {status['current_level']} - {status['level_description']}")
        print(f"Difficulties: {', '.join(status['difficulties'])}")
        print(f"Target for next level: {status['target_accuracy']:.0%}")

        print(f"\nProgress:")
        print(f"  Evals at current level: {status['evals_at_current_level']}")

        if status['avg_accuracy_current_level']:
            print(f"  Avg accuracy (last 3): {status['avg_accuracy_current_level']:.2%}")

        print(f"  Should progress: {status['should_progress']}")
        print(f"  Reason: {status['progression_reason']}")

        print(f"\nRecent accuracy:")
        for r in status['recent_accuracy']:
            print(f"  Step {r['step']}: {r['acc']:.2%}")

        print(f"\nTotal progressions: {status['total_progressions']}")
        print("="*80 + "\n")

    elif args.command == 'record':
        manager.record_accuracy(args.accuracy, args.step)
        print(f"✅ Recorded: {args.accuracy:.2%} at step {args.step}")

    elif args.command == 'check':
        progressed, new_level = manager.check_and_progress(args.accuracy, args.step)

        if progressed:
            print(f"\n📈 PROGRESSED to Level {new_level['level']}!")
            print(f"   {new_level['description']}")
            print(f"   Difficulties: {', '.join(new_level['difficulties'])}\n")
        else:
            print(f"\n✅ Recorded accuracy, no progression\n")

    elif args.command == 'progress':
        new_level = manager.progress_to_next_level()
        print(f"\n📈 Manually progressed to Level {new_level['level']}")
        print(f"   {new_level['description']}\n")

    elif args.command == 'reset':
        manager.reset()
        print("\n🔄 Curriculum reset to Level 1\n")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
