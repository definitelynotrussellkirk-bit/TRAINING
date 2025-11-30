#!/usr/bin/env python3
"""
Data Manager - Unified coordinator for data generation, testing, and queueing

Responsibilities:
1. Monitor queue depth
2. Request data generation from skill APIs when needed
3. Run quality tests on generated data
4. Queue approved data for training
5. Track metrics and performance

Updated 2025-11-25: Now uses local skill APIs (singleSKILL) instead of remote GPU.
- SkillAPIClient: Connects to SYLLO (8080) and Binary (8090) servers
- CurriculumManager: Manages difficulty progression per skill
- Active skill: Configured in curriculum (defaults to "syllo")
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager.skill_api_client import SkillAPIClient, syllo_to_training_format, SKILL_APIS
from data_manager.curriculum_manager import CurriculumManager
from data_manager.quality_checker import QualityChecker
from core.training_queue import TrainingQueue

# Events system (global announcements)
try:
    from events import (
        emit, data_generating_event, data_generated_event,
        data_queued_event, quality_pass_event, quality_fail_event
    )
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False
    def emit(*args, **kwargs): pass

logger = logging.getLogger(__name__)


class DataManager:
    """
    Unified Data Manager

    Coordinates data generation, quality testing, and queue management.
    Uses singleSKILL APIs (local) for data generation with curriculum-based difficulty.
    """

    def __init__(self, base_dir: Path, config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.config = config

        # Initialize quality checker
        self.quality_checker = QualityChecker(
            min_length=config.get("min_length", 10),
            max_length=config.get("max_length", 4096)
        )

        self.queue = TrainingQueue(str(base_dir))

        # Curriculum Manager (manages difficulty progression for all skills)
        self.curriculum = CurriculumManager(base_dir, config)
        active_skill = self.curriculum.state.get("active_skill", "syllo")
        level_info = self.curriculum.get_current_level(active_skill)
        logger.info(f"✅ Curriculum enabled: {active_skill} Level {level_info['level']} ({level_info['name']})")

        # Skill API clients (lazy-initialized)
        self._skill_clients: Dict[str, SkillAPIClient] = {}

        # Directories
        self.inbox_dir = self.base_dir / "inbox"
        self.reports_dir = self.base_dir / "data_manager" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.last_gen_time = 0
        self.generation_count = 0
        self.approval_rate = 1.0

        # Stats
        self.stats = {
            "total_generated": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "generation_time_avg": 0,
            "quality_scores": []
        }

    def _get_skill_client(self, skill: str) -> SkillAPIClient:
        """Get or create skill API client (lazy initialization)."""
        if skill not in self._skill_clients:
            self._skill_clients[skill] = SkillAPIClient(skill)
        return self._skill_clients[skill]

    def check_skill_api_status(self, skill: Optional[str] = None) -> bool:
        """Check if skill API server is available."""
        if skill is None:
            skill = self.curriculum.state.get("active_skill", "syllo")

        client = self._get_skill_client(skill)
        skill_config = SKILL_APIS[skill]

        logger.info(f"Checking {skill_config['name']} API at {skill_config['base_url']}")

        if not client.health_check():
            logger.error(f"❌ {skill} API server is not available")
            server_script = skill_config['server_script']
            singleskill_dir = Path(server_script).parents[1]
            logger.error(f"   Start with: cd {singleskill_dir} && python3 {server_script} --port {skill_config['base_url'].split(':')[-1]}")
            return False

        try:
            info = client.get_info()
            logger.info(f"✅ {skill_config['name']} API online:")
            logger.info(f"   Skill: {info.get('skill_name', skill)}")
            logger.info(f"   Levels: {info.get('total_levels', '?')}")

            # Show current curriculum level
            level_info = self.curriculum.get_current_level(skill)
            logger.info(f"   Current level: {level_info['level']} ({level_info['name']})")
        except Exception as e:
            logger.warning(f"Could not get server info: {e}")

        return True

    # Legacy alias for backward compatibility
    def check_remote_status(self) -> bool:
        """Legacy method - use check_skill_api_status instead."""
        return self.check_skill_api_status()

    def should_generate(self) -> Tuple[bool, str]:
        """
        Determine if we should generate new data

        Returns:
            (should_generate, reason)
        """
        auto_cfg = self.config.get("auto_generate", {})

        if not auto_cfg.get("enabled", False):
            return False, "Auto-generation disabled in config"

        # Check queue depth
        queue_status = self.queue.get_queue_status()
        threshold = auto_cfg.get("threshold", 0)

        if queue_status["total_queued"] > threshold:
            return False, f"Queue depth ({queue_status['total_queued']}) > threshold ({threshold})"

        # Check if currently processing
        if queue_status["processing"] > 0:
            return False, "Already processing a file"

        # Check cooldown
        cooldown = auto_cfg.get("cooldown_sec", 180)
        time_since_last = time.time() - self.last_gen_time

        if time_since_last < cooldown:
            return False, f"Cooldown: {cooldown - int(time_since_last)}s remaining"

        # Check skill API availability
        active_skill = self.curriculum.state.get("active_skill", "syllo")
        client = self._get_skill_client(active_skill)
        if not client.health_check():
            return False, f"{active_skill} API server unavailable"

        return True, "All checks passed"

    def generate_batch(self, count: int, seed: Optional[int] = None) -> Tuple[bool, List[Dict], Dict]:
        """
        Generate a batch of training data using skill API.

        Uses curriculum to determine:
        - Active skill (syllo or binary)
        - Current difficulty level

        Returns:
            (success, data, metadata)
        """
        active_skill = self.curriculum.state.get("active_skill", "syllo")
        gen_params = self.curriculum.get_generation_params(active_skill, count)
        level = gen_params["level"]
        level_info = self.curriculum.get_current_level(active_skill)

        logger.info(f"🎲 Generating {count:,} {active_skill} examples at Level {level} ({level_info['name']})")

        start_time = time.time()

        try:
            client = self._get_skill_client(active_skill)

            # Build request params
            params = {"count": count, "level": level}
            if seed is not None:
                params["seed"] = seed

            response = client.generate(**params)
            generation_time = time.time() - start_time

            # Convert to training format based on skill
            training_data = []
            if active_skill == "syllo" or active_skill == "sy":
                for puzzle in response.get("puzzles", []):
                    training_data.append(syllo_to_training_format(puzzle))
            elif active_skill == "binary" or active_skill == "bin":
                from data_manager.skill_api_client import binary_to_training_format
                level_info_dict = response.get("level_info", {})
                for sample in response.get("samples", []):
                    training_data.append(binary_to_training_format(sample, level_info_dict))

            logger.info(f"✅ Generated {len(training_data):,} examples in {generation_time:.1f}s")
            logger.info(f"   Rate: {len(training_data) / generation_time:.1f} examples/sec")

            metadata = {
                "skill": active_skill,
                "level": level,
                "level_name": level_info["name"],
                "count": len(training_data),
                "generation_time": generation_time,
                "rate": len(training_data) / generation_time if generation_time > 0 else 0,
                "seed": seed,
                "timestamp": datetime.now().isoformat()
            }

            self.stats["total_generated"] += len(training_data)

            return True, training_data, metadata

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, [], {"error": str(e)}

    def test_quality(self, data: List[Dict], skill: str) -> Tuple[bool, Dict]:
        """
        Run quality tests on generated data using skill-specific profile.

        Args:
            data: List of training examples
            skill: Skill name (e.g., "binary", "syllo") - REQUIRED

        Returns:
            (approved, report)

        Raises:
            UnknownSkillError: If skill has no validation profile
        """
        logger.info(f"🧪 Running quality tests on {len(data):,} {skill} examples...")

        passed, report = self.quality_checker.check_all(data, skill=skill)

        if passed:
            logger.info(f"✅ Quality tests PASSED")
            logger.info(f"   {report['passed_checks']}/{report['total_checks']} checks passed")
            logger.info(f"   Recommendation: {report['recommendation']}")
            self.stats["total_approved"] += len(data)
        else:
            logger.warning(f"❌ Quality tests FAILED")
            logger.warning(f"   {report['passed_checks']}/{report['total_checks']} checks passed")
            logger.warning(f"   Recommendation: {report['recommendation']}")
            self.stats["total_rejected"] += len(data)

        # Track quality score
        quality_score = report['passed_checks'] / report['total_checks']
        self.stats["quality_scores"].append(quality_score)

        return passed, report

    def queue_data(self, data: List[Dict], metadata: Dict, priority: str = "normal") -> bool:
        """
        Queue approved data for training

        Returns:
            True if successfully queued
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        skill = metadata.get("skill", "syllo")
        level = metadata.get("level", 1)
        filename = f"train_{skill}_level{level}_{len(data)}_{timestamp}.jsonl"
        filepath = self.inbox_dir / filename

        logger.info(f"📝 Writing {len(data):,} examples to {filename}")

        try:
            # Write JSONL
            with open(filepath, 'w') as f:
                for example in data:
                    f.write(json.dumps(example) + '\n')

            # Add to queue
            self.queue.add_to_queue(filepath, priority)

            # Save metadata
            meta_file = self.reports_dir / f"{filename}.meta.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Queued: {filename} (priority: {priority})")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to queue data: {e}")
            return False

    def generate_and_queue(self, force: bool = False) -> bool:
        """
        Main workflow: Generate → Test → Queue

        Returns:
            True if successfully generated and queued
        """
        # Check if we should generate
        if not force:
            should_gen, reason = self.should_generate()
            if not should_gen:
                logger.debug(f"Skipping generation: {reason}")
                return False

        # Get configuration
        auto_cfg = self.config.get("auto_generate", {})
        count = auto_cfg.get("count", 100000)
        seed = auto_cfg.get("seed")
        priority = auto_cfg.get("priority", "normal")

        # Get active skill and level for event emission
        active_skill = self.curriculum.state.get("active_skill", "syllo")
        level_info = self.curriculum.get_current_level(active_skill)
        level = level_info.get("level", 1)

        # Emit generation starting event
        emit(data_generating_event(skill=active_skill, count=count, level=level))

        # Step 1: Generate
        start_time = time.time()
        success, data, metadata = self.generate_batch(count, seed)
        duration = time.time() - start_time

        if not success or not data:
            logger.error("Generation failed or returned no data")
            return False

        # Emit generation complete event
        emit(data_generated_event(skill=active_skill, count=len(data), duration=duration))

        # Step 2: Test quality (skill-aware validation)
        approved, report = self.test_quality(data, skill=active_skill)

        # Save quality report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"quality_report_{timestamp}.json"
        self.quality_checker.save_report(report, report_file)

        # Add report to metadata
        metadata["quality_report"] = report
        metadata["approved"] = approved

        # Emit quality test result event
        tests_passed = report.get("tests_passed", 0)
        total_tests = report.get("total_tests", 0)
        if approved:
            emit(quality_pass_event(tests_passed, total_tests))
        else:
            failures = report.get("failed_tests", [])
            emit(quality_fail_event(tests_passed, total_tests, failures))

        # Step 3: Queue if approved
        if approved:
            queued = self.queue_data(data, metadata, priority)
            if queued:
                self.last_gen_time = time.time()
                self.generation_count += 1

                # Emit queued event
                filename = metadata.get("filename", f"batch_{self.generation_count}")
                emit(data_queued_event(filename=filename, count=len(data), priority=priority))

                logger.info(f"🎉 Successfully generated, tested, and queued batch #{self.generation_count}")
                return True
        else:
            logger.warning("⚠️  Data failed quality checks - NOT queued for training")

            # Save rejected data for analysis
            reject_file = self.base_dir / "data_manager" / "rejected" / f"rejected_{timestamp}.jsonl"
            reject_file.parent.mkdir(parents=True, exist_ok=True)

            with open(reject_file, 'w') as f:
                for example in data:
                    f.write(json.dumps(example) + '\n')

            logger.info(f"📁 Rejected data saved to {reject_file} for analysis")

        return False

    def get_active_skill(self) -> str:
        """Get the currently active skill from curriculum."""
        return self.curriculum.state.get("active_skill", "syllo")

    def get_stats(self) -> Dict:
        """Get data manager statistics including curriculum status."""
        avg_quality = (sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"])
                      if self.stats["quality_scores"] else 0.0)

        return {
            "generation_count": self.generation_count,
            "total_generated": self.stats["total_generated"],
            "total_approved": self.stats["total_approved"],
            "total_rejected": self.stats["total_rejected"],
            "approval_rate": (self.stats["total_approved"] / self.stats["total_generated"]
                            if self.stats["total_generated"] > 0 else 0.0),
            "avg_quality_score": avg_quality,
            "queue_status": self.queue.get_queue_status(),
            "curriculum": self.curriculum.get_status()
        }


def main():
    """CLI for Data Manager"""
    import argparse

    parser = argparse.ArgumentParser(description="Data Manager - Generate and test training data")
    parser.add_argument('--base-dir', default=None, help='Base directory (auto-detect from core.paths)')
    parser.add_argument('--config', help='Config file (default: base-dir/config.json)')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Check status
    subparsers.add_parser('status', help='Check remote GPU status')

    # Generate data
    gen_parser = subparsers.add_parser('generate', help='Generate and queue data')
    gen_parser.add_argument('--force', action='store_true', help='Force generation (ignore checks)')
    gen_parser.add_argument('--count', type=int, help='Override example count')
    gen_parser.add_argument('--seed', type=int, help='Random seed')

    # Stats
    subparsers.add_parser('stats', help='Show statistics')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

    config_file = Path(args.config) if args.config else base_dir / "config.json"

    with open(config_file) as f:
        config = json.load(f)

    # Create manager
    manager = DataManager(base_dir, config)

    if args.command == 'status':
        print("\n" + "="*80)
        print("DATA MANAGER STATUS")
        print("="*80 + "\n")

        # Check skill API
        active_skill = manager.curriculum.state.get("active_skill", "syllo")
        if manager.check_skill_api_status(active_skill):
            print(f"\n✅ {active_skill.upper()} API is online and ready\n")
        else:
            print(f"\n❌ {active_skill.upper()} API is offline\n")

        stats = manager.get_stats()

        # Curriculum status
        curriculum = stats.get("curriculum", {})
        print("📚 Curriculum Status:")
        print(f"   Active skill: {curriculum.get('active_skill', 'syllo')}")
        for skill_name, skill_status in curriculum.get("skills", {}).items():
            level = skill_status.get("current_level", 1)
            level_name = skill_status.get("level_name", "Unknown")
            total = skill_status.get("total_levels", "?")
            print(f"   {skill_name}: Level {level}/{total} ({level_name})")
        print()

        print(f"Generation Count: {stats['generation_count']}")
        print(f"Total Generated:  {stats['total_generated']:,}")
        print(f"Total Approved:   {stats['total_approved']:,}")
        print(f"Total Rejected:   {stats['total_rejected']:,}")
        print(f"Approval Rate:    {stats['approval_rate']*100:.1f}%")
        print(f"Avg Quality:      {stats['avg_quality_score']*100:.1f}%")
        print("\nQueue Status:")
        for k, v in stats['queue_status'].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    print(f"  {k2}: {v2}")
            else:
                print(f"  {k}: {v}")
        print("="*80 + "\n")

    elif args.command == 'generate':
        # Override config if args provided
        if args.count:
            manager.config["auto_generate"]["count"] = args.count

        success = manager.generate_and_queue(force=args.force)

        if success:
            print("\n✅ Successfully generated, tested, and queued data\n")
        else:
            print("\n❌ Generation failed or data was rejected\n")

    elif args.command == 'stats':
        stats = manager.get_stats()
        print("\n" + "="*80)
        print("DATA MANAGER STATISTICS")
        print("="*80 + "\n")
        print(json.dumps(stats, indent=2))
        print("\n" + "="*80 + "\n")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
