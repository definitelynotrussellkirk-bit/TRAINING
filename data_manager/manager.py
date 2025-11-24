#!/usr/bin/env python3
"""
Data Manager - Unified coordinator for data generation, testing, and queueing

Responsibilities:
1. Monitor queue depth
2. Request data generation from remote GPU when needed
3. Run quality tests on generated data
4. Queue approved data for training
5. Track metrics and performance
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

from data_manager.remote_client import RemoteGPUClient
from data_manager.quality_checker import QualityChecker
from core.training_queue import TrainingQueue

logger = logging.getLogger(__name__)


class DataManager:
    """
    Unified Data Manager

    Coordinates data generation, quality testing, and queue management.
    Acts as both pipeline orchestrator and test suite.
    """

    def __init__(self, base_dir: Path, config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.config = config

        # Initialize components
        auto_cfg = config.get("auto_generate", {})
        self.remote_client = RemoteGPUClient(
            host=auto_cfg.get("host", "192.168.x.x"),
            port=auto_cfg.get("port", 8765),
            timeout=auto_cfg.get("timeout", 300)
        )

        self.quality_checker = QualityChecker(
            min_length=config.get("min_length", 10),
            max_length=config.get("max_length", 4096)
        )

        self.queue = TrainingQueue(str(base_dir))

        # Curriculum Manager (for adaptive difficulty)
        try:
            from curriculum_manager import CurriculumManager
            self.curriculum = CurriculumManager(base_dir, config)
            logger.info(f"✅ Curriculum enabled: Level {self.curriculum.state['current_level']}")
        except Exception as e:
            logger.warning(f"Curriculum disabled: {e}")
            self.curriculum = None

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

    def check_remote_status(self) -> bool:
        """Check if remote GPU server is available"""
        logger.info(f"Checking remote GPU at {self.remote_client.host}:{self.remote_client.port}")

        if not self.remote_client.is_available():
            logger.error("❌ Remote GPU server is not available")
            return False

        health = self.remote_client.health_check()
        gpu_info = health.get("gpu", {})

        logger.info(f"✅ Remote GPU online:")
        logger.info(f"   Device: {gpu_info.get('device_name', 'Unknown')}")
        logger.info(f"   VRAM: {gpu_info.get('memory_allocated_gb', 0):.1f}GB / 24GB")
        logger.info(f"   Active model: {health.get('active_model', 'None')}")
        logger.info(f"   Worker busy: {health.get('worker_busy', False)}")

        return True

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

        # Check remote availability
        if not self.remote_client.is_available():
            return False, "Remote GPU server unavailable"

        return True, "All checks passed"

    def generate_batch(self, count: int, seed: Optional[int] = None) -> Tuple[bool, List[Dict], Dict]:
        """
        Generate a batch of training data

        Returns:
            (success, data, metadata)
        """
        logger.info(f"🎲 Generating {count:,} examples from remote GPU...")

        start_time = time.time()

        try:
            auto_cfg = self.config.get("auto_generate", {})
            payload = auto_cfg.get("payload", {}).copy()

            # Add curriculum difficulty if enabled
            if self.curriculum:
                curriculum_config = self.curriculum.get_generation_config()
                # TEMPORARY: Remote API doesn't support curriculum yet
                # Just log the intended config but don't send it
                logger.info(f"   📚 Curriculum Level {curriculum_config['level']}: {curriculum_config['level_name']}")
                logger.info(f"   Difficulties: {', '.join(curriculum_config['difficulties'])}")
                logger.info(f"   ⚠️  Remote API doesn't support curriculum - generating with default mix")

            raw_data = self.remote_client.generate_data(
                count=count,
                seed=seed,
                payload=payload
            )

            generation_time = time.time() - start_time

            logger.info(f"✅ Generated {len(raw_data):,} examples in {generation_time:.1f}s")
            logger.info(f"   Rate: {len(raw_data) / generation_time:.1f} examples/sec")

            # Convert to training format
            training_data = self._convert_to_training_format(raw_data)

            metadata = {
                "count": len(training_data),
                "generation_time": generation_time,
                "rate": len(training_data) / generation_time,
                "seed": seed,
                "timestamp": datetime.now().isoformat()
            }

            self.stats["total_generated"] += len(training_data)

            return True, training_data, metadata

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return False, [], {"error": str(e)}

    def test_quality(self, data: List[Dict]) -> Tuple[bool, Dict]:
        """
        Run quality tests on generated data

        Returns:
            (approved, report)
        """
        logger.info(f"🧪 Running quality tests on {len(data):,} examples...")

        passed, report = self.quality_checker.check_all(data)

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
        filename = f"syllo_autogen_{timestamp}_count{len(data)}.jsonl"
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

        # Step 1: Generate
        success, data, metadata = self.generate_batch(count, seed)
        if not success or not data:
            logger.error("Generation failed or returned no data")
            return False

        # Step 2: Test quality
        approved, report = self.test_quality(data)

        # Save quality report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"quality_report_{timestamp}.json"
        self.quality_checker.save_report(report, report_file)

        # Add report to metadata
        metadata["quality_report"] = report
        metadata["approved"] = approved

        # Step 3: Queue if approved
        if approved:
            queued = self.queue_data(data, metadata, priority)
            if queued:
                self.last_gen_time = time.time()
                self.generation_count += 1
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

    def get_stats(self) -> Dict:
        """Get data manager statistics"""
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
            "remote_gpu": self.remote_client.get_memory_usage() if self.remote_client.is_available() else {}
        }

    def _convert_to_training_format(self, raw_data: List[Dict]) -> List[Dict]:
        """Convert raw generated data to training format"""
        # This depends on what format the remote server returns
        # For now, assume it returns data in the correct format
        # Override this method if conversion is needed
        return raw_data


def main():
    """CLI for Data Manager"""
    import argparse

    parser = argparse.ArgumentParser(description="Data Manager - Generate and test training data")
    parser.add_argument('--base-dir', default='/path/to/training', help='Base directory')
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

        if manager.check_remote_status():
            print("\n✅ Remote GPU is online and ready\n")
        else:
            print("\n❌ Remote GPU is offline or unreachable\n")

        stats = manager.get_stats()
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
