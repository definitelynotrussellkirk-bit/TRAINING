#!/usr/bin/env python3
"""
Data Generation Automation v2.0 - Multi-Skill API-Based Generation

Auto-generates training data based on queue status.
Supports multiple skills via HTTP API with level-based difficulty.

Supported Skills:
- SYLLO: Word puzzle generation (10 levels with signal degradation)
- Binary Arithmetic: Binary math problems (7 levels)
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Skill configuration
SKILLS = {
    'syllo': {
        'name': 'SYLLO Word Puzzles',
        'api_url': 'http://localhost:8080',
        'api_host': 'localhost',  # For local generation
        'levels': 10,  # 10-level signal degradation system
        'default_level': 5,  # Intermediate (middle of 10 levels)
        'weight': 0.7,  # 70% of generated data
    },
    'binary': {
        'name': 'Binary Arithmetic',
        'api_url': 'http://localhost:8090',
        'api_host': 'localhost',
        'levels': 7,
        'default_level': 3,  # Two-digit (6-bit)
        'weight': 0.3,  # 30% of generated data
    }
}


class DataGenerationAutomation:
    def __init__(self, base_dir: str = None, interval: int = 300):
        if base_dir is None:
            from core.paths import require_base_dir
            base_dir = str(require_base_dir())
        self.base_dir = Path(base_dir)
        self.interval = interval
        self.queue_dir = self.base_dir / "queue"
        self.skills = SKILLS

    def count_queue_files(self) -> int:
        """Count total files in queue"""
        total = 0
        for priority in ['high', 'normal', 'low']:
            priority_dir = self.queue_dir / priority
            if priority_dir.exists():
                total += len(list(priority_dir.glob('*.jsonl')))
        return total

    def get_skill_info(self, skill_key: str) -> dict:
        """Query skill API for metadata"""
        skill = self.skills.get(skill_key)
        if not skill:
            return None

        try:
            response = requests.get(f"{skill['api_url']}/info", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get {skill_key} info: {e}")

        return None

    def get_curriculum_level(self, skill_key: str) -> int:
        """
        Get recommended difficulty level from curriculum optimizer

        Maps curriculum difficulty weights to skill levels:
        - High easy weight (>50%) -> Lower levels (1-2)
        - Balanced (30-50% each) -> Middle levels (3-4)
        - High hard weight (>40%) -> Upper levels (4-5 for SYLLO, 5-7 for binary)
        """
        status_file = self.base_dir / "status/curriculum_optimization.json"

        if status_file.exists():
            try:
                with open(status_file) as f:
                    data = json.load(f)
                    if 'recommended_strategy' in data and 'params' in data['recommended_strategy']:
                        weights = data['recommended_strategy']['params'].get('difficulty_weights', {})

                        easy = weights.get('easy', 0.33)
                        hard = weights.get('hard', 0.33)

                        skill = self.skills[skill_key]
                        max_level = skill['levels']

                        # Map curriculum difficulty to skill level
                        if easy > 0.50:  # Focus on easy
                            return max(1, int(max_level * 0.3))  # Level 1-2
                        elif hard > 0.40:  # Increase challenge
                            return min(max_level, int(max_level * 0.8))  # Level 4-5 or 5-7
                        else:  # Balanced
                            return skill['default_level']  # Middle levels
            except Exception as e:
                logger.warning(f"Failed to read curriculum: {e}")

        return self.skills[skill_key]['default_level']

    def generate_skill_data(
        self,
        skill_key: str,
        count: int,
        priority: str = "normal"
    ) -> dict:
        """Generate training data for a specific skill via API"""

        skill = self.skills.get(skill_key)
        if not skill:
            logger.error(f"Unknown skill: {skill_key}")
            return None

        # Get curriculum-recommended level
        level = self.get_curriculum_level(skill_key)

        # Create output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"auto_{skill_key}_level{level}_{timestamp}_count{count}.jsonl"
        output_file = self.queue_dir / priority / filename

        logger.info(f"🎯 Generating {skill['name']} data:")
        logger.info(f"   Level: {level}/{skill['levels']}")
        logger.info(f"   Count: {count:,}")
        logger.info(f"   API: {skill['api_url']}")

        try:
            # Call skill API
            response = requests.post(
                f"{skill['api_url']}/generate",
                json={
                    "count": count,
                    "level": level,
                    "seed": int(datetime.now().timestamp()) % 100000
                },
                timeout=600  # 10 minute timeout for large generations
            )

            if response.status_code == 200:
                data = response.json()

                # Write conversations to JSONL
                with open(output_file, 'w') as f:
                    for conversation in data.get('conversations', []):
                        f.write(json.dumps(conversation) + '\n')

                logger.info(f"✅ Generated: {filename}")

                return {
                    'success': True,
                    'skill': skill_key,
                    'level': level,
                    'count': len(data.get('conversations', [])),
                    'output_file': str(output_file),
                    'filename': filename
                }
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {'success': False, 'error': f"HTTP {response.status_code}"}

        except requests.Timeout:
            logger.error("Generation timed out after 10 minutes")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {'success': False, 'error': str(e)}

    def generate_mixed_batch(self, total_count: int = 100000, priority: str = "normal"):
        """
        Generate a mixed batch across all skills based on configured weights

        Example: 100k total -> 70k SYLLO + 30k Binary
        """
        logger.info(f"📦 Generating mixed batch: {total_count:,} total examples")

        results = []

        for skill_key, skill in self.skills.items():
            skill_count = int(total_count * skill['weight'])

            if skill_count > 0:
                result = self.generate_skill_data(skill_key, skill_count, priority)
                if result:
                    results.append(result)

        # Summary
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Generation Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Successful: {len(successful)}/{len(results)}")

        for result in successful:
            logger.info(f"  ✓ {result['filename']} ({result['count']:,} examples)")

        if failed:
            for result in failed:
                logger.info(f"  ✗ {result.get('skill', 'unknown')}: {result.get('error', 'unknown')}")

        return results

    def run_continuous(self):
        """Run continuous data generation loop"""
        logger.info("=" * 80)
        logger.info("🎲 DATA GENERATION AUTOMATION v2.0 - MULTI-SKILL API MODE")
        logger.info("=" * 80)
        logger.info(f"Check interval: {self.interval}s")
        logger.info(f"Supported skills: {', '.join(s['name'] for s in self.skills.values())}")
        logger.info("=" * 80)

        # Test API connectivity
        for skill_key, skill in self.skills.items():
            info = self.get_skill_info(skill_key)
            if info:
                logger.info(f"✓ {skill['name']}: {skill['levels']} levels available")
            else:
                logger.warning(f"⚠ {skill['name']}: API not responding")

        logger.info("=" * 80)

        while True:
            try:
                queue_count = self.count_queue_files()
                logger.info(f"\n📊 Queue Status: {queue_count} files")

                if queue_count < 2:
                    logger.warning(f"⚠️  Queue low ({queue_count} files) - generating large batch")
                    self.generate_mixed_batch(total_count=100000, priority="normal")
                elif queue_count < 5:
                    logger.info(f"Queue moderate ({queue_count} files) - pre-generating")
                    self.generate_mixed_batch(total_count=50000, priority="low")
                else:
                    logger.info(f"✓ Queue healthy ({queue_count} files)")

                logger.info(f"\n💤 Next check in {self.interval}s...\n")
                time.sleep(self.interval)

            except KeyboardInterrupt:
                logger.info("\n🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                time.sleep(self.interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Multi-skill data generation automation')
    parser.add_argument('--base-dir', default=None, help='Base directory (default: auto-detected)')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    parser.add_argument('--test', action='store_true', help='Test generation and exit')
    parser.add_argument('--skill', choices=['syllo', 'binary', 'mixed'], default='mixed',
                        help='Skill to generate (default: mixed)')
    parser.add_argument('--count', type=int, default=100, help='Example count for test mode')

    args = parser.parse_args()

    daemon = DataGenerationAutomation(args.base_dir, args.interval)

    if args.test:
        logger.info("🧪 TEST MODE - Single generation")
        if args.skill == 'mixed':
            daemon.generate_mixed_batch(args.count, "normal")
        else:
            daemon.generate_skill_data(args.skill, args.count, "normal")
    else:
        daemon.run_continuous()
