#!/usr/bin/env python3
"""
Data Generation Automation - Auto-generates training data based on queue status
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataGenerationAutomation:
    def __init__(self, base_dir: str = "/path/to/training", interval: int = 300):
        self.base_dir = Path(base_dir)
        self.interval = interval
        self.queue_dir = self.base_dir / "queue"
        self.generation_script = Path("/path/to/skills/skill_syllo_variant/scripts/export_training_data.py")

    def count_queue_files(self) -> int:
        """Count total files in queue"""
        total = 0
        for priority in ['high', 'normal', 'low']:
            priority_dir = self.queue_dir / priority
            if priority_dir.exists():
                total += len(list(priority_dir.glob('*.jsonl')))
        return total

    def get_curriculum_difficulty(self) -> dict:
        """Get current curriculum difficulty settings"""
        status_file = self.base_dir / "status/curriculum_optimization.json"
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
                if 'recommended_strategy' in data and 'params' in data['recommended_strategy']:
                    return data['recommended_strategy']['params'].get('difficulty_weights', {
                        'easy': 0.33, 'medium': 0.33, 'hard': 0.34
                    })
        return {'easy': 0.33, 'medium': 0.33, 'hard': 0.34}

    def generate_data(self, count: int = 100000, priority: str = "normal"):
        """Generate new training data"""
        difficulty = self.get_curriculum_difficulty()
        diff_str = f"easy:{difficulty.get('easy', 0.33)},medium:{difficulty.get('medium', 0.33)},hard:{difficulty.get('hard', 0.34)}"
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.queue_dir / priority / f"auto_gen_{timestamp}_count{count}.jsonl"
        
        logger.info(f"Generating {count} examples with difficulty: {diff_str}")
        
        cmd = [
            "python3", str(self.generation_script),
            "--count", str(count),
            "--difficulty", diff_str,
            "--output", str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info(f"✅ Generated: {output_file}")
            return output_file
        else:
            logger.error(f"Generation failed: {result.stderr}")
            return None

    def run_continuous(self):
        """Run continuous data generation loop"""
        logger.info("🎲 DATA GENERATION AUTOMATION - STARTING")
        logger.info(f"Check interval: {self.interval}s")
        logger.info("=" * 80)
        
        while True:
            try:
                queue_count = self.count_queue_files()
                logger.info(f"Queue: {queue_count} files")
                
                if queue_count < 2:
                    logger.warning(f"⚠️  Queue low ({queue_count} files) - triggering generation")
                    self.generate_data(count=100000, priority="normal")
                elif queue_count < 5:
                    logger.info(f"Queue moderate ({queue_count} files) - pre-generating")
                    self.generate_data(count=50000, priority="low")
                else:
                    logger.info(f"Queue healthy ({queue_count} files)")
                
                logger.info(f"Next check in {self.interval}s...")
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(self.interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='/path/to/training')
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    
    daemon = DataGenerationAutomation(args.base_dir, args.interval)
    daemon.run_continuous()
