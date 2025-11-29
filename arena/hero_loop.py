#!/usr/bin/env python3
"""
Hero Loop - The Autonomous Training Daemon

Heroes never sleep. They either train on available data or
generate new training data based on their skill priorities.

Usage:
    python3 -m arena.hero_loop titan-qwen3-4b/campaign-001
    python3 -m arena.hero_loop dio-qwen3-0.6b/campaign-001
"""

import os
import sys
import json
import time
import signal
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir
from arena.factory import create_trainer, load_hero_config, load_campaign_config


class HeroLoop:
    """
    Autonomous training loop for a hero campaign.
    
    The hero loop implements the core game mechanic:
    - If data exists in queue → Train (gain XP)
    - If queue empty → Generate data (seek adventures)
    - Repeat forever (heroes never rest)
    """
    
    def __init__(self, campaign_path: Path, poll_interval: int = 30):
        """
        Initialize the hero loop.
        
        Args:
            campaign_path: Path to campaign directory
            poll_interval: Seconds between queue checks when idle
        """
        self.campaign_path = campaign_path
        self.poll_interval = poll_interval
        self.running = False
        
        # Set up logging
        self.logger = logging.getLogger(f"hero_loop.{campaign_path.name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
            ))
            self.logger.addHandler(handler)
        
        # Load configs
        self.campaign_config = load_campaign_config(campaign_path)
        self.hero_id = self.campaign_config.get("hero_id", campaign_path.parent.name)
        self.hero_config = load_hero_config(self.hero_id)
        
        # Set up directories
        self.data_dir = campaign_path / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.completed_dir = campaign_path / "data" / "completed"
        self.completed_dir.mkdir(exist_ok=True)
        
        # Create trainer
        self.trainer = create_trainer(self.hero_config, campaign_path)
        
        # Stats
        self.total_steps = 0
        self.files_processed = 0
        self.generations_triggered = 0
        
        self.logger.info(f"Hero Loop initialized for {self.hero_config.get('name', 'Unknown')}")
        self.logger.info(f"  Campaign: {campaign_path}")
        self.logger.info(f"  Trainer: {self.trainer.name}")
    
    def _get_pending_files(self) -> List[Path]:
        """Get training files waiting in the queue."""
        files = list(self.data_dir.glob("training_*.jsonl"))
        # Sort by modification time (oldest first)
        return sorted(files, key=lambda p: p.stat().st_mtime)
    
    def _generate_data(self) -> Optional[Path]:
        """
        Generate training data based on skill priorities.
        
        Returns:
            Path to generated data file, or None if generation failed
        """
        idle_config = self.hero_config.get("idle_behavior", {})
        if not idle_config.get("enabled", True):
            self.logger.info("Idle generation disabled")
            return None
        
        skill_priorities = idle_config.get("skill_priorities", {})
        gen_config = idle_config.get("generation", {})
        batch_size = gen_config.get("batch_size", 1000)
        levels = gen_config.get("levels", [1, 2, 3])
        
        self.logger.info(f"Generating training data (batch={batch_size}, levels={levels})")
        
        # Port mapping for skills
        skill_ports = {"sy": 8080, "bin": 8090}
        
        all_samples = []
        
        for skill, priority in skill_priorities.items():
            if priority <= 0:
                continue
            
            port = skill_ports.get(skill)
            if not port:
                self.logger.warning(f"Unknown skill: {skill}")
                continue
            
            count_for_skill = int(batch_size * priority)
            count_per_level = count_for_skill // len(levels)
            
            for level in levels:
                try:
                    url = f"http://localhost:{port}/generate"
                    payload = {"level": level, "count": count_per_level}
                    
                    self.logger.info(f"  {skill.upper()} L{level}: requesting {count_per_level} samples")
                    
                    resp = requests.post(url, json=payload, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    items = data.get("samples", []) or data.get("puzzles", [])
                    for item in items:
                        prompt = item.get("user_prompt", "")
                        response = item.get("assistant_response", "")
                        if prompt and response:
                            all_samples.append({
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": response}
                                ]
                            })
                    
                    self.logger.info(f"    Got {len(items)} samples")
                    
                except Exception as e:
                    self.logger.error(f"  {skill.upper()} L{level}: failed - {e}")
        
        if not all_samples:
            self.logger.error("No samples generated. Are skill servers running?")
            return None
        
        # Shuffle and save
        import random
        random.shuffle(all_samples)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"training_{timestamp}.jsonl"
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        self.logger.info(f"Generated {len(all_samples)} samples -> {output_path.name}")
        self.generations_triggered += 1
        
        return output_path
    
    def _process_file(self, data_path: Path) -> bool:
        """
        Process a single training file.
        
        Returns:
            True if training succeeded, False otherwise
        """
        self.logger.info(f"Training on: {data_path.name}")
        
        result = self.trainer.train(data_path)
        
        if result.success:
            self.total_steps += result.steps_completed
            self.files_processed += 1
            
            self.logger.info(f"Training complete: {result.steps_completed} steps, loss={result.final_loss:.4f}")
            
            # Move to completed
            completed_path = self.completed_dir / data_path.name
            data_path.rename(completed_path)
            self.logger.info(f"  Moved to: {completed_path}")
            
            return True
        else:
            self.logger.error(f"Training failed: {result.error_message}")
            # Move to failed directory
            failed_dir = self.campaign_path / "data" / "failed"
            failed_dir.mkdir(exist_ok=True)
            failed_path = failed_dir / data_path.name
            data_path.rename(failed_path)
            return False
    
    def run_once(self) -> bool:
        """
        Run one iteration of the hero loop.
        
        Returns:
            True if work was done, False if idle
        """
        # Check for pending files
        pending = self._get_pending_files()
        
        if pending:
            # Train on oldest file
            self._process_file(pending[0])
            return True
        
        # Queue empty - check if we should generate
        idle_config = self.hero_config.get("idle_behavior", {})
        min_depth = idle_config.get("generation", {}).get("min_queue_depth", 1)
        
        if len(pending) < min_depth:
            generated = self._generate_data()
            if generated:
                return True
        
        return False
    
    def run(self):
        """
        Run the hero loop forever.
        
        Handles SIGTERM/SIGINT for graceful shutdown.
        """
        self.running = True
        
        def signal_handler(signum, frame):
            self.logger.info("Shutdown signal received")
            self.running = False
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        self.logger.info("=" * 60)
        self.logger.info(f"HERO LOOP STARTED: {self.hero_config.get('name', 'Unknown')}")
        self.logger.info("=" * 60)
        self.logger.info("The hero never sleeps. Training continuously.")
        self.logger.info("Press Ctrl+C to stop.")
        self.logger.info("")
        
        iteration = 0
        while self.running:
            iteration += 1
            
            try:
                did_work = self.run_once()
                
                if not did_work:
                    self.logger.info(f"[Idle] Queue empty, waiting {self.poll_interval}s... (iteration {iteration})")
                    time.sleep(self.poll_interval)
                else:
                    # Brief pause between files
                    time.sleep(2)
                    
            except Exception as e:
                self.logger.exception(f"Error in hero loop: {e}")
                time.sleep(self.poll_interval)
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("HERO LOOP STOPPED")
        self.logger.info(f"  Files processed: {self.files_processed}")
        self.logger.info(f"  Total steps: {self.total_steps}")
        self.logger.info(f"  Data generations: {self.generations_triggered}")
        self.logger.info("=" * 60)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the autonomous hero loop")
    parser.add_argument("campaign", help="Campaign path (e.g., titan-qwen3-4b/campaign-001)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between idle checks")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit")
    
    args = parser.parse_args()
    
    base_dir = get_base_dir()
    campaign_path = base_dir / "campaigns" / args.campaign
    
    if not campaign_path.exists():
        print(f"ERROR: Campaign not found: {campaign_path}")
        sys.exit(1)
    
    loop = HeroLoop(campaign_path, poll_interval=args.poll_interval)
    
    if args.once:
        loop.run_once()
    else:
        loop.run()


if __name__ == "__main__":
    main()
