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
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir
from core.job import Job
from core.job_logger import JobLogger
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

        # Global inbox for dropping files (checked in addition to data_dir)
        self.inbox_dir = campaign_path.parent.parent.parent / "inbox"
        
        # Create trainer
        self.trainer = create_trainer(self.hero_config, campaign_path)

        # Job logger for tracking training jobs
        status_dir = campaign_path / "status"
        status_dir.mkdir(exist_ok=True)
        self.job_logger = JobLogger(status_dir / "job_history.jsonl")
        self.current_job: Optional[Job] = None

        # Stats
        self.total_steps = 0
        self.files_processed = 0
        self.generations_triggered = 0
        
        self.logger.info(f"Hero Loop initialized for {self.hero_config.get('name', 'Unknown')}")
        self.logger.info(f"  Campaign: {campaign_path}")
        self.logger.info(f"  Trainer: {self.trainer.name}")
    
    def _get_pending_files(self) -> List[Path]:
        """Get training files waiting in the queue.

        Checks both the campaign's data directory and the global inbox.
        Files from inbox are moved to data_dir with priority prefix.

        Priority folders (processed in order):
          inbox/high/*.jsonl   → training_0high_*.jsonl  (first)
          inbox/normal/*.jsonl → training_1normal_*.jsonl
          inbox/*.jsonl        → training_1normal_*.jsonl
          inbox/low/*.jsonl    → training_2low_*.jsonl   (last)
        """
        import shutil

        # Priority mapping: folder name -> (sort_prefix, display_name)
        PRIORITY_MAP = {
            "high": ("0high", "HIGH"),
            "normal": ("1normal", "NORMAL"),
            "low": ("2low", "LOW"),
        }

        # Process inbox with priority folders
        if self.inbox_dir.exists():
            # Check priority subdirs first
            for priority_name, (prefix, display) in PRIORITY_MAP.items():
                priority_dir = self.inbox_dir / priority_name
                if priority_dir.exists():
                    for inbox_file in priority_dir.glob("*.jsonl"):
                        if inbox_file.name.endswith("_stats.json"):
                            continue
                        new_name = f"training_{prefix}_{inbox_file.name}"
                        dest = self.data_dir / new_name
                        self.logger.info(f"[Inbox/{display}] Moving {inbox_file.name} → {dest.name}")
                        shutil.move(str(inbox_file), str(dest))

            # Root inbox files get normal priority
            for inbox_file in self.inbox_dir.glob("*.jsonl"):
                if inbox_file.name.endswith("_stats.json"):
                    continue
                new_name = f"training_1normal_{inbox_file.name}"
                dest = self.data_dir / new_name
                self.logger.info(f"[Inbox] Moving {inbox_file.name} → {dest.name}")
                shutil.move(str(inbox_file), str(dest))

        files = list(self.data_dir.glob("training_*.jsonl"))
        # Sort by filename (priority prefix) then by mtime within same priority
        return sorted(files, key=lambda p: (p.name.split('_')[1] if '_' in p.name else '1normal', p.stat().st_mtime))
    
    def _generate_data(self) -> Optional[Path]:
        """
        Generate training data based on skill priorities.
        
        Returns:
            Path to generated data file, or None if generation failed
        """
        idle_config = self.hero_config.get("idle_behavior", {})
        if not idle_config.get("enabled", True):
            self.logger.info("Idle behavior disabled")
            return None

        skill_priorities = idle_config.get("skill_priorities", {})
        gen_config = idle_config.get("generation", {})

        # Check if generation is explicitly disabled
        if not gen_config.get("enabled", True):
            self.logger.debug("Auto-generation disabled in hero config")
            return None

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

        Creates a Job object to track the training lifecycle.

        Returns:
            True if training succeeded, False otherwise
        """
        self.logger.info(f"Training on: {data_path.name}")

        # Create Job and log queued state
        num_examples = sum(1 for _ in open(data_path, 'r', encoding='utf-8'))
        job = Job.from_file(
            path=str(data_path),
            hero_id=self.hero_id,
            campaign_id=self.campaign_path.name,
        )
        job.num_examples = num_examples
        self.job_logger.log_job(job)  # Log queued
        self.current_job = job

        # Mark as started
        job.start()
        self.job_logger.log_job(job)  # Log processing

        # Execute training
        result = self.trainer.train(data_path)

        if result.success:
            self.total_steps += result.steps_completed
            self.files_processed += 1

            self.logger.info(f"Training complete: {result.steps_completed} steps, loss={result.final_loss:.4f}")

            # Mark job as completed
            job.complete(
                final_step=result.steps_completed,
                final_loss=result.final_loss or 0.0,
            )
            self.job_logger.log_job(job)  # Log completed

            # Move to completed
            completed_path = self.completed_dir / data_path.name
            data_path.rename(completed_path)
            self.logger.info(f"  Moved to: {completed_path}")

            self.current_job = None
            return True
        else:
            self.logger.error(f"Training failed: {result.error_message}")

            # Mark job as failed
            job.fail(result.error_message or "Unknown error")
            self.job_logger.log_job(job)  # Log failed

            # Move to failed directory
            failed_dir = self.campaign_path / "data" / "failed"
            failed_dir.mkdir(exist_ok=True)
            failed_path = failed_dir / data_path.name
            data_path.rename(failed_path)

            self.current_job = None
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
            # Clean up GPU memory immediately on signal
            self._cleanup_gpu()

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
        
        # Final cleanup
        self._cleanup_gpu()

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("HERO LOOP STOPPED")
        self.logger.info(f"  Files processed: {self.files_processed}")
        self.logger.info(f"  Total steps: {self.total_steps}")
        self.logger.info(f"  Data generations: {self.generations_triggered}")
        self.logger.info("=" * 60)

    def _cleanup_gpu(self):
        """Clean up GPU memory to prevent zombie processes holding VRAM."""
        try:
            # Clear trainer reference if exists
            if hasattr(self, 'trainer') and self.trainer is not None:
                if hasattr(self.trainer, 'model'):
                    del self.trainer.model
                if hasattr(self.trainer, 'engine') and self.trainer.engine is not None:
                    if hasattr(self.trainer.engine, 'model'):
                        del self.trainer.engine.model
                    if hasattr(self.trainer.engine, 'tokenizer'):
                        del self.trainer.engine.tokenizer
                del self.trainer
                self.trainer = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.info("GPU memory cleaned up")
        except Exception as e:
            self.logger.warning(f"GPU cleanup warning: {e}")


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

    # Validate against active_campaign.json to prevent mismatches
    active_file = base_dir / "control" / "active_campaign.json"
    if active_file.exists():
        try:
            active_data = json.load(open(active_file))
            active_path = active_data.get("campaign_path", "")
            if active_path and active_path != args.campaign:
                print(f"")
                print(f"╔══════════════════════════════════════════════════════════════════╗")
                print(f"║  ⚠️  CAMPAIGN MISMATCH WARNING                                    ║")
                print(f"╠══════════════════════════════════════════════════════════════════╣")
                print(f"║  You are starting: {args.campaign:<43} ║")
                print(f"║  Active campaign:  {active_path:<43} ║")
                print(f"╠══════════════════════════════════════════════════════════════════╣")
                print(f"║  This may cause confusion! Use one of:                           ║")
                print(f"║    1. python3 scripts/start_hero_loop.py (uses active campaign)  ║")
                print(f"║    2. Update active_campaign.json to match                       ║")
                print(f"╚══════════════════════════════════════════════════════════════════╝")
                print(f"")
                print(f"Continuing in 5 seconds... Press Ctrl+C to abort.")
                time.sleep(5)
        except Exception:
            pass  # Ignore errors reading active_campaign.json

    loop = HeroLoop(campaign_path, poll_interval=args.poll_interval)
    
    if args.once:
        loop.run_once()
    else:
        loop.run()


if __name__ == "__main__":
    main()
