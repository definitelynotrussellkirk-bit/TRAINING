#!/usr/bin/env python3
"""
SYLLO L1 Data Generator

Generates Level 1 SYLLO training data using prompt+solution pairs (correctly matched formats).
Runs as daemon, generates when queue is low.

Usage:
    python3 monitoring/syllo_l1_generator.py --daemon --interval 300 --threshold 2
    python3 monitoring/syllo_l1_generator.py --generate 1000  # One-shot
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SYLLO_API = "http://localhost:8080"
BASE_DIR = Path("/path/to/training")
QUEUE_DIR = BASE_DIR / "queue"
STATUS_FILE = BASE_DIR / "status" / "syllo_l1_generator.json"

# Generator versioning for lineage tracking
GENERATOR_ID = "syllo_l1_generator"
GENERATOR_VERSION = "1.0.0"


def count_queue_files() -> int:
    """Count .jsonl files in queue"""
    total = 0
    for priority in ['high', 'normal', 'low']:
        pdir = QUEUE_DIR / priority
        if pdir.exists():
            total += len(list(pdir.glob('*.jsonl')))
    return total


def generate_batch(count: int, level: int = 1, priority: str = "high") -> dict:
    """Generate a batch of L1 SYLLO data using correct format."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"train_SYLLO_L{level}_{timestamp}.jsonl"
    output_file = QUEUE_DIR / priority / filename
    
    logger.info(f"Generating {count} L{level} examples...")
    
    examples = []
    batch_size = 50
    generated = 0
    
    while generated < count:
        batch = min(batch_size, count - generated)
        try:
            resp = requests.post(
                f"{SYLLO_API}/generate",
                json={"level": level, "count": batch},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            
            for puzzle in data.get("puzzles", []):
                prompt = puzzle.get("prompt", "")
                # Use 'solution' NOT 'solution_text' - matches prompt format!
                solution = puzzle.get("solution", "")
                
                if not prompt or not solution:
                    continue
                
                example = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": solution}
                    ],
                    "metadata": {
                        "source": "syllo_api",
                        "generator_id": GENERATOR_ID,
                        "generator_version": GENERATOR_VERSION,
                        "level": level,
                        "puzzle_id": puzzle.get("puzzle_id", ""),
                        "output_variant": puzzle.get("output_variant", ""),
                        "words": [w.get("label", "") for w in puzzle.get("words", [])]
                    }
                }
                examples.append(example)
                generated += 1
            
            if generated % 100 == 0:
                logger.info(f"  {generated}/{count} examples...")
                
        except Exception as e:
            logger.error(f"API error: {e}")
            time.sleep(5)
            continue
    
    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    logger.info(f"✅ Wrote {len(examples)} examples to {filename}")
    
    # Update status
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "last_generation": {
            "timestamp": datetime.now().isoformat(),
            "count": len(examples),
            "level": level,
            "file": filename
        },
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)
    
    return {
        "success": True,
        "count": len(examples),
        "file": str(output_file)
    }


def get_curriculum_level() -> int:
    """Get current SYLLO level from curriculum state."""
    try:
        state_file = BASE_DIR / "data_manager" / "curriculum_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            return state.get("skills", {}).get("syllo", {}).get("current_level", 1)
    except Exception as e:
        logger.warning(f"Could not read curriculum state: {e}")
    return 1


def run_daemon(interval: int, threshold: int, batch_size: int, level: int):
    """Run as daemon, generating when queue drops below threshold.

    If level=0, follows curriculum (auto-advances with model).
    Otherwise uses fixed level.
    """
    follow_curriculum = (level == 0)

    if follow_curriculum:
        logger.info("Starting SYLLO generator daemon (following curriculum)")
    else:
        logger.info(f"Starting SYLLO L{level} generator daemon (fixed level)")
    logger.info(f"  Interval: {interval}s")
    logger.info(f"  Threshold: {threshold} files")
    logger.info(f"  Batch size: {batch_size}")

    # Write PID
    pid_file = BASE_DIR / ".pids" / "syllo_l1_generator.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    while True:
        try:
            queue_count = count_queue_files()

            if queue_count < threshold:
                # Get level (from curriculum or fixed)
                gen_level = get_curriculum_level() if follow_curriculum else level
                logger.info(f"Queue low ({queue_count} < {threshold}), generating L{gen_level}...")
                generate_batch(batch_size, level=gen_level)
            else:
                logger.info(f"Queue OK ({queue_count} files), sleeping...")

            time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="SYLLO L1 Data Generator")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--generate", type=int, metavar="N", help="Generate N examples and exit")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--threshold", type=int, default=2, help="Min queue files before generating")
    parser.add_argument("--batch-size", type=int, default=1000, help="Examples per generation")
    parser.add_argument("--level", type=int, default=0, help="SYLLO level (1-10, or 0 to follow curriculum)")
    
    args = parser.parse_args()
    
    if args.generate:
        result = generate_batch(args.generate, level=args.level)
        print(json.dumps(result, indent=2))
    elif args.daemon:
        run_daemon(args.interval, args.threshold, args.batch_size, args.level)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
