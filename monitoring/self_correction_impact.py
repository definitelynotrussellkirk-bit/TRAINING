#!/usr/bin/env python3
"""
Self-Correction Impact Monitor
==============================

Tracks whether correction training data reduces targeted error rates.

Answers the question: "Are our self-correction efforts actually working?"

How it works:
1. Reads error patterns from logs/error_patterns/*.json
2. Re-tests sample prompts against current deployed model via 3090 API
3. Compares current error rate to historical rate
4. Writes results to status/self_correction_impact.json

GPU Usage: None (uses 3090 API)
ROI: Provides visibility into self-correction effectiveness
"""

import json
import time
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ImpactMonitor - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfCorrectionImpactMonitor:
    """
    Measures impact of self-correction by re-testing error patterns.
    """

    def __init__(
        self,
        api_url: str = "http://192.168.x.x:8765",
        base_dir: str = "/path/to/training",
        samples_per_pattern: int = 10,
        check_interval: int = 3600  # 1 hour default
    ):
        self.api_url = api_url
        self.base_dir = Path(base_dir)
        self.samples_per_pattern = samples_per_pattern
        self.check_interval = check_interval

        # Paths
        self.patterns_dir = self.base_dir / "logs" / "error_patterns"
        self.status_file = self.base_dir / "status" / "self_correction_impact.json"

        # State
        self.status = self._load_status()

    def _load_status(self) -> Dict:
        """Load previous status"""
        if self.status_file.exists():
            try:
                with open(self.status_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Could not load status file, starting fresh")
        return {
            "measurements": [],
            "pattern_history": {},  # pattern_type -> [error_rates over time]
            "last_updated": None
        }

    def _save_status(self):
        """Save status"""
        self.status["last_updated"] = datetime.now().isoformat()
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def load_error_patterns(self) -> Dict[str, List[Dict]]:
        """
        Load all historical error patterns.
        Returns: {error_type: [list of error examples]}
        """
        patterns = defaultdict(list)

        if not self.patterns_dir.exists():
            logger.warning(f"Patterns directory not found: {self.patterns_dir}")
            return patterns

        pattern_files = sorted(self.patterns_dir.glob("patterns_*.json"))
        if not pattern_files:
            logger.warning(f"No pattern files found in {self.patterns_dir}")
            return patterns

        for pattern_file in pattern_files:
            try:
                with open(pattern_file) as f:
                    data = json.load(f)

                for pattern in data.get("patterns", []):
                    error_type = pattern.get("error_type", "Unknown")
                    samples = pattern.get("sample_problems", [])
                    for sample in samples:
                        patterns[error_type].append({
                            "problem": sample,
                            "source_file": pattern_file.name,
                            "frequency": pattern.get("frequency", 0)
                        })
            except Exception as e:
                logger.warning(f"Failed to load {pattern_file}: {e}")

        logger.info(f"Loaded {len(patterns)} error pattern types")
        return dict(patterns)

    def call_api(self, prompt: str) -> Optional[str]:
        """Call 3090 API for inference"""
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "model": "Qwen3-0.6B",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.1
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
            return None
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from model output"""
        patterns = [
            r'Answer:\s*(\w+)',
            r'answer:\s*(\w+)',
            r'Therefore:\s*(\w+)',
            r'Conclusion:\s*(\w+)',
            r'\*\*(\w+)\*\*$',
            r'(?:Valid|Invalid|True|False|Yes|No)$'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.groups():
                    return match.group(1).strip().lower()
                else:
                    return match.group(0).strip().lower()

        # Fallback: last word if it's a known answer type
        words = text.strip().split()
        if words:
            last_word = words[-1].lower().strip('.,!?')
            if last_word in ['valid', 'invalid', 'true', 'false', 'yes', 'no']:
                return last_word
        return None

    def test_pattern_samples(
        self,
        error_type: str,
        samples: List[Dict]
    ) -> Dict:
        """
        Test samples for a specific error pattern.
        Returns error rate and details.
        """
        # Sample subset
        test_samples = random.sample(
            samples,
            min(self.samples_per_pattern, len(samples))
        )

        errors = 0
        tested = 0
        no_answer = 0

        for sample in test_samples:
            prompt = sample.get("problem", "")
            if not prompt:
                continue

            response = self.call_api(prompt)
            if response is None:
                continue

            tested += 1

            # Check if model can extract a clear answer
            model_answer = self.extract_answer(response)
            if model_answer is None:
                errors += 1  # Couldn't extract clear answer
                no_answer += 1

        error_rate = errors / tested if tested > 0 else 0.0

        return {
            "error_type": error_type,
            "tested": tested,
            "errors": errors,
            "no_answer": no_answer,
            "error_rate": error_rate
        }

    def measure_impact(self) -> Optional[Dict]:
        """
        Run full impact measurement across all known error patterns.
        """
        logger.info("Starting impact measurement...")

        # Load patterns
        patterns = self.load_error_patterns()
        if not patterns:
            logger.warning("No error patterns found - need to run self-correction loop first")
            return None

        # Get current checkpoint info
        checkpoint_info = self._get_checkpoint_info()
        logger.info(f"Current checkpoint: step {checkpoint_info.get('step')}")

        # Test each pattern type
        results = []
        for error_type, samples in patterns.items():
            if len(samples) < 3:  # Skip rare patterns
                continue

            logger.info(f"Testing pattern: {error_type} ({len(samples)} samples available)")
            result = self.test_pattern_samples(error_type, samples)
            results.append(result)

            # Update pattern history
            if error_type not in self.status["pattern_history"]:
                self.status["pattern_history"][error_type] = []

            self.status["pattern_history"][error_type].append({
                "timestamp": datetime.now().isoformat(),
                "checkpoint": checkpoint_info.get("step"),
                "error_rate": result["error_rate"],
                "tested": result["tested"]
            })

            # Keep only last 20 measurements per pattern
            if len(self.status["pattern_history"][error_type]) > 20:
                self.status["pattern_history"][error_type] = \
                    self.status["pattern_history"][error_type][-20:]

        # Build measurement record
        measurement = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": checkpoint_info,
            "patterns_tested": len(results),
            "results": results,
            "summary": self._compute_summary(results)
        }

        # Keep only last 50 measurements
        self.status["measurements"].append(measurement)
        if len(self.status["measurements"]) > 50:
            self.status["measurements"] = self.status["measurements"][-50:]

        self._save_status()

        logger.info(f"Impact measurement complete: {len(results)} patterns tested")
        self._log_summary(measurement["summary"])

        return measurement

    def _get_checkpoint_info(self) -> Dict:
        """Get current checkpoint info from training status"""
        status_file = self.base_dir / "status" / "training_status.json"
        if status_file.exists():
            try:
                with open(status_file) as f:
                    data = json.load(f)
                    return {
                        "step": data.get("current_step", 0),
                        "model": data.get("model_name", "unknown")
                    }
            except Exception:
                pass
        return {"step": 0, "model": "unknown"}

    def _compute_summary(self, results: List[Dict]) -> Dict:
        """Compute summary statistics"""
        if not results:
            return {"avg_error_rate": 0, "improving": 0, "regressing": 0, "stable": 0}

        avg_error_rate = sum(r["error_rate"] for r in results) / len(results)

        # Count improving vs regressing patterns
        improving = 0
        regressing = 0

        for result in results:
            error_type = result["error_type"]
            history = self.status["pattern_history"].get(error_type, [])

            if len(history) >= 2:
                prev_rate = history[-2]["error_rate"]
                curr_rate = result["error_rate"]

                if curr_rate < prev_rate - 0.05:  # 5% improvement threshold
                    improving += 1
                elif curr_rate > prev_rate + 0.05:  # 5% regression threshold
                    regressing += 1

        return {
            "avg_error_rate": round(avg_error_rate, 4),
            "improving": improving,
            "regressing": regressing,
            "stable": len(results) - improving - regressing
        }

    def _log_summary(self, summary: Dict):
        """Log summary to console"""
        logger.info("=" * 60)
        logger.info("IMPACT MEASUREMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Average error rate: {summary['avg_error_rate']:.1%}")
        logger.info(f"Improving patterns: {summary['improving']}")
        logger.info(f"Regressing patterns: {summary['regressing']}")
        logger.info(f"Stable patterns: {summary['stable']}")
        logger.info("=" * 60)

    def run_continuous(self):
        """Run continuous monitoring loop"""
        logger.info("=" * 60)
        logger.info("SELF-CORRECTION IMPACT MONITOR - CONTINUOUS MODE")
        logger.info("=" * 60)
        logger.info(f"Check interval: {self.check_interval}s ({self.check_interval/3600:.1f} hours)")
        logger.info(f"Samples per pattern: {self.samples_per_pattern}")
        logger.info("=" * 60)

        while True:
            try:
                self.measure_impact()
                logger.info(f"Next measurement in {self.check_interval}s...")
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

    def print_status(self):
        """Print current status"""
        print("\n" + "=" * 60)
        print("SELF-CORRECTION IMPACT STATUS")
        print("=" * 60)

        print(f"\nMeasurements recorded: {len(self.status['measurements'])}")
        print(f"Pattern types tracked: {len(self.status['pattern_history'])}")

        if self.status['measurements']:
            latest = self.status['measurements'][-1]
            summary = latest.get('summary', {})
            print(f"\nLatest Measurement:")
            print(f"  Timestamp: {latest.get('timestamp', 'N/A')}")
            print(f"  Patterns tested: {latest.get('patterns_tested', 0)}")
            print(f"  Avg error rate: {summary.get('avg_error_rate', 0):.1%}")
            print(f"  Improving: {summary.get('improving', 0)}")
            print(f"  Regressing: {summary.get('regressing', 0)}")
            print(f"  Stable: {summary.get('stable', 0)}")

        # Show top patterns by improvement
        if self.status['pattern_history']:
            print("\nPattern Trends (last 2 measurements):")
            for pattern_type, history in list(self.status['pattern_history'].items())[:5]:
                if len(history) >= 2:
                    prev = history[-2]['error_rate']
                    curr = history[-1]['error_rate']
                    delta = curr - prev
                    trend = "↓" if delta < -0.01 else ("↑" if delta > 0.01 else "→")
                    print(f"  {trend} {pattern_type[:40]}: {prev:.1%} → {curr:.1%}")

        print("=" * 60 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-Correction Impact Monitor - tracks if corrections reduce errors"
    )
    parser.add_argument('--api-url', default='http://192.168.x.x:8765',
                       help='API URL for 3090')
    parser.add_argument('--base-dir', default='/path/to/training',
                       help='Base directory')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Check interval in seconds (default: 3600 = 1 hour)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Samples per pattern (default: 10)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    parser.add_argument('--status', action='store_true',
                       help='Print status and exit')

    args = parser.parse_args()

    monitor = SelfCorrectionImpactMonitor(
        api_url=args.api_url,
        base_dir=args.base_dir,
        samples_per_pattern=args.samples,
        check_interval=args.interval
    )

    if args.status:
        monitor.print_status()
    elif args.once:
        result = monitor.measure_impact()
        if result:
            print(json.dumps(result["summary"], indent=2))
    else:
        monitor.run_continuous()


if __name__ == "__main__":
    main()
