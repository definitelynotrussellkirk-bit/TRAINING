#!/usr/bin/env python3
"""
Self-Correction Impact Monitor
==============================

NOTE: Consider using guild/sparring.py for new self-correction workflows.
This module can still be used to measure impact of ANY correction data,
including the new Sparring system.

Tracks whether correction training data reduces targeted error rates.

Answers the question: "Are our self-correction efforts actually working?"

How it works:
1. Reads error patterns from logs/error_patterns/*.json
2. Re-tests sample prompts against current deployed model via 3090 API
3. Compares current error rate to historical rate
4. Writes results to status/self_correction_impact.json

GPU Usage: None (uses 3090 API)
ROI: Provides visibility into self-correction effectiveness

See also: guild/sparring.py for generating correction training data
"""

import json
import time
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.paths import get_base_dir, get_status_dir, get_remote_api_url
except ImportError:
    def get_base_dir():
        return Path("/path/to/training")
    def get_status_dir():
        return get_base_dir() / "status"
    def get_remote_api_url():
        return "http://192.168.x.x:8765"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ImpactMonitor - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfCorrectionImpactMonitor:
    """
    Measures impact of self-correction by re-testing error patterns.

    Enhanced to track:
    - Individual correction batch effectiveness
    - Correlation between corrections applied and error rate changes
    - "Did this correction help?" score per batch
    """

    def __init__(
        self,
        api_url: str = None,
        base_dir: str = None,
        samples_per_pattern: int = 10,
        check_interval: int = 3600  # 1 hour default
    ):
        self.api_url = api_url or get_remote_api_url()
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.samples_per_pattern = samples_per_pattern
        self.check_interval = check_interval

        # Paths
        self.patterns_dir = self.base_dir / "logs" / "error_patterns"
        self.status_file = self.base_dir / "status" / "self_correction_impact.json"
        self.self_correction_status = self.base_dir / "status" / "self_correction.json"
        self.corrections_dir = self.base_dir / "queue" / "corrections"

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
            "batch_effectiveness": [],  # tracks whether each correction batch helped
            "correction_batches_tracked": [],  # correction runs we've tracked
            "last_updated": None
        }

    def _save_status(self):
        """Save status"""
        self.status["last_updated"] = datetime.now().isoformat()
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def get_correction_runs(self) -> List[Dict]:
        """Load correction runs from self_correction.json"""
        if not self.self_correction_status.exists():
            return []

        try:
            with open(self.self_correction_status) as f:
                data = json.load(f)
            return data.get("correction_runs", [])
        except Exception as e:
            logger.warning(f"Could not load correction runs: {e}")
            return []

    def get_new_correction_batches(self) -> List[Dict]:
        """Find correction batches we haven't tracked yet"""
        correction_runs = self.get_correction_runs()
        tracked_timestamps = set(self.status.get("correction_batches_tracked", []))

        new_batches = []
        for run in correction_runs:
            timestamp = run.get("timestamp")
            if timestamp and timestamp not in tracked_timestamps:
                # Get error patterns from this batch
                patterns = run.get("error_patterns", [])
                new_batches.append({
                    "timestamp": timestamp,
                    "corrections_generated": run.get("corrections_generated", 0),
                    "errors_captured": run.get("errors_captured", 0),
                    "error_types": [p.get("type") for p in patterns if p.get("type")]
                })

        return new_batches

    def correlate_batch_with_improvement(
        self,
        batch: Dict,
        pre_measurement: Optional[Dict],
        post_measurement: Optional[Dict]
    ) -> Dict:
        """
        Correlate a correction batch with observed error rate changes.

        Returns effectiveness assessment:
        - improved: error rates decreased for targeted patterns
        - neutral: no significant change
        - regressed: error rates increased
        """
        if not pre_measurement or not post_measurement:
            return {
                "batch_timestamp": batch.get("timestamp"),
                "effectiveness": "unknown",
                "reason": "insufficient measurements"
            }

        pre_results = {r["error_type"]: r["error_rate"] for r in pre_measurement.get("results", [])}
        post_results = {r["error_type"]: r["error_rate"] for r in post_measurement.get("results", [])}

        # Check error types this batch targeted
        targeted_types = batch.get("error_types", [])

        improvements = 0
        regressions = 0
        total_delta = 0.0
        compared = 0

        for error_type in targeted_types:
            if error_type in pre_results and error_type in post_results:
                pre_rate = pre_results[error_type]
                post_rate = post_results[error_type]
                delta = post_rate - pre_rate

                total_delta += delta
                compared += 1

                if delta < -0.05:  # 5% improvement threshold
                    improvements += 1
                elif delta > 0.05:  # 5% regression threshold
                    regressions += 1

        if compared == 0:
            effectiveness = "unknown"
            reason = "no comparable patterns"
        elif improvements > regressions:
            effectiveness = "improved"
            reason = f"{improvements} patterns improved, {regressions} regressed"
        elif regressions > improvements:
            effectiveness = "regressed"
            reason = f"{improvements} patterns improved, {regressions} regressed"
        else:
            effectiveness = "neutral"
            reason = f"no significant change ({compared} patterns compared)"

        avg_delta = total_delta / compared if compared > 0 else 0

        return {
            "batch_timestamp": batch.get("timestamp"),
            "corrections_generated": batch.get("corrections_generated", 0),
            "error_types_targeted": targeted_types,
            "patterns_compared": compared,
            "improvements": improvements,
            "regressions": regressions,
            "avg_error_delta": round(avg_delta, 4),
            "effectiveness": effectiveness,
            "reason": reason,
            "assessed_at": datetime.now().isoformat()
        }

    def assess_new_batches(self) -> List[Dict]:
        """
        Assess effectiveness of new correction batches.
        Links correction batches to error rate changes.
        """
        new_batches = self.get_new_correction_batches()
        if not new_batches:
            return []

        measurements = self.status.get("measurements", [])
        if len(measurements) < 2:
            logger.info("Need at least 2 measurements to assess batch effectiveness")
            return []

        assessments = []

        for batch in new_batches:
            batch_time = batch.get("timestamp")
            if not batch_time:
                continue

            # Find measurements before and after this batch
            pre_measurement = None
            post_measurement = None

            for m in measurements:
                m_time = m.get("timestamp")
                if m_time:
                    if m_time < batch_time:
                        pre_measurement = m
                    elif m_time >= batch_time and post_measurement is None:
                        post_measurement = m

            assessment = self.correlate_batch_with_improvement(
                batch, pre_measurement, post_measurement
            )
            assessments.append(assessment)

            # Mark as tracked
            if batch_time not in self.status["correction_batches_tracked"]:
                self.status["correction_batches_tracked"].append(batch_time)

            # Store effectiveness
            self.status["batch_effectiveness"].append(assessment)

            # Keep only last 50 assessments
            if len(self.status["batch_effectiveness"]) > 50:
                self.status["batch_effectiveness"] = self.status["batch_effectiveness"][-50:]

            logger.info(f"Batch {batch_time}: {assessment['effectiveness']} "
                       f"({assessment['reason']})")

        self._save_status()
        return assessments

    def get_effectiveness_summary(self) -> Dict:
        """Get overall effectiveness summary of self-correction system"""
        assessments = self.status.get("batch_effectiveness", [])

        if not assessments:
            return {
                "total_batches_assessed": 0,
                "effectiveness_score": None,
                "verdict": "no data"
            }

        improved = sum(1 for a in assessments if a.get("effectiveness") == "improved")
        regressed = sum(1 for a in assessments if a.get("effectiveness") == "regressed")
        neutral = sum(1 for a in assessments if a.get("effectiveness") == "neutral")
        unknown = sum(1 for a in assessments if a.get("effectiveness") == "unknown")

        total = len(assessments)
        known = improved + regressed + neutral

        # Calculate effectiveness score (0-100)
        if known > 0:
            # Score: improved=100, neutral=50, regressed=0
            score = ((improved * 100) + (neutral * 50) + (regressed * 0)) / known
        else:
            score = None

        # Verdict
        if score is None:
            verdict = "insufficient data"
        elif score >= 70:
            verdict = "highly effective"
        elif score >= 50:
            verdict = "moderately effective"
        elif score >= 30:
            verdict = "marginally effective"
        else:
            verdict = "ineffective - review approach"

        return {
            "total_batches_assessed": total,
            "improved": improved,
            "neutral": neutral,
            "regressed": regressed,
            "unknown": unknown,
            "effectiveness_score": round(score, 1) if score else None,
            "verdict": verdict,
            "recent_trend": self._get_recent_trend(assessments[-5:]) if assessments else None
        }

    def _get_recent_trend(self, recent: List[Dict]) -> str:
        """Get trend from recent assessments"""
        if not recent:
            return "no data"

        improved = sum(1 for a in recent if a.get("effectiveness") == "improved")
        regressed = sum(1 for a in recent if a.get("effectiveness") == "regressed")

        if improved > regressed:
            return "improving"
        elif regressed > improved:
            return "declining"
        else:
            return "stable"

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

        # Also assess any new correction batches
        new_assessments = self.assess_new_batches()
        if new_assessments:
            logger.info(f"Assessed {len(new_assessments)} new correction batches")
            effectiveness = self.get_effectiveness_summary()
            logger.info(f"Overall effectiveness: {effectiveness['verdict']} "
                       f"(score: {effectiveness['effectiveness_score']})")

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
                    trend = "\u2193" if delta < -0.01 else ("\u2191" if delta > 0.01 else "\u2192")
                    print(f"  {trend} {pattern_type[:40]}: {prev:.1%} \u2192 {curr:.1%}")

        # Show batch effectiveness summary (NEW)
        effectiveness = self.get_effectiveness_summary()
        print("\n" + "-" * 60)
        print("CORRECTION BATCH EFFECTIVENESS:")
        print("-" * 60)
        print(f"  Batches assessed: {effectiveness['total_batches_assessed']}")

        if effectiveness['total_batches_assessed'] > 0:
            print(f"  Improved:  {effectiveness['improved']}")
            print(f"  Neutral:   {effectiveness['neutral']}")
            print(f"  Regressed: {effectiveness['regressed']}")
            print(f"  Unknown:   {effectiveness['unknown']}")
            if effectiveness['effectiveness_score'] is not None:
                print(f"\n  Effectiveness Score: {effectiveness['effectiveness_score']}/100")
            print(f"  Verdict: {effectiveness['verdict'].upper()}")
            if effectiveness.get('recent_trend'):
                print(f"  Recent Trend: {effectiveness['recent_trend']}")
        else:
            print("  No batches assessed yet - need more measurements")

        print("=" * 60 + "\n")


def main():
    """Main entry point"""
    import argparse

    # Get defaults from paths module
    default_api = get_remote_api_url()
    default_base = str(get_base_dir())

    parser = argparse.ArgumentParser(
        description="Self-Correction Impact Monitor - tracks if corrections reduce errors"
    )
    parser.add_argument('--api-url', default=None,
                       help=f'API URL for 3090 (default: {default_api})')
    parser.add_argument('--base-dir', default=None,
                       help=f'Base directory (default: auto-detected)')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Check interval in seconds (default: 3600 = 1 hour)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Samples per pattern (default: 10)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    parser.add_argument('--status', action='store_true',
                       help='Print status and exit')
    parser.add_argument('--assess-batches', action='store_true',
                       help='Assess new correction batches and show effectiveness')

    args = parser.parse_args()

    monitor = SelfCorrectionImpactMonitor(
        api_url=args.api_url,  # Will use get_remote_api_url() if None
        base_dir=args.base_dir,  # Will use get_base_dir() if None
        samples_per_pattern=args.samples,
        check_interval=args.interval
    )

    if args.status:
        monitor.print_status()
    elif args.assess_batches:
        # Just assess batches without running full measurement
        assessments = monitor.assess_new_batches()
        effectiveness = monitor.get_effectiveness_summary()
        print(json.dumps({
            "new_assessments": len(assessments),
            "effectiveness": effectiveness
        }, indent=2))
    elif args.once:
        result = monitor.measure_impact()
        if result:
            summary = result["summary"]
            effectiveness = monitor.get_effectiveness_summary()
            print(json.dumps({
                "measurement_summary": summary,
                "effectiveness": effectiveness
            }, indent=2))
    else:
        monitor.run_continuous()


if __name__ == "__main__":
    main()
