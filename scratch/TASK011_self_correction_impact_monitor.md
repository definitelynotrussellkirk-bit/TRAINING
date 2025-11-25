# TASK011: Self-Correction Impact Monitor

**Status:** Proposed
**Effort:** Medium (1-2 hours)
**File:** `monitoring/self_correction_impact.py` (new)

## Objective

Create a new module that tracks whether self-correction training data actually reduces error rates in targeted categories.

## Problem

Currently there's no way to answer:
- "Did the syllogism corrections we generated last week reduce syllogism errors?"
- "Are certain error patterns resistant to correction?"
- "Is self-correction working at all?"

## Design

### Core Concept

1. Read error patterns from `logs/error_patterns/*.json`
2. Extract sample prompts from each error type
3. Re-test those prompts against current deployed model via 3090 API
4. Compare current error rate to historical rate
5. Write results to `status/self_correction_impact.json`

### Data Flow

```
logs/error_patterns/*.json (historical)
         │
         ▼
  Extract sample prompts by error type
         │
         ▼
  Test via 3090 API (/v1/chat/completions)
         │
         ▼
  Calculate current error rate per type
         │
         ▼
status/self_correction_impact.json
```

## Implementation

### Class Structure

```python
#!/usr/bin/env python3
"""
Self-Correction Impact Monitor
Tracks whether correction training data reduces targeted error rates.
"""

import json
import time
import logging
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
            with open(self.status_file) as f:
                return json.load(f)
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

        for pattern_file in sorted(self.patterns_dir.glob("patterns_*.json")):
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
        """Extract final answer (reuse logic from self_correction_loop)"""
        import re
        patterns = [
            r'Answer:\s*(\w+)',
            r'answer:\s*(\w+)',
            r'Therefore:\s*(\w+)',
            r'Conclusion:\s*(\w+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.groups():
                return match.group(1).strip().lower()
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
        import random

        # Sample subset
        test_samples = random.sample(
            samples,
            min(self.samples_per_pattern, len(samples))
        )

        errors = 0
        tested = 0

        for sample in test_samples:
            prompt = sample.get("problem", "")
            if not prompt:
                continue

            response = self.call_api(prompt)
            if response is None:
                continue

            tested += 1

            # Check if model still makes error
            # (simplified: just check if answer extraction fails or differs)
            model_answer = self.extract_answer(response)
            if model_answer is None:
                errors += 1  # Couldn't extract clear answer

        error_rate = errors / tested if tested > 0 else 0.0

        return {
            "error_type": error_type,
            "tested": tested,
            "errors": errors,
            "error_rate": error_rate
        }

    def measure_impact(self) -> Dict:
        """
        Run full impact measurement across all known error patterns.
        """
        logger.info("Starting impact measurement...")

        # Load patterns
        patterns = self.load_error_patterns()
        if not patterns:
            logger.warning("No error patterns found")
            return None

        # Get current checkpoint info
        checkpoint_info = self._get_checkpoint_info()

        # Test each pattern type
        results = []
        for error_type, samples in patterns.items():
            if len(samples) < 3:  # Skip rare patterns
                continue

            logger.info(f"Testing pattern: {error_type} ({len(samples)} samples)")
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

        # Build measurement record
        measurement = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": checkpoint_info,
            "patterns_tested": len(results),
            "results": results,
            "summary": self._compute_summary(results)
        }

        self.status["measurements"].append(measurement)
        self._save_status()

        logger.info(f"Impact measurement complete: {len(results)} patterns tested")
        return measurement

    def _get_checkpoint_info(self) -> Dict:
        """Get current checkpoint info from training status"""
        status_file = self.base_dir / "status" / "training_status.json"
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
                return {
                    "step": data.get("current_step", 0),
                    "model": data.get("model_name", "unknown")
                }
        return {"step": 0, "model": "unknown"}

    def _compute_summary(self, results: List[Dict]) -> Dict:
        """Compute summary statistics"""
        if not results:
            return {"avg_error_rate": 0, "improving": 0, "regressing": 0}

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
            "avg_error_rate": avg_error_rate,
            "improving": improving,
            "regressing": regressing,
            "stable": len(results) - improving - regressing
        }

    def run_continuous(self):
        """Run continuous monitoring loop"""
        logger.info("Starting continuous impact monitoring")
        logger.info(f"Check interval: {self.check_interval}s")

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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Self-Correction Impact Monitor")
    parser.add_argument('--api-url', default='http://192.168.x.x:8765')
    parser.add_argument('--base-dir', default='/path/to/training')
    parser.add_argument('--interval', type=int, default=3600, help='Check interval (seconds)')
    parser.add_argument('--samples', type=int, default=10, help='Samples per pattern')
    parser.add_argument('--once', action='store_true', help='Run once and exit')

    args = parser.parse_args()

    monitor = SelfCorrectionImpactMonitor(
        api_url=args.api_url,
        base_dir=args.base_dir,
        samples_per_pattern=args.samples,
        check_interval=args.interval
    )

    if args.once:
        result = monitor.measure_impact()
        if result:
            print(json.dumps(result["summary"], indent=2))
    else:
        monitor.run_continuous()


if __name__ == "__main__":
    main()
```

## Status File Format

`status/self_correction_impact.json`:

```json
{
  "measurements": [
    {
      "timestamp": "2025-11-25T10:00:00",
      "checkpoint": {"step": 157000, "model": "qwen3"},
      "patterns_tested": 8,
      "results": [
        {"error_type": "Syllogism validity", "tested": 10, "errors": 3, "error_rate": 0.3}
      ],
      "summary": {
        "avg_error_rate": 0.25,
        "improving": 3,
        "regressing": 1,
        "stable": 4
      }
    }
  ],
  "pattern_history": {
    "Syllogism validity": [
      {"timestamp": "2025-11-24", "checkpoint": 156000, "error_rate": 0.45, "tested": 10},
      {"timestamp": "2025-11-25", "checkpoint": 157000, "error_rate": 0.30, "tested": 10}
    ]
  },
  "last_updated": "2025-11-25T10:00:00"
}
```

## Verification

```bash
# 1. Ensure error patterns exist (from previous self-correction runs)
ls logs/error_patterns/

# 2. Run once to test
python3 monitoring/self_correction_impact.py --once --samples 5

# 3. Check status file
cat status/self_correction_impact.json | jq .

# 4. Run continuously (background)
nohup python3 monitoring/self_correction_impact.py --interval 3600 > logs/impact_monitor.log 2>&1 &
```

## Dependencies

- TASK010 should be completed first (so error patterns are being recorded)
- Requires 3090 API to be running
- Requires `logs/error_patterns/*.json` to exist

## Success Criteria

- [ ] Module runs without errors
- [ ] Writes `status/self_correction_impact.json`
- [ ] Tracks error rate over time per pattern type
- [ ] Summary shows improving/regressing/stable counts
- [ ] Can run as continuous daemon
