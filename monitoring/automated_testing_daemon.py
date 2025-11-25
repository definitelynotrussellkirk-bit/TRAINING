#!/usr/bin/env python3
"""
Automated Testing Daemon - Continuously runs fixed test suite against checkpoints
Provides real-time quality metrics and regression detection
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomatedTestingDaemon:
    """
    Continuously tests checkpoints against fixed validation suite
    """

    def __init__(
        self,
        base_dir: str = "/path/to/training",
        api_url: str = "http://192.168.x.x:8765",
        interval: int = 600,  # 10 minutes
        validation_file: str = None
    ):
        self.base_dir = Path(base_dir)
        self.api_url = api_url
        self.interval = interval

        # Load validation data
        if validation_file is None:
            validation_file = self.base_dir / "data/validation"

        self.validation_data = self.load_validation_suite(Path(validation_file))

        # State
        self.test_history = []
        self.checkpoint_scores = defaultdict(dict)

        # Output files
        self.status_file = self.base_dir / "status/automated_testing.json"
        self.history_file = self.base_dir / "status/test_history.jsonl"

        # Ensure output dirs exist
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def load_validation_suite(self, validation_dir: Path) -> List[Dict]:
        """Load fixed validation dataset"""
        validation_data = []

        if validation_dir.is_file():
            # Single file
            with open(validation_dir) as f:
                for line in f:
                    if line.strip():
                        validation_data.append(json.loads(line))
        elif validation_dir.is_dir():
            # Directory with easy/medium/hard splits
            for difficulty_file in ['easy.jsonl', 'medium.jsonl', 'hard.jsonl']:
                filepath = validation_dir / difficulty_file
                if filepath.exists():
                    with open(filepath) as f:
                        for line in f:
                            if line.strip():
                                example = json.loads(line)
                                example['difficulty'] = difficulty_file.replace('.jsonl', '')
                                validation_data.append(example)

        logger.info(f"Loaded {len(validation_data)} validation examples")

        return validation_data

    def get_current_checkpoint_path(self) -> Optional[Path]:
        """Get path to current checkpoint"""
        # Check training status for current checkpoint
        status_file = self.base_dir / "status/training_status.json"

        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)

            current_step = status.get('current_step', 0)

            # Find checkpoint directory
            checkpoints_dir = self.base_dir / "models/checkpoints"
            if checkpoints_dir.exists():
                # Look for checkpoint matching current step
                checkpoint_dirs = sorted(checkpoints_dir.glob("checkpoint-*"))
                if checkpoint_dirs:
                    # Get latest checkpoint
                    return checkpoint_dirs[-1]

        # Fallback to current_model
        current_model = self.base_dir / "models/current_model"
        if current_model.exists():
            return current_model

        return None

    def call_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.1
    ) -> Dict:
        """Call 3090 API for inference"""
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "model": "Qwen3-0.6B",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    text = result['choices'][0]['message']['content']
                    return {"text": text, "response": text}
                return result
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from model output"""
        import re

        patterns = [
            r'Answer:\s*(\w+)',
            r'answer:\s*(\w+)',
            r'Therefore:\s*(\w+)',
            r'Conclusion:\s*(\w+)',
            r'(?:Valid|Invalid|True|False|Yes|No)$'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.groups():
                    return match.group(1).strip().lower()
                else:
                    return match.group(0).strip().lower()

        # Fallback: last word
        words = text.strip().split()
        if words:
            last_word = words[-1].lower().strip('.,!?')
            if last_word in ['valid', 'invalid', 'true', 'false', 'yes', 'no']:
                return last_word

        return None

    def test_example(self, example: Dict) -> Dict:
        """Test a single example"""
        # Build prompt
        prompt = example.get('input', example.get('problem', ''))

        # Call API
        start_time = time.time()
        result = self.call_api(prompt, max_tokens=300, temperature=0.1)
        latency_ms = (time.time() - start_time) * 1000

        if 'error' in result:
            return {
                'success': False,
                'error': result['error'],
                'latency_ms': latency_ms
            }

        # Extract answer
        response_text = result.get('text', result.get('response', ''))
        model_answer = self.extract_answer(response_text)

        # Get ground truth
        correct_answer = example.get('output', example.get('answer', ''))
        if isinstance(correct_answer, str):
            truth = self.extract_answer(correct_answer)
            if truth:
                correct_answer = truth

        # Compare
        model_answer_norm = str(model_answer).lower().strip() if model_answer else ''
        correct_answer_norm = str(correct_answer).lower().strip()

        is_correct = model_answer_norm == correct_answer_norm

        return {
            'success': True,
            'correct': is_correct,
            'model_answer': model_answer,
            'model_response': response_text,
            'correct_answer': correct_answer,
            'input': prompt,
            'latency_ms': latency_ms,
            'difficulty': example.get('difficulty', 'unknown')
        }

    def run_test_suite(self, checkpoint_name: str) -> Dict[str, Any]:
        """Run full test suite on current checkpoint"""
        logger.info(f"Running test suite on checkpoint: {checkpoint_name}")

        results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': checkpoint_name,
            'total': 0,
            'correct': 0,
            'failed': 0,
            'errors': 0,
            'accuracy': 0.0,
            'by_difficulty': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'avg_latency_ms': 0,
            'examples': []
        }

        latencies = []

        for i, example in enumerate(self.validation_data):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(self.validation_data)}")

            test_result = self.test_example(example)
            results['total'] += 1

            if not test_result['success']:
                results['errors'] += 1
                logger.warning(f"  Error: {test_result.get('error')}")
                continue

            difficulty = test_result.get('difficulty', 'unknown')
            results['by_difficulty'][difficulty]['total'] += 1

            if test_result['correct']:
                results['correct'] += 1
                results['by_difficulty'][difficulty]['correct'] += 1
            else:
                results['failed'] += 1
                # Store failed examples for analysis
                results['examples'].append({
                    'input': test_result['input'][:200],
                    'expected': test_result['correct_answer'],
                    'got': test_result['model_answer'],
                    'difficulty': difficulty
                })

            latencies.append(test_result['latency_ms'])

        # Calculate metrics
        if results['total'] > 0:
            results['accuracy'] = results['correct'] / results['total']

        if latencies:
            results['avg_latency_ms'] = sum(latencies) / len(latencies)

        # Calculate per-difficulty accuracy
        for difficulty, stats in results['by_difficulty'].items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']

        logger.info(f"  Results: {results['correct']}/{results['total']} correct ({results['accuracy']:.1%})")

        return results

    def detect_regression(self, current_results: Dict, previous_results: Optional[Dict]) -> Optional[Dict]:
        """Detect if there's a regression compared to previous test"""
        if previous_results is None:
            return None

        current_acc = current_results['accuracy']
        previous_acc = previous_results['accuracy']

        accuracy_drop = previous_acc - current_acc

        if accuracy_drop > 0.05:  # 5% drop
            return {
                'regression_detected': True,
                'severity': 'critical' if accuracy_drop > 0.15 else 'warning',
                'accuracy_drop': accuracy_drop,
                'previous_accuracy': previous_acc,
                'current_accuracy': current_acc,
                'checkpoint': current_results['checkpoint'],
                'previous_checkpoint': previous_results['checkpoint']
            }

        return None

    def save_results(self, results: Dict):
        """Save test results"""
        # Save to history
        with open(self.history_file, 'a') as f:
            f.write(json.dumps(results) + '\n')

        # Save latest to status file
        status = {
            'last_test': results['timestamp'],
            'checkpoint': results['checkpoint'],
            'accuracy': results['accuracy'],
            'total_examples': results['total'],
            'correct': results['correct'],
            'failed': results['failed'],
            'by_difficulty': dict(results['by_difficulty']),
            'avg_latency_ms': results['avg_latency_ms']
        }

        # Add regression info if detected
        if len(self.test_history) > 0:
            regression = self.detect_regression(results, self.test_history[-1])
            if regression:
                status['regression'] = regression

        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)

        # Add to history
        self.test_history.append(results)

        # Keep only last 100 results in memory
        if len(self.test_history) > 100:
            self.test_history = self.test_history[-100:]

    def run_continuous(self):
        """Run continuous testing loop"""
        logger.info("=" * 80)
        logger.info("🧪 AUTOMATED TESTING DAEMON - STARTING")
        logger.info("=" * 80)
        logger.info(f"Validation examples: {len(self.validation_data)}")
        logger.info(f"Check interval: {self.interval}s")
        logger.info(f"API URL: {self.api_url}")
        logger.info("=" * 80)

        last_checkpoint = None

        while True:
            try:
                # Get current checkpoint
                checkpoint_path = self.get_current_checkpoint_path()

                if checkpoint_path:
                    checkpoint_name = checkpoint_path.name

                    # Only test if checkpoint changed
                    if checkpoint_name != last_checkpoint:
                        logger.info(f"\n📊 New checkpoint detected: {checkpoint_name}")

                        # Run test suite
                        results = self.run_test_suite(checkpoint_name)

                        # Save results
                        self.save_results(results)

                        # Check for regression
                        if len(self.test_history) > 1:
                            regression = self.detect_regression(results, self.test_history[-2])
                            if regression:
                                logger.warning("⚠️  REGRESSION DETECTED!")
                                logger.warning(f"   Accuracy dropped by {regression['accuracy_drop']:.1%}")
                                logger.warning(f"   {regression['previous_accuracy']:.1%} → {regression['current_accuracy']:.1%}")

                        last_checkpoint = checkpoint_name
                    else:
                        logger.info(f"No new checkpoint (current: {checkpoint_name})")
                else:
                    logger.warning("No checkpoint found")

                logger.info(f"\nNext check in {self.interval}s...")
                time.sleep(self.interval)

            except KeyboardInterrupt:
                logger.info("\n🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in testing loop: {e}")
                time.sleep(self.interval)


def run_with_scheduler(scheduler_url: str, interval: int, base_dir: str, validation_file: str = None):
    """Run as a scheduler client - submit tasks instead of executing directly."""
    import sys
    sys.path.insert(0, base_dir)
    from monitoring.task_client import TaskClient

    client = TaskClient(scheduler_url)

    logger.info("="*60)
    logger.info("Automated Testing Daemon - Scheduler Mode")
    logger.info("="*60)
    logger.info(f"Scheduler: {scheduler_url}")
    logger.info(f"Interval: {interval}s")

    if not client.is_healthy():
        logger.error("Scheduler not available!")
        return

    while True:
        try:
            logger.info("\nSubmitting automated_testing task to scheduler...")

            params = {"base_dir": base_dir}
            if validation_file:
                params["validation_file"] = validation_file

            result = client.submit_and_wait(
                task_type="automated_test",
                params=params,
                priority=2,  # NORMAL
                timeout=900.0  # 15 min timeout (testing takes longer)
            )

            if result:
                logger.info(f"Task completed: {result.get('status')}")
                if result.get('result'):
                    r = result['result']
                    logger.info(f"  Total: {r.get('total', 0)}")
                    logger.info(f"  Correct: {r.get('correct', 0)}")
                    logger.info(f"  Accuracy: {r.get('accuracy', 0)*100:.1f}%")
            else:
                logger.warning("Task failed or timed out")

            logger.info(f"\n💤 Next submission in {interval}s...")
            time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\nStopping...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(interval)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Testing Daemon")
    parser.add_argument('--base-dir', default='/path/to/training',
                       help='Base directory')
    parser.add_argument('--api-url', default='http://192.168.x.x:8765',
                       help='API URL for 3090')
    parser.add_argument('--interval', type=int, default=600,
                       help='Check interval (seconds)')
    parser.add_argument('--validation-file', type=str,
                       help='Path to validation dataset')
    parser.add_argument('--use-scheduler', action='store_true',
                       help='Submit tasks to GPU Task Scheduler instead of running directly')
    parser.add_argument('--scheduler-url', default='http://192.168.x.x:8766',
                       help='GPU Task Scheduler URL')

    args = parser.parse_args()

    # Scheduler mode
    if args.use_scheduler:
        run_with_scheduler(args.scheduler_url, args.interval, args.base_dir, args.validation_file)
        return

    # Create daemon
    daemon = AutomatedTestingDaemon(
        base_dir=args.base_dir,
        api_url=args.api_url,
        interval=args.interval,
        validation_file=args.validation_file
    )

    # Run continuously
    daemon.run_continuous()


if __name__ == "__main__":
    main()
