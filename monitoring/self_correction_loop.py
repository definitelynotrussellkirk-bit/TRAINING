#!/usr/bin/env python3
"""
DEPRECATED - Use guild/sparring.py instead

This module is superseded by the new Sparring system (2025-11-27).
See: guild/sparring.py - "Sparring with the Trainers"

The new system:
- Cleaner 3-example-per-mistake format
- Always HIGH priority queue (data becomes stale)
- Dedicated validator: guild/sparring_validator.py
- Task registry for scheduling: guild/task_registry.py

Usage of new system:
    python3 guild/sparring.py --skill binary --count 100

---
ORIGINAL: Self-Correction Loop - Learn from model mistakes
Creates high-value training data from errors via automated correction
"""

import warnings
warnings.warn(
    "self_correction_loop.py is DEPRECATED. Use guild/sparring.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import os
import requests
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import logging
import re

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir, get_remote_api_url

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfCorrectionLoop:
    """
    Automated system that:
    1. Tests examples with checkpoint
    2. Captures errors
    3. Analyzes what went wrong
    4. Generates correction training examples
    5. Mines error patterns
    """

    def __init__(
        self,
        api_url: str = None,
        base_dir: str = None,
        batch_size: int = 100,
        error_threshold: int = 50,  # Min errors before analysis
        model: str = None,  # Model ID to use (None = auto-detect)
    ):
        self.api_url = api_url or get_remote_api_url()
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.batch_size = batch_size
        self.error_threshold = error_threshold

        # DYNAMIC: Model ID comes from --model arg or auto-detected from 3090
        # Examples: "checkpoint-177000", "Qwen3-0.6B-base"
        self.model_id = model  # Will auto-detect in call_api if None
        self._model_was_specified = model is not None

        # State tracking
        self.error_cache = []
        self.pattern_db = defaultdict(list)
        self.stats = {
            'tested': 0,
            'correct': 0,
            'incorrect': 0,
            'corrections_generated': 0,
            'patterns_found': 0
        }

        # Directories
        self.setup_directories()

        # Status tracking (for plugin compatibility)
        self.status_file = self.base_dir / "status" / "self_correction.json"
        self.status = self._load_status()

    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'queue/unvalidated',
            'queue/normal',
            'queue/corrections',
            'queue/rejected',
            'logs/error_patterns',
            'logs/corrections',
            'status'
        ]
        for d in dirs:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)

    def _load_status(self) -> Dict:
        """Load previous status if exists"""
        if self.status_file.exists():
            try:
                with open(self.status_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Could not load status file, starting fresh")
        return {
            "correction_runs": [],
            "total_errors_captured": 0,
            "total_corrections_generated": 0,
            "last_updated": None
        }

    def _save_status(self):
        """Save status to JSON"""
        self.status["last_updated"] = datetime.now().isoformat()
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def update_status(self, patterns: List[Dict] = None):
        """
        Update status file after pipeline run.
        Format matches what SelfCorrectionPlugin expects.
        """
        # Build error_patterns list for this run
        error_patterns = []
        if patterns:
            for p in patterns[:10]:  # Top 10 patterns
                error_patterns.append({
                    "type": p.get("error_type", "Unknown"),
                    "count": p.get("frequency", 0),
                    "description": p.get("sample_problems", [""])[0][:100] if p.get("sample_problems") else ""
                })

        # Create run record
        # Determine which model was used (may have been auto-detected during run)
        model_used = self.model_id or self.get_active_model() or "unknown"
        run_record = {
            "timestamp": datetime.now().isoformat(),
            # DYNAMIC: Model ID used for testing (from --model arg or auto-detected)
            "model": model_used,
            "model_specified": self._model_was_specified,
            "errors_captured": self.stats["incorrect"],
            "corrections_generated": self.stats["corrections_generated"],
            "error_patterns": error_patterns,
            "tested": self.stats["tested"],
            "correct": self.stats["correct"],
            "incorrect": self.stats["incorrect"]
        }

        # Update status
        self.status["correction_runs"].append(run_record)
        self.status["total_errors_captured"] += self.stats["incorrect"]
        self.status["total_corrections_generated"] += self.stats["corrections_generated"]

        self._save_status()
        logger.info(f"Status updated: {self.status_file}")

    def get_active_model(self) -> Optional[str]:
        """Get first available model from the pool"""
        try:
            response = requests.get(
                f"{self.api_url}/models/pool",
                headers={"X-API-Key": os.environ.get("INFERENCE_ADMIN_KEY", "admin123")},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("models"):
                    return data["models"][0]["model_id"]
            return None
        except Exception as e:
            logger.warning(f"Failed to get active model: {e}")
            return None

    def call_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
        model_id: Optional[str] = None
    ) -> Dict:
        """Call 3090 API for inference"""
        # Use instance model_id, then parameter, then auto-detect
        # DYNAMIC: Model ID like "checkpoint-177000" or "Qwen3-0.6B-base"
        if model_id is None:
            model_id = self.model_id or self.get_active_model()
            if model_id is None:
                return {"error": "No model loaded in pool"}

        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={"X-API-Key": os.environ.get("INFERENCE_ADMIN_KEY", "admin123")},
                json={
                    "model": model_id,
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
                # Extract text from OpenAI format
                if 'choices' in result and len(result['choices']) > 0:
                    text = result['choices'][0]['message']['content']
                    return {"text": text, "response": text}
                return result
            else:
                logger.error(f"API error: {response.status_code}")
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {"error": str(e)}

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from model output"""
        # Look for "Answer: X" pattern
        patterns = [
            r'Answer:\s*(\w+)',
            r'answer:\s*(\w+)',
            r'Therefore:\s*(\w+)',
            r'Conclusion:\s*(\w+)',
            r'\*\*(\w+)\*\*$',  # Bold answer at end
            r'(?:Valid|Invalid|True|False|Yes|No)$'  # Standalone at end
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

    def test_example(self, example: Dict) -> Dict:
        """Test a single example with checkpoint"""
        # Build prompt
        prompt = example.get('input', example.get('problem', ''))

        # Call API
        result = self.call_api(prompt, max_tokens=300, temperature=0.1)

        if 'error' in result:
            return {
                'success': False,
                'error': result['error']
            }

        # Extract answer
        response_text = result.get('text', result.get('response', ''))
        model_answer = self.extract_answer(response_text)

        # Get ground truth
        correct_answer = example.get('output', example.get('answer', ''))
        if isinstance(correct_answer, str):
            # Extract answer from output if it's full reasoning
            truth = self.extract_answer(correct_answer)
            if truth:
                correct_answer = truth

        # Normalize for comparison
        model_answer_norm = str(model_answer).lower().strip() if model_answer else ''
        correct_answer_norm = str(correct_answer).lower().strip()

        is_correct = model_answer_norm == correct_answer_norm

        return {
            'success': True,
            'correct': is_correct,
            'model_answer': model_answer,
            'model_response': response_text,
            'correct_answer': correct_answer,
            'input': prompt
        }

    def validate_and_capture_errors(self, examples: List[Dict]) -> Dict:
        """Stage 2: Test examples and capture errors"""
        logger.info(f"Testing {len(examples)} examples...")

        results = {
            'correct': [],
            'errors': []
        }

        for i, example in enumerate(examples):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(examples)}")

            test_result = self.test_example(example)
            self.stats['tested'] += 1

            if not test_result['success']:
                logger.warning(f"  Test failed: {test_result.get('error')}")
                continue

            if test_result['correct']:
                # Correct answer - move to normal queue
                self.stats['correct'] += 1
                results['correct'].append(example)
            else:
                # Wrong answer - capture for correction
                self.stats['incorrect'] += 1
                error = {
                    'problem': test_result['input'],
                    'correct_answer': test_result['correct_answer'],
                    'model_answer': test_result['model_answer'],
                    'model_response': test_result['model_response'],
                    'timestamp': datetime.now().isoformat(),
                    'original_example': example
                }
                results['errors'].append(error)
                self.error_cache.append(error)

        accuracy = self.stats['correct'] / max(self.stats['tested'], 1) * 100
        logger.info(f"  Results: {self.stats['correct']}/{self.stats['tested']} correct ({accuracy:.1f}%)")
        logger.info(f"  Errors captured: {len(results['errors'])}")

        return results

    def analyze_error(self, error: Dict) -> str:
        """Stage 3: Identify what went wrong"""
        analysis_prompt = f"""Analyze this logical reasoning error:

Problem: {error['problem']}
Correct answer: {error['correct_answer']}
Model answered: {error['model_answer']}

What logical error did the model make? Respond in one short sentence.

Error type:"""

        result = self.call_api(analysis_prompt, max_tokens=100, temperature=0.3)

        if 'error' in result:
            return "Unknown error"

        error_type = result.get('text', result.get('response', 'Unknown error')).strip()

        # Clean up response
        error_type = error_type.split('\n')[0]  # First line only
        error_type = error_type[:200]  # Max length

        return error_type

    def generate_correction(self, error: Dict, error_type: str) -> Dict:
        """Stage 4: Create correction training example"""
        correction_prompt = f"""Create a training example that teaches correct reasoning.

Problem: {error['problem']}
Common error: {error_type}
Correct answer: {error['correct_answer']}

Generate output showing:
1. Step-by-step correct reasoning
2. Warning about the common error
3. Final answer

Output:"""

        result = self.call_api(correction_prompt, max_tokens=400, temperature=0.7)

        if 'error' in result:
            logger.error(f"Correction generation failed: {result['error']}")
            return None

        correction = result.get('text', result.get('response', ''))

        # Create training example
        training_example = {
            "input": error['problem'],
            "output": correction.strip(),
            "metadata": {
                "source": "self_correction",
                "error_type": error_type,
                "original_wrong_answer": error['model_answer'],
                "correct_answer": error['correct_answer'],
                "generated_at": datetime.now().isoformat()
            }
        }

        return training_example

    def process_error_batch(self):
        """Analyze and correct a batch of errors"""
        if len(self.error_cache) < self.error_threshold:
            return

        logger.info(f"Processing {len(self.error_cache)} errors...")

        corrections = []

        for i, error in enumerate(self.error_cache):
            if i % 10 == 0:
                logger.info(f"  Analyzing error {i}/{len(self.error_cache)}")

            # Analyze error
            error_type = self.analyze_error(error)

            # Track pattern
            self.pattern_db[error_type].append(error)

            # Generate correction
            correction = self.generate_correction(error, error_type)

            if correction:
                corrections.append(correction)
                self.stats['corrections_generated'] += 1

        # Save corrections
        if corrections:
            self.save_corrections(corrections)
            logger.info(f"  ✅ Generated {len(corrections)} correction examples")

        # Mine patterns
        patterns = self.mine_patterns()
        if patterns:
            self.report_patterns(patterns)

        # Clear cache
        self.error_cache = []

    def save_corrections(self, corrections: List[Dict]):
        """Save correction examples to queue"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.base_dir / f"queue/corrections/corrections_{timestamp}.jsonl"

        with output_file.open('w') as f:
            for correction in corrections:
                f.write(json.dumps(correction) + '\n')

        logger.info(f"  Saved corrections to: {output_file}")

    def mine_patterns(self) -> List[Dict]:
        """Stage 5: Find systematic error patterns"""
        patterns = []

        total = sum(len(examples) for examples in self.pattern_db.values())

        for error_type, examples in self.pattern_db.items():
            if len(examples) >= 3:  # Significant pattern
                patterns.append({
                    'error_type': error_type,
                    'frequency': len(examples),
                    'percentage': len(examples) / max(total, 1) * 100,
                    'sample_problems': [e['problem'][:100] for e in examples[:3]]
                })

        # Sort by frequency
        patterns.sort(key=lambda x: x['frequency'], reverse=True)

        self.stats['patterns_found'] = len(patterns)

        return patterns

    def report_patterns(self, patterns: List[Dict]):
        """Generate and save pattern report"""
        logger.info("\n" + "=" * 80)
        logger.info("🔍 ERROR PATTERN ANALYSIS")
        logger.info("=" * 80)

        for i, pattern in enumerate(patterns[:10], 1):
            logger.info(f"{i}. {pattern['error_type']}")
            logger.info(f"   Frequency: {pattern['frequency']} ({pattern['percentage']:.1f}%)")

        logger.info("=" * 80)

        # Save detailed report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.base_dir / f"logs/error_patterns/patterns_{timestamp}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'patterns': patterns
        }

        with report_file.open('w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Pattern report saved: {report_file}")

    def load_examples_from_file(self, filepath: Path) -> List[Dict]:
        """Load examples from JSONL file"""
        examples = []

        with filepath.open('r') as f:
            for line in f:
                if line.strip():
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")

        return examples

    def run_validation_pipeline(self, input_file: Path):
        """Run complete validation pipeline on a file"""
        logger.info("=" * 80)
        logger.info(f"🔄 SELF-CORRECTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Input: {input_file}")
        logger.info("")

        # Load examples
        examples = self.load_examples_from_file(input_file)
        logger.info(f"Loaded {len(examples)} examples")

        # Validate and capture errors
        results = self.validate_and_capture_errors(examples)

        # Save correct examples to normal queue
        if results['correct']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.base_dir / f"queue/normal/validated_{timestamp}.jsonl"

            with output_file.open('w') as f:
                for example in results['correct']:
                    f.write(json.dumps(example) + '\n')

            logger.info(f"✅ Saved {len(results['correct'])} correct examples to: {output_file}")

        # Process errors if threshold reached
        if len(self.error_cache) >= self.error_threshold:
            self.process_error_batch()

        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 SESSION STATS")
        logger.info("=" * 80)
        logger.info(f"Total tested: {self.stats['tested']}")
        logger.info(f"Correct: {self.stats['correct']} ({self.stats['correct']/max(self.stats['tested'],1)*100:.1f}%)")
        logger.info(f"Incorrect: {self.stats['incorrect']} ({self.stats['incorrect']/max(self.stats['tested'],1)*100:.1f}%)")
        logger.info(f"Corrections generated: {self.stats['corrections_generated']}")
        logger.info(f"Patterns found: {self.stats['patterns_found']}")
        logger.info("=" * 80)

        # Update status file for plugin compatibility
        patterns = self.mine_patterns() if self.pattern_db else []
        self.update_status(patterns)

    def run_continuous(self, interval: int = 300):
        """Run continuous self-correction loop"""
        logger.info("=" * 80)
        logger.info("🔄 SELF-CORRECTION LOOP - CONTINUOUS MODE")
        logger.info("=" * 80)
        logger.info(f"Check interval: {interval}s")
        logger.info(f"Error threshold: {self.error_threshold}")
        logger.info("=" * 80)

        unvalidated_dir = self.base_dir / "queue/unvalidated"

        while True:
            try:
                # Check for unvalidated files
                files = sorted(unvalidated_dir.glob("*.jsonl"))

                if files:
                    logger.info(f"\n📂 Found {len(files)} unvalidated files")

                    for filepath in files:
                        logger.info(f"\nProcessing: {filepath.name}")

                        # Process file
                        self.run_validation_pipeline(filepath)

                        # Move to processed
                        processed_dir = unvalidated_dir / "processed"
                        processed_dir.mkdir(exist_ok=True)
                        filepath.rename(processed_dir / filepath.name)

                logger.info(f"\nNext check in {interval}s...")
                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("\n🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in loop: {e}")
                time.sleep(interval)


def run_with_scheduler(scheduler_url: str, interval: int, base_dir: str, error_threshold: int):
    """Run as a scheduler client - submit tasks instead of executing directly."""
    import sys
    sys.path.insert(0, str(Path(base_dir)))
    from monitoring.task_client import TaskClient

    client = TaskClient(scheduler_url)

    logger.info("="*60)
    logger.info("Self-Correction Loop - Scheduler Mode")
    logger.info("="*60)
    logger.info(f"Scheduler: {scheduler_url}")
    logger.info(f"Interval: {interval}s")
    logger.info(f"Error threshold: {error_threshold}")

    if not client.is_healthy():
        logger.error("Scheduler not available!")
        return

    while True:
        try:
            logger.info("\nSubmitting self_correction task to scheduler...")

            result = client.submit_and_wait(
                task_type="self_correction",
                params={
                    "base_dir": base_dir,
                    "error_threshold": error_threshold,
                    "batch_size": 100
                },
                priority=2,  # NORMAL
                timeout=600.0  # 10 min timeout
            )

            if result:
                logger.info(f"Task completed: {result.get('status')}")
                if result.get('result'):
                    r = result['result']
                    logger.info(f"  Tested: {r.get('tested', 0)}")
                    logger.info(f"  Errors captured: {r.get('errors_captured', 0)}")
                    logger.info(f"  Corrections generated: {r.get('corrections_generated', 0)}")
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

    parser = argparse.ArgumentParser(description="Self-Correction Loop")
    parser.add_argument('--api-url', default=None,
                       help='API URL for 3090 (auto-detect if not set)')
    parser.add_argument('--base-dir', default=None,
                       help='Base directory (auto-detect if not set)')
    parser.add_argument('--file', type=str,
                       help='Process specific file')
    parser.add_argument('--continuous', action='store_true',
                       help='Run in continuous mode')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval for continuous mode (seconds)')
    parser.add_argument('--error-threshold', type=int, default=50,
                       help='Min errors before analysis')
    parser.add_argument('--model', type=str, default=None,
                       help="Model ID to use (e.g., 'checkpoint-177000', 'Qwen3-0.6B-base'). "
                            "If not specified, auto-detects from inference server.")
    parser.add_argument('--use-scheduler', action='store_true',
                       help='Submit tasks to GPU Task Scheduler instead of running directly')
    parser.add_argument('--scheduler-url', default='http://192.168.x.x:8766',
                       help='GPU Task Scheduler URL')

    args = parser.parse_args()

    # Scheduler mode
    if args.use_scheduler:
        run_with_scheduler(args.scheduler_url, args.interval, args.base_dir, args.error_threshold)
        return

    # Create loop
    loop = SelfCorrectionLoop(
        api_url=args.api_url,
        base_dir=args.base_dir,
        error_threshold=args.error_threshold,
        model=args.model,  # DYNAMIC: Model ID like "checkpoint-177000" or "Qwen3-0.6B-base"
    )

    if args.file:
        # Process single file
        loop.run_validation_pipeline(Path(args.file))
    elif args.continuous:
        # Run continuously
        loop.run_continuous(interval=args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
