#!/usr/bin/env python3
"""
Metacognition Trainer - Teach the model to recognize and correct errors.

Creates two types of training data:
1. Discrimination: "Is this answer correct?" → "Yes" / "No" (50/50 split)
2. Correction: "What is the correct answer?" → correct answer

No model-generated reasoning - purely factual answers from ground truth.

Usage:
    # Generate from validation data
    python3 metacognition_trainer.py --validation-dir data/validation/primitives

    # Process curriculum eval results
    python3 metacognition_trainer.py --eval-results status/curriculum_eval.json

    # Run continuously, process new eval results
    python3 metacognition_trainer.py --continuous --interval 600
"""

import json
import random
import requests
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.paths import get_base_dir, get_remote_api_url
except ImportError:
    def get_base_dir():
        return Path("/path/to/training")
    def get_remote_api_url():
        return "http://192.168.x.x:8765"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetacognitionTrainer:
    """
    Generate metacognition training data:
    - Discrimination: Is this answer right or wrong?
    - Correction: What's the correct answer?
    """

    def __init__(
        self,
        api_url: str = None,
        base_dir: str = None,
        batch_size: int = 50,
    ):
        self.api_url = api_url or get_remote_api_url()
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.batch_size = batch_size

        # Stats
        self.stats = {
            'questions_tested': 0,
            'correct_answers': 0,
            'wrong_answers': 0,
            'discrimination_examples': 0,
            'correction_examples': 0,
        }

        # Storage for wrong/correct answers
        self.wrong_answers: List[Dict] = []
        self.correct_answers: List[Dict] = []

        # Status file
        self.status_file = self.base_dir / "status" / "metacognition_trainer.json"
        self.status = self._load_status()

    def _load_status(self) -> Dict:
        """Load previous status."""
        if self.status_file.exists():
            try:
                with open(self.status_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "runs": [],
            "total_discrimination": 0,
            "total_correction": 0,
            "last_updated": None
        }

    def _save_status(self):
        """Save status."""
        self.status["last_updated"] = datetime.now().isoformat()
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def call_api(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Call inference API."""
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "model": "Qwen3-0.6B",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    return result['choices'][0]['message']['content']
            return None
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.
        Looks for common answer patterns in model output.
        """
        if not answer:
            return ""

        text = str(answer).strip().lower()

        # Look for explicit answer patterns
        import re
        patterns = [
            r'answer[:\s]+([a-z0-9]+)',
            r'the answer is[:\s]+([a-z0-9]+)',
            r'result[:\s]+([a-z0-9]+)',
            r'\*\*([a-z0-9]+)\*\*',  # Bold
            r'^([a-z0-9]+)$',  # Just the answer
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip('.,!?:;')

        # Fallback: look for known answer types in the text
        known_answers = ['yes', 'no', 'true', 'false', 'even', 'odd', 'valid', 'invalid']
        for ans in known_answers:
            if ans in text:
                return ans

        # Last resort: first word that's alphanumeric
        words = re.findall(r'\b[a-z0-9]+\b', text)
        if words:
            return words[0]

        return ""

    def extract_short_answer(self, model_response: str, expected: str) -> str:
        """
        Extract a short answer from model response for display.
        If model gave a long response, try to find the actual answer part.
        """
        normalized = self.normalize_answer(model_response)
        if normalized:
            return normalized

        # If normalization failed, truncate
        if len(model_response) > 50:
            return model_response[:50] + "..."
        return model_response

    def test_question(self, question: str, expected: str) -> Dict:
        """
        Test a single question against the model.
        Returns dict with question, expected, model_answer, is_correct.
        """
        model_response = self.call_api(question)
        if model_response is None:
            return None

        model_answer = self.normalize_answer(model_response)
        expected_norm = self.normalize_answer(expected)

        is_correct = model_answer == expected_norm

        return {
            "question": question,
            "expected": expected,
            "model_answer": model_response.strip() if model_response else "",
            "model_answer_normalized": model_answer,
            "is_correct": is_correct
        }

    def create_discrimination_example(
        self,
        question: str,
        given_answer: str,
        is_correct: bool,
        expected_answer: str = None
    ) -> Dict:
        """
        Create a discrimination training example.
        "Is this answer correct?" → "Yes" or "No"
        Uses short answer for cleaner training data.
        """
        # Use short answer for display (unless it's ground truth "Yes" case)
        if is_correct:
            short_answer = given_answer  # Ground truth, already clean
        else:
            short_answer = self.extract_short_answer(given_answer, expected_answer or "")

        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer given: {short_answer}\n\nIs this answer correct?"
                },
                {
                    "role": "assistant",
                    "content": "Yes" if is_correct else "No"
                }
            ],
            "metadata": {
                "task": "discrimination",
                "is_correct": is_correct,
                "source": "metacognition_trainer",
                "generated_at": datetime.now().isoformat()
            }
        }

    def create_correction_example(
        self,
        question: str,
        wrong_answer: str,
        correct_answer: str
    ) -> Dict:
        """
        Create a correction training example.
        "What is the correct answer?" → correct answer
        Uses short/normalized wrong answer for cleaner training data.
        """
        # Use short answer for display
        short_wrong = self.extract_short_answer(wrong_answer, correct_answer)

        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Question: {question}\nIncorrect answer: {short_wrong}\n\nWhat is the correct answer?"
                },
                {
                    "role": "assistant",
                    "content": correct_answer
                }
            ],
            "metadata": {
                "task": "correction",
                "wrong_answer_short": short_wrong,
                "source": "metacognition_trainer",
                "generated_at": datetime.now().isoformat()
            }
        }

    def process_validation_file(self, filepath: Path) -> int:
        """
        Process a validation JSONL file.
        Tests each example, captures right/wrong answers.
        Returns number of examples processed.
        """
        count = 0

        with open(filepath) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract question and expected answer
                question = example.get("user_prompt", example.get("prompt", ""))
                expected = example.get("expected_answer", example.get("answer", ""))

                if not question or not expected:
                    continue

                # Test against model
                result = self.test_question(question, expected)
                if result is None:
                    continue

                self.stats['questions_tested'] += 1

                if result['is_correct']:
                    self.stats['correct_answers'] += 1
                    self.correct_answers.append(result)
                else:
                    self.stats['wrong_answers'] += 1
                    self.wrong_answers.append(result)

                count += 1

                if count % 10 == 0:
                    logger.info(f"  Processed {count} examples...")

        return count

    def generate_training_data(self, output_dir: Path = None) -> Dict[str, Path]:
        """
        Generate balanced discrimination + correction training data.

        Discrimination: 50% correct answers ("Yes"), 50% wrong answers ("No")
        Correction: All wrong answers with their corrections
        """
        if output_dir is None:
            output_dir = self.base_dir / "inbox"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}

        # === DISCRIMINATION DATA (50/50 split) ===
        discrimination_examples = []

        # Add wrong answer examples (answer: "No")
        for item in self.wrong_answers:
            ex = self.create_discrimination_example(
                question=item['question'],
                given_answer=item['model_answer'],
                is_correct=False,
                expected_answer=item.get('expected', '')
            )
            discrimination_examples.append(ex)

        # Add correct answer examples (answer: "Yes")
        # Sample to match wrong count for 50/50 balance
        num_wrong = len(self.wrong_answers)
        correct_sample = self.correct_answers[:num_wrong]  # Match count
        if len(self.correct_answers) > num_wrong:
            random.seed(42)
            correct_sample = random.sample(self.correct_answers, num_wrong)

        for item in correct_sample:
            ex = self.create_discrimination_example(
                question=item['question'],
                given_answer=item['model_answer'],
                is_correct=True
            )
            discrimination_examples.append(ex)

        # Shuffle discrimination examples
        random.shuffle(discrimination_examples)

        if discrimination_examples:
            disc_file = output_dir / f"metacog_discrimination_{timestamp}.jsonl"
            with open(disc_file, 'w') as f:
                for ex in discrimination_examples:
                    f.write(json.dumps(ex) + '\n')
            output_files['discrimination'] = disc_file
            self.stats['discrimination_examples'] = len(discrimination_examples)
            logger.info(f"✅ Wrote {len(discrimination_examples)} discrimination examples to {disc_file}")

        # === CORRECTION DATA ===
        correction_examples = []

        for item in self.wrong_answers:
            ex = self.create_correction_example(
                question=item['question'],
                wrong_answer=item['model_answer'],
                correct_answer=item['expected']
            )
            correction_examples.append(ex)

        if correction_examples:
            corr_file = output_dir / f"metacog_correction_{timestamp}.jsonl"
            with open(corr_file, 'w') as f:
                for ex in correction_examples:
                    f.write(json.dumps(ex) + '\n')
            output_files['correction'] = corr_file
            self.stats['correction_examples'] = len(correction_examples)
            logger.info(f"✅ Wrote {len(correction_examples)} correction examples to {corr_file}")

        return output_files

    def run_on_validation_dir(self, validation_dir: Path, sample_size: int = 100):
        """
        Run on validation directory.

        Strategy:
        - Test sample_size examples to find WRONG answers (for "No" + corrections)
        - Use ground truth directly for "Yes" examples (no inference needed!)
        """
        logger.info("="*60)
        logger.info("METACOGNITION TRAINER")
        logger.info("="*60)
        logger.info(f"Validation dir: {validation_dir}")
        logger.info(f"Sample size for testing: {sample_size}")

        # Collect all validation files
        files = list(validation_dir.rglob("*.jsonl"))
        if not files:
            logger.error(f"No .jsonl files found in {validation_dir}")
            return

        logger.info(f"Found {len(files)} validation files")

        # Load all examples
        all_examples = []
        for filepath in files:
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        try:
                            ex = json.loads(line)
                            question = ex.get("user_prompt", ex.get("prompt", ""))
                            answer = ex.get("expected_answer", ex.get("answer", ""))
                            if question and answer:
                                all_examples.append({
                                    "user_prompt": question,
                                    "expected_answer": answer,
                                    "source_file": filepath.name
                                })
                        except json.JSONDecodeError:
                            pass

        logger.info(f"Loaded {len(all_examples)} total examples")

        # Sample for testing (to find wrong answers)
        random.seed(42)
        if len(all_examples) > sample_size:
            test_sample = random.sample(all_examples, sample_size)
        else:
            test_sample = all_examples

        logger.info(f"\n=== Phase 1: Finding wrong answers ({len(test_sample)} tests) ===")

        # Test to find wrong answers
        for i, example in enumerate(test_sample, 1):
            result = self.test_question(
                example['user_prompt'],
                example['expected_answer']
            )
            if result is None:
                continue

            self.stats['questions_tested'] += 1

            if result['is_correct']:
                self.stats['correct_answers'] += 1
                # Don't store - we'll use ground truth for "Yes" examples
            else:
                self.stats['wrong_answers'] += 1
                self.wrong_answers.append(result)

            if i % 20 == 0:
                acc = self.stats['correct_answers'] / max(self.stats['questions_tested'], 1)
                logger.info(f"  [{i}/{len(test_sample)}] Accuracy: {acc:.1%}, Wrong: {len(self.wrong_answers)}")

        # For "Yes" examples: use ground truth directly (no inference needed!)
        # Sample same count as wrong answers for 50/50 balance
        num_wrong = len(self.wrong_answers)
        logger.info(f"\n=== Phase 2: Sampling {num_wrong} ground truth examples for 'Yes' ===")

        # Exclude examples we already tested (avoid overlap)
        tested_questions = {r['question'] for r in self.wrong_answers}
        untested = [ex for ex in all_examples if ex['user_prompt'] not in tested_questions]

        if len(untested) >= num_wrong:
            ground_truth_sample = random.sample(untested, num_wrong)
        else:
            ground_truth_sample = untested

        # Convert ground truth to correct_answers format
        for ex in ground_truth_sample:
            self.correct_answers.append({
                "question": ex['user_prompt'],
                "expected": ex['expected_answer'],
                "model_answer": ex['expected_answer'],  # Ground truth IS the answer
                "is_correct": True
            })

        logger.info(f"  Added {len(self.correct_answers)} ground truth 'Yes' examples")

        # Generate training data
        logger.info("\n=== Phase 3: Generating training data ===")
        output_files = self.generate_training_data()

        # Update status
        self._update_run_status(output_files)

        # Print summary
        self._print_summary()

    def _update_run_status(self, output_files: Dict[str, Path]):
        """Update status file with run results."""
        run = {
            "timestamp": datetime.now().isoformat(),
            "questions_tested": self.stats['questions_tested'],
            "correct_answers": self.stats['correct_answers'],
            "wrong_answers": self.stats['wrong_answers'],
            "discrimination_examples": self.stats['discrimination_examples'],
            "correction_examples": self.stats['correction_examples'],
            "accuracy": self.stats['correct_answers'] / max(self.stats['questions_tested'], 1),
            "output_files": {k: str(v) for k, v in output_files.items()}
        }
        self.status["runs"].append(run)
        self.status["total_discrimination"] += self.stats['discrimination_examples']
        self.status["total_correction"] += self.stats['correction_examples']
        self._save_status()

    def _print_summary(self):
        """Print run summary."""
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Questions tested: {self.stats['questions_tested']}")
        logger.info(f"Correct answers: {self.stats['correct_answers']}")
        logger.info(f"Wrong answers: {self.stats['wrong_answers']}")
        acc = self.stats['correct_answers'] / max(self.stats['questions_tested'], 1)
        logger.info(f"Accuracy: {acc:.1%}")
        logger.info("")
        logger.info(f"Generated training data:")
        logger.info(f"  Discrimination examples: {self.stats['discrimination_examples']}")
        logger.info(f"  Correction examples: {self.stats['correction_examples']}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Metacognition Trainer")
    parser.add_argument('--api-url', default='http://192.168.x.x:8765',
                       help='Inference API URL')
    parser.add_argument('--base-dir', default='/path/to/training',
                       help='Base directory')
    parser.add_argument('--validation-dir', type=Path,
                       help='Validation directory to process')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of examples to sample and test')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for training data (default: inbox/)')

    args = parser.parse_args()

    trainer = MetacognitionTrainer(
        api_url=args.api_url,
        base_dir=args.base_dir
    )

    if args.validation_dir:
        trainer.run_on_validation_dir(
            validation_dir=args.validation_dir,
            sample_size=args.sample_size
        )
    else:
        # Default: run on primitives
        default_dir = Path(args.base_dir) / "data/validation/primitives"
        if default_dir.exists():
            trainer.run_on_validation_dir(default_dir, sample_size=args.sample_size)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
