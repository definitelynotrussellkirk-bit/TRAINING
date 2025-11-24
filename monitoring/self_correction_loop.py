#!/usr/bin/env python3
"""
Self-Correction Loop - Learn from model mistakes
Creates high-value training data from errors via automated correction
"""

import json
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import logging
import re

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
        api_url: str = "http://192.168.x.x:8765",
        base_dir: str = "/path/to/training",
        batch_size: int = 100,
        error_threshold: int = 50  # Min errors before analysis
    ):
        self.api_url = api_url
        self.base_dir = Path(base_dir)
        self.batch_size = batch_size
        self.error_threshold = error_threshold

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

    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'queue/unvalidated',
            'queue/normal',
            'queue/corrections',
            'queue/rejected',
            'logs/error_patterns',
            'logs/corrections'
        ]
        for d in dirs:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)

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


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Self-Correction Loop")
    parser.add_argument('--api-url', default='http://192.168.x.x:8765',
                       help='API URL for 3090')
    parser.add_argument('--base-dir', default='/path/to/training',
                       help='Base directory')
    parser.add_argument('--file', type=str,
                       help='Process specific file')
    parser.add_argument('--continuous', action='store_true',
                       help='Run in continuous mode')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval for continuous mode (seconds)')
    parser.add_argument('--error-threshold', type=int, default=50,
                       help='Min errors before analysis')

    args = parser.parse_args()

    # Create loop
    loop = SelfCorrectionLoop(
        api_url=args.api_url,
        base_dir=args.base_dir,
        error_threshold=args.error_threshold
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
