#!/usr/bin/env python3
"""
Self-Correction Training Pipeline

Generates training data that teaches models to:
1. Answer questions
2. Self-evaluate their answers
3. Correct mistakes based on programmatic feedback

Flow:
    prompt → answer_1 → error_code → self_eval → corrected_answer

Creates 3 separate training examples per question:
    1. prompt → answer_1
    2. prompt + answer_1 + error_code → self_eval
    3. prompt + error_code → corrected_answer
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
from pathlib import Path
import argparse


class ErrorCodeGenerator:
    """Generate consistent, programmatic error codes by comparing answers."""

    def __init__(self):
        self.error_codes = []

    def analyze(self, answer: str, golden: str) -> List[str]:
        """
        Generate error codes by comparing answer to golden.

        Returns list of error codes that provide hints without giving away answer.
        """
        self.error_codes = []

        answer = answer.strip()
        golden = golden.strip()

        # 1. Length analysis
        self._check_length(answer, golden)

        # 2. Token similarity
        self._check_token_similarity(answer, golden)

        # 3. Numeric content
        self._check_numeric_content(answer, golden)

        # 4. Structure analysis
        self._check_structure(answer, golden)

        # 5. Key elements
        self._check_key_elements(answer, golden)

        return self.error_codes

    def _check_length(self, answer: str, golden: str):
        """Check length differences."""
        answer_tokens = len(answer.split())
        golden_tokens = len(golden.split())
        diff = answer_tokens - golden_tokens

        if abs(diff) > 5:  # Significant difference
            if diff > 0:
                self.error_codes.append(f"LENGTH_LONG_BY_{abs(diff)}_TOKENS")
            else:
                self.error_codes.append(f"LENGTH_SHORT_BY_{abs(diff)}_TOKENS")

        # Character length
        char_diff = len(answer) - len(golden)
        if abs(char_diff) > 50:
            if char_diff > 0:
                self.error_codes.append(f"TOO_VERBOSE")
            else:
                self.error_codes.append(f"TOO_BRIEF")

    def _check_token_similarity(self, answer: str, golden: str):
        """Check token-level similarity."""
        # Use SequenceMatcher for similarity ratio
        ratio = SequenceMatcher(None, answer.lower(), golden.lower()).ratio()
        similarity_pct = int(ratio * 100)

        if similarity_pct < 30:
            self.error_codes.append(f"VERY_DIFFERENT_CONTENT")
        elif similarity_pct < 60:
            self.error_codes.append(f"SIMILARITY_{similarity_pct}%")
        elif similarity_pct < 95:
            self.error_codes.append(f"CLOSE_BUT_INACCURATE")

    def _check_numeric_content(self, answer: str, golden: str):
        """Check for numeric errors."""
        answer_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
        golden_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', golden))

        if golden_nums and not answer_nums:
            self.error_codes.append("MISSING_NUMBERS")
        elif answer_nums != golden_nums:
            missing = golden_nums - answer_nums
            extra = answer_nums - golden_nums

            if missing:
                self.error_codes.append(f"NUMERIC_ERROR_MISSING_{len(missing)}_VALUES")
            if extra:
                self.error_codes.append(f"NUMERIC_ERROR_EXTRA_{len(extra)}_VALUES")

    def _check_structure(self, answer: str, golden: str):
        """Check structural patterns."""
        # Check for lists
        answer_has_list = bool(re.search(r'^\s*[-*\d]+\.?\s+', answer, re.MULTILINE))
        golden_has_list = bool(re.search(r'^\s*[-*\d]+\.?\s+', golden, re.MULTILINE))

        if golden_has_list and not answer_has_list:
            self.error_codes.append("STRUCTURE_SHOULD_BE_LIST")
        elif answer_has_list and not golden_has_list:
            self.error_codes.append("STRUCTURE_SHOULD_BE_PROSE")

        # Check for code blocks
        answer_has_code = '```' in answer or '    ' in answer
        golden_has_code = '```' in golden or '    ' in golden

        if golden_has_code and not answer_has_code:
            self.error_codes.append("MISSING_CODE_BLOCK")
        elif answer_has_code and not golden_has_code:
            self.error_codes.append("UNEXPECTED_CODE_BLOCK")

        # Check for JSON
        answer_has_json = '{' in answer and '}' in answer
        golden_has_json = '{' in golden and '}' in golden

        if golden_has_json and not answer_has_json:
            self.error_codes.append("FORMAT_SHOULD_BE_JSON")

    def _check_key_elements(self, answer: str, golden: str):
        """Check for key phrases/elements."""
        # Extract potential key phrases (words that appear in golden but not answer)
        answer_lower = answer.lower()
        golden_words = set(w for w in golden.lower().split() if len(w) > 4)
        answer_words = set(w for w in answer_lower.split() if len(w) > 4)

        missing = golden_words - answer_words
        extra = answer_words - golden_words

        if len(missing) > len(golden_words) * 0.3:  # Missing >30% of key words
            self.error_codes.append(f"MISSING_KEY_CONCEPTS_{len(missing)}_TERMS")

        if len(extra) > len(answer_words) * 0.5:  # >50% extra words
            self.error_codes.append("OFF_TOPIC_CONTENT")

        # Check if answer is completely empty
        if not answer or len(answer.strip()) == 0:
            self.error_codes.append("PRIMARY_ANSWER_MISSING")


class SelfCorrectionPipeline:
    """Generate self-correction training data."""

    def __init__(self, model_inference_fn=None):
        """
        Args:
            model_inference_fn: Function that takes prompt and returns model output
                              If None, we expect pre-generated answers
        """
        self.model_inference_fn = model_inference_fn
        self.error_generator = ErrorCodeGenerator()

    def generate_from_qa_pair(
        self,
        prompt: str,
        golden_answer: str,
        initial_answer: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate self-correction training examples from a Q&A pair.

        Args:
            prompt: The question/prompt
            golden_answer: The correct answer
            initial_answer: Model's first attempt (if None, will generate)

        Returns:
            List of 3 training examples (or 1 if answer was correct)
        """
        # Step 1: Get initial answer
        if initial_answer is None:
            if self.model_inference_fn is None:
                raise ValueError("Need either initial_answer or model_inference_fn")
            initial_answer = self.model_inference_fn(prompt)

        # Step 2: Compare to golden and generate error codes
        error_codes = self.error_generator.analyze(initial_answer, golden_answer)

        # If no errors, create self-evaluation showing it was correct
        if not error_codes:
            self_eval_prompt = f"""{prompt}

Your previous answer:
{initial_answer}

Was this correct?"""

            return [{
                "prompt": self_eval_prompt,
                "response": "This was correct.",
                "type": "self_evaluation",
                "correct": True
            }]

        # Step 3: Create training examples
        examples = []

        # Example 1: Self-evaluation (teaches to recognize errors)
        # Show the model its previous answer and ask if it was correct
        self_eval_prompt = f"""{prompt}

Your previous answer:
{initial_answer}

Was this correct?"""

        self_eval_response = "This was not correct."

        examples.append({
            "prompt": self_eval_prompt,
            "response": self_eval_response,
            "type": "self_evaluation",
            "error_codes": error_codes
        })

        # Example 2: Correction (teaches fixing based on hints)
        # Show the model it was wrong, provide hints, ask for correct answer
        error_hint = " | ".join(error_codes)
        correction_prompt = f"""{prompt}

Your previous answer:
{initial_answer}

This was not correct. Error indicators: {error_hint}

Provide the correct answer:"""

        examples.append({
            "prompt": correction_prompt,
            "response": golden_answer,
            "type": "correction",
            "error_codes": error_codes
        })

        return examples

    def process_dataset(
        self,
        input_file: str,
        output_file: str,
        initial_answers_file: Optional[str] = None
    ):
        """
        Process a dataset of Q&A pairs into self-correction training data.

        Args:
            input_file: Path to .jsonl with {"prompt": ..., "response": ...}
            output_file: Path to write training examples
            initial_answers_file: Optional .jsonl with pre-generated answers
        """
        # Load golden Q&A pairs
        with open(input_file) as f:
            golden_pairs = [json.loads(line) for line in f]

        # Load initial answers if provided
        initial_answers = {}
        if initial_answers_file:
            with open(initial_answers_file) as f:
                for line in f:
                    data = json.loads(line)
                    # Assume format: {"prompt": ..., "answer": ...}
                    initial_answers[data['prompt']] = data['answer']

        # Process each pair
        all_examples = []
        stats = {
            'total_questions': len(golden_pairs),
            'correct_first_try': 0,
            'needed_correction': 0,
            'examples_generated': 0
        }

        for pair in golden_pairs:
            prompt = pair['prompt']
            golden = pair['response']
            initial = initial_answers.get(prompt)

            examples = self.generate_from_qa_pair(prompt, golden, initial)

            if examples[0].get('correct', False):
                stats['correct_first_try'] += 1
            else:
                stats['needed_correction'] += 1

            all_examples.extend(examples)
            stats['examples_generated'] += len(examples)

        # Write output
        with open(output_file, 'w') as f:
            for example in all_examples:
                # Convert to standard training format
                training_example = {
                    "prompt": example['prompt'],
                    "response": example['response']
                }
                f.write(json.dumps(training_example) + '\n')

        # Write metadata
        meta_file = output_file.replace('.jsonl', '_metadata.json')
        with open(meta_file, 'w') as f:
            json.dump({
                'stats': stats,
                'source_file': input_file,
                'examples_breakdown': {
                    'self_evaluations_correct': stats['correct_first_try'],
                    'self_evaluations_incorrect': stats['needed_correction'],
                    'corrections': stats['needed_correction']
                }
            }, f, indent=2)

        print(f"Generated {stats['examples_generated']} training examples")
        print(f"  Correct first try: {stats['correct_first_try']}")
        print(f"  Needed correction: {stats['needed_correction']}")
        print(f"  Output: {output_file}")
        print(f"  Metadata: {meta_file}")


def generate_initial_answers(input_file: str, output_file: str, model_fn):
    """Helper to generate initial answers from a model."""
    with open(input_file) as f:
        pairs = [json.loads(line) for line in f]

    with open(output_file, 'w') as f:
        for i, pair in enumerate(pairs):
            prompt = pair['prompt']
            answer = model_fn(prompt)

            f.write(json.dumps({
                "prompt": prompt,
                "answer": answer
            }) + '\n')

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{len(pairs)} answers...")

    print(f"Initial answers saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate self-correction training data')
    parser.add_argument('--input', required=True, help='Input .jsonl with golden Q&A pairs')
    parser.add_argument('--output', required=True, help='Output .jsonl for training')
    parser.add_argument('--initial-answers', help='Optional: pre-generated initial answers')
    parser.add_argument('--test-error-codes', action='store_true',
                       help='Test error code generation on sample data')

    args = parser.parse_args()

    if args.test_error_codes:
        # Test the error code system
        print("\n=== Testing Error Code Generation ===\n")

        test_cases = [
            {
                "name": "Too short",
                "answer": "The answer is 42.",
                "golden": "The answer is 42 because it represents the ultimate answer to life, the universe, and everything according to Douglas Adams' Hitchhiker's Guide to the Galaxy."
            },
            {
                "name": "Wrong numbers",
                "answer": "The calculation gives us 156.",
                "golden": "The calculation gives us 142."
            },
            {
                "name": "Missing structure",
                "answer": "We need apples oranges and bananas.",
                "golden": "We need:\n1. Apples\n2. Oranges\n3. Bananas"
            },
            {
                "name": "Off topic",
                "answer": "Python is a great programming language with dynamic typing.",
                "golden": "The capital of France is Paris."
            }
        ]

        gen = ErrorCodeGenerator()
        for case in test_cases:
            print(f"Test: {case['name']}")
            print(f"  Answer: {case['answer'][:50]}...")
            print(f"  Golden: {case['golden'][:50]}...")
            codes = gen.analyze(case['answer'], case['golden'])
            print(f"  Error codes: {codes}")
            print()

    else:
        # Generate training data
        pipeline = SelfCorrectionPipeline()
        pipeline.process_dataset(
            input_file=args.input,
            output_file=args.output,
            initial_answers_file=args.initial_answers
        )
