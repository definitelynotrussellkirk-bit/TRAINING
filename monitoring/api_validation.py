#!/usr/bin/env python3
"""
API-Based Validation Test - Uses existing API server

Much faster than direct inference - uses optimized API endpoint.
Measures timing and accuracy via HTTP requests.
"""

import argparse
import json
import time
from pathlib import Path
import requests
from typing import Tuple, List

def run_inference_via_api(prompt: str, api_url: str = "http://192.168.x.x:5001/v1/chat/completions") -> str:
    """Run inference via API"""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2048
    }

    response = requests.post(api_url, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data['choices'][0]['message']['content'].strip()


def test_validation_file(
    api_url: str,
    validation_file: Path,
    num_samples: int
) -> Tuple[int, int, float, List[dict]]:
    """Test model on validation file via API

    Returns: (correct, total, avg_time, results)
    """
    with open(validation_file) as f:
        examples = [json.loads(line) for line in f]

    # Limit samples
    examples = examples[:num_samples]

    correct = 0
    total = 0
    times = []
    results = []

    for i, example in enumerate(examples, 1):
        prompt = example['text']
        expected = example['expected_answer']

        start = time.time()
        try:
            response = run_inference_via_api(prompt, api_url)
            elapsed = time.time() - start
            times.append(elapsed)

            # Extract answer
            if "The answer is:" in response:
                answer = response.split("The answer is:")[-1].strip().rstrip('.')
            else:
                answer = response.strip()

            is_correct = answer == expected
            if is_correct:
                correct += 1
            total += 1

            results.append({
                'prompt': prompt[:100],
                'expected': expected,
                'got': answer,
                'correct': is_correct,
                'time': elapsed
            })

            print(f"  [{i}/{len(examples)}] {'✓' if is_correct else '✗'} ({elapsed:.2f}s)")

        except Exception as e:
            print(f"  [{i}/{len(examples)}] ERROR: {e}")
            total += 1
            results.append({
                'prompt': prompt[:100],
                'expected': expected,
                'got': f"ERROR: {e}",
                'correct': False,
                'time': 0
            })

    avg_time = sum(times) / len(times) if times else 0
    return correct, total, avg_time, results


def main():
    parser = argparse.ArgumentParser(description='Run validation via API')
    parser.add_argument('--api-url', default='http://192.168.x.x:5001/v1/chat/completions',
                       help='API endpoint URL')
    parser.add_argument('--samples', type=int, default=10,
                       help='Samples per difficulty level')
    parser.add_argument('--output', type=Path, default=Path('/tmp/api_validation.json'),
                       help='Output JSON file')

    args = parser.parse_args()

    # Find validation files
    base_dir = Path.home() / 'TRAINING'
    validation_dir = base_dir / 'data' / 'validation'

    difficulties = ['easy', 'medium', 'hard']

    print(f"API Validation Test")
    print(f"  API: {args.api_url}")
    print(f"  Samples per difficulty: {args.samples}")
    print()

    overall_start = time.time()
    all_results = {}

    for difficulty in difficulties:
        # Use the samples files
        val_file = validation_dir / f'{difficulty}_{args.samples}.jsonl'

        if not val_file.exists():
            print(f"⚠️  {difficulty}: File not found: {val_file}")
            continue

        print(f"Testing {difficulty}...")
        correct, total, avg_time, results = test_validation_file(
            args.api_url,
            val_file,
            args.samples
        )

        accuracy = correct / total if total > 0 else 0

        all_results[difficulty] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'avg_time': avg_time,
            'throughput': 1.0 / avg_time if avg_time > 0 else 0,
            'results': results
        }

        print(f"  {difficulty}: {correct}/{total} = {accuracy*100:.1f}% (avg {avg_time:.2f}s/ex)")
        print()

    overall_time = time.time() - overall_start

    # Calculate overall accuracy
    total_correct = sum(r['correct'] for r in all_results.values())
    total_examples = sum(r['total'] for r in all_results.values())
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0

    # Summary
    summary = {
        'timestamp': time.time(),
        'api_url': args.api_url,
        'samples_per_difficulty': args.samples,
        'overall_accuracy': overall_accuracy,
        'overall_time': overall_time,
        'by_difficulty': all_results
    }

    # Save results
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Overall: {total_correct}/{total_examples} = {overall_accuracy*100:.1f}%")
    print(f"Total time: {overall_time:.1f}s")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
