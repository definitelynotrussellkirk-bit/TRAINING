#!/usr/bin/env python3
"""
Quick Validation Test - Direct Model Inference

Runs validation without needing an API server.
Measures timing and accuracy directly.
"""

import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List

def load_model(model_path: str):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str, max_tokens: int = 2048) -> str:
    """Run a single inference"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens (skip input)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def test_validation_file(
    model,
    tokenizer,
    validation_file: Path,
    max_samples: int = None,
    verbose: bool = False
) -> Tuple[int, int, float, List[dict]]:
    """
    Test a validation file

    Returns:
        (correct, total, elapsed_time, results_list)
    """
    print(f"\n  Testing: {validation_file.name} (max_samples={max_samples})")

    examples = []
    with open(validation_file, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            examples.append(json.loads(line))

    correct = 0
    total = len(examples)
    results = []
    start_time = time.time()

    for i, example in enumerate(examples):
        if i % 10 == 0:
            print(f"    Progress: {i}/{total}")

        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_msg = messages[0]['content']
        expected = messages[1]['content']

        # Run inference
        actual = run_inference(model, tokenizer, user_msg)

        # Simple exact match (can be improved)
        is_correct = actual.strip() == expected.strip()
        if is_correct:
            correct += 1

        results.append({
            'index': i,
            'expected': expected,
            'actual': actual,
            'correct': is_correct
        })

        if verbose and not is_correct:
            print(f"    INCORRECT #{i}:")
            print(f"      Expected: {expected[:100]}...")
            print(f"      Actual:   {actual[:100]}...")

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    throughput = total / elapsed if elapsed > 0 else 0

    print(f"  Result: {correct}/{total} correct ({accuracy:.1%}) in {elapsed:.1f}s ({throughput:.2f} ex/s)")

    return correct, total, elapsed, results


def main():
    parser = argparse.ArgumentParser(description="Quick validation test")
    parser.add_argument('--model-path', type=Path, required=True, help='Path to model')
    parser.add_argument('--validation-dir', type=Path, default=Path('data/validation'))
    parser.add_argument('--samples', type=int, default=10, help='Samples per difficulty')
    parser.add_argument('--difficulties', type=str, default='easy,medium,hard')
    parser.add_argument('--verbose', action='store_true', help='Show incorrect examples')
    parser.add_argument('--output', type=Path, help='Save results to JSON')

    args = parser.parse_args()

    print(f"Quick Validation Test")
    print(f"  Model: {args.model_path}")
    print(f"  Samples per difficulty: {args.samples}")
    print(f"  Validation dir: {args.validation_dir}")

    # Load model
    model, tokenizer = load_model(args.model_path)

    # Test each difficulty
    difficulties = args.difficulties.split(',')
    all_results = {}
    total_time = 0

    for diff in difficulties:
        val_file = args.validation_dir / f"{diff}.jsonl"
        if not val_file.exists():
            print(f"\n  Skipping {diff} (file not found)")
            continue

        correct, total, elapsed, results = test_validation_file(
            model, tokenizer, val_file,
            max_samples=args.samples,
            verbose=args.verbose
        )

        all_results[diff] = {
            'correct': correct,
            'total': total,
            'accuracy': correct / total if total > 0 else 0,
            'elapsed_time': elapsed,
            'throughput': total / elapsed if elapsed > 0 else 0,
            'examples': results if args.output else []
        }

        total_time += elapsed

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for diff in difficulties:
        if diff in all_results:
            r = all_results[diff]
            print(f"{diff.upper():8} : {r['accuracy']:6.1%}  ({r['correct']:2d}/{r['total']:2d})  "
                  f"{r['elapsed_time']:6.1f}s @ {r['throughput']:5.2f} ex/s")

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Average throughput: {sum(r['total'] for r in all_results.values()) / total_time:.2f} ex/s")

    # Save results
    if args.output:
        output_data = {
            'timestamp': time.time(),
            'model_path': str(args.model_path),
            'samples_per_difficulty': args.samples,
            'total_time': total_time,
            'results': all_results
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
