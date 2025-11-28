#!/usr/bin/env python3
"""
Validation Benchmarking System

Tests different validation set sizes to understand:
- Compute cost (time, throughput)
- Statistical power (confidence intervals, significance testing)
- Cost/benefit tradeoffs

Usage:
    python3 validation_benchmark.py --checkpoint-dir models/current_model --sizes 50,100,200
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import sys
import requests

@dataclass
class BenchmarkResult:
    """Single benchmark run results"""
    sample_size: int
    difficulty: str
    elapsed_time: float
    examples_per_second: float
    total_examples: int
    correct: int
    accuracy: float
    confidence_interval_95: Tuple[float, float]

@dataclass
class CostBenefitMetrics:
    """Cost/benefit analysis for a sample size"""
    sample_size: int
    time_per_run: float
    runs_per_hour: int
    statistical_power: float  # Ability to detect 5% accuracy change
    recommended: bool
    notes: str


def calculate_confidence_interval(correct: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for binomial proportion"""
    if total == 0:
        return (0.0, 0.0)

    from math import sqrt

    # Z-score for 95% confidence
    z = 1.96 if confidence == 0.95 else 2.576  # 99% confidence

    p = correct / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator

    return (max(0, center - margin), min(1, center + margin))


def calculate_statistical_power(n: int, p1: float = 0.7, p2: float = 0.75, alpha: float = 0.05) -> float:
    """
    Calculate statistical power to detect accuracy difference.

    Args:
        n: Sample size
        p1: Baseline accuracy (e.g., 70%)
        p2: Target accuracy (e.g., 75% - a 5% improvement)
        alpha: Significance level

    Returns:
        Power (probability of detecting the difference)
    """
    from math import sqrt

    # Pooled proportion
    p_pooled = (p1 + p2) / 2

    # Standard error under null hypothesis
    se_null = sqrt(2 * p_pooled * (1 - p_pooled) / n)

    # Standard error under alternative hypothesis
    se_alt = sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)

    # Z-score for alpha (two-tailed test)
    z_alpha = 1.96

    # Z-score for power calculation
    z_power = (abs(p1 - p2) - z_alpha * se_null) / se_alt

    # Convert to power (approximation using normal distribution)
    # For simplicity, using a rough approximation
    if z_power < -3:
        return 0.001
    elif z_power > 3:
        return 0.999
    else:
        # Approximate using normal CDF
        # This is a rough approximation
        from math import erf
        power = 0.5 + 0.5 * erf(z_power / sqrt(2))
        return power


def test_validation_set(
    validation_file: Path,
    api_url: str = "http://inference.local:5001/v1/chat/completions",
    max_samples: int = None
) -> Tuple[int, int, float]:
    """
    Test a validation set and return (correct, total, elapsed_time)

    Args:
        validation_file: Path to .jsonl validation file
        api_url: API endpoint
        max_samples: Maximum number of samples to test (None = all)

    Returns:
        (correct_count, total_count, elapsed_seconds)
    """
    print(f"  Testing: {validation_file.name} (max_samples={max_samples})")

    examples = []
    with open(validation_file, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            examples.append(json.loads(line))

    correct = 0
    total = len(examples)
    start_time = time.time()

    for i, example in enumerate(examples):
        if i % 10 == 0:
            print(f"    Progress: {i}/{total}")

        # Extract user prompt and expected response
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_msg = messages[0]['content']
        expected = messages[1]['content']

        # Call API
        try:
            response = requests.post(
                api_url,
                json={
                    "messages": [{"role": "user", "content": user_msg}],
                    "temperature": 0.0,
                    "max_tokens": 2048
                },
                timeout=30
            )

            if response.status_code == 200:
                actual = response.json()['choices'][0]['message']['content']

                # Simple exact match for now (can be improved)
                if actual.strip() == expected.strip():
                    correct += 1
            else:
                print(f"    API error: {response.status_code}")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"  Completed: {correct}/{total} correct in {elapsed:.1f}s")

    return correct, total, elapsed


def run_benchmark(
    validation_dir: Path,
    sample_sizes: List[int],
    api_url: str = "http://inference.local:5001/v1/chat/completions"
) -> List[BenchmarkResult]:
    """Run benchmarks across different sample sizes"""
    results = []

    difficulties = ['easy', 'medium', 'hard']

    for size in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking sample size: {size}")
        print(f"{'='*60}")

        for diff in difficulties:
            val_file = validation_dir / f"{diff}.jsonl"
            if not val_file.exists():
                print(f"  Skipping {diff} (file not found)")
                continue

            correct, total, elapsed = test_validation_set(val_file, api_url, max_samples=size)

            if total > 0:
                accuracy = correct / total
                ci = calculate_confidence_interval(correct, total)
                throughput = total / elapsed if elapsed > 0 else 0

                result = BenchmarkResult(
                    sample_size=size,
                    difficulty=diff,
                    elapsed_time=elapsed,
                    examples_per_second=throughput,
                    total_examples=total,
                    correct=correct,
                    accuracy=accuracy,
                    confidence_interval_95=ci
                )
                results.append(result)

                print(f"  {diff.upper()}: {accuracy:.1%} [{ci[0]:.1%}, {ci[1]:.1%}] @ {throughput:.1f} ex/s")

    return results


def analyze_cost_benefit(results: List[BenchmarkResult]) -> List[CostBenefitMetrics]:
    """Analyze cost/benefit tradeoffs"""
    # Group by sample size
    size_groups = {}
    for r in results:
        if r.sample_size not in size_groups:
            size_groups[r.sample_size] = []
        size_groups[r.sample_size].append(r)

    analysis = []

    for size in sorted(size_groups.keys()):
        group = size_groups[size]

        # Average time across difficulties
        avg_time = sum(r.elapsed_time for r in group) / len(group)
        runs_per_hour = int(3600 / avg_time) if avg_time > 0 else 0

        # Statistical power to detect 5% accuracy improvement
        power = calculate_statistical_power(size, p1=0.70, p2=0.75)

        # Recommendation logic
        if size < 50:
            recommended = False
            notes = "Too small - insufficient statistical power"
        elif size > 200 and avg_time > 60:
            recommended = False
            notes = "Too slow - diminishing returns"
        elif power > 0.8:
            recommended = True
            notes = "Good balance of speed and statistical power"
        else:
            recommended = False
            notes = f"Moderate power ({power:.0%}) - consider larger sample"

        metrics = CostBenefitMetrics(
            sample_size=size,
            time_per_run=avg_time,
            runs_per_hour=runs_per_hour,
            statistical_power=power,
            recommended=recommended,
            notes=notes
        )
        analysis.append(metrics)

    return analysis


def print_summary(results: List[BenchmarkResult], analysis: List[CostBenefitMetrics]):
    """Print summary report"""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    print("\n1. ACCURACY BY DIFFICULTY & SAMPLE SIZE")
    print("-" * 80)

    # Group by difficulty
    diff_groups = {}
    for r in results:
        if r.difficulty not in diff_groups:
            diff_groups[r.difficulty] = []
        diff_groups[r.difficulty].append(r)

    for diff in ['easy', 'medium', 'hard']:
        if diff not in diff_groups:
            continue

        print(f"\n{diff.upper()}:")
        for r in sorted(diff_groups[diff], key=lambda x: x.sample_size):
            ci_width = r.confidence_interval_95[1] - r.confidence_interval_95[0]
            print(f"  n={r.sample_size:3d}: {r.accuracy:6.1%} ± {ci_width/2:5.1%}  "
                  f"[{r.confidence_interval_95[0]:.1%}, {r.confidence_interval_95[1]:.1%}]  "
                  f"({r.elapsed_time:5.1f}s @ {r.examples_per_second:4.1f} ex/s)")

    print(f"\n{'='*80}")
    print("2. COST/BENEFIT ANALYSIS")
    print("-" * 80)
    print(f"{'Size':>6} {'Time/Run':>10} {'Runs/Hr':>9} {'Power':>7} {'Rec':>5} {'Notes':<40}")
    print("-" * 80)

    for m in analysis:
        rec_mark = "✓" if m.recommended else " "
        print(f"{m.sample_size:6d} {m.time_per_run:9.1f}s {m.runs_per_hour:8d}/hr "
              f"{m.statistical_power:6.0%}  {rec_mark:>4}  {m.notes}")

    print(f"\n{'='*80}")
    print("3. RECOMMENDATIONS")
    print("-" * 80)

    recommended = [m for m in analysis if m.recommended]
    if recommended:
        best = max(recommended, key=lambda x: x.statistical_power)
        print(f"\nRecommended sample size: {best.sample_size}")
        print(f"  - Time per run: {best.time_per_run:.1f}s")
        print(f"  - Runs per hour: {best.runs_per_hour}/hr")
        print(f"  - Statistical power: {best.statistical_power:.0%} (to detect 5% improvement)")
        print(f"  - {best.notes}")
    else:
        print("\nNo optimal size found in tested range. Consider:")
        print("  - Increase sample size for more power")
        print("  - Decrease sample size for faster iteration")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Validation benchmarking system")
    parser.add_argument(
        '--validation-dir',
        type=Path,
        default=Path('data/validation'),
        help='Directory containing validation .jsonl files'
    )
    parser.add_argument(
        '--sizes',
        type=str,
        default='50,100,200',
        help='Comma-separated sample sizes to test (e.g., "50,100,200")'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://inference.local:5001/v1/chat/completions',
        help='API endpoint URL'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    # Parse sizes
    sample_sizes = [int(s.strip()) for s in args.sizes.split(',')]

    print(f"Validation Benchmark")
    print(f"  Validation dir: {args.validation_dir}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  API URL: {args.api_url}")

    # Run benchmarks
    results = run_benchmark(args.validation_dir, sample_sizes, args.api_url)

    # Analyze
    analysis = analyze_cost_benefit(results)

    # Print summary
    print_summary(results, analysis)

    # Save results
    if args.output:
        output_data = {
            'timestamp': time.time(),
            'sample_sizes': sample_sizes,
            'results': [asdict(r) for r in results],
            'analysis': [asdict(a) for a in analysis]
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
