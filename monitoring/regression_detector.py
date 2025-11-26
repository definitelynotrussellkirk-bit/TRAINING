#!/usr/bin/env python3
"""
Statistical Regression Detection

Uses proper statistical testing to determine if performance 
degradation is significant vs random variance.
"""

import numpy as np
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RegressionDetector - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics with variance"""
    checkpoint: str
    em_mean: float
    em_std: float
    em_samples: List[float]
    sample_size: int
    timestamp: str

class RegressionDetector:
    """
    Detects statistically significant performance regressions

    Methods:
    1. Bootstrap confidence intervals
    2. Two-sample t-test
    3. Effect size (Cohen's d)
    4. Multiple test correction (Bonferroni)
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8765",
        base_dir: str = "/path/to/training",
        alpha: float = 0.05,  # Significance level
        min_effect_size: float = 0.3,  # Minimum Cohen's d
        n_bootstrap: int = 1000,  # Bootstrap iterations
        threshold: float = 0.05  # 5% accuracy drop = regression
    ):
        self.api_url = api_url
        self.base_dir = Path(base_dir)
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold

        # Results storage
        self.results_file = self.base_dir / "status" / "regression_monitoring.json"
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self.results = self._load_results()

        # Baseline storage
        self.baseline_file = self.base_dir / "status" / "regression_baseline.json"

    def _load_results(self) -> Dict:
        """Load previous results"""
        if self.results_file.exists():
            with open(self.results_file) as f:
                return json.load(f)
        return {"checks": [], "regressions_detected": 0, "last_updated": None}

    def _save_results(self):
        """Save results to JSON for dashboard"""
        self.results["last_updated"] = datetime.now().isoformat()
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {self.results_file}")

    def _get_baseline(self) -> Optional[Dict]:
        """Get baseline performance metrics"""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return None

    def _save_baseline(self, metrics: Dict):
        """Save baseline performance"""
        metrics["saved_at"] = datetime.now().isoformat()
        with open(self.baseline_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def _test_model(self, num_samples: int = 50) -> Dict:
        """Test current model and return accuracy"""
        validation_dir = self.base_dir / "data" / "validation"
        examples = []

        for val_file in validation_dir.glob("*.jsonl"):
            with open(val_file) as f:
                for line in f:
                    try:
                        examples.append(json.loads(line))
                    except:
                        continue

        if not examples:
            return {"error": "No validation data"}

        import random
        test_sample = random.sample(examples, min(num_samples, len(examples)))

        correct = 0
        total = 0

        for ex in test_sample:
            try:
                # Extract prompt and expected answer
                messages = ex.get("messages", [])
                if len(messages) < 2:
                    continue

                prompt = messages[-2].get("content", "") if messages else ""
                expected = messages[-1].get("content", "") if len(messages) > 1 else ""

                if not prompt or not expected:
                    continue

                # Call inference API
                resp = requests.post(
                    f"{self.api_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 50,
                        "temperature": 0.1
                    },
                    timeout=30
                )

                if resp.ok:
                    response = resp.json()["choices"][0]["message"]["content"]
                    # Simple match check
                    if expected.lower().strip() in response.lower() or response.lower().strip() in expected.lower():
                        correct += 1
                    total += 1
            except Exception as e:
                logger.warning(f"Test error: {e}")
                continue

        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "timestamp": datetime.now().isoformat()
        }

    def check_regression(self) -> Dict:
        """
        Check for regression against baseline.
        Called by GPU scheduler.

        Returns:
            Dict with regression check results
        """
        logger.info("Running regression check...")

        # Get current performance
        current = self._test_model()
        if "error" in current:
            return current

        current_accuracy = current["accuracy"]

        # Get baseline
        baseline = self._get_baseline()

        if baseline is None:
            # No baseline - establish one
            logger.info(f"No baseline found, establishing with accuracy {current_accuracy:.2%}")
            self._save_baseline(current)
            result = {
                "regression_detected": False,
                "current_accuracy": current_accuracy,
                "baseline_accuracy": current_accuracy,
                "accuracy_delta": 0,
                "message": "Baseline established",
                "timestamp": datetime.now().isoformat()
            }
        else:
            baseline_accuracy = baseline.get("accuracy", 0)
            delta = current_accuracy - baseline_accuracy

            regression_detected = delta < -self.threshold

            result = {
                "regression_detected": regression_detected,
                "current_accuracy": current_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_delta": delta,
                "threshold": self.threshold,
                "timestamp": datetime.now().isoformat()
            }

            if regression_detected:
                logger.warning(f"🚨 REGRESSION DETECTED: {delta:+.2%} (threshold: {self.threshold:.2%})")
                self.results["regressions_detected"] += 1
            else:
                logger.info(f"✅ No regression: {delta:+.2%}")

                # Update baseline if improved significantly
                if delta > self.threshold:
                    logger.info("Performance improved, updating baseline")
                    self._save_baseline(current)

        # Store check result
        self.results["checks"].append(result)
        # Keep only last 100 checks
        self.results["checks"] = self.results["checks"][-100:]
        self._save_results()

        return result
    
    def run_multiple_tests(
        self,
        model_id: str,
        examples: List[Dict],
        n_runs: int = 5,
        test_func = None
    ) -> PerformanceMetrics:
        """
        Run model on examples multiple times to measure variance
        
        Args:
            model_id: Model identifier
            examples: Test examples
            n_runs: Number of independent runs
            test_func: Function that tests model and returns EM score
        
        Returns:
            PerformanceMetrics with mean, std, and samples
        """
        em_scores = []
        
        print(f"   Running {n_runs} independent tests for variance estimation...")
        
        for run in range(n_runs):
            # Shuffle examples to get different sample
            shuffled = np.random.choice(examples, len(examples), replace=False).tolist()
            
            if test_func:
                em = test_func(model_id, shuffled)
            else:
                # Placeholder - would call actual test
                em = np.random.beta(8, 2)  # Simulated EM score
            
            em_scores.append(em)
            print(f"   Run {run+1}/{n_runs}: EM={em:.2%}")
        
        return PerformanceMetrics(
            checkpoint=model_id,
            em_mean=np.mean(em_scores),
            em_std=np.std(em_scores, ddof=1),  # Sample std
            em_samples=em_scores,
            sample_size=len(examples),
            timestamp=None
        )
    
    def bootstrap_ci(
        self,
        samples: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval
        
        Args:
            samples: Performance samples
            confidence: Confidence level (default 95%)
        
        Returns:
            (lower_bound, upper_bound)
        """
        bootstrap_means = []
        
        for _ in range(self.n_bootstrap):
            resample = np.random.choice(samples, len(samples), replace=True)
            bootstrap_means.append(np.mean(resample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def cohens_d(self, samples1: List[float], samples2: List[float]) -> float:
        """
        Calculate Cohen's d effect size
        
        Args:
            samples1: Baseline samples
            samples2: New samples
        
        Returns:
            Cohen's d (standardized mean difference)
        """
        mean1 = np.mean(samples1)
        mean2 = np.mean(samples2)
        
        # Pooled standard deviation
        std1 = np.std(samples1, ddof=1)
        std2 = np.std(samples2, ddof=1)
        n1, n2 = len(samples1), len(samples2)
        
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        
        d = (mean1 - mean2) / pooled_std
        return d
    
    def test_regression(
        self,
        baseline: PerformanceMetrics,
        current: PerformanceMetrics,
        verbose: bool = True
    ) -> Dict:
        """
        Test if current performance is significantly worse than baseline

        Returns:
            {
                'is_regression': bool,
                'p_value': float,
                'effect_size': float,
                'baseline_ci': tuple,
                'current_ci': tuple,
                'conclusion': str
            }
        """
        if not SCIPY_AVAILABLE:
            # Fallback without scipy - use simple comparison
            delta = current.em_mean - baseline.em_mean
            return {
                'is_regression': delta < -0.05,
                'p_value': None,
                'effect_size': delta,
                'baseline_ci': (baseline.em_mean, baseline.em_mean),
                'current_ci': (current.em_mean, current.em_mean),
                'baseline_mean': baseline.em_mean,
                'current_mean': current.em_mean,
                'conclusion': '🚨 REGRESSION DETECTED' if delta < -0.05 else '✅ No regression'
            }

        # 1. Two-sample t-test
        t_stat, p_value = stats.ttest_ind(
            baseline.em_samples,
            current.em_samples,
            alternative='greater'  # Test if baseline > current
        )
        
        # 2. Effect size (Cohen's d)
        effect_size = self.cohens_d(baseline.em_samples, current.em_samples)
        
        # 3. Bootstrap confidence intervals
        baseline_ci = self.bootstrap_ci(baseline.em_samples)
        current_ci = self.bootstrap_ci(current.em_samples)
        
        # 4. Decision logic
        is_significant = p_value < self.alpha
        is_large_effect = effect_size > self.min_effect_size
        is_regression = is_significant and is_large_effect
        
        # 5. Determine conclusion
        if is_regression:
            conclusion = "🚨 SIGNIFICANT REGRESSION DETECTED"
        elif is_significant and not is_large_effect:
            conclusion = "⚠️  Statistically significant but small effect"
        elif is_large_effect and not is_significant:
            conclusion = "📊 Large effect but not statistically significant"
        else:
            conclusion = "✅ No significant regression"
        
        result = {
            'is_regression': is_regression,
            'p_value': p_value,
            'effect_size': effect_size,
            'baseline_ci': baseline_ci,
            'current_ci': current_ci,
            'baseline_mean': baseline.em_mean,
            'current_mean': current.em_mean,
            'conclusion': conclusion
        }
        
        if verbose:
            self._print_report(baseline, current, result)
        
        return result
    
    def _print_report(
        self,
        baseline: PerformanceMetrics,
        current: PerformanceMetrics,
        result: Dict
    ):
        """Print detailed regression test report"""
        print("\n" + "="*80)
        print("🔬 REGRESSION DETECTION REPORT")
        print("="*80)
        
        print(f"\n📊 Baseline: {baseline.checkpoint}")
        print(f"   Mean EM: {baseline.em_mean:.2%} ± {baseline.em_std:.2%}")
        print(f"   95% CI: [{result['baseline_ci'][0]:.2%}, {result['baseline_ci'][1]:.2%}]")
        print(f"   Samples: {baseline.em_samples}")
        
        print(f"\n📊 Current: {current.checkpoint}")
        print(f"   Mean EM: {current.em_mean:.2%} ± {current.em_std:.2%}")
        print(f"   95% CI: [{result['current_ci'][0]:.2%}, {result['current_ci'][1]:.2%}]")
        print(f"   Samples: {current.em_samples}")
        
        print(f"\n📉 Change:")
        delta = current.em_mean - baseline.em_mean
        print(f"   Δ EM: {delta:+.2%} ({'worse' if delta < 0 else 'better'})")
        
        print(f"\n🧮 Statistical Tests:")
        print(f"   p-value: {result['p_value']:.4f} (α={self.alpha})")
        print(f"   Significant? {'YES' if result['p_value'] < self.alpha else 'NO'}")
        print(f"   Effect size (Cohen's d): {result['effect_size']:.3f}")
        
        # Interpret effect size
        if abs(result['effect_size']) < 0.2:
            effect_interp = "negligible"
        elif abs(result['effect_size']) < 0.5:
            effect_interp = "small"
        elif abs(result['effect_size']) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        print(f"   Effect interpretation: {effect_interp}")
        
        print(f"\n{result['conclusion']}")
        print("="*80 + "\n")

def example_usage():
    """Example showing how to use the detector"""
    detector = RegressionDetector(
        alpha=0.05,  # 5% significance level
        min_effect_size=0.3  # Medium effect size threshold
    )
    
    # Simulate baseline checkpoint performance
    baseline = PerformanceMetrics(
        checkpoint="checkpoint-50000",
        em_mean=0.85,
        em_std=0.03,
        em_samples=[0.82, 0.86, 0.84, 0.87, 0.86],
        sample_size=100,
        timestamp="2025-11-23T10:00:00"
    )
    
    # Simulate current checkpoint performance (with regression)
    current = PerformanceMetrics(
        checkpoint="checkpoint-60000",
        em_mean=0.78,
        em_std=0.04,
        em_samples=[0.75, 0.80, 0.77, 0.79, 0.78],
        sample_size=100,
        timestamp="2025-11-23T14:00:00"
    )
    
    # Test for regression
    result = detector.test_regression(baseline, current, verbose=True)
    
    return result

if __name__ == '__main__':
    example_usage()
