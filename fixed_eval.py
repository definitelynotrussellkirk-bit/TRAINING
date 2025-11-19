#!/usr/bin/env python3
"""
Fixed Evaluation Set - Ground Truth Signal

Maintains a held-out validation set that is NEVER trained on.
Provides true signal vs noisy streaming loss on training data.

Metrics computed:
- EM (Exact Match): % of perfect matches
- CE (Cross Entropy): Teacher-forced loss (no label smoothing)
- ECE (Expected Calibration Error): Confidence calibration
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

import torch
from transformers import AutoTokenizer


class FixedEvalSet:
    """Manages a fixed validation set for ground truth performance measurement."""

    def __init__(self, eval_file: Path, tokenizer, max_samples=2000, device='cuda'):
        """
        Load fixed eval set from JSONL.

        Args:
            eval_file: Path to .jsonl file with evaluation examples
            tokenizer: HuggingFace tokenizer
            max_samples: Maximum number of examples to load
            device: Device to run evaluation on
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.device = device

        # Tracking for best checkpoint
        self.best_em = 0.0
        self.best_checkpoint = None
        self.eval_history = []  # List of {step, em, ce, ece}

        # Load examples from file
        if not eval_file.exists():
            print(f"⚠️  Fixed eval file not found: {eval_file}")
            print(f"    Create one to get ground truth metrics!")
            return

        with open(eval_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    example = json.loads(line)
                    if 'messages' in example:
                        self.examples.append(example)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON at line {i+1}")

        print(f"📊 Loaded {len(self.examples)} fixed eval examples from {eval_file.name}")

    def evaluate(self, model, current_step: int) -> Dict:
        """
        Run full evaluation: EM, CE, ECE.

        Args:
            model: The model to evaluate
            current_step: Current global step (for tracking)

        Returns:
            Dict with keys: em, ce, ece, best_ckpt, em_trend, ece_rise
        """
        if len(self.examples) == 0:
            return self._empty_result()

        model.eval()
        results = {
            'correct': 0,
            'total': len(self.examples),
            'ce_sum': 0.0,
            'token_probs': [],  # For ECE calculation
            'token_correct': []  # For ECE calculation
        }

        print(f"\n{'='*80}")
        print(f"🎯 FIXED EVAL - Step {current_step:,}")
        print(f"{'='*80}")

        with torch.no_grad():
            for i, example in enumerate(self.examples):
                if i % 100 == 0 and i > 0:
                    print(f"   Progress: {i}/{len(self.examples)} examples...")

                messages = example['messages']
                user_msg = next((m['content'] for m in messages if m['role'] == 'user'), None)
                golden_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), None)

                if not user_msg or not golden_msg:
                    continue

                try:
                    # 1. Generate (greedy for EM)
                    prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_msg}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,  # Shorter for eval speed
                        temperature=0.0,  # Greedy
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    generated = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()

                    # 2. Exact Match
                    exact_match = (generated == golden_msg.strip())
                    results['correct'] += int(exact_match)

                    # 3. Cross-Entropy (teacher-forced)
                    full_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                    loss_outputs = model(**full_inputs, labels=full_inputs['input_ids'])
                    results['ce_sum'] += loss_outputs.loss.item()

                    # 4. Token-level probabilities for ECE
                    logits = loss_outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    max_probs = probs.max(dim=-1).values.cpu().numpy()
                    target_ids = full_inputs['input_ids'][0].cpu().numpy()
                    pred_ids = logits.argmax(dim=-1)[0].cpu().numpy()

                    # Store for ECE calculation
                    for prob, target, pred in zip(max_probs, target_ids[1:], pred_ids[:-1]):
                        results['token_probs'].append(prob)
                        results['token_correct'].append(int(target == pred))

                except Exception as e:
                    print(f"Warning: Eval failed for example {i}: {e}")
                    continue

        # Compute aggregate metrics
        em = results['correct'] / results['total']
        ce = results['ce_sum'] / results['total']
        ece = self._compute_ece(results['token_probs'], results['token_correct'])

        # Track best checkpoint
        if em > self.best_em:
            self.best_em = em
            self.best_checkpoint = f"checkpoint-{current_step}"
            print(f"🏆 NEW BEST MODEL: EM={em*100:.1f}% at step {current_step}")

        # Add to history
        self.eval_history.append({
            'step': current_step,
            'em': em,
            'ce': ce,
            'ece': ece
        })

        # Analyze trends
        em_trend = self._get_em_trend()
        ece_rise = self._check_ece_rise()

        model.train()  # Back to training mode

        result = {
            'em': em,
            'ce': ce,
            'ece': ece,
            'best_ckpt': self.best_checkpoint,
            'em_trend': em_trend,
            'ece_rise': ece_rise
        }

        print(f"📊 Results:")
        print(f"   EM:  {em*100:.1f}% ({results['correct']}/{results['total']})")
        print(f"   CE:  {ce:.3f}")
        print(f"   ECE: {ece:.3f}")
        print(f"   Best: {self.best_checkpoint} (EM={self.best_em*100:.1f}%)")
        print(f"   Trend: {em_trend}")
        print(f"{'='*80}\n")

        return result

    def _compute_ece(self, probs: List[float], correct: List[int], n_bins=10) -> float:
        """
        Compute Expected Calibration Error.

        ECE measures how well confidence matches accuracy.
        Low ECE = model knows when it's right/wrong.
        """
        if len(probs) == 0:
            return 0.0

        probs = np.array(probs)
        correct = np.array(correct)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue

            bin_acc = correct[mask].mean()
            bin_conf = probs[mask].mean()
            bin_size = mask.sum() / len(probs)

            ece += bin_size * abs(bin_acc - bin_conf)

        return ece

    def _get_em_trend(self) -> str:
        """
        Analyze EM trend over last 3 evaluations.

        Returns: 'improving', 'plateau', or 'regressing'
        """
        if len(self.eval_history) < 2:
            return 'improving'  # Not enough data

        recent = self.eval_history[-3:]
        if len(recent) < 3:
            # Just compare last 2
            if recent[-1]['em'] > recent[-2]['em'] + 0.005:  # +0.5%
                return 'improving'
            elif recent[-1]['em'] < recent[-2]['em'] - 0.005:
                return 'regressing'
            else:
                return 'plateau'

        # Compare recent average vs older
        recent_avg = np.mean([x['em'] for x in recent])
        older_avg = np.mean([x['em'] for x in self.eval_history[-6:-3]]) if len(self.eval_history) >= 6 else recent_avg

        improvement = (recent_avg - older_avg)

        if improvement > 0.01:  # > 1% improvement
            return 'improving'
        elif improvement < -0.01:  # > 1% regression
            return 'regressing'
        else:
            return 'plateau'

    def _check_ece_rise(self) -> bool:
        """Check if ECE has risen significantly (calibration degrading)."""
        if len(self.eval_history) < 2:
            return False

        current_ece = self.eval_history[-1]['ece']
        prev_ece = self.eval_history[-2]['ece']

        # ECE increased by more than 20%?
        if current_ece > prev_ece * 1.2 and current_ece > 0.15:
            return True

        return False

    def _empty_result(self) -> Dict:
        """Return empty result when no eval data."""
        return {
            'em': 0.0,
            'ce': 0.0,
            'ece': 0.0,
            'best_ckpt': None,
            'em_trend': 'unknown',
            'ece_rise': False
        }


def create_fixed_eval_from_train(train_file: Path, output_file: Path, n_samples=2000):
    """
    Helper: Create fixed eval set by sampling from training data.

    WARNING: This samples FROM the training data, so it's not truly held-out!
    Better to use completely separate examples.

    Args:
        train_file: Path to training .jsonl
        output_file: Where to save fixed eval set
        n_samples: Number of samples to extract
    """
    import random

    examples = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except:
                pass

    # Random sample
    if len(examples) > n_samples:
        selected = random.sample(examples, n_samples)
    else:
        selected = examples

    # Write to output
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in selected:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✅ Created fixed eval set: {output_file}")
    print(f"   {len(selected)} examples")


if __name__ == '__main__':
    # Example usage
    print("Fixed Eval Set - Example Usage")
    print("="*80)
    print("\nTo use in training:")
    print("1. Create fixed_eval.jsonl with held-out examples")
    print("2. In train.py:")
    print("   from fixed_eval import FixedEvalSet")
    print("   fixed_eval = FixedEvalSet('fixed_eval.jsonl', tokenizer)")
    print("   # Every 500 steps:")
    print("   results = fixed_eval.evaluate(model, current_step)")
    print("   status_writer.update_fixed_eval(results)")
    print("\n" + "="*80)
