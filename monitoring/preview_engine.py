#!/usr/bin/env python3
"""
Live Preview Engine - Periodic model testing during training

Runs inference on validation samples via 3090 API and tracks quality metrics.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib import request, error as urllib_error
from datetime import datetime
import re


class PreviewEngine:
    """
    Manages live preview inference during training

    Features:
    - Loads validation data from data/validation/
    - Sends prompts to 3090 API for inference
    - Calculates Exact Match (EM) and metrics
    - Stores preview history with timestamps
    - Provides latest preview for control room display
    """

    def __init__(
        self,
        base_dir: Path,
        remote_api_url: str = "http://192.168.x.x:8765",
        validation_file: str = "data/validation/syllo_validation_20.jsonl",
        preview_count: int = 5,
        max_tokens: int = 2048,
        temperature: float = 0.1
    ):
        self.base_dir = Path(base_dir)
        self.remote_api_url = remote_api_url
        self.validation_file = self.base_dir / validation_file
        self.preview_count = preview_count
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Storage
        self.preview_history_dir = self.base_dir / "data" / "preview_history"
        self.preview_history_dir.mkdir(parents=True, exist_ok=True)

        self.latest_preview_file = self.base_dir / "status" / "latest_preview.json"
        self.latest_preview_file.parent.mkdir(parents=True, exist_ok=True)

        # Load validation data
        self.validation_data: List[Dict] = []
        self._load_validation_data()

    def _load_validation_data(self):
        """Load validation dataset"""
        if not self.validation_file.exists():
            print(f"  ⚠ Validation file not found: {self.validation_file}")
            return

        with open(self.validation_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.validation_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        print(f"  ✓ Loaded {len(self.validation_data)} validation samples")

    def _call_3090_api(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        Call 3090 inference API

        Returns dict with:
        - generated_text: str
        - inference_time_ms: float
        - error: Optional[str]
        """
        try:
            payload = {
                "model": "Qwen3-0.6B",
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

            req = request.Request(
                f"{self.remote_api_url}/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            start_time = time.time()
            with request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            inference_time = (time.time() - start_time) * 1000

            # Extract text from OpenAI format
            generated_text = result['choices'][0]['message']['content']

            return {
                'generated_text': generated_text,
                'inference_time_ms': inference_time,
                'error': None
            }

        except Exception as e:
            return {
                'generated_text': '',
                'inference_time_ms': 0.0,
                'error': str(e)
            }

    def _calculate_exact_match(self, generated: str, expected: str) -> bool:
        """
        Calculate exact match for SYLLO puzzles

        Extracts JSON from generated text and compares solutions
        """
        try:
            # Extract JSON from generated text (may have extra text)
            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if not json_match:
                return False

            generated_json = json.loads(json_match.group(0))
            expected_json = json.loads(expected)

            # Compare solutions array
            gen_solutions = generated_json.get('solutions', [])
            exp_solutions = expected_json.get('solutions', [])

            if len(gen_solutions) != len(exp_solutions):
                return False

            # Check each answer
            for gen_sol, exp_sol in zip(gen_solutions, exp_solutions):
                if gen_sol.get('answer') != exp_sol.get('answer'):
                    return False

            return True

        except Exception:
            return False

    def run_preview(
        self,
        training_step: int,
        checkpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run preview inference on random validation samples

        Args:
            training_step: Current training step
            checkpoint_id: Optional checkpoint identifier

        Returns:
            Dict with preview results and metrics
        """
        if not self.validation_data:
            return {
                'error': 'No validation data loaded',
                'timestamp': datetime.now().isoformat()
            }

        # Sample random examples
        samples = random.sample(
            self.validation_data,
            min(self.preview_count, len(self.validation_data))
        )

        results = []
        em_scores = []
        inference_times = []

        print(f"\n  🔍 Running preview at step {training_step}...")

        for i, sample in enumerate(samples, 1):
            messages = sample['messages']
            user_content = messages[0]['content']
            expected_output = messages[1]['content']

            # Get metadata
            metadata = sample.get('metadata', {})
            puzzle_id = metadata.get('puzzle_id', 'unknown')
            difficulty = metadata.get('difficulty', 'unknown')

            print(f"     [{i}/{len(samples)}] {puzzle_id} ({difficulty})")

            # Call inference API
            response = self._call_3090_api([{"role": "user", "content": user_content}])

            if response['error']:
                print(f"       ⚠ API error: {response['error']}")
                continue

            generated = response['generated_text']
            inference_time = response['inference_time_ms']

            # Calculate EM
            exact_match = self._calculate_exact_match(generated, expected_output)
            em_scores.append(1.0 if exact_match else 0.0)
            inference_times.append(inference_time)

            # Store result
            result = {
                'puzzle_id': puzzle_id,
                'difficulty': difficulty,
                'prompt': user_content[:200] + '...',  # Truncate for storage
                'expected': expected_output,
                'generated': generated,
                'exact_match': exact_match,
                'inference_time_ms': inference_time
            }
            results.append(result)

            # Print status
            status = '✓' if exact_match else '✗'
            print(f"       {status} EM: {exact_match} ({inference_time:.0f}ms)")

        # Calculate aggregate metrics
        em_rate = sum(em_scores) / len(em_scores) if em_scores else 0.0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0

        preview_data = {
            'training_step': training_step,
            'checkpoint_id': checkpoint_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'exact_match_rate': em_rate,
                'avg_inference_time_ms': avg_inference_time,
                'samples_tested': len(results)
            },
            'results': results
        }

        # Save to history
        history_file = self.preview_history_dir / f"preview_step_{training_step:06d}.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(preview_data, f, indent=2)

        # Update latest preview
        with open(self.latest_preview_file, 'w', encoding='utf-8') as f:
            json.dump(preview_data, f, indent=2)

        print(f"  ✓ Preview complete: EM={em_rate:.1%}, Avg time={avg_inference_time:.0f}ms")

        return preview_data

    def get_latest_preview(self) -> Optional[Dict[str, Any]]:
        """Get latest preview results"""
        if not self.latest_preview_file.exists():
            return None

        with open(self.latest_preview_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_preview_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent preview history"""
        history_files = sorted(self.preview_history_dir.glob("preview_step_*.json"))
        history_files = history_files[-limit:]  # Last N files

        history = []
        for file in history_files:
            with open(file, 'r', encoding='utf-8') as f:
                history.append(json.load(f))

        return history

    def get_em_trend(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get EM trend over recent previews

        Returns list of {step, em_rate, timestamp}
        """
        history = self.get_preview_history(limit=limit)

        trend = []
        for preview in history:
            trend.append({
                'step': preview['training_step'],
                'em_rate': preview['metrics']['exact_match_rate'],
                'timestamp': preview['timestamp']
            })

        return trend


def main():
    """CLI for testing preview engine"""
    import argparse

    parser = argparse.ArgumentParser(description='Run preview inference')
    parser.add_argument('--base-dir', default='/path/to/training', help='Base directory')
    parser.add_argument('--step', type=int, default=0, help='Training step number')
    parser.add_argument('--count', type=int, default=5, help='Number of samples to test')
    parser.add_argument('--show-history', action='store_true', help='Show preview history')
    parser.add_argument('--show-trend', action='store_true', help='Show EM trend')

    args = parser.parse_args()

    engine = PreviewEngine(
        base_dir=args.base_dir,
        preview_count=args.count
    )

    if args.show_history:
        history = engine.get_preview_history(limit=10)
        print(f"\nPreview History ({len(history)} entries):")
        for preview in history:
            step = preview['training_step']
            em = preview['metrics']['exact_match_rate']
            ts = preview['timestamp']
            print(f"  Step {step:6d}: EM={em:.1%}  ({ts})")
        return

    if args.show_trend:
        trend = engine.get_em_trend(limit=20)
        print(f"\nEM Trend ({len(trend)} points):")
        for point in trend:
            print(f"  Step {point['step']:6d}: {point['em_rate']:.1%}")
        return

    # Run preview
    results = engine.run_preview(training_step=args.step)

    if 'error' in results:
        print(f"\n❌ Error: {results['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Preview Results - Step {results['training_step']}")
    print(f"{'='*60}")
    print(f"EM Rate: {results['metrics']['exact_match_rate']:.1%}")
    print(f"Samples: {results['metrics']['samples_tested']}")
    print(f"Avg Time: {results['metrics']['avg_inference_time_ms']:.0f}ms")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
