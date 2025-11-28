#!/usr/bin/env python3
"""
Prediction Viewer Engine - Interactive model prediction inspection

Generates predictions with full checkpoint tracking, semantic analysis,
and human grading support.

Extends preview_engine.py with:
- Checkpoint tracking (path + step)
- Semantic match detection (thinking tags, format)
- Difficulty detection (easy/medium/hard from word count)
- Human grading API support
- Contract-compliant JSON format
"""

import json
import random
import time
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib import request, error as urllib_error
from datetime import datetime


class PredictionViewerEngine:
    """
    Enhanced prediction generator for interactive viewing

    Features:
    - Full checkpoint tracking (path, step, timestamp)
    - Semantic analysis (thinking tags, format, structure)
    - Difficulty detection from puzzle complexity
    - Human grading support
    - Contract-compliant storage format
    """

    def __init__(
        self,
        base_dir: Path = None,
        remote_api_url: str = None,
        validation_file: str = "data/validation/syllo_validation_20.jsonl",
        prediction_count: int = 10,
        max_tokens: int = 2048,
        temperature: float = 0.1
    ):
        if base_dir is None:
            from core.paths import require_base_dir
            base_dir = require_base_dir()
        if remote_api_url is None:
            try:
                from core.hosts import get_service_url
                remote_api_url = get_service_url("inference")
            except (ImportError, Exception):
                remote_api_url = "http://192.168.x.x:8765"
        self.base_dir = Path(base_dir)
        self.remote_api_url = remote_api_url
        self.validation_file = self.base_dir / validation_file
        self.prediction_count = prediction_count
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Storage
        self.predictions_file = self.base_dir / "status" / "latest_predictions.json"
        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)

        self.gradings_file = self.base_dir / "status" / "prediction_gradings.json"

        # History
        self.history_dir = self.base_dir / "data" / "prediction_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

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

    def _detect_checkpoint(self) -> Dict[str, Any]:
        """
        Detect current checkpoint being used by 3090

        Returns checkpoint info: {path, step, training_step, last_updated, model_name, status}
        """
        # Check for checkpoint info from training status
        status_file = self.base_dir / "status" / "training_status.json"

        checkpoint_info = {
            "path": str(self.base_dir / "current_model"),
            "step": None,
            "training_step": None,
            "last_updated": datetime.now().isoformat(),
            "model_name": "Qwen3-0.6B",
            "status": "unknown"
        }

        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                    current_step = status.get("current_step")
                    checkpoint_info["step"] = current_step
                    checkpoint_info["training_step"] = current_step
                    checkpoint_info["status"] = "active" if status.get("status") == "training" else "idle"
            except Exception:
                pass

        # Try to detect step from checkpoint directory
        checkpoint_dir = self.base_dir / "current_model"
        if checkpoint_dir.exists():
            # Look for checkpoint-NNNN directories
            checkpoint_dirs = list(checkpoint_dir.glob("checkpoint-*"))
            if checkpoint_dirs:
                # Get latest checkpoint
                latest = sorted(checkpoint_dirs)[-1]
                try:
                    step = int(latest.name.split('-')[1])
                    if checkpoint_info["step"] is None:
                        checkpoint_info["step"] = step
                        checkpoint_info["training_step"] = step
                except (IndexError, ValueError):
                    pass

        return checkpoint_info

    def _detect_difficulty(self, prompt: str) -> str:
        """
        Detect puzzle difficulty from word count

        Rules:
        - Easy: 4 words in question
        - Medium: 5-6 words
        - Hard: 7+ words
        """
        # Extract the question (last part after "Solve:" or similar)
        lines = prompt.strip().split('\n')
        question_line = None

        for line in reversed(lines):
            if line.strip() and not line.startswith('#'):
                question_line = line.strip()
                break

        if not question_line:
            return "unknown"

        # Count words
        words = question_line.split()
        word_count = len(words)

        if word_count <= 4:
            return "easy"
        elif word_count <= 6:
            return "medium"
        else:
            return "hard"

    def _has_thinking_tags(self, text: str) -> bool:
        """Check if output contains <think></think> tags"""
        return bool(re.search(r'<think>.*?</think>', text, re.DOTALL | re.IGNORECASE))

    def _check_semantic_match(self, generated: str, expected: str) -> bool:
        """
        Check semantic match - output is well-formed even if answer is wrong

        Criteria:
        - Contains valid JSON structure
        - Has 'solutions' array
        - Each solution has 'question' and 'answer' fields
        - Format matches expected structure
        """
        try:
            # Extract JSON from generated text
            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if not json_match:
                return False

            generated_json = json.loads(json_match.group(0))
            expected_json = json.loads(expected)

            # Check structure
            if 'solutions' not in generated_json:
                return False

            gen_solutions = generated_json['solutions']
            exp_solutions = expected_json['solutions']

            # Must have same number of questions
            if len(gen_solutions) != len(exp_solutions):
                return False

            # Each solution must have required fields
            for sol in gen_solutions:
                if 'question' not in sol or 'answer' not in sol:
                    return False

            return True

        except Exception:
            return False

    def _calculate_exact_match(self, generated: str, expected: str) -> bool:
        """
        Calculate exact match - answers are correct

        Compares solution answers
        """
        try:
            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if not json_match:
                return False

            generated_json = json.loads(json_match.group(0))
            expected_json = json.loads(expected)

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

    def generate_predictions(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_step: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions with full metadata

        Args:
            checkpoint_path: Optional override for checkpoint path
            checkpoint_step: Optional override for checkpoint step

        Returns:
            Full prediction data matching contract spec
        """
        if not self.validation_data:
            return {
                'error': 'No validation data loaded',
                'timestamp': datetime.now().isoformat()
            }

        # Detect checkpoint
        checkpoint_info = self._detect_checkpoint()
        if checkpoint_path:
            checkpoint_info["path"] = checkpoint_path
        if checkpoint_step is not None:
            checkpoint_info["step"] = checkpoint_step

        print(f"\n  🔍 Generating predictions...")
        print(f"     Checkpoint: {checkpoint_info['path']}")
        print(f"     Step: {checkpoint_info['step']}")

        # Sample random examples
        samples = random.sample(
            self.validation_data,
            min(self.prediction_count, len(self.validation_data))
        )

        predictions = []
        stats = {
            "total": 0,
            "exact_match": 0,
            "semantic_match": 0,
            "has_thinking": 0,
            "by_difficulty": {
                "easy": {"total": 0, "correct": 0},
                "medium": {"total": 0, "correct": 0},
                "hard": {"total": 0, "correct": 0}
            }
        }

        for i, sample in enumerate(samples, 1):
            messages = sample['messages']
            user_content = messages[0]['content']
            expected_output = messages[1]['content']

            # Get metadata
            metadata = sample.get('metadata', {})
            puzzle_id = metadata.get('puzzle_id', f'puzzle_{i}')

            # Detect difficulty
            difficulty = self._detect_difficulty(user_content)

            print(f"     [{i}/{len(samples)}] {puzzle_id} ({difficulty})")

            # Call inference API
            response = self._call_3090_api([{"role": "user", "content": user_content}])

            if response['error']:
                print(f"       ⚠ API error: {response['error']}")
                continue

            generated = response['generated_text']
            inference_time = response['inference_time_ms']

            # Analyze prediction
            exact_match = self._calculate_exact_match(generated, expected_output)
            semantic_match = self._check_semantic_match(generated, expected_output)
            has_thinking = self._has_thinking_tags(generated)

            # Update stats
            stats["total"] += 1
            if exact_match:
                stats["exact_match"] += 1
            if semantic_match:
                stats["semantic_match"] += 1
            if has_thinking:
                stats["has_thinking"] += 1

            stats["by_difficulty"][difficulty]["total"] += 1
            if exact_match:
                stats["by_difficulty"][difficulty]["correct"] += 1

            # Create prediction entry matching contract spec
            pred_timestamp = datetime.now()
            prediction = {
                "id": f"pred_{pred_timestamp.strftime('%Y%m%d_%H%M%S')}_{i:03d}",
                "difficulty": difficulty,
                "prompt": user_content,
                "expected_answer": expected_output,
                "model_output": generated,
                "extracted_answer": None,  # Could extract JSON structure here
                "metrics": {
                    "exact_match": exact_match,
                    "semantic_match": semantic_match,
                    "has_thinking_tags": has_thinking,
                    "format_valid": semantic_match,  # If semantic match works, format is valid
                    "completion_time_ms": int(inference_time)
                },
                "manual_grade": None,
                "timestamp": pred_timestamp.isoformat()
            }
            predictions.append(prediction)

            # Print status
            em_status = '✓' if exact_match else '✗'
            sm_status = '✓' if semantic_match else '✗'
            think_status = '💭' if has_thinking else '  '
            print(f"       {em_status} EM:{exact_match} SM:{semantic_match} {think_status} ({inference_time:.0f}ms)")

        # Calculate accuracy rates
        accuracy_auto = stats["exact_match"] / stats["total"] if stats["total"] > 0 else 0.0
        semantic_rate = stats["semantic_match"] / stats["total"] if stats["total"] > 0 else 0.0
        thinking_rate = stats["has_thinking"] / stats["total"] if stats["total"] > 0 else 0.0

        # Build full output
        output = {
            "checkpoint": checkpoint_info,
            "predictions": predictions,
            "stats": {
                "total": stats["total"],
                "accuracy_auto": accuracy_auto,
                "accuracy_human": None,  # Updated when graded
                "semantic_match_rate": semantic_rate,
                "thinking_tag_rate": thinking_rate,
                "by_difficulty": stats["by_difficulty"]
            },
            "generated_at": datetime.now().isoformat()
        }

        # Save to latest predictions
        with open(self.predictions_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        # Save to history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = self.history_dir / f"predictions_{timestamp}.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        print(f"\n  ✓ Generated {stats['total']} predictions")
        print(f"     Accuracy (auto): {accuracy_auto:.1%}")
        print(f"     Semantic match: {semantic_rate:.1%}")
        print(f"     With thinking: {thinking_rate:.1%}")

        return output

    def get_latest_predictions(self) -> Optional[Dict[str, Any]]:
        """Get latest predictions"""
        if not self.predictions_file.exists():
            return None

        with open(self.predictions_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_grading(self, prediction_id: str, grade: str, notes: str = ""):
        """
        Save human grading for a prediction

        Args:
            prediction_id: Prediction ID
            grade: "correct", "incorrect", or "unsure"
            notes: Optional grading notes
        """
        # Load current predictions
        predictions_data = self.get_latest_predictions()
        if not predictions_data:
            return

        # Find and update prediction
        updated = False
        for pred in predictions_data["predictions"]:
            if pred["id"] == prediction_id:
                pred["human_grade"] = grade
                pred["human_grade_notes"] = notes
                pred["graded_at"] = datetime.now().isoformat()
                updated = True
                break

        if not updated:
            return

        # Recalculate human accuracy
        graded = [p for p in predictions_data["predictions"] if p["human_grade"] is not None]
        if graded:
            correct = sum(1 for p in graded if p["human_grade"] == "correct")
            predictions_data["stats"]["accuracy_human"] = correct / len(graded)

        # Save updated predictions
        with open(self.predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, indent=2)

        # Append to gradings log
        gradings = []
        if self.gradings_file.exists():
            with open(self.gradings_file, 'r') as f:
                gradings = json.load(f)

        gradings.append({
            "prediction_id": prediction_id,
            "grade": grade,
            "notes": notes,
            "graded_at": datetime.now().isoformat(),
            "checkpoint_step": predictions_data["checkpoint"]["step"]
        })

        with open(self.gradings_file, 'w', encoding='utf-8') as f:
            json.dump(gradings, f, indent=2)

        print(f"  ✓ Saved grading for {prediction_id}: {grade}")


def main():
    """CLI for prediction viewer engine"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate predictions for interactive viewing')
    parser.add_argument('--base-dir', default=None, help='Base directory (default: auto-detected)')
    parser.add_argument('--count', type=int, default=10, help='Number of predictions to generate')
    parser.add_argument('--checkpoint-path', help='Override checkpoint path')
    parser.add_argument('--checkpoint-step', type=int, help='Override checkpoint step')
    parser.add_argument('--show-latest', action='store_true', help='Show latest predictions')

    args = parser.parse_args()

    engine = PredictionViewerEngine(
        base_dir=args.base_dir,
        prediction_count=args.count
    )

    if args.show_latest:
        latest = engine.get_latest_predictions()
        if not latest:
            print("No predictions found")
            return

        print(f"\nLatest Predictions:")
        print(f"  Checkpoint: {latest['checkpoint']['path']}")
        print(f"  Step: {latest['checkpoint']['step']}")
        print(f"  Generated: {latest['generated_at']}")
        print(f"  Total: {latest['stats']['total']}")
        print(f"  Accuracy (auto): {latest['stats']['accuracy_auto']:.1%}")
        if latest['stats']['accuracy_human'] is not None:
            print(f"  Accuracy (human): {latest['stats']['accuracy_human']:.1%}")
        return

    # Generate predictions
    result = engine.generate_predictions(
        checkpoint_path=args.checkpoint_path,
        checkpoint_step=args.checkpoint_step
    )

    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Predictions Generated")
    print(f"{'='*60}")
    print(f"Saved to: {engine.predictions_file}")
    print(f"View at: http://localhost:8080/ui/predictions.html")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
