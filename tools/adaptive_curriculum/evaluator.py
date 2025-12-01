#!/usr/bin/env python3
"""Evaluation harness for testing model on generated data."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable
from urllib import request, error


@dataclass
class EvalExample:
    """Single evaluation example."""
    input_text: str
    expected_output: str
    metadata: Dict


@dataclass
class EvalBatch:
    """Batch of examples for evaluation."""
    generator_id: str
    difficulty_level: int
    examples: List[EvalExample]
    toggles: Dict[str, any]


class ModelEvaluator:
    """Evaluates model performance on test sets.

    Supports both local model inference and remote API calls.
    """

    def __init__(self, inference_url: str = "http://192.168.x.x:8000  # TODO: Use core.hosts.get_service_url("inference")/generate",
                 timeout: int = 300):
        """Initialize evaluator.

        Args:
            inference_url: URL for model inference endpoint
            timeout: Request timeout in seconds
        """
        self.inference_url = inference_url
        self.timeout = timeout

    def evaluate_batch(self, batch: EvalBatch,
                       custom_judge: Optional[Callable[[str, str], bool]] = None) -> Dict:
        """Evaluate model on a batch of examples.

        Args:
            batch: Batch of examples to evaluate
            custom_judge: Optional function(prediction, expected) -> is_correct
                         If None, uses exact string match

        Returns:
            Dict with keys: correct_count, total_count, accuracy, predictions
        """
        predictions = []
        correct_count = 0

        for example in batch.examples:
            # Get model prediction
            prediction = self._call_model(example.input_text)

            # Judge correctness
            if custom_judge:
                is_correct = custom_judge(prediction, example.expected_output)
            else:
                is_correct = self._exact_match(prediction, example.expected_output)

            predictions.append({
                "input": example.input_text,
                "expected": example.expected_output,
                "predicted": prediction,
                "correct": is_correct,
                "metadata": example.metadata
            })

            if is_correct:
                correct_count += 1

        total_count = len(batch.examples)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        return {
            "generator_id": batch.generator_id,
            "difficulty_level": batch.difficulty_level,
            "correct_count": correct_count,
            "total_count": total_count,
            "accuracy": accuracy,
            "toggles": batch.toggles,
            "predictions": predictions
        }

    def _call_model(self, prompt: str) -> str:
        """Call model inference API.

        Args:
            prompt: Input prompt

        Returns:
            Model's generated text
        """
        payload = {
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 0.1,  # Low temp for evaluation
            "stop": None
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        req = request.Request(
            self.inference_url,
            data=data,
            headers=headers,
            method="POST"
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                result = json.loads(body)

                # Extract text from various response formats
                if isinstance(result, dict):
                    return result.get("text", result.get("response", str(result)))
                return str(result)

        except error.URLError as exc:
            raise RuntimeError(f"Inference API call failed: {exc}") from exc

    @staticmethod
    def _exact_match(prediction: str, expected: str) -> bool:
        """Default judge: exact string match (case-insensitive, stripped)."""
        return prediction.strip().lower() == expected.strip().lower()

    @staticmethod
    def json_judge(prediction: str, expected: str) -> bool:
        """Judge for JSON outputs: parse and compare."""
        try:
            pred_obj = json.loads(prediction)
            exp_obj = json.loads(expected)
            return pred_obj == exp_obj
        except json.JSONDecodeError:
            return False

    @staticmethod
    def contains_judge(prediction: str, expected: str) -> bool:
        """Judge: expected is substring of prediction."""
        return expected.lower() in prediction.lower()


class EvalSetBuilder:
    """Builds evaluation sets from generator outputs."""

    def __init__(self, eval_sets_dir: Path):
        """Initialize builder.

        Args:
            eval_sets_dir: Directory to store eval sets
        """
        self.eval_sets_dir = eval_sets_dir
        self.eval_sets_dir.mkdir(parents=True, exist_ok=True)

    def create_eval_set(self, generator_id: str, difficulty: int,
                        training_file: Path, eval_size: int = 100) -> Path:
        """Create evaluation set by sampling from training file.

        Args:
            generator_id: Generator identifier
            difficulty: Difficulty level
            training_file: Path to training JSONL file
            eval_size: Number of examples to sample

        Returns:
            Path to created eval set file
        """
        # Load training examples
        examples = []
        with training_file.open("r") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        # Sample for eval set
        if len(examples) > eval_size:
            sampled = random.sample(examples, eval_size)
        else:
            sampled = examples

        # Save eval set
        eval_filename = f"{generator_id}_diff{difficulty}_eval.jsonl"
        eval_path = self.eval_sets_dir / eval_filename

        with eval_path.open("w") as f:
            for example in sampled:
                f.write(json.dumps(example) + "\n")

        return eval_path

    def load_eval_batch(self, eval_file: Path) -> EvalBatch:
        """Load evaluation batch from file.

        Expects JSONL with format:
        {"messages": [...], "metadata": {"generator_id": ..., "difficulty_level": ..., "toggles": ...}}

        Returns:
            EvalBatch ready for evaluation
        """
        examples = []
        metadata = None

        with eval_file.open("r") as f:
            for line in f:
                if not line.strip():
                    continue

                entry = json.loads(line)
                messages = entry.get("messages", [])

                # Extract input/output from messages
                input_text = None
                expected_output = None
                for msg in messages:
                    if msg.get("role") == "user":
                        input_text = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        expected_output = msg.get("content", "")

                if input_text and expected_output:
                    examples.append(EvalExample(
                        input_text=input_text,
                        expected_output=expected_output,
                        metadata=entry.get("metadata", {})
                    ))

                # Extract batch metadata from first example
                if metadata is None:
                    metadata = entry.get("metadata", {})

        if not examples:
            raise ValueError(f"No valid examples found in {eval_file}")

        generator_id = metadata.get("generator_id", "unknown")
        difficulty_level = metadata.get("difficulty_level", 0)
        toggles = metadata.get("toggles", {})

        return EvalBatch(
            generator_id=generator_id,
            difficulty_level=difficulty_level,
            examples=examples,
            toggles=toggles
        )


class PeriodicEvaluator:
    """Runs periodic evaluations during training."""

    def __init__(self, evaluator: ModelEvaluator,
                 eval_sets_dir: Path,
                 results_dir: Path):
        """Initialize periodic evaluator.

        Args:
            evaluator: Model evaluator instance
            eval_sets_dir: Directory containing eval sets
            results_dir: Directory to store results
        """
        self.evaluator = evaluator
        self.eval_sets_dir = eval_sets_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_all_evals(self, custom_judge: Optional[Callable] = None) -> List[Dict]:
        """Run evaluation on all available eval sets.

        Args:
            custom_judge: Optional custom judging function

        Returns:
            List of evaluation results
        """
        results = []

        # Find all eval sets
        eval_files = list(self.eval_sets_dir.glob("*_eval.jsonl"))

        for eval_file in eval_files:
            try:
                # Load batch
                builder = EvalSetBuilder(self.eval_sets_dir)
                batch = builder.load_eval_batch(eval_file)

                # Evaluate
                result = self.evaluator.evaluate_batch(batch, custom_judge)
                result["eval_file"] = str(eval_file)

                results.append(result)

            except Exception as exc:
                print(f"Error evaluating {eval_file}: {exc}")
                continue

        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"eval_results_{timestamp}.json"

        with results_file.open("w") as f:
            json.dump({
                "timestamp": timestamp,
                "results": results
            }, f, indent=2)

        return results
