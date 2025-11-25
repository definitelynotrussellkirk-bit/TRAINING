#!/usr/bin/env python3
"""
Hard Example Tracker - Track model performance on canonical difficult problems.

Maintains a "rogues gallery" of hard examples and tracks:
- When each example gets fixed
- What new failure modes appear
- Error type evolution over training

Usage:
    # Evaluate current checkpoint against hard examples
    python3 hard_example_tracker.py --evaluate

    # Show board (history of ✓/✗ per checkpoint)
    python3 hard_example_tracker.py --show-board

    # Add new hard example
    python3 hard_example_tracker.py --add --prompt "..." --expected "..." --category "..."

Output:
    config/hard_examples.json - Canonical hard examples
    status/hard_example_board.json - Performance history
    status/visualizations/hard_example_board.png - Visual board
"""

import argparse
import json
import logging
import os
import sys
import requests
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HardExample:
    """A canonical hard example for tracking."""
    id: str
    prompt: str
    expected: str
    category: str
    difficulty: str = "hard"
    notes: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of evaluating one hard example."""
    example_id: str
    correct: bool
    model_answer: Optional[str] = None
    error_type: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None


@dataclass
class BoardEntry:
    """One checkpoint's evaluation of all hard examples."""
    checkpoint: str
    step: int
    timestamp: str
    results: Dict[str, bool]  # example_id -> correct
    total_correct: int
    total: int
    accuracy: float
    error_types: Dict[str, int]  # error_type -> count


# Default hard examples for SYLLO reasoning
DEFAULT_HARD_EXAMPLES = [
    HardExample(
        id="negation_basic",
        prompt="No cats are dogs. All pets are cats. Are any pets dogs?",
        expected="No",
        category="negation",
        notes="Basic negation with universal quantifier"
    ),
    HardExample(
        id="double_negation",
        prompt="It is not true that no birds can fly. Can some birds fly?",
        expected="Yes",
        category="double_negation",
        notes="Double negation resolution"
    ),
    HardExample(
        id="quantifier_scope",
        prompt="All dogs bark. Some animals are dogs. Do all animals bark?",
        expected="No",
        category="quantifier",
        notes="Quantifier scope - 'all' vs 'some' confusion"
    ),
    HardExample(
        id="modus_tollens",
        prompt="If it rains, the ground is wet. The ground is not wet. Is it raining?",
        expected="No",
        category="modus_tollens",
        notes="Classic modus tollens"
    ),
    HardExample(
        id="transitivity",
        prompt="All A are B. All B are C. All C are D. Are all A also D?",
        expected="Yes",
        category="transitivity",
        notes="Multi-step transitivity"
    ),
    HardExample(
        id="contradiction_detection",
        prompt="All swans are white. This swan is black. Is this consistent?",
        expected="No",
        category="contradiction",
        notes="Detect logical contradiction"
    ),
    HardExample(
        id="existential_trap",
        prompt="Some birds can fly. Penguins are birds. Can penguins fly?",
        expected="Cannot determine",
        category="existential",
        notes="'Some' doesn't imply 'all' - common trap"
    ),
    HardExample(
        id="negation_chain",
        prompt="No fish are mammals. No mammals are reptiles. Are any fish reptiles?",
        expected="Cannot determine",
        category="negation_chain",
        notes="Negation doesn't chain transitively"
    ),
    HardExample(
        id="hidden_universal",
        prompt="Dogs bark. Rover is a dog. Does Rover bark?",
        expected="Yes",
        category="implicit_quantifier",
        notes="Implicit 'all' in generic statement"
    ),
    HardExample(
        id="vacuous_truth",
        prompt="All unicorns are purple. Are there any unicorns?",
        expected="Cannot determine",
        category="vacuous",
        notes="Statement about empty set"
    ),
]


class HardExampleTracker:
    """Track model performance on canonical hard examples."""

    def __init__(
        self,
        base_dir: str = "/path/to/training",
        api_url: str = "http://192.168.x.x:8765"
    ):
        self.base_dir = Path(base_dir)
        self.api_url = api_url

        # Paths
        self.examples_file = self.base_dir / "config" / "hard_examples.json"
        self.board_file = self.base_dir / "status" / "hard_example_board.json"
        self.viz_dir = self.base_dir / "status" / "visualizations"

        # Ensure directories exist
        self.examples_file.parent.mkdir(parents=True, exist_ok=True)
        self.board_file.parent.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize examples
        self.examples = self._load_examples()
        self.board = self._load_board()

    def _load_examples(self) -> List[HardExample]:
        """Load hard examples from file or use defaults."""
        if self.examples_file.exists():
            with open(self.examples_file) as f:
                data = json.load(f)
                return [HardExample(**e) for e in data]
        else:
            # Save defaults
            self._save_examples(DEFAULT_HARD_EXAMPLES)
            return DEFAULT_HARD_EXAMPLES

    def _save_examples(self, examples: List[HardExample]) -> None:
        """Save hard examples to file."""
        with open(self.examples_file, 'w') as f:
            json.dump([asdict(e) for e in examples], f, indent=2)

    def _load_board(self) -> Dict[str, Any]:
        """Load evaluation board from file."""
        if self.board_file.exists():
            with open(self.board_file) as f:
                return json.load(f)
        return {"entries": [], "examples": [e.id for e in self.examples]}

    def _save_board(self) -> None:
        """Save evaluation board to file."""
        with open(self.board_file, 'w') as f:
            json.dump(self.board, f, indent=2)

    def add_example(self, example: HardExample) -> None:
        """Add a new hard example."""
        # Check for duplicate ID
        existing_ids = {e.id for e in self.examples}
        if example.id in existing_ids:
            logger.warning(f"Example {example.id} already exists, updating")
            self.examples = [e for e in self.examples if e.id != example.id]

        self.examples.append(example)
        self._save_examples(self.examples)
        logger.info(f"Added hard example: {example.id}")

    def call_api(self, prompt: str) -> Dict[str, Any]:
        """Call inference API."""
        try:
            # Format as SYLLO problem
            full_prompt = f"Solve this logic problem. Answer with just Yes, No, or Cannot determine.\n\nProblem: {prompt}\n\nAnswer:"

            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": full_prompt}],
                    "max_tokens": 50,
                    "temperature": 0.0
                },
                timeout=30
            )

            if response.ok:
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                return {
                    "text": text,
                    "latency_ms": response.elapsed.total_seconds() * 1000
                }
            else:
                return {"error": f"API error: {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}

    def extract_answer(self, response: str) -> str:
        """Extract answer from model response."""
        response = response.lower().strip()

        # Look for explicit answers
        if "cannot determine" in response or "cannot be determined" in response:
            return "Cannot determine"
        elif response.startswith("yes") or "yes." in response or "yes," in response:
            return "Yes"
        elif response.startswith("no") or "no." in response or "no," in response:
            return "No"
        else:
            # Try to find answer in response
            for word in ["yes", "no"]:
                if word in response.split():
                    return word.capitalize()
            return response[:50]  # Return truncated response

    def classify_error(self, expected: str, got: str, category: str) -> str:
        """Classify the type of error made."""
        expected = expected.lower()
        got = got.lower()

        if "cannot determine" in expected and "cannot determine" not in got:
            return "over_confident"  # Said yes/no when should be uncertain
        elif "cannot determine" in got and "cannot determine" not in expected:
            return "under_confident"  # Said uncertain when answer exists
        elif expected == "yes" and got == "no":
            return "false_negative"
        elif expected == "no" and got == "yes":
            return "false_positive"
        else:
            return f"wrong_{category}"

    def evaluate_example(self, example: HardExample) -> EvaluationResult:
        """Evaluate a single hard example."""
        result = self.call_api(example.prompt)

        if "error" in result:
            return EvaluationResult(
                example_id=example.id,
                correct=False,
                error_type="api_error"
            )

        model_answer = self.extract_answer(result.get("text", ""))
        expected = example.expected.lower()
        got = model_answer.lower()

        # Normalize for comparison
        correct = (
            (expected == "yes" and got == "yes") or
            (expected == "no" and got == "no") or
            ("cannot determine" in expected and "cannot determine" in got)
        )

        error_type = None if correct else self.classify_error(
            example.expected, model_answer, example.category
        )

        return EvaluationResult(
            example_id=example.id,
            correct=correct,
            model_answer=model_answer,
            error_type=error_type,
            latency_ms=result.get("latency_ms")
        )

    def evaluate_all(self, checkpoint_name: str = "current") -> BoardEntry:
        """Evaluate all hard examples."""
        # Get current step
        step = 0
        training_status = self.base_dir / "status" / "training_status.json"
        if training_status.exists():
            with open(training_status) as f:
                data = json.load(f)
                step = data.get("current_step", 0)

        results = {}
        error_types = {}
        total_correct = 0

        logger.info(f"Evaluating {len(self.examples)} hard examples...")

        for example in self.examples:
            result = self.evaluate_example(example)
            results[example.id] = result.correct

            if result.correct:
                total_correct += 1
                logger.info(f"  ✓ {example.id}")
            else:
                error_type = result.error_type or "unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1
                logger.info(f"  ✗ {example.id} ({error_type})")

        entry = BoardEntry(
            checkpoint=checkpoint_name,
            step=step,
            timestamp=datetime.now().isoformat(),
            results=results,
            total_correct=total_correct,
            total=len(self.examples),
            accuracy=total_correct / len(self.examples) if self.examples else 0,
            error_types=error_types
        )

        # Add to board
        self.board["entries"].append(asdict(entry))
        self._save_board()

        logger.info(f"\nResults: {total_correct}/{len(self.examples)} ({entry.accuracy:.1%})")

        return entry

    def show_board(self) -> None:
        """Display the evaluation board."""
        entries = self.board.get("entries", [])
        example_ids = [e.id for e in self.examples]

        if not entries:
            print("No evaluations yet. Run --evaluate first.")
            return

        # Header
        print("\n" + "=" * 80)
        print("HARD EXAMPLE BOARD")
        print("=" * 80)

        # Column headers
        header = f"{'Example':<25} | " + " | ".join(
            f"{e['checkpoint'][:8]:^8}" for e in entries[-5:]  # Last 5
        )
        print(header)
        print("-" * len(header))

        # Rows
        for ex_id in example_ids:
            row = f"{ex_id:<25} | "
            for entry in entries[-5:]:
                result = entry.get("results", {}).get(ex_id, None)
                if result is True:
                    row += f"{'✓':^8} | "
                elif result is False:
                    row += f"{'✗':^8} | "
                else:
                    row += f"{'?':^8} | "
            print(row)

        # Summary
        print("-" * len(header))
        summary = f"{'TOTAL':<25} | "
        for entry in entries[-5:]:
            acc = entry.get("accuracy", 0) * 100
            summary += f"{acc:^7.0f}% | "
        print(summary)

    def generate_board_image(self) -> Path:
        """Generate visual board as PNG."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.error("matplotlib required")
            return None

        entries = self.board.get("entries", [])[-10:]  # Last 10
        example_ids = [e.id for e in self.examples]

        if not entries:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create grid
        n_examples = len(example_ids)
        n_checkpoints = len(entries)

        # Colors
        colors = {True: '#2ecc71', False: '#e74c3c', None: '#95a5a6'}

        for i, ex_id in enumerate(example_ids):
            for j, entry in enumerate(entries):
                result = entry.get("results", {}).get(ex_id, None)
                color = colors[result]
                rect = mpatches.Rectangle(
                    (j, n_examples - i - 1), 1, 1,
                    facecolor=color, edgecolor='white', linewidth=2
                )
                ax.add_patch(rect)

        # Labels
        ax.set_xlim(0, n_checkpoints)
        ax.set_ylim(0, n_examples)

        ax.set_xticks([i + 0.5 for i in range(n_checkpoints)])
        ax.set_xticklabels([f"Step\n{e['step']}" for e in entries], fontsize=8)

        ax.set_yticks([i + 0.5 for i in range(n_examples)])
        ax.set_yticklabels(reversed(example_ids), fontsize=8)

        ax.set_title("Hard Example Board", fontsize=14)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#2ecc71', label='Correct'),
            mpatches.Patch(facecolor='#e74c3c', label='Wrong'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        output_path = self.viz_dir / "hard_example_board.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved board image to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Hard Example Tracker")
    parser.add_argument('--base-dir', default='/path/to/training',
                       help='Base directory')
    parser.add_argument('--api-url', default='http://192.168.x.x:8765',
                       help='Inference API URL')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate current checkpoint')
    parser.add_argument('--show-board', action='store_true',
                       help='Show evaluation board')
    parser.add_argument('--generate-image', action='store_true',
                       help='Generate board image')
    parser.add_argument('--add', action='store_true',
                       help='Add new hard example')
    parser.add_argument('--prompt', type=str,
                       help='Prompt for new example')
    parser.add_argument('--expected', type=str,
                       help='Expected answer for new example')
    parser.add_argument('--category', type=str,
                       help='Category for new example')
    parser.add_argument('--checkpoint', type=str, default='current',
                       help='Checkpoint name for evaluation')

    args = parser.parse_args()

    tracker = HardExampleTracker(args.base_dir, args.api_url)

    if args.add:
        if not all([args.prompt, args.expected, args.category]):
            print("--add requires --prompt, --expected, and --category")
            return

        example = HardExample(
            id=f"{args.category}_{len(tracker.examples)}",
            prompt=args.prompt,
            expected=args.expected,
            category=args.category
        )
        tracker.add_example(example)

    elif args.evaluate:
        tracker.evaluate_all(args.checkpoint)
        tracker.show_board()

    elif args.show_board:
        tracker.show_board()

    elif args.generate_image:
        tracker.generate_board_image()

    else:
        # Default: show current examples
        print(f"Hard Examples ({len(tracker.examples)}):")
        for ex in tracker.examples:
            print(f"  - {ex.id}: {ex.category}")
        print(f"\nUse --evaluate to test, --show-board to see history")


if __name__ == "__main__":
    main()
