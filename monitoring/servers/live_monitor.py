#!/usr/bin/env python3
"""
Live Inference Monitor

THE KILLER FEATURE!

Shows what the model is learning DURING training.
Every N steps, run inference and display:
  - Input (what model sees)
  - Expected (what it should say)
  - Predicted (what it actually says)
  - Accuracy tracking over time

This would have caught the composition issue DURING training
instead of waiting 9 hours to find out the model learned formatting.
"""

import torch
import random
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from transformers import LogitsProcessorList
from logit_penalty import reset_processor_states


@dataclass
class InferenceResult:
    """Result of a single inference run."""
    step: int
    example_id: int
    input_text: str
    expected: str
    predicted: str
    match: bool
    timestamp: str
    system_prompt: Optional[str] = None
    full_prompt: Optional[str] = None


class LiveInferenceMonitor:
    """Monitor model learning during training with live inference."""

    def __init__(
        self,
        model,
        tokenizer,
        val_examples: List[Dict],
        num_samples: int = 5,
        max_new_tokens: int = 2048,  # Increased to capture full outputs
        logits_processor=None,
        self_correction_generator=None  # Optional self-correction training data generator
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.all_val_examples = val_examples  # Keep full list
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.history: List[InferenceResult] = []
        self.used_indices = set()  # Track which examples we've used
        self.logits_processor = LogitsProcessorList(logits_processor or [])
        self.self_correction_generator = self_correction_generator  # Auto-generate correction training data

    def run_inference(self, step: int) -> List[InferenceResult]:
        """Run inference on NEW random validation examples (no repeats)."""
        # Skip if num_samples is 0 (inference disabled)
        if self.num_samples == 0:
            return []

        # Sample fresh examples that haven't been used yet
        available_indices = [i for i in range(len(self.all_val_examples)) if i not in self.used_indices]

        # If we've used all examples, reset
        if len(available_indices) < self.num_samples:
            print(f"\n📊 Evaluated all examples, resetting pool...")
            self.used_indices.clear()
            available_indices = list(range(len(self.all_val_examples)))

        # Sample new random indices
        sampled_indices = random.sample(available_indices, min(self.num_samples, len(available_indices)))
        self.used_indices.update(sampled_indices)

        # Get the examples
        val_examples = [self.all_val_examples[i] for i in sampled_indices]

        # Switch to eval mode
        self.model.eval()
        results = []

        with torch.no_grad():
            for i, example in enumerate(val_examples):
                # Extract messages
                messages_raw = example['messages']

                # Extract system prompt if exists
                system_prompt = None
                user_content = None
                expected = None

                for msg in messages_raw:
                    role = msg.get('role', '')
                    content = str(msg.get('content', ''))

                    if role == 'system':
                        system_prompt = content
                    elif role == 'user':
                        user_content = content
                    elif role == 'assistant':
                        expected = content.strip()

                # Fallback if structure is simple [user, assistant]
                if user_content is None:
                    user_content = str(messages_raw[0]['content'])
                if expected is None:
                    expected = str(messages_raw[1]['content']).strip()

                # Build prompt using chat template (with system if present)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_content})

                try:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    # Fallback if chat template fails
                    if system_prompt:
                        text = f"System: {system_prompt}\nUser: {user_content}\nAssistant:"
                    else:
                        text = f"User: {user_content}\nAssistant:"

                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

                # Generate
                reset_processor_states(self.logits_processor)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    logits_processor=self.logits_processor,
                    min_new_tokens=1
                )

                # Decode
                predicted = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                # Check match
                match = expected.strip() == predicted.strip()

                result = InferenceResult(
                    step=step,
                    example_id=i,
                    input_text=user_content,
                    expected=expected,
                    predicted=predicted,
                    match=match,
                    timestamp=datetime.now().isoformat(),
                    system_prompt=system_prompt,
                    full_prompt=text
                )

                results.append(result)
                self.history.append(result)

        # Switch back to train mode
        self.model.train()

        # Auto-generate self-correction training data if enabled
        if self.self_correction_generator and results:
            try:
                self.self_correction_generator.process_inference_results(results, step)
            except Exception as e:
                print(f"⚠️  Self-correction generation error: {e}")

        return results

    def display_results(self, results: List[InferenceResult], step: int, total_steps: int):
        """Display inference results in terminal."""
        print("\n" + "=" * 80)
        print(f"🔍 LIVE INFERENCE - Step {step:,} / {total_steps:,} ({step/total_steps*100:.1f}%)")
        print("=" * 80)

        for result in results:
            status = "✅" if result.match else "❌"

            print(f"\n{status} Example {result.example_id + 1}/{len(results)}:")

            # Show SYSTEM PROMPT if present
            if result.system_prompt:
                print("┌" + "─" * 78 + "┐")
                print("│ SYSTEM PROMPT:                                                             │")
                print("├" + "─" * 78 + "┤")
                for line in result.system_prompt.split('\n')[:3]:
                    if len(line) > 76:
                        line = line[:73] + "..."
                    print(f"│ {line:<76} │")
                if result.system_prompt.count('\n') > 3:
                    print("│ ...                                                                        │")
                print("└" + "─" * 78 + "┘")

            # Show USER INPUT
            print("┌" + "─" * 78 + "┐")
            print("│ USER INPUT:                                                                │")
            print("├" + "─" * 78 + "┤")

            # Show first 5 lines of input
            for line in result.input_text.split('\n')[:5]:
                # Truncate long lines
                if len(line) > 76:
                    line = line[:73] + "..."
                print(f"│ {line:<76} │")

            if result.input_text.count('\n') > 5:
                print("│ ...                                                                        │")

            print("└" + "─" * 78 + "┘")

            # Show FULL PROMPT (what actually gets tokenized)
            if result.full_prompt:
                print("┌" + "─" * 78 + "┐")
                print("│ FULL FORMATTED PROMPT (tokenized):                                        │")
                print("├" + "─" * 78 + "┤")
                for line in result.full_prompt.split('\n')[:4]:
                    if len(line) > 76:
                        line = line[:73] + "..."
                    print(f"│ {line:<76} │")
                if result.full_prompt.count('\n') > 4:
                    print("│ ...                                                                        │")
                print("└" + "─" * 78 + "┘")

            # Show EXPECTED
            print("┌" + "─" * 78 + "┐")
            print("│ EXPECTED (training target):                                               │")
            print("├" + "─" * 78 + "┤")

            for line in result.expected.split('\n')[:3]:
                if len(line) > 76:
                    line = line[:73] + "..."
                print(f"│ {line:<76} │")

            print("└" + "─" * 78 + "┘")

            # Show MODEL OUTPUT
            print("┌" + "─" * 78 + "┐")
            print("│ MODEL OUTPUT (current guess):                                             │")
            print("├" + "─" * 78 + "┤")

            for line in result.predicted.split('\n')[:3]:
                if len(line) > 76:
                    line = line[:73] + "..."
                print(f"│ {line:<76} │")

            print("└" + "─" * 78 + "┘")

            if not result.match:
                print("\n⚠️  MISMATCH DETECTED!")
                # Show diff
                if len(result.expected) < 100 and len(result.predicted) < 100:
                    print(f"   Expected: '{result.expected}'")
                    print(f"   Got:      '{result.predicted}'")

        # Calculate accuracy
        accuracy = sum(r.match for r in results) / len(results) * 100
        print("\n" + "─" * 80)
        print(f"📊 Accuracy: {accuracy:.1f}% ({sum(r.match for r in results)}/{len(results)} correct)")

        # Show trend if we have history
        if len(self.history) >= len(self.val_examples) * 2:
            recent_steps = sorted(set(r.step for r in self.history))[-3:]
            if len(recent_steps) >= 2:
                print("\n📈 Accuracy Trend:")
                for s in recent_steps:
                    step_results = [r for r in self.history if r.step == s]
                    step_acc = sum(r.match for r in step_results) / len(step_results) * 100
                    bars = "█" * int(step_acc / 5)
                    print(f"   Step {s:>6}: {bars:<20} {step_acc:.1f}%")

        print("=" * 80 + "\n")

        # Warning if accuracy is low
        if accuracy < 50:
            print("⚠️⚠️⚠️ WARNING: Accuracy below 50%!")
            print("   → Model may not be learning correctly")
            print("   → Consider checking:")
            print("      • Training data quality (run validator.py)")
            print("      • Learning rate (may be too high/low)")
            print("      • Dataset has correct examples")
            print()

        # Save results to JSON for web dashboard
        self.save_inference_results(results, step, total_steps, accuracy)

        return accuracy

    def save_inference_results(self, results: List[InferenceResult], step: int, total_steps: int, accuracy: float):
        """Save inference results to JSON file for web dashboard."""
        try:
            # Path to save JSON file
            output_dir = Path(__file__).parent / "current_model" / "status"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "live_inference.json"

            # Build JSON structure
            examples = []
            for result in results:
                examples.append({
                    "example_id": result.example_id,
                    "system_prompt": result.system_prompt,
                    "user_input": result.input_text,
                    "formatted_prompt": result.full_prompt,
                    "expected": result.expected,
                    "predicted": result.predicted,
                    "match": result.match,
                    "timestamp": result.timestamp
                })

            # Calculate accuracy trend
            trend = []
            if len(self.history) >= len(self.val_examples) * 2:
                recent_steps = sorted(set(r.step for r in self.history))[-3:]
                for s in recent_steps:
                    step_results = [r for r in self.history if r.step == s]
                    step_acc = sum(r.match for r in step_results) / len(step_results) * 100
                    trend.append({"step": s, "accuracy": step_acc})

            data = {
                "status": "inference_complete",
                "step": step,
                "total_steps": total_steps,
                "progress_percent": round(step / total_steps * 100, 1) if total_steps > 0 else 0,
                "accuracy": round(accuracy, 1),
                "num_examples": len(results),
                "examples": examples,
                "accuracy_trend": trend,
                "timestamp": datetime.now().isoformat()
            }

            # Write to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"\n⚠️ Warning: Could not save inference results to JSON: {e}")

    def get_accuracy_history(self) -> List[tuple]:
        """Get accuracy over time."""
        steps = sorted(set(r.step for r in self.history))
        history = []

        for step in steps:
            step_results = [r for r in self.history if r.step == step]
            accuracy = sum(r.match for r in step_results) / len(step_results) * 100
            history.append((step, accuracy))

        return history

    def plot_accuracy(self, width: int = 60):
        """Simple ASCII plot of accuracy over time."""
        history = self.get_accuracy_history()
        if len(history) < 2:
            return

        print("\n" + "=" * 80)
        print("ACCURACY OVER TIME")
        print("=" * 80)

        max_step = max(h[0] for h in history)
        max_acc = 100

        for acc_level in range(100, -1, -10):
            line = f"{acc_level:3d}% │"

            for step, acc in history:
                x_pos = int((step / max_step) * (width - 1))
                if abs(acc - acc_level) < 5:
                    line += "█"
                else:
                    line += " "

            print(line)

        # X-axis
        print("     └" + "─" * width)
        print(f"      0{' ' * (width - 10)}{max_step:>8}")
        print(f"{'Steps':^{width + 6}}")
        print("=" * 80 + "\n")


def test_monitor():
    """Test the monitor with dummy data."""
    print("Testing Live Inference Monitor (dummy mode)")
    print()

    # Create dummy examples
    val_examples = [
        {
            "messages": [
                {"role": "user", "content": "Find items with is_cold"},
                {"role": "assistant", "content": "Result: banana, water"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Count items with is_heavy"},
                {"role": "assistant", "content": "Result: 5"}
            ]
        }
    ]

    # Simulate monitoring
    class DummyModel:
        def eval(self): pass
        def train(self): pass
        device = "cpu"

        def generate(self, **kwargs):
            # Return dummy tokens
            return torch.tensor([[1, 2, 3, 4]])

    class DummyTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "User: " + messages[0]['content']

        def __call__(self, text, return_tensors=None):
            class DummyInputs(dict):
                def __init__(self):
                    super().__init__()
                    self['input_ids'] = torch.tensor([[1, 2, 3]])
                    self.input_ids = self['input_ids']
                def to(self, device):
                    return self
            return DummyInputs()

        def decode(self, tokens, skip_special_tokens=False):
            # Vary output based on "training progress"
            import random
            if random.random() > 0.5:
                return "Result: banana, water"  # Correct
            else:
                return "Result: wrong answer"  # Wrong

    model = DummyModel()
    tokenizer = DummyTokenizer()

    monitor = LiveInferenceMonitor(model, tokenizer, val_examples, num_samples=2)

    # Simulate training steps
    for step in [0, 100, 200, 300, 400, 500]:
        results = monitor.run_inference(step)
        monitor.display_results(results, step, total_steps=500)

        import time
        time.sleep(0.5)

    # Show plot
    monitor.plot_accuracy()


if __name__ == "__main__":
    test_monitor()
