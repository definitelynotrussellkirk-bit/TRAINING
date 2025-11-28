#!/usr/bin/env python3
"""
Model Comparison Tool

Compare two model checkpoints side-by-side for:
- Inference speed
- Quality on test prompts
- Model size
- Memory usage

Usage: python3 compare_models.py model1_path model2_path [--test-file test.jsonl]
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import sys

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    print("Error: Required packages not installed")
    print("Install with: pip install transformers peft torch")
    sys.exit(1)

class ModelComparator:
    def __init__(self, base_model_path: str = None):
        self.base_model_path = base_model_path or "model_qwen25"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, adapter_path: str):
        """Load model with adapter"""
        print(f"Loading base model from {self.base_model_path}...")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

        if Path(adapter_path).exists():
            print(f"Loading adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)

        return model, tokenizer

    def get_model_size(self, model_path: str) -> float:
        """Calculate model size in GB"""
        total_size = 0
        path = Path(model_path)

        if not path.exists():
            return 0.0

        for file in path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size

        return total_size / (1024 ** 3)

    def measure_inference_speed(self, model, tokenizer, prompts: List[str], num_runs: int = 5) -> Dict:
        """Measure inference speed"""
        times = []
        tokens_generated = []

        for prompt in prompts[:num_runs]:
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            end = time.time()

            times.append(end - start)
            tokens_generated.append(outputs.shape[1] - inputs['input_ids'].shape[1])

        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'avg_tokens': sum(tokens_generated) / len(tokens_generated),
            'tokens_per_sec': sum(tokens_generated) / sum(times)
        }

    def test_quality(self, model, tokenizer, test_examples: List[Dict]) -> Dict:
        """Test model quality on examples"""
        results = []

        for example in test_examples:
            prompt = example['messages'][0]['content'] if 'messages' in example else example.get('prompt', '')
            expected = example['messages'][1]['content'] if 'messages' in example else example.get('answer', '')

            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            matches = response.strip().lower() == expected.strip().lower()

            results.append({
                'prompt': prompt[:100],
                'expected': expected[:100],
                'generated': response[:100],
                'matches': matches
            })

        accuracy = sum(1 for r in results if r['matches']) / len(results) * 100

        return {
            'accuracy': accuracy,
            'total': len(results),
            'correct': sum(1 for r in results if r['matches']),
            'examples': results
        }

    def compare(self, model1_path: str, model2_path: str, test_file: str = None):
        """Compare two models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        # Model sizes
        print("\n📦 Model Sizes:")
        size1 = self.get_model_size(model1_path)
        size2 = self.get_model_size(model2_path)
        print(f"  Model 1: {size1:.2f} GB ({model1_path})")
        print(f"  Model 2: {size2:.2f} GB ({model2_path})")
        print(f"  Difference: {abs(size1 - size2):.2f} GB")

        # Load test examples
        test_examples = []
        if test_file and Path(test_file).exists():
            with open(test_file) as f:
                for line in f:
                    test_examples.append(json.loads(line))
                    if len(test_examples) >= 10:  # Limit to 10 for speed
                        break
        else:
            # Default test prompts
            test_examples = [
                {'messages': [{'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant', 'content': '4'}]},
                {'messages': [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'Paris'}]},
                {'messages': [{'role': 'user', 'content': 'List three colors.'}, {'role': 'assistant', 'content': 'Red, blue, green'}]},
            ]

        print(f"\n🧪 Running tests with {len(test_examples)} examples...")

        # Load and test Model 1
        print("\n⏳ Testing Model 1...")
        model1, tokenizer1 = self.load_model(model1_path)

        prompts = [ex['messages'][0]['content'] if 'messages' in ex else ex.get('prompt', '') for ex in test_examples]

        speed1 = self.measure_inference_speed(model1, tokenizer1, prompts)
        quality1 = self.test_quality(model1, tokenizer1, test_examples)

        del model1
        torch.cuda.empty_cache()

        # Load and test Model 2
        print("⏳ Testing Model 2...")
        model2, tokenizer2 = self.load_model(model2_path)

        speed2 = self.measure_inference_speed(model2, tokenizer2, prompts)
        quality2 = self.test_quality(model2, tokenizer2, test_examples)

        del model2
        torch.cuda.empty_cache()

        # Print comparison
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)

        print("\n⚡ Inference Speed:")
        print(f"  Model 1: {speed1['avg_time']:.3f}s avg ({speed1['tokens_per_sec']:.1f} tok/s)")
        print(f"  Model 2: {speed2['avg_time']:.3f}s avg ({speed2['tokens_per_sec']:.1f} tok/s)")

        faster = "Model 1" if speed1['avg_time'] < speed2['avg_time'] else "Model 2"
        speedup = max(speed1['avg_time'], speed2['avg_time']) / min(speed1['avg_time'], speed2['avg_time'])
        print(f"  Winner: {faster} ({speedup:.2f}x faster)")

        print("\n🎯 Quality (Accuracy):")
        print(f"  Model 1: {quality1['accuracy']:.1f}% ({quality1['correct']}/{quality1['total']})")
        print(f"  Model 2: {quality2['accuracy']:.1f}% ({quality2['correct']}/{quality2['total']})")

        if quality1['accuracy'] > quality2['accuracy']:
            print(f"  Winner: Model 1 (+{quality1['accuracy'] - quality2['accuracy']:.1f}%)")
        elif quality2['accuracy'] > quality1['accuracy']:
            print(f"  Winner: Model 2 (+{quality2['accuracy'] - quality1['accuracy']:.1f}%)")
        else:
            print(f"  Winner: Tie")

        print("\n📊 Example Comparison:")
        for i, (ex1, ex2) in enumerate(zip(quality1['examples'][:3], quality2['examples'][:3])):
            print(f"\n  Example {i+1}: {ex1['prompt']}")
            print(f"    Expected: {ex1['expected']}")
            print(f"    Model 1:  {ex1['generated']} {'✓' if ex1['matches'] else '✗'}")
            print(f"    Model 2:  {ex2['generated']} {'✓' if ex2['matches'] else '✗'}")

        print("\n💡 Recommendation:")

        # Score models
        score1 = 0
        score2 = 0

        if speed1['avg_time'] < speed2['avg_time']:
            score1 += 1
        else:
            score2 += 1

        if quality1['accuracy'] > quality2['accuracy']:
            score1 += 2  # Quality worth more than speed
        elif quality2['accuracy'] > quality1['accuracy']:
            score2 += 2

        if size1 < size2:
            score1 += 0.5  # Size is a tiebreaker
        else:
            score2 += 0.5

        if score1 > score2:
            print(f"  ⭐ Model 1 is better overall (score: {score1} vs {score2})")
            print(f"     Use: {model1_path}")
        elif score2 > score1:
            print(f"  ⭐ Model 2 is better overall (score: {score2} vs {score1})")
            print(f"     Use: {model2_path}")
        else:
            print(f"  ⭐ Models are equivalent (score: {score1} vs {score2})")
            print(f"     Either model is fine")

        print("\n" + "="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Compare two model checkpoints")
    parser.add_argument('model1', help='Path to first model/adapter')
    parser.add_argument('model2', help='Path to second model/adapter')
    parser.add_argument('--test-file', help='JSONL file with test examples')
    parser.add_argument('--base-model', default='model_qwen25', help='Base model path')

    args = parser.parse_args()

    comparator = ModelComparator(args.base_model)
    comparator.compare(args.model1, args.model2, args.test_file)

if __name__ == '__main__':
    main()
