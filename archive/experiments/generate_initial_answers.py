#!/usr/bin/env python3
"""
Generate initial answers from current model for self-correction pipeline.

This script loads your trained model and generates first-attempt answers
for all prompts in a dataset.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse


def load_model(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_answer(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """Generate answer from model."""

    # Format prompt (adjust based on your training format)
    formatted_prompt = f"{prompt}\n\nAnswer:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and extract answer
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt to get just the answer
    answer = full_output[len(formatted_prompt):].strip()

    return answer


def generate_initial_answers(
    input_file: str,
    output_file: str,
    model_path: str,
    max_samples: int = None,
    temperature: float = 0.7
):
    """Generate initial answers for all prompts."""

    # Load model
    model, tokenizer = load_model(model_path)

    # Load prompts
    with open(input_file) as f:
        data = [json.loads(line) for line in f]

    if max_samples:
        data = data[:max_samples]

    print(f"Generating answers for {len(data)} prompts...")

    # Generate answers
    results = []
    for item in tqdm(data):
        prompt = item['prompt']

        try:
            answer = generate_answer(prompt, model, tokenizer, temperature=temperature)

            results.append({
                "prompt": prompt,
                "answer": answer,
                "golden": item.get('response', '')  # Include golden for reference
            })

        except Exception as e:
            print(f"Error generating for prompt: {prompt[:50]}...")
            print(f"  Error: {e}")
            results.append({
                "prompt": prompt,
                "answer": "",
                "error": str(e)
            })

    # Save results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Saved {len(results)} answers to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate initial answers from model')
    parser.add_argument('--input', required=True,
                       help='Input .jsonl with prompts (and golden responses)')
    parser.add_argument('--output', required=True,
                       help='Output .jsonl for initial answers')
    parser.add_argument('--model', default='/path/to/training/current_model',
                       help='Path to model (default: current_model/)')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum number of samples to process')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')

    args = parser.parse_args()

    generate_initial_answers(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model,
        max_samples=args.max_samples,
        temperature=args.temperature
    )
