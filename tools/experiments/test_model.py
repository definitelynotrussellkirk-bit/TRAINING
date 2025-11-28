#!/usr/bin/env python3
"""
Quick Model Tester

Simple script to test the trained model with interactive prompts.
Run after training completes to see if model learned.

Usage:
    python3 test_model.py
    python3 test_model.py --model current_model/
    python3 test_model.py --prompt "Your prompt here"
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_model(base_model_path: str, adapter_path: str = None):
    """Load base model and optional adapter."""
    print(f"Loading base model from: {base_model_path}")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load adapter if provided
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("No adapter loaded - using base model only")
        model = base_model

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate response from model."""
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode (skip the input prompt)
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response

def interactive_mode(model, tokenizer):
    """Interactive testing mode."""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODEL TESTER")
    print("=" * 80)
    print("\nType your prompts below. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("\n🔵 Prompt: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break

            response = generate(model, tokenizer, prompt)

            print("\n🤖 Response:")
            print("-" * 80)
            print(response)
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument("--base-model", default="/path/to/training/consolidated_models/20251119_152444",
                       help="Path to base model")
    parser.add_argument("--adapter", default="/path/to/training/current_model",
                       help="Path to adapter (optional)")
    parser.add_argument("--prompt", type=str, help="Single prompt to test (non-interactive)")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.base_model, args.adapter)

    if args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}")
        response = generate(model, tokenizer, args.prompt, args.max_tokens)
        print(f"\nResponse:\n{response}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()
