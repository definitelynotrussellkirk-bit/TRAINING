#!/usr/bin/env python3
"""
GPU Inference Worker
Loads model and runs actual inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json

class InferenceWorker:
    def __init__(self, model_path: str, device='cuda'):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loaded_model_id = None

    def load_model(self, model_id: str = 'Qwen3-0.6B'):
        """Load model into memory"""
        if self.loaded_model_id == model_id and self.model is not None:
            return  # Already loaded

        print(f'Loading model: {model_id}')
        model_dir = self.model_path / model_id

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        self.loaded_model_id = model_id
        print(f'✓ Model loaded: {model_id}')

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> dict:
        """Run inference"""
        if self.model is None:
            raise RuntimeError('Model not loaded')

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):]  # Remove prompt

        return {
            'generated_text': generated_text,
            'prompt_tokens': len(inputs['input_ids'][0]),
            'completion_tokens': len(outputs[0]) - len(inputs['input_ids'][0]),
            'total_tokens': len(outputs[0])
        }

# Global worker instance
_worker = None

def get_worker():
    global _worker
    if _worker is None:
        models_dir = Path.home() / 'llm' / 'models'
        _worker = InferenceWorker(models_dir)
    return _worker
