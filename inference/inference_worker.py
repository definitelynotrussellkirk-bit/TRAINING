#!/usr/bin/env python3
"""
GPU Inference Worker - Multi-Model Pool
Manages multiple models in VRAM with explicit model selection
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from collections import OrderedDict
import json


class ModelEntry:
    """Single model entry in the pool"""
    def __init__(self, model_id: str, model, tokenizer, path: str):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.path = path
        self.loaded_at = datetime.now().isoformat()
        self.last_used = datetime.now()
        self.request_count = 0

    def touch(self):
        """Update last used timestamp"""
        self.last_used = datetime.now()
        self.request_count += 1


class ModelPool:
    """
    Multi-model inference pool with LRU eviction.

    - Keeps multiple models loaded in VRAM
    - Requires explicit model selection (no defaults)
    - Every response tagged with model_id
    - LRU eviction when pool is full
    """

    def __init__(self, models_dir: str, max_models: int = 3, device: str = 'cuda'):
        self.models_dir = Path(models_dir)
        self.max_models = max_models
        self.device = device
        self.pool: OrderedDict[str, ModelEntry] = OrderedDict()

    def get_pool_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        models = []
        total_vram = 0

        for model_id, entry in self.pool.items():
            # Estimate VRAM per model
            vram_mb = sum(p.numel() * p.element_size() for p in entry.model.parameters()) / 1e6
            total_vram += vram_mb

            models.append({
                "model_id": model_id,
                "path": entry.path,
                "loaded_at": entry.loaded_at,
                "last_used": entry.last_used.isoformat(),
                "request_count": entry.request_count,
                "vram_mb": round(vram_mb, 1)
            })

        return {
            "loaded_count": len(self.pool),
            "max_models": self.max_models,
            "total_vram_mb": round(total_vram, 1),
            "models": models
        }

    def is_loaded(self, model_id: str) -> bool:
        """Check if model is loaded"""
        return model_id in self.pool

    def _evict_lru(self):
        """Evict least recently used model"""
        if not self.pool:
            return

        # Find LRU model
        lru_id = min(self.pool.keys(), key=lambda k: self.pool[k].last_used)
        entry = self.pool.pop(lru_id)

        print(f"🗑️ Evicting LRU model: {lru_id} (last used: {entry.last_used})")

        # Free memory
        del entry.model
        del entry.tokenizer
        torch.cuda.empty_cache()

    def load_model(self, model_id: str, path: Optional[str] = None) -> bool:
        """
        Load a model into the pool.

        Args:
            model_id: Unique identifier (e.g., "checkpoint-175000", "Qwen3-0.6B")
            path: Full path to model. If None, uses models_dir/model_id

        Returns:
            True if loaded (or already loaded), False on error
        """
        # Already loaded - just touch it
        if model_id in self.pool:
            self.pool[model_id].touch()
            # Move to end (most recent)
            self.pool.move_to_end(model_id)
            print(f"✓ Model already loaded: {model_id}")
            return True

        # Evict if at capacity
        while len(self.pool) >= self.max_models:
            self._evict_lru()

        # Determine path
        model_path = Path(path) if path else self.models_dir / model_id

        if not model_path.exists():
            print(f"❌ Model path not found: {model_path}")
            return False

        print(f"📦 Loading model: {model_id} from {model_path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )

            entry = ModelEntry(model_id, model, tokenizer, str(model_path))
            self.pool[model_id] = entry

            vram_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
            print(f"✓ Model loaded: {model_id} ({vram_mb:.0f}MB VRAM)")

            return True

        except Exception as e:
            print(f"❌ Failed to load {model_id}: {e}")
            return False

    def unload_model(self, model_id: str) -> bool:
        """Explicitly unload a model"""
        if model_id not in self.pool:
            return False

        entry = self.pool.pop(model_id)
        del entry.model
        del entry.tokenizer
        torch.cuda.empty_cache()

        print(f"✓ Unloaded model: {model_id}")
        return True

    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        messages: list = None
    ) -> Dict[str, Any]:
        """
        Generate text using a specific model.

        REQUIRES explicit model_id - no defaults.
        Response always includes model_id for tracking.

        Args:
            prompt: Raw text prompt (legacy)
            messages: List of {"role": "user/assistant", "content": "..."} (preferred)
        """
        if model_id not in self.pool:
            return {
                'error': f'Model not loaded: {model_id}',
                'model_id': model_id,
                'available_models': list(self.pool.keys())
            }

        entry = self.pool[model_id]
        entry.touch()
        self.pool.move_to_end(model_id)  # Update LRU order

        # Use chat template if messages provided (matches training format)
        if messages:
            prompt = entry.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Tokenize
        inputs = entry.tokenizer(prompt, return_tensors='pt').to(self.device)

        # Generate
        with torch.no_grad():
            outputs = entry.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=entry.tokenizer.eos_token_id
            )

        # Decode only the generated tokens (not the prompt)
        prompt_length = len(inputs['input_ids'][0])
        generated_text = entry.tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True
        )

        return {
            'generated_text': generated_text,
            'model_id': model_id,  # Always include model tracking
            'model_path': entry.path,
            'prompt_tokens': len(inputs['input_ids'][0]),
            'completion_tokens': len(outputs[0]) - len(inputs['input_ids'][0]),
            'total_tokens': len(outputs[0]),
            'timestamp': datetime.now().isoformat()
        }


# ============================================================
# Legacy compatibility layer
# ============================================================

class InferenceWorker:
    """
    Legacy single-model interface.
    Now wraps ModelPool for backward compatibility.
    """
    def __init__(self, model_path: str, device='cuda'):
        self.model_path = Path(model_path)
        self.device = device
        self.pool = ModelPool(model_path, max_models=3, device=device)

        # Legacy attributes for compatibility
        self.model = None  # Deprecated
        self.tokenizer = None  # Deprecated
        self.loaded_model_id = None
        self.model_loaded_at = None

    def load_model(self, model_id: str = 'Qwen3-0.6B'):
        """Legacy: Load model (now uses pool)"""
        success = self.pool.load_model(model_id)
        if success:
            self.loaded_model_id = model_id
            entry = self.pool.pool[model_id]
            self.model = entry.model
            self.tokenizer = entry.tokenizer
            self.model_loaded_at = entry.loaded_at

    def load_model_from_path(self, full_path: str, model_id: str):
        """Legacy: Load model from explicit path"""
        success = self.pool.load_model(model_id, path=full_path)
        if success:
            self.loaded_model_id = model_id
            entry = self.pool.pool[model_id]
            self.model = entry.model
            self.tokenizer = entry.tokenizer
            self.model_loaded_at = entry.loaded_at

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> dict:
        """Legacy generate - uses currently loaded model"""
        if not self.loaded_model_id:
            raise RuntimeError('No model loaded')

        return self.pool.generate(
            model_id=self.loaded_model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )


# Global instances
_worker = None
_pool = None


def get_worker() -> InferenceWorker:
    """Get legacy single-model worker (backward compatible)"""
    global _worker
    if _worker is None:
        models_dir = Path.home() / 'llm' / 'models'
        _worker = InferenceWorker(models_dir)
    return _worker


def get_pool() -> ModelPool:
    """Get multi-model pool (preferred)"""
    global _pool
    if _pool is None:
        models_dir = Path.home() / 'llm' / 'models'
        _pool = ModelPool(models_dir, max_models=3)
    return _pool
