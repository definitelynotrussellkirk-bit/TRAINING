#!/usr/bin/env python3
"""
GPU Inference Worker - Multi-Model Pool
Manages multiple models in VRAM with explicit model selection

Supports:
- Full model checkpoints (model.safetensors)
- PEFT/LoRA adapters (adapter_model.safetensors)
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Tuple
from collections import OrderedDict
import json

# Optional PEFT support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None
    PeftConfig = None


# ============================================================
# Base Model Path Remapping
# ============================================================
# Maps trainer machine paths → inference machine paths
# This allows PEFT adapters trained on 4090 to load on 3090

# Get paths from environment or use sensible defaults
_HOME = os.path.expanduser("~")
_LOCAL_MODELS = os.environ.get("INFERENCE_MODELS_DIR", os.path.join(_HOME, "llm", "models"))
_NAS_MODELS = os.environ.get("INFERENCE_NAS_MODELS", "/mnt/synology/data/models")

# Search paths for base models (in priority order)
# 1. Local SSD for hot models
# 2. Synology NAS mount for overflow (55TB available)
BASE_MODEL_SEARCH_PATHS = [
    _LOCAL_MODELS,
    _NAS_MODELS,
]

# Build path mappings dynamically - maps any training machine paths to local inference paths
# Handles paths like /home/*/Desktop/TRAINING/models/* and /home/*/TRAINING/models/*
def _build_model_mapping() -> dict:
    """Build path mappings from training machine paths to inference local paths."""
    # Known model names that might be referenced by training configs
    model_names = [
        "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
        "Qwen3-4B-Instruct-2507", "Qwen2.5-3B", "Qwen2.5-7B",
    ]

    mapping = {}

    # Get training user from env or default to current user
    train_user = os.environ.get("TRAINING_USER", os.environ.get("USER", os.getlogin()))

    # Common training path patterns - built from config, not hardcoded
    home_base = "/home"  # Linux standard, not user-specific
    train_patterns = [
        f"{home_base}/{train_user}/Desktop/TRAINING/models/{{model}}",
        f"{home_base}/{train_user}/TRAINING/models/{{model}}",
    ]

    for model in model_names:
        local_path = os.path.join(_LOCAL_MODELS, model)
        for pattern in train_patterns:
            train_path = pattern.format(model=model)
            mapping[train_path] = local_path

    return mapping

BASE_MODEL_MAPPING = _build_model_mapping()

def resolve_base_model_path(path: str) -> str:
    """
    Resolve a base model path, remapping trainer paths to local paths if needed.

    Search order:
    1. Direct mapping from BASE_MODEL_MAPPING (if path exists)
    2. Original path (if exists - same machine scenario)
    3. Search in BASE_MODEL_SEARCH_PATHS by model name
    4. Fallback to HuggingFace model ID
    """
    # Check direct mapping first
    if path in BASE_MODEL_MAPPING:
        resolved = BASE_MODEL_MAPPING[path]
        if Path(resolved).exists():
            print(f"   📍 Remapped base model path: {path} → {resolved}")
            return resolved

    # Check if original path exists (same machine)
    if Path(path).exists():
        return path

    # Search in known base model locations by model name
    model_name = Path(path).name
    for search_path in BASE_MODEL_SEARCH_PATHS:
        candidate = Path(search_path) / model_name
        if candidate.exists():
            print(f"   📍 Found base model in search path: {candidate}")
            return str(candidate)

    # Try HuggingFace model ID as last resort
    hf_mapping = {
        "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
        "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
        "Qwen3-4B": "Qwen/Qwen3-4B",
        "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct",  # Local variant → HF base
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    }

    if model_name in hf_mapping:
        hf_id = hf_mapping[model_name]
        print(f"   🌐 Using HuggingFace model: {hf_id} (local path not found: {path})")
        return hf_id

    # Return original path (will likely fail, but with clear error)
    print(f"   ⚠️ Base model path not found and no mapping: {path}")
    return path


class ModelEntry:
    """Single model entry in the pool"""
    def __init__(self, model_id: str, model, tokenizer, path: str,
                 is_peft: bool = False, base_model_path: Optional[str] = None):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.path = path
        self.is_peft = is_peft
        self.base_model_path = base_model_path
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
                "vram_mb": round(vram_mb, 1),
                "is_peft": entry.is_peft,
                "base_model_path": entry.base_model_path
            })

        return {
            "loaded_count": len(self.pool),
            "max_models": self.max_models,
            "total_vram_mb": round(total_vram, 1),
            "peft_available": PEFT_AVAILABLE,
            "models": models
        }

    def is_loaded(self, model_id: str) -> bool:
        """Check if model is loaded"""
        return model_id in self.pool

    def _is_peft_checkpoint(self, model_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Check if a model path is a PEFT/LoRA adapter checkpoint.

        Returns:
            Tuple of (is_peft, base_model_path)
        """
        adapter_config_path = model_path / "adapter_config.json"
        if not adapter_config_path.exists():
            return False, None

        try:
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
            return True, base_model_path
        except Exception as e:
            print(f"Warning: Could not read adapter_config.json: {e}")
            return False, None

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

    def load_model(self, model_id: str, path: Optional[str] = None,
                   load_in_4bit: bool = False) -> bool:
        """
        Load a model into the pool.

        Supports both full model checkpoints and PEFT/LoRA adapters.
        PEFT adapters are automatically detected via adapter_config.json.

        Args:
            model_id: Unique identifier (e.g., "checkpoint-175000", "Qwen3-0.6B")
            path: Full path to model. If None, uses models_dir/model_id
            load_in_4bit: Force 4-bit quantization (auto-detected for QLoRA adapters)

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

        # Check if this is a PEFT adapter checkpoint
        is_peft, base_model_path = self._is_peft_checkpoint(model_path)

        if is_peft:
            return self._load_peft_model(model_id, model_path, base_model_path, load_in_4bit)
        else:
            return self._load_full_model(model_id, model_path, load_in_4bit)

    def _load_full_model(self, model_id: str, model_path: Path,
                         load_in_4bit: bool = False) -> bool:
        """Load a full model checkpoint."""
        print(f"📦 Loading full model: {model_id} from {model_path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # IMPORTANT: Use device_map="cuda:0" instead of "auto" to prevent split-device issues.
            # device_map="auto" can place embedding layers on CPU while other layers on CUDA,
            # causing "Expected all tensors to be on the same device" errors during index_select.
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "device_map": "cuda:0"  # Force all on CUDA to avoid device mismatch
            }

            if load_in_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            try:
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # Fall back to device_map="auto" if CUDA OOM
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    print(f"   ⚠️ CUDA OOM with device_map='cuda:0', falling back to 'auto'")
                    load_kwargs["device_map"] = "auto"
                    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
                else:
                    raise

            entry = ModelEntry(model_id, model, tokenizer, str(model_path),
                             is_peft=False, base_model_path=None)
            self.pool[model_id] = entry

            vram_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
            print(f"✓ Full model loaded: {model_id} ({vram_mb:.0f}MB VRAM)")

            return True

        except Exception as e:
            print(f"❌ Failed to load full model {model_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_peft_model(self, model_id: str, adapter_path: Path,
                         base_model_path: Optional[str],
                         load_in_4bit: bool = False) -> bool:
        """
        Load a PEFT/LoRA adapter checkpoint.

        This loads the base model first, then applies the adapter.
        For QLoRA adapters, the base model is loaded in 4-bit.
        """
        if not PEFT_AVAILABLE:
            print(f"❌ PEFT not available. Install with: pip install peft")
            print(f"   Attempting to load {model_id} as full model instead...")
            # Fall back to trying to load as full model (will fail for pure adapters)
            return self._load_full_model(model_id, adapter_path, load_in_4bit)

        if not base_model_path:
            print(f"❌ No base_model_name_or_path in adapter_config.json for {model_id}")
            return False

        print(f"📦 Loading PEFT adapter: {model_id}")
        print(f"   Adapter: {adapter_path}")
        print(f"   Base model (original): {base_model_path}")

        # Resolve base model path (handles remapping between machines)
        resolved_base_path = resolve_base_model_path(base_model_path)
        if resolved_base_path != base_model_path:
            print(f"   Base model (resolved): {resolved_base_path}")

        try:
            # Load tokenizer from adapter path (has any custom tokens)
            # Fall back to base model if adapter doesn't have tokenizer
            tokenizer_path = adapter_path if (adapter_path / "tokenizer_config.json").exists() else resolved_base_path
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )

            # Check if this was a QLoRA adapter (trained with 4-bit base)
            # We detect this by checking the adapter_config for quantization hints
            adapter_config_path = adapter_path / "adapter_config.json"
            use_4bit = load_in_4bit
            if adapter_config_path.exists():
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                # If adapter was trained with quantized base, we should load the same way
                # Common indicators: presence of certain fields or naming conventions
                if "qlora" in str(adapter_path).lower() or adapter_config.get("use_qalora", False):
                    use_4bit = True
                    print(f"   Detected QLoRA adapter, loading base model in 4-bit")

            # AUTO-QUANTIZE: For 7B+ models on 24GB VRAM, auto-enable 4-bit to prevent
            # device mismatch errors from memory pressure causing CPU offloading
            if not use_4bit:
                base_model_name = Path(resolved_base_path).name.lower()
                # Detect model size from name - look for patterns like "8b", "7b", "13b"
                # Must handle names like "qwen3-8b" where we want 8, not 3
                import re
                # Find all numbers followed by 'b' and take the largest (model size)
                size_matches = re.findall(r'(\d+)b', base_model_name)
                if size_matches:
                    model_size_b = max(int(s) for s in size_matches)
                    if model_size_b >= 7:
                        use_4bit = True
                        print(f"   🔧 Auto-enabling 4-bit quantization for {model_size_b}B model (prevents OOM/device mismatch)")

            # Load base model
            # IMPORTANT: Use device_map="cuda:0" instead of "auto" to prevent split-device issues.
            # device_map="auto" can place embedding layers on CPU while other layers on CUDA,
            # causing "Expected all tensors to be on the same device" errors during index_select.
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "device_map": "cuda:0"  # Force all on CUDA to avoid device mismatch
            }

            if use_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            try:
                base_model = AutoModelForCausalLM.from_pretrained(resolved_base_path, **load_kwargs)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # Fall back to device_map="auto" if CUDA OOM
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    print(f"   ⚠️ CUDA OOM with device_map='cuda:0', falling back to 'auto'")
                    load_kwargs["device_map"] = "auto"
                    base_model = AutoModelForCausalLM.from_pretrained(resolved_base_path, **load_kwargs)
                else:
                    raise

            # Load PEFT adapter on top of base model
            model = PeftModel.from_pretrained(base_model, adapter_path)

            # Put model in eval mode
            model.eval()

            entry = ModelEntry(model_id, model, tokenizer, str(adapter_path),
                             is_peft=True, base_model_path=base_model_path)
            self.pool[model_id] = entry

            # Calculate VRAM (includes both base and adapter)
            vram_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
            print(f"✓ PEFT adapter loaded: {model_id} ({vram_mb:.0f}MB VRAM, 4-bit={use_4bit})")

            return True

        except Exception as e:
            print(f"❌ Failed to load PEFT adapter {model_id}: {e}")
            import traceback
            traceback.print_exc()
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
