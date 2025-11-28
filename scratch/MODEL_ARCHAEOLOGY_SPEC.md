# Model Archaeology: Interpretability Job System

**Author:** Claude + Russ
**Date:** 2025-11-28
**Status:** Design Spec - Ready for Implementation

---

## 1. Vision: The Lorekeeper of the Model's Brain

Model Archaeology treats interpretability as **periodic, heavy-but-safe jobs** that:

1. Load a checkpoint (quantized/bf16 for efficiency)
2. Run a fixed probe dataset
3. Compute structural metrics (weight norms, activation stats)
4. Track **drift** across checkpoints (which layers changed most?)
5. Write stable JSON artifacts for visualization
6. Emit Battle Log events for the timeline

The 3090 becomes the **Archaeologist** - a specialist worker that digs through checkpoints to understand how the model's brain is changing.

### What We Learn

| Insight | Question Answered |
|---------|-------------------|
| **Layer Drift** | Which layers changed most between checkpoints? |
| **Weight Stability** | Are any layers "frozen" or "thrashing"? |
| **Activation Health** | Are activations staying in reasonable ranges? |
| **Skill Impact** | Which layers change when BIN vs SY improves? |
| **Collapse Detection** | Early warning of mode collapse (uniform activations) |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINER (4090)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Training     │  │ Checkpoint   │  │ Job Dispatcher       │   │
│  │ Daemon       ├──► Ledger       ├──► (POST /api/jobs)     │   │
│  └──────────────┘  └──────────────┘  └──────────┬───────────┘   │
│                                                  │               │
│  ┌──────────────┐  ┌──────────────┐             │               │
│  │ Tavern UI    │◄─┤ Analysis API │             │               │
│  │ /analysis    │  │ /api/analysis│             │               │
│  └──────────────┘  └──────────────┘             │               │
└─────────────────────────────────────────────────│───────────────┘
                                                  │
                                                  │ (claim job)
                                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ARCHAEOLOGIST (3090)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Claiming     │  │ Model        │  │ analysis/            │   │
│  │ Worker       ├──► Loader       ├──► layer_stats.py       │   │
│  │ (analytics)  │  │ (bf16)       │  │ layer_drift.py       │   │
│  └──────────────┘  └──────────────┘  └──────────┬───────────┘   │
│                                                  │               │
│                                      ┌───────────▼───────────┐   │
│                                      │ campaigns/.../        │   │
│                                      │   analysis/           │   │
│                                      │     layer_stats/      │   │
│                                      │     layer_drift/      │   │
│                                      └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. New Job Types

### 3.1 `layer_stats` Job

Computes weight norms, activation statistics, and optional drift for a single checkpoint.

```python
# In jobs/registry.py

"layer_stats": JobTypeConfig(
    name="layer_stats",
    description="Compute per-layer weight and activation stats for a checkpoint",
    required_fields=["campaign_id", "hero_id", "checkpoint_path"],
    optional_fields=[
        "model_ref",                  # e.g., 'qwen3-0.6b', 'qwen3-4b'
        "reference_checkpoint_path",  # For drift calculation (optional)
        "probe_dataset",              # e.g., 'probes/core_v1.jsonl'
        "max_probe_tokens",           # Limit probe size (default: 4096)
        "output_path",                # Override default analysis dir
        "compute_activations",        # default: True
        "compute_drift",              # default: True if reference provided
    ],
    payload_version=1,
    default_timeout=1800,       # 30 min - model loading is slow
    max_attempts=1,             # Don't retry - fix the issue
    retryable_errors=[],
    allowed_roles=["analytics"],
    requires_gpu=True,          # Need GPU for activation computation
    max_pending=10,
    max_running=1,              # Only one at a time (VRAM constraint)
    queue_full_policy="warn",
)
```

### 3.2 `layer_drift` Job (Optional - Can Merge with layer_stats)

Compares two checkpoints without recomputing activations.

```python
"layer_drift": JobTypeConfig(
    name="layer_drift",
    description="Compare layer weight drift between two checkpoints",
    required_fields=[
        "campaign_id",
        "hero_id",
        "base_checkpoint_path",
        "target_checkpoint_path",
    ],
    optional_fields=[
        "metrics",              # ['l2', 'cosine', 'frobenius']
        "output_path",
    ],
    payload_version=1,
    default_timeout=600,        # 10 min - no inference needed
    max_attempts=1,
    retryable_errors=[],
    allowed_roles=["analytics"],
    requires_gpu=False,         # Can be CPU-only (loads state_dicts)
    max_pending=20,
    max_running=2,
    queue_full_policy="allow",
)
```

---

## 4. Device Configuration

### 4.1 Update `config/devices.json`

Add `analytics` role to inference3090:

```json
{
  "inference3090": {
    "hostname": "192.168.x.x",
    "description": "Inference server with RTX 3090 - also handles analytics",
    "roles": [
      "inference",
      "eval_worker",
      "analytics",       // <-- ADD THIS
      "storage_hot"
    ],
    "gpus": [
      {
        "name": "RTX 3090",
        "count": 1,
        "vram_gb": 24
      }
    ],
    ...
  }
}
```

### 4.2 Start Analytics Worker on 3090

```bash
# On 3090 machine
python3 -m workers.claiming_worker \
    --device inference3090 \
    --server http://trainer4090.local:8767 \
    --roles analytics,eval_worker \
    --inference-url http://localhost:8765
```

---

## 5. Analysis Module

### 5.1 Directory Structure

```
analysis/
├── __init__.py
├── layer_stats.py        # Core computation functions
├── layer_drift.py        # Drift-specific computations
├── probe_datasets.py     # Probe dataset management
├── model_loader.py       # Efficient model loading for analysis
└── schemas.py            # Pydantic/dataclass schemas
```

### 5.2 `analysis/schemas.py`

```python
"""Data schemas for Model Archaeology."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class LayerWeightStats:
    """Weight statistics for a single layer."""
    name: str
    layer_type: str  # 'attention', 'mlp', 'embedding', 'norm'

    # Weight norms by submodule
    weight_norms: Dict[str, float] = field(default_factory=dict)
    # e.g., {'q_proj': 12.34, 'k_proj': 11.2, 'v_proj': 10.8, 'o_proj': 9.5}

    # Bias norms (if present)
    bias_norms: Dict[str, float] = field(default_factory=dict)

    # Total parameter count
    param_count: int = 0


@dataclass
class LayerActivationStats:
    """Activation statistics for a single layer."""
    name: str
    mean: float
    std: float
    max: float
    min: float

    # Distribution info
    percentile_1: float = 0.0
    percentile_99: float = 0.0

    # Dead neuron detection
    zero_fraction: float = 0.0  # % of activations that are exactly 0


@dataclass
class LayerDriftStats:
    """Drift statistics for a single layer."""
    name: str

    # Per-submodule drift
    weight_l2: Dict[str, float] = field(default_factory=dict)
    # e.g., {'q_proj': 0.023, 'k_proj': 0.018, ...}

    weight_cosine: Dict[str, float] = field(default_factory=dict)
    # e.g., {'q_proj': 0.9991, 'k_proj': 0.9993, ...}

    # Aggregated
    total_l2: float = 0.0
    total_cosine: float = 0.0


@dataclass
class GlobalStats:
    """Aggregated statistics across all layers."""
    avg_weight_norm: float
    max_weight_norm: float
    min_weight_norm: float

    # Drift (if computed)
    avg_drift_l2: Optional[float] = None
    max_drift_l2: Optional[float] = None
    most_changed_layer: Optional[str] = None
    least_changed_layer: Optional[str] = None


@dataclass
class ProbeInfo:
    """Information about the probe dataset used."""
    dataset_id: str
    num_sequences: int
    total_tokens: int
    source_file: str


@dataclass
class LayerStatsResult:
    """Complete result from a layer_stats job."""
    # Identifiers
    version: int = 1
    campaign_id: str = ""
    hero_id: str = ""
    checkpoint_path: str = ""
    checkpoint_step: int = 0
    model_ref: str = ""

    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Weight stats
    weight_stats: Dict[str, LayerWeightStats] = field(default_factory=dict)
    global_weight_stats: Optional[GlobalStats] = None

    # Activation stats (optional)
    activation_stats: Dict[str, LayerActivationStats] = field(default_factory=dict)
    probe_info: Optional[ProbeInfo] = None

    # Drift stats (optional)
    drift_stats: Dict[str, LayerDriftStats] = field(default_factory=dict)
    reference_checkpoint_path: Optional[str] = None
    global_drift_stats: Optional[GlobalStats] = None

    # Metadata
    compute_duration_sec: float = 0.0
    device: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        import dataclasses
        return dataclasses.asdict(self)
```

### 5.3 `analysis/layer_stats.py`

```python
"""
Layer Statistics Computation

Computes:
- Per-layer weight norms (L2, Frobenius)
- Per-submodule breakdown (q_proj, k_proj, etc.)
- Activation statistics over probe dataset
- Drift vs reference checkpoint
"""

import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from .schemas import (
    LayerWeightStats,
    LayerActivationStats,
    LayerDriftStats,
    GlobalStats,
    ProbeInfo,
    LayerStatsResult,
)

logger = logging.getLogger("analysis.layer_stats")


# =============================================================================
# WEIGHT STATISTICS
# =============================================================================

def compute_weight_stats(
    model: torch.nn.Module,
    model_ref: str = "unknown",
) -> Tuple[Dict[str, LayerWeightStats], GlobalStats]:
    """
    Compute per-layer weight statistics.

    Args:
        model: The loaded model
        model_ref: Model reference string (for layer name patterns)

    Returns:
        Tuple of (layer_stats_dict, global_stats)
    """
    layer_stats = {}
    all_norms = []

    # Map common submodule patterns
    submodule_patterns = {
        'q_proj': 'attention',
        'k_proj': 'attention',
        'v_proj': 'attention',
        'o_proj': 'attention',
        'gate_proj': 'mlp',
        'up_proj': 'mlp',
        'down_proj': 'mlp',
        'embed_tokens': 'embedding',
        'lm_head': 'output',
        'input_layernorm': 'norm',
        'post_attention_layernorm': 'norm',
        'norm': 'norm',
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Parse layer info
        parts = name.split('.')

        # Determine layer type
        layer_type = 'unknown'
        for pattern, ltype in submodule_patterns.items():
            if pattern in name:
                layer_type = ltype
                break

        # Get containing layer name (e.g., "model.layers.0")
        layer_name = '.'.join(parts[:-1]) if len(parts) > 1 else name

        # Compute norm
        norm = param.data.norm(2).item()
        all_norms.append(norm)

        # Initialize layer if needed
        if layer_name not in layer_stats:
            layer_stats[layer_name] = LayerWeightStats(
                name=layer_name,
                layer_type=layer_type,
            )

        # Add to appropriate category
        submodule = parts[-1] if len(parts) > 1 else name
        if 'weight' in submodule or submodule in ['weight', 'embed_tokens', 'lm_head']:
            layer_stats[layer_name].weight_norms[submodule] = norm
        elif 'bias' in submodule:
            layer_stats[layer_name].bias_norms[submodule] = norm

        layer_stats[layer_name].param_count += param.numel()

    # Compute global stats
    global_stats = GlobalStats(
        avg_weight_norm=sum(all_norms) / len(all_norms) if all_norms else 0,
        max_weight_norm=max(all_norms) if all_norms else 0,
        min_weight_norm=min(all_norms) if all_norms else 0,
    )

    logger.info(f"Computed weight stats for {len(layer_stats)} layers")
    return layer_stats, global_stats


# =============================================================================
# ACTIVATION STATISTICS
# =============================================================================

def compute_activation_stats(
    model: torch.nn.Module,
    tokenizer,
    probe_sequences: List[str],
    max_tokens: int = 4096,
    device: str = "cuda",
) -> Tuple[Dict[str, LayerActivationStats], ProbeInfo]:
    """
    Compute activation statistics by running probes through the model.

    Args:
        model: The loaded model (in eval mode)
        tokenizer: Tokenizer for encoding probes
        probe_sequences: List of probe strings
        max_tokens: Max total tokens to process
        device: Device to run on

    Returns:
        Tuple of (activation_stats_dict, probe_info)
    """
    model.eval()
    activation_stats = {}
    hooks = []
    activations = {}  # layer_name -> list of activation tensors

    def make_hook(name):
        def hook(module, input, output):
            # Get output tensor
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            # Store flattened activations
            if name not in activations:
                activations[name] = []

            # Detach and move to CPU to save VRAM
            activations[name].append(out.detach().cpu())
        return hook

    # Register hooks on transformer layers
    for name, module in model.named_modules():
        # Hook into layer outputs
        if 'layers.' in name and name.count('.') == 2:  # model.layers.N
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Process probes
    total_tokens = 0
    num_sequences = 0

    with torch.no_grad():
        for seq in probe_sequences:
            if total_tokens >= max_tokens:
                break

            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            model(**inputs)

            total_tokens += inputs['input_ids'].shape[1]
            num_sequences += 1

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute statistics
    for name, acts in activations.items():
        if not acts:
            continue

        # Concatenate all activations
        all_acts = torch.cat([a.flatten() for a in acts])

        activation_stats[name] = LayerActivationStats(
            name=name,
            mean=all_acts.mean().item(),
            std=all_acts.std().item(),
            max=all_acts.max().item(),
            min=all_acts.min().item(),
            percentile_1=torch.quantile(all_acts.float(), 0.01).item(),
            percentile_99=torch.quantile(all_acts.float(), 0.99).item(),
            zero_fraction=(all_acts == 0).float().mean().item(),
        )

    probe_info = ProbeInfo(
        dataset_id="dynamic",
        num_sequences=num_sequences,
        total_tokens=total_tokens,
        source_file="inline",
    )

    logger.info(f"Computed activation stats from {num_sequences} sequences, {total_tokens} tokens")
    return activation_stats, probe_info


# =============================================================================
# DRIFT COMPUTATION
# =============================================================================

def compute_weight_drift(
    model: torch.nn.Module,
    reference_state_dict: Dict[str, torch.Tensor],
    model_ref: str = "unknown",
) -> Tuple[Dict[str, LayerDriftStats], GlobalStats]:
    """
    Compute weight drift between current model and reference.

    Args:
        model: Current model
        reference_state_dict: State dict from reference checkpoint
        model_ref: Model reference string

    Returns:
        Tuple of (drift_stats_dict, global_stats)
    """
    drift_stats = {}
    all_l2_drifts = []
    all_cosine_sims = []

    for name, param in model.named_parameters():
        if name not in reference_state_dict:
            continue

        ref_param = reference_state_dict[name]

        # Ensure same device
        ref_param = ref_param.to(param.device)

        # Compute L2 distance
        l2_diff = (param.data - ref_param).norm(2).item()
        all_l2_drifts.append(l2_diff)

        # Compute cosine similarity (flatten both)
        flat_cur = param.data.flatten()
        flat_ref = ref_param.flatten()
        cosine = torch.nn.functional.cosine_similarity(
            flat_cur.unsqueeze(0),
            flat_ref.unsqueeze(0)
        ).item()
        all_cosine_sims.append(cosine)

        # Get layer name
        parts = name.split('.')
        layer_name = '.'.join(parts[:-1]) if len(parts) > 1 else name
        submodule = parts[-1] if len(parts) > 1 else name

        if layer_name not in drift_stats:
            drift_stats[layer_name] = LayerDriftStats(name=layer_name)

        drift_stats[layer_name].weight_l2[submodule] = l2_diff
        drift_stats[layer_name].weight_cosine[submodule] = cosine

    # Compute per-layer totals
    for layer_name, stats in drift_stats.items():
        if stats.weight_l2:
            stats.total_l2 = sum(stats.weight_l2.values())
        if stats.weight_cosine:
            stats.total_cosine = sum(stats.weight_cosine.values()) / len(stats.weight_cosine)

    # Find most/least changed layers
    sorted_by_drift = sorted(
        [(name, stats.total_l2) for name, stats in drift_stats.items()],
        key=lambda x: x[1],
        reverse=True
    )

    global_stats = GlobalStats(
        avg_weight_norm=0,  # Not relevant for drift
        max_weight_norm=0,
        min_weight_norm=0,
        avg_drift_l2=sum(all_l2_drifts) / len(all_l2_drifts) if all_l2_drifts else 0,
        max_drift_l2=max(all_l2_drifts) if all_l2_drifts else 0,
        most_changed_layer=sorted_by_drift[0][0] if sorted_by_drift else None,
        least_changed_layer=sorted_by_drift[-1][0] if sorted_by_drift else None,
    )

    logger.info(f"Computed drift for {len(drift_stats)} layers")
    logger.info(f"Most changed: {global_stats.most_changed_layer}")
    logger.info(f"Least changed: {global_stats.least_changed_layer}")

    return drift_stats, global_stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_layer_stats_analysis(
    checkpoint_path: str,
    campaign_id: str,
    hero_id: str,
    model_ref: str = "qwen3-0.6b",
    reference_checkpoint_path: Optional[str] = None,
    probe_sequences: Optional[List[str]] = None,
    max_probe_tokens: int = 4096,
    compute_activations: bool = True,
    device: str = "cuda",
) -> LayerStatsResult:
    """
    Run complete layer stats analysis.

    Args:
        checkpoint_path: Path to checkpoint to analyze
        campaign_id: Campaign identifier
        hero_id: Hero identifier
        model_ref: Model reference for architecture detection
        reference_checkpoint_path: Optional reference for drift
        probe_sequences: Optional probe dataset
        max_probe_tokens: Max tokens for activation probing
        compute_activations: Whether to compute activation stats
        device: Device to use

    Returns:
        LayerStatsResult with all computed statistics
    """
    import time
    from .model_loader import load_model_for_analysis, load_tokenizer

    start_time = time.time()

    result = LayerStatsResult(
        campaign_id=campaign_id,
        hero_id=hero_id,
        checkpoint_path=checkpoint_path,
        model_ref=model_ref,
        device=device,
    )

    # Extract step from checkpoint path
    ckpt_name = Path(checkpoint_path).name
    if 'checkpoint-' in ckpt_name:
        try:
            result.checkpoint_step = int(ckpt_name.split('-')[1].split('_')[0])
        except:
            pass

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_model_for_analysis(checkpoint_path, model_ref, device=device)

    # Compute weight stats
    logger.info("Computing weight statistics...")
    weight_stats, global_weight = compute_weight_stats(model, model_ref)
    result.weight_stats = weight_stats
    result.global_weight_stats = global_weight

    # Compute activation stats
    if compute_activations and probe_sequences:
        logger.info("Computing activation statistics...")
        tokenizer = load_tokenizer(model_ref)
        act_stats, probe_info = compute_activation_stats(
            model, tokenizer, probe_sequences, max_probe_tokens, device
        )
        result.activation_stats = act_stats
        result.probe_info = probe_info

    # Compute drift
    if reference_checkpoint_path:
        logger.info(f"Computing drift vs {reference_checkpoint_path}")
        ref_state = torch.load(
            Path(reference_checkpoint_path) / "pytorch_model.bin",
            map_location=device
        )
        drift_stats, global_drift = compute_weight_drift(model, ref_state, model_ref)
        result.drift_stats = drift_stats
        result.global_drift_stats = global_drift
        result.reference_checkpoint_path = reference_checkpoint_path

    result.compute_duration_sec = time.time() - start_time
    logger.info(f"Analysis complete in {result.compute_duration_sec:.1f}s")

    return result
```

### 5.4 `analysis/model_loader.py`

```python
"""Efficient model loading for analysis."""

import logging
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger("analysis.model_loader")

# Model architectures
MODEL_CONFIGS = {
    "qwen3-0.6b": {
        "base_model": "Qwen/Qwen3-0.6B",
        "model_class": "Qwen2ForCausalLM",
        "tokenizer_class": "AutoTokenizer",
    },
    "qwen3-4b": {
        "base_model": "Qwen/Qwen3-4B",
        "model_class": "Qwen2ForCausalLM",
        "tokenizer_class": "AutoTokenizer",
    },
}


def load_model_for_analysis(
    checkpoint_path: str,
    model_ref: str = "qwen3-0.6b",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """
    Load model for analysis with memory efficiency.

    Uses bf16 precision to reduce memory footprint.
    Does NOT use 8-bit quantization to ensure accurate weight norms.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    ckpt_path = Path(checkpoint_path)

    # Load config
    config = AutoConfig.from_pretrained(str(ckpt_path), trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path),
        config=config,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    model.eval()

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {param_count/1e9:.2f}B parameters")

    return model


def load_tokenizer(model_ref: str = "qwen3-0.6b"):
    """Load tokenizer for the model."""
    from transformers import AutoTokenizer

    config = MODEL_CONFIGS.get(model_ref, MODEL_CONFIGS["qwen3-0.6b"])

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
    )

    return tokenizer


def load_reference_state_dict(
    checkpoint_path: str,
    device: str = "cpu",
) -> dict:
    """Load only the state dict from a checkpoint (memory efficient)."""
    ckpt_path = Path(checkpoint_path)

    # Try different file patterns
    patterns = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model-00001-of-*.bin",
    ]

    for pattern in patterns:
        matches = list(ckpt_path.glob(pattern))
        if matches:
            break

    if not matches:
        raise FileNotFoundError(f"No model files found in {checkpoint_path}")

    # Load safetensors if available (faster)
    if matches[0].suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(matches[0]), device=device)

    # Otherwise load pytorch
    return torch.load(str(matches[0]), map_location=device)
```

### 5.5 `analysis/probe_datasets.py`

```python
"""Probe dataset management for activation analysis."""

import json
from pathlib import Path
from typing import List, Optional

# Default probes - a fixed set for consistency
DEFAULT_PROBES = [
    # Math reasoning
    "What is 15 + 27?",
    "If x + 5 = 12, what is x?",
    "Calculate 144 / 12.",

    # Logic
    "If all cats are mammals and all mammals breathe air, do cats breathe air?",
    "True or False: If A implies B, and B is false, then A must be false.",

    # Language
    "What is the opposite of 'happy'?",
    "Complete this sentence: The quick brown fox jumps over the lazy ___.",
    "What rhymes with 'cat'?",

    # Binary (if BIN skill)
    "Convert 5 to binary.",
    "What is ①①① in decimal?",

    # Syllacrostic (if SY skill)
    "What are the first letters of: Apple, Banana, Cherry?",
]


def get_default_probes() -> List[str]:
    """Get default probe sequences."""
    return DEFAULT_PROBES.copy()


def load_probe_dataset(
    dataset_id: str,
    probes_dir: Optional[Path] = None,
    max_probes: int = 256,
) -> List[str]:
    """
    Load a probe dataset by ID.

    Args:
        dataset_id: Either 'default' or path to jsonl file
        probes_dir: Base directory for probe files
        max_probes: Maximum number of probes to load

    Returns:
        List of probe strings
    """
    if dataset_id == "default":
        return get_default_probes()[:max_probes]

    # Try to load from file
    if probes_dir:
        probe_file = probes_dir / dataset_id
    else:
        probe_file = Path(dataset_id)

    if not probe_file.exists():
        raise FileNotFoundError(f"Probe dataset not found: {probe_file}")

    probes = []
    with open(probe_file) as f:
        for line in f:
            if len(probes) >= max_probes:
                break
            data = json.loads(line)
            # Support different formats
            if "prompt" in data:
                probes.append(data["prompt"])
            elif "text" in data:
                probes.append(data["text"])
            elif "instruction" in data:
                probes.append(data["instruction"])

    return probes
```

---

## 6. Worker Integration

### 6.1 Update `workers/claiming_worker.py`

Add handlers for `layer_stats` and `layer_drift` jobs:

```python
# In ClaimingWorker._execute_job(), add:

elif job_type == "layer_stats":
    result = self._execute_layer_stats(payload)
elif job_type == "layer_drift":
    result = self._execute_layer_drift(payload)


# Add new methods:

def _execute_layer_stats(self, payload: Dict) -> Dict:
    """Execute a layer stats job."""
    from analysis.layer_stats import run_layer_stats_analysis
    from analysis.probe_datasets import load_probe_dataset, get_default_probes

    campaign_id = payload["campaign_id"]
    hero_id = payload["hero_id"]
    checkpoint_path = payload["checkpoint_path"]
    model_ref = payload.get("model_ref", "qwen3-0.6b")
    reference_path = payload.get("reference_checkpoint_path")
    probe_dataset = payload.get("probe_dataset", "default")
    max_tokens = payload.get("max_probe_tokens", 4096)
    compute_act = payload.get("compute_activations", True)

    logger.info(f"Running layer_stats: campaign={campaign_id}, ckpt={checkpoint_path}")

    # Load probes
    if probe_dataset == "default":
        probes = get_default_probes()
    else:
        probes = load_probe_dataset(probe_dataset)

    # Run analysis
    result = run_layer_stats_analysis(
        checkpoint_path=checkpoint_path,
        campaign_id=campaign_id,
        hero_id=hero_id,
        model_ref=model_ref,
        reference_checkpoint_path=reference_path,
        probe_sequences=probes if compute_act else None,
        max_probe_tokens=max_tokens,
        compute_activations=compute_act,
    )

    # Save to campaign analysis dir
    output_path = payload.get("output_path")
    if not output_path:
        output_path = self._get_analysis_path(campaign_id, hero_id, "layer_stats")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filename based on checkpoint step
    filename = f"ckpt-{result.checkpoint_step:06d}.layer_stats.json"
    filepath = output_path / filename

    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Saved layer stats to {filepath}")

    return {
        "success": True,
        "output_path": str(filepath),
        "checkpoint_step": result.checkpoint_step,
        "num_layers": len(result.weight_stats),
        "has_activations": bool(result.activation_stats),
        "has_drift": bool(result.drift_stats),
        "most_changed_layer": (
            result.global_drift_stats.most_changed_layer
            if result.global_drift_stats else None
        ),
        "duration_sec": result.compute_duration_sec,
    }


def _execute_layer_drift(self, payload: Dict) -> Dict:
    """Execute a layer drift comparison job."""
    from analysis.layer_stats import compute_weight_drift
    from analysis.model_loader import load_reference_state_dict

    campaign_id = payload["campaign_id"]
    hero_id = payload["hero_id"]
    base_path = payload["base_checkpoint_path"]
    target_path = payload["target_checkpoint_path"]

    logger.info(f"Computing drift: {base_path} -> {target_path}")

    # Load both state dicts (CPU to save VRAM)
    base_state = load_reference_state_dict(base_path, device="cpu")
    target_state = load_reference_state_dict(target_path, device="cpu")

    # Compute drift
    drift_stats = {}
    all_l2 = []

    for name in base_state:
        if name not in target_state:
            continue

        base_param = base_state[name]
        target_param = target_state[name]

        l2 = (target_param - base_param).norm(2).item()
        all_l2.append(l2)

        layer_name = '.'.join(name.split('.')[:-1])
        if layer_name not in drift_stats:
            drift_stats[layer_name] = {"l2_total": 0, "params": {}}

        drift_stats[layer_name]["params"][name] = l2
        drift_stats[layer_name]["l2_total"] += l2

    # Find extremes
    sorted_layers = sorted(
        drift_stats.items(),
        key=lambda x: x[1]["l2_total"],
        reverse=True
    )

    result = {
        "success": True,
        "base_checkpoint": base_path,
        "target_checkpoint": target_path,
        "layer_drift": drift_stats,
        "avg_drift_l2": sum(all_l2) / len(all_l2) if all_l2 else 0,
        "max_drift_l2": max(all_l2) if all_l2 else 0,
        "most_changed_layer": sorted_layers[0][0] if sorted_layers else None,
        "least_changed_layer": sorted_layers[-1][0] if sorted_layers else None,
    }

    # Save if output path specified
    output_path = payload.get("output_path")
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

    return result


def _get_analysis_path(self, campaign_id: str, hero_id: str, analysis_type: str) -> Path:
    """Get the analysis directory path for a campaign."""
    # Default: campaigns/{hero_id}/{campaign_id}/analysis/{type}/
    base = os.environ.get("TRAINING_BASE_DIR", "/path/to/training")
    return Path(base) / "campaigns" / hero_id / campaign_id / "analysis" / analysis_type
```

---

## 7. Storage Paths & File Format

### 7.1 Directory Layout

```
campaigns/
  {hero_id}/
    {campaign_id}/
      analysis/
        layer_stats/
          ckpt-000100.layer_stats.json
          ckpt-000200.layer_stats.json
          ckpt-001000.layer_stats.json
        layer_drift/
          drift-000100-to-000200.json
          drift-000200-to-001000.json
        summaries/
          layer_drift_timeline.json     # Aggregated time series
          stability_report.json         # Which layers are stable
```

### 7.2 `layer_stats.json` Schema

```json
{
  "version": 1,
  "campaign_id": "campaign-001",
  "hero_id": "dio-qwen3-0.6b",
  "checkpoint_path": "/path/to/checkpoint-183000",
  "checkpoint_step": 183000,
  "model_ref": "qwen3-0.6b",
  "created_at": "2025-11-28T14:30:00.000Z",

  "weight_stats": {
    "model.layers.0": {
      "name": "model.layers.0",
      "layer_type": "transformer_block",
      "weight_norms": {
        "self_attn.q_proj.weight": 12.34,
        "self_attn.k_proj.weight": 11.89,
        "self_attn.v_proj.weight": 10.45,
        "self_attn.o_proj.weight": 9.87,
        "mlp.gate_proj.weight": 15.23,
        "mlp.up_proj.weight": 14.98,
        "mlp.down_proj.weight": 13.45
      },
      "bias_norms": {},
      "param_count": 4194304
    },
    "...": "..."
  },

  "global_weight_stats": {
    "avg_weight_norm": 12.34,
    "max_weight_norm": 18.92,
    "min_weight_norm": 3.45,
    "avg_drift_l2": null,
    "max_drift_l2": null,
    "most_changed_layer": null,
    "least_changed_layer": null
  },

  "activation_stats": {
    "model.layers.0": {
      "name": "model.layers.0",
      "mean": 0.0023,
      "std": 0.987,
      "max": 6.12,
      "min": -5.98,
      "percentile_1": -2.34,
      "percentile_99": 2.56,
      "zero_fraction": 0.0001
    },
    "...": "..."
  },

  "probe_info": {
    "dataset_id": "default",
    "num_sequences": 12,
    "total_tokens": 384,
    "source_file": "inline"
  },

  "drift_stats": {
    "model.layers.0": {
      "name": "model.layers.0",
      "weight_l2": {
        "self_attn.q_proj.weight": 0.023,
        "self_attn.k_proj.weight": 0.018
      },
      "weight_cosine": {
        "self_attn.q_proj.weight": 0.9991,
        "self_attn.k_proj.weight": 0.9993
      },
      "total_l2": 0.156,
      "total_cosine": 0.9987
    }
  },

  "reference_checkpoint_path": "/path/to/checkpoint-180000",

  "global_drift_stats": {
    "avg_drift_l2": 0.05,
    "max_drift_l2": 0.11,
    "most_changed_layer": "model.layers.15",
    "least_changed_layer": "model.embed_tokens"
  },

  "compute_duration_sec": 145.3,
  "device": "cuda:0"
}
```

---

## 8. API Endpoints

### 8.1 VaultKeeper/Tavern Server Additions

Add to `vault/server.py` or `tavern/server.py`:

```python
# =============================================================================
# ANALYSIS API
# =============================================================================

@app.route('/api/analysis/<campaign_id>/layer_stats')
def list_layer_stats(campaign_id):
    """List available layer stats for a campaign."""
    hero_id = request.args.get('hero_id', 'dio-qwen3-0.6b')

    analysis_dir = BASE_DIR / "campaigns" / hero_id / campaign_id / "analysis" / "layer_stats"

    if not analysis_dir.exists():
        return jsonify({"stats": [], "count": 0})

    stats = []
    for f in sorted(analysis_dir.glob("*.layer_stats.json")):
        # Read just the summary fields
        with open(f) as fp:
            data = json.load(fp)

        stats.append({
            "checkpoint_step": data.get("checkpoint_step", 0),
            "created_at": data.get("created_at"),
            "has_drift": bool(data.get("drift_stats")),
            "has_activations": bool(data.get("activation_stats")),
            "most_changed_layer": (
                data.get("global_drift_stats", {}).get("most_changed_layer")
            ),
            "avg_weight_norm": data.get("global_weight_stats", {}).get("avg_weight_norm"),
            "filename": f.name,
        })

    return jsonify({
        "stats": stats,
        "count": len(stats),
        "campaign_id": campaign_id,
        "hero_id": hero_id,
    })


@app.route('/api/analysis/<campaign_id>/layer_stats/<checkpoint_name>')
def get_layer_stats(campaign_id, checkpoint_name):
    """Get full layer stats for a specific checkpoint."""
    hero_id = request.args.get('hero_id', 'dio-qwen3-0.6b')

    analysis_dir = BASE_DIR / "campaigns" / hero_id / campaign_id / "analysis" / "layer_stats"

    # Handle both "183000" and "ckpt-183000.layer_stats.json" formats
    if checkpoint_name.isdigit():
        filename = f"ckpt-{int(checkpoint_name):06d}.layer_stats.json"
    else:
        filename = checkpoint_name

    filepath = analysis_dir / filename

    if not filepath.exists():
        return jsonify({"error": f"Not found: {filename}"}), 404

    with open(filepath) as f:
        data = json.load(f)

    return jsonify(data)


@app.route('/api/analysis/<campaign_id>/drift_timeline')
def get_drift_timeline(campaign_id):
    """
    Get drift time series for visualization.

    Returns per-layer drift over checkpoints.
    """
    hero_id = request.args.get('hero_id', 'dio-qwen3-0.6b')
    layer_filter = request.args.get('layers')  # comma-separated layer names

    analysis_dir = BASE_DIR / "campaigns" / hero_id / campaign_id / "analysis" / "layer_stats"

    if not analysis_dir.exists():
        return jsonify({"error": "No analysis data found"}), 404

    # Collect all layer stats with drift data
    timeline = {
        "checkpoints": [],
        "layers": {},
    }

    for f in sorted(analysis_dir.glob("*.layer_stats.json")):
        with open(f) as fp:
            data = json.load(fp)

        if not data.get("drift_stats"):
            continue

        step = data.get("checkpoint_step", 0)
        timeline["checkpoints"].append(step)

        for layer_name, drift in data["drift_stats"].items():
            if layer_filter:
                allowed = layer_filter.split(',')
                if not any(a in layer_name for a in allowed):
                    continue

            if layer_name not in timeline["layers"]:
                timeline["layers"][layer_name] = {
                    "name": layer_name,
                    "drift_l2": [],
                    "drift_cosine": [],
                }

            timeline["layers"][layer_name]["drift_l2"].append(drift.get("total_l2", 0))
            timeline["layers"][layer_name]["drift_cosine"].append(drift.get("total_cosine", 1))

    return jsonify(timeline)


@app.route('/api/analysis/<campaign_id>/top_movers')
def get_top_movers(campaign_id):
    """Get layers that changed the most across training."""
    hero_id = request.args.get('hero_id', 'dio-qwen3-0.6b')
    top_n = int(request.args.get('n', 10))

    analysis_dir = BASE_DIR / "campaigns" / hero_id / campaign_id / "analysis" / "layer_stats"

    if not analysis_dir.exists():
        return jsonify({"error": "No analysis data found"}), 404

    # Accumulate total drift per layer
    layer_drift = {}

    for f in sorted(analysis_dir.glob("*.layer_stats.json")):
        with open(f) as fp:
            data = json.load(fp)

        if not data.get("drift_stats"):
            continue

        for layer_name, drift in data["drift_stats"].items():
            if layer_name not in layer_drift:
                layer_drift[layer_name] = 0
            layer_drift[layer_name] += drift.get("total_l2", 0)

    # Sort by total drift
    sorted_layers = sorted(
        layer_drift.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return jsonify({
        "top_movers": [
            {"layer": name, "total_drift": drift}
            for name, drift in sorted_layers[:top_n]
        ],
        "most_stable": [
            {"layer": name, "total_drift": drift}
            for name, drift in sorted_layers[-top_n:][::-1]
        ],
    })
```

---

## 9. Tavern UI

### 9.1 New Analysis Page

Create `tavern/templates/analysis.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Model Archaeology | Realm of Training</title>
    <link rel="stylesheet" href="/static/css/tavern.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <nav class="tavern-nav">
        <a href="/">Tavern</a>
        <a href="/quests">Quests</a>
        <a href="/jobs">Jobs</a>
        <a href="/analysis" class="active">Archaeology</a>
        <a href="/oracle">Oracle</a>
        <a href="/vault">Vault</a>
    </nav>

    <main class="analysis-container">
        <h1>Model Archaeology</h1>
        <p class="subtitle">Excavating the Hero's Neural Pathways</p>

        <!-- Campaign Selector -->
        <div class="campaign-selector">
            <label>Campaign:</label>
            <select id="campaign-select">
                <option value="campaign-001">DIO's First Campaign</option>
            </select>
        </div>

        <!-- Summary Cards -->
        <div class="summary-cards">
            <div class="card">
                <h3>Checkpoints Analyzed</h3>
                <div class="value" id="analyzed-count">-</div>
            </div>
            <div class="card">
                <h3>Most Changed Layer</h3>
                <div class="value" id="most-changed">-</div>
            </div>
            <div class="card">
                <h3>Most Stable Layer</h3>
                <div class="value" id="most-stable">-</div>
            </div>
            <div class="card">
                <h3>Average Drift</h3>
                <div class="value" id="avg-drift">-</div>
            </div>
        </div>

        <!-- Layer Drift Heatmap -->
        <section class="heatmap-section">
            <h2>Layer Drift Heatmap</h2>
            <p>Colors show how much each layer changed vs previous checkpoint</p>
            <div id="heatmap-container">
                <canvas id="drift-heatmap"></canvas>
            </div>
        </section>

        <!-- Top Movers -->
        <section class="movers-section">
            <h2>Top Movers vs Most Stable</h2>
            <div class="movers-grid">
                <div class="movers-column">
                    <h3>Most Changed (Cumulative)</h3>
                    <table id="top-movers-table">
                        <thead><tr><th>Layer</th><th>Total Drift</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="movers-column">
                    <h3>Most Stable</h3>
                    <table id="most-stable-table">
                        <thead><tr><th>Layer</th><th>Total Drift</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Per-Layer Detail -->
        <section class="layer-detail-section">
            <h2>Layer Detail</h2>
            <select id="layer-select">
                <option value="">Select a layer...</option>
            </select>
            <div id="layer-detail">
                <canvas id="layer-drift-chart"></canvas>
            </div>
        </section>

        <!-- Manual Trigger -->
        <section class="trigger-section">
            <h2>Run Analysis</h2>
            <div class="trigger-form">
                <label>Checkpoint Step:</label>
                <input type="number" id="checkpoint-step" placeholder="e.g., 183000">

                <label>Reference Step (optional):</label>
                <input type="number" id="reference-step" placeholder="e.g., 180000">

                <button id="run-analysis-btn" class="btn-primary">
                    Run Layer Stats Analysis
                </button>
            </div>
            <div id="trigger-status"></div>
        </section>

        <!-- Checkpoint List -->
        <section class="checkpoint-list-section">
            <h2>Analyzed Checkpoints</h2>
            <table id="checkpoints-table">
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Analyzed At</th>
                        <th>Has Drift</th>
                        <th>Most Changed</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </section>
    </main>

    <script src="/static/js/analysis.js"></script>
</body>
</html>
```

### 9.2 `static/js/analysis.js`

```javascript
// Analysis page JavaScript

const API_BASE = '';
let currentCampaign = 'campaign-001';
let heroId = 'dio-qwen3-0.6b';

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    loadLayerStats();
    loadTopMovers();
    loadDriftTimeline();

    document.getElementById('run-analysis-btn').addEventListener('click', runAnalysis);
    document.getElementById('campaign-select').addEventListener('change', (e) => {
        currentCampaign = e.target.value;
        loadLayerStats();
        loadTopMovers();
        loadDriftTimeline();
    });
});

async function loadLayerStats() {
    const response = await fetch(
        `${API_BASE}/api/analysis/${currentCampaign}/layer_stats?hero_id=${heroId}`
    );
    const data = await response.json();

    document.getElementById('analyzed-count').textContent = data.count;

    // Populate table
    const tbody = document.querySelector('#checkpoints-table tbody');
    tbody.innerHTML = '';

    for (const stat of data.stats) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${stat.checkpoint_step}</td>
            <td>${new Date(stat.created_at).toLocaleString()}</td>
            <td>${stat.has_drift ? '✓' : '-'}</td>
            <td>${stat.most_changed_layer || '-'}</td>
            <td>
                <button onclick="viewDetail(${stat.checkpoint_step})">View</button>
            </td>
        `;
        tbody.appendChild(row);
    }
}

async function loadTopMovers() {
    const response = await fetch(
        `${API_BASE}/api/analysis/${currentCampaign}/top_movers?hero_id=${heroId}&n=10`
    );
    const data = await response.json();

    // Update summary cards
    if (data.top_movers && data.top_movers.length > 0) {
        document.getElementById('most-changed').textContent =
            data.top_movers[0].layer.split('.').pop();
    }
    if (data.most_stable && data.most_stable.length > 0) {
        document.getElementById('most-stable').textContent =
            data.most_stable[0].layer.split('.').pop();
    }

    // Populate tables
    const topTable = document.querySelector('#top-movers-table tbody');
    topTable.innerHTML = '';
    for (const item of data.top_movers || []) {
        topTable.innerHTML += `
            <tr>
                <td>${item.layer}</td>
                <td>${item.total_drift.toFixed(4)}</td>
            </tr>
        `;
    }

    const stableTable = document.querySelector('#most-stable-table tbody');
    stableTable.innerHTML = '';
    for (const item of data.most_stable || []) {
        stableTable.innerHTML += `
            <tr>
                <td>${item.layer}</td>
                <td>${item.total_drift.toFixed(4)}</td>
            </tr>
        `;
    }

    // Populate layer selector
    const select = document.getElementById('layer-select');
    select.innerHTML = '<option value="">Select a layer...</option>';
    for (const item of data.top_movers || []) {
        select.innerHTML += `<option value="${item.layer}">${item.layer}</option>`;
    }
}

async function loadDriftTimeline() {
    const response = await fetch(
        `${API_BASE}/api/analysis/${currentCampaign}/drift_timeline?hero_id=${heroId}`
    );
    const data = await response.json();

    if (!data.checkpoints || data.checkpoints.length === 0) {
        return;
    }

    // Calculate average drift
    let totalDrift = 0;
    let count = 0;
    for (const layer of Object.values(data.layers)) {
        for (const d of layer.drift_l2) {
            totalDrift += d;
            count++;
        }
    }
    document.getElementById('avg-drift').textContent =
        count > 0 ? (totalDrift / count).toFixed(4) : '-';

    // Create heatmap
    createDriftHeatmap(data);
}

function createDriftHeatmap(data) {
    const ctx = document.getElementById('drift-heatmap').getContext('2d');

    // Prepare data for heatmap
    const layers = Object.keys(data.layers).slice(0, 20); // Top 20 layers
    const checkpoints = data.checkpoints;

    const heatmapData = [];
    for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        const drifts = data.layers[layer].drift_l2;
        for (let j = 0; j < checkpoints.length; j++) {
            heatmapData.push({
                x: checkpoints[j],
                y: i,
                v: drifts[j] || 0
            });
        }
    }

    // Simple bar chart as fallback (heatmap requires plugin)
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: checkpoints.map(c => `Step ${c}`),
            datasets: layers.slice(0, 5).map((layer, i) => ({
                label: layer.split('.').slice(-2).join('.'),
                data: data.layers[layer].drift_l2,
                backgroundColor: `hsl(${i * 60}, 70%, 50%)`,
            }))
        },
        options: {
            responsive: true,
            scales: {
                y: { title: { display: true, text: 'L2 Drift' } },
                x: { title: { display: true, text: 'Checkpoint' } }
            }
        }
    });
}

async function runAnalysis() {
    const step = document.getElementById('checkpoint-step').value;
    const refStep = document.getElementById('reference-step').value;

    if (!step) {
        alert('Please enter a checkpoint step');
        return;
    }

    const statusEl = document.getElementById('trigger-status');
    statusEl.textContent = 'Submitting job...';

    // Construct checkpoint path (this may need adjustment based on your setup)
    const checkpointPath = `/path/to/campaigns/${heroId}/${currentCampaign}/checkpoints/checkpoint-${step}`;

    const payload = {
        job_type: 'layer_stats',
        payload: {
            campaign_id: currentCampaign,
            hero_id: heroId,
            checkpoint_path: checkpointPath,
        }
    };

    if (refStep) {
        payload.payload.reference_checkpoint_path =
            `/path/to/campaigns/${heroId}/${currentCampaign}/checkpoints/checkpoint-${refStep}`;
    }

    try {
        const response = await fetch(`${API_BASE}/api/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (result.accepted) {
            statusEl.textContent = `Job submitted: ${result.job_id}`;
            statusEl.className = 'status-success';
        } else {
            statusEl.textContent = `Failed: ${result.message}`;
            statusEl.className = 'status-error';
        }
    } catch (e) {
        statusEl.textContent = `Error: ${e.message}`;
        statusEl.className = 'status-error';
    }
}

async function viewDetail(step) {
    const response = await fetch(
        `${API_BASE}/api/analysis/${currentCampaign}/layer_stats/${step}?hero_id=${heroId}`
    );
    const data = await response.json();

    console.log('Layer stats detail:', data);
    alert(`Checkpoint ${step}\nLayers: ${Object.keys(data.weight_stats || {}).length}\nCompute time: ${data.compute_duration_sec?.toFixed(1)}s`);
}
```

---

## 10. Events Integration (Battle Log)

### 10.1 Add Event Types

In `events/types.py`:

```python
class EventType(str, Enum):
    # ... existing events ...

    # Analytics Events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"


# Factory functions
def analysis_started_event(job_type: str, checkpoint_step: int) -> Event:
    """Analysis job started."""
    return Event(
        type=EventType.ANALYSIS_STARTED,
        message=f"Analyzing checkpoint {checkpoint_step} ({job_type})",
        severity=Severity.INFO,
        source="archaeologist",
        data={"job_type": job_type, "checkpoint_step": checkpoint_step},
    )


def analysis_completed_event(
    job_type: str,
    checkpoint_step: int,
    most_changed_layer: str = None,
    duration_sec: float = 0,
) -> Event:
    """Analysis job completed."""
    msg = f"Analysis complete for checkpoint {checkpoint_step}"
    if most_changed_layer:
        msg += f" (most changed: {most_changed_layer})"

    return Event(
        type=EventType.ANALYSIS_COMPLETED,
        message=msg,
        severity=Severity.SUCCESS,
        source="archaeologist",
        data={
            "job_type": job_type,
            "checkpoint_step": checkpoint_step,
            "most_changed_layer": most_changed_layer,
            "duration_sec": duration_sec,
        },
    )
```

### 10.2 Emit Events from Worker

In the worker's `_execute_layer_stats`:

```python
def _execute_layer_stats(self, payload: Dict) -> Dict:
    from events import get_broadcaster, analysis_started_event, analysis_completed_event

    broadcaster = get_broadcaster()
    checkpoint_step = ... # extract from path

    # Emit start event
    broadcaster.emit(analysis_started_event("layer_stats", checkpoint_step))

    try:
        # ... run analysis ...
        result = run_layer_stats_analysis(...)

        # Emit completion event
        broadcaster.emit(analysis_completed_event(
            "layer_stats",
            result.checkpoint_step,
            most_changed_layer=result.global_drift_stats.most_changed_layer if result.global_drift_stats else None,
            duration_sec=result.compute_duration_sec,
        ))

        return {...}
    except Exception as e:
        broadcaster.emit(Event(
            type=EventType.ANALYSIS_FAILED,
            message=f"Analysis failed: {e}",
            severity=Severity.ERROR,
            source="archaeologist",
        ))
        raise
```

---

## 11. Job Triggering

### 11.1 Automatic Trigger on Checkpoint Save

In `trainer/monitoring/callbacks.py`, in `on_save`:

```python
def on_save(self, args, state, control, **kwargs):
    # ... existing ledger + eval queue logic ...

    # Queue layer stats analysis (optional, based on config)
    if self.config.get("auto_analyze_checkpoints", False):
        self._queue_layer_stats(checkpoint_path, state.global_step)


def _queue_layer_stats(self, checkpoint_path: str, step: int):
    """Queue a layer_stats job for this checkpoint."""
    import requests

    # Get reference checkpoint (previous analyzed checkpoint)
    reference = self._get_last_analyzed_checkpoint()

    payload = {
        "job_type": "layer_stats",
        "payload": {
            "campaign_id": self.campaign_id,
            "hero_id": self.hero_id,
            "checkpoint_path": checkpoint_path,
            "reference_checkpoint_path": reference,
            "model_ref": self.model_ref,
        },
        "priority": "low",
    }

    try:
        response = requests.post(
            f"{self.job_server_url}/api/jobs",
            json=payload,
            timeout=5,
        )
        if response.status_code == 200:
            logger.info(f"Queued layer_stats job for step {step}")
    except Exception as e:
        logger.warning(f"Failed to queue layer_stats: {e}")
```

### 11.2 Manual Trigger via CLI

Create `scripts/run_layer_stats.py`:

```python
#!/usr/bin/env python3
"""Manually trigger layer stats analysis."""

import argparse
import requests
import sys

def main():
    parser = argparse.ArgumentParser(description="Trigger layer stats analysis")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--reference", help="Reference checkpoint path")
    parser.add_argument("--campaign", default="campaign-001")
    parser.add_argument("--hero", default="dio-qwen3-0.6b")
    parser.add_argument("--server", default="http://localhost:8767")

    args = parser.parse_args()

    payload = {
        "job_type": "layer_stats",
        "payload": {
            "campaign_id": args.campaign,
            "hero_id": args.hero,
            "checkpoint_path": args.checkpoint,
        }
    }

    if args.reference:
        payload["payload"]["reference_checkpoint_path"] = args.reference

    response = requests.post(f"{args.server}/api/jobs", json=payload)
    result = response.json()

    if result.get("accepted"):
        print(f"Job submitted: {result['job_id']}")
    else:
        print(f"Failed: {result.get('message', 'unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## 12. Testing Strategy

### 12.1 Unit Tests

```python
# tests/test_layer_stats.py

import pytest
import torch
from analysis.layer_stats import compute_weight_stats, compute_weight_drift
from analysis.schemas import LayerStatsResult


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)


def test_compute_weight_stats():
    model = MockModel()
    stats, global_stats = compute_weight_stats(model)

    assert len(stats) >= 2
    assert global_stats.avg_weight_norm > 0
    assert global_stats.max_weight_norm >= global_stats.min_weight_norm


def test_compute_weight_drift():
    model = MockModel()

    # Create reference state (same as current)
    ref_state = {k: v.clone() for k, v in model.state_dict().items()}

    drift, global_drift = compute_weight_drift(model, ref_state)

    # Drift should be zero for identical weights
    assert global_drift.avg_drift_l2 < 1e-6


def test_layer_stats_result_serialization():
    result = LayerStatsResult(
        campaign_id="test",
        hero_id="test-hero",
        checkpoint_path="/tmp/test",
    )

    d = result.to_dict()
    assert d["campaign_id"] == "test"
    assert "version" in d
```

### 12.2 Integration Tests

```python
# tests/test_layer_stats_integration.py

import pytest
from pathlib import Path

@pytest.mark.integration
def test_full_layer_stats_analysis():
    """Test complete analysis pipeline on a real checkpoint."""
    from analysis.layer_stats import run_layer_stats_analysis
    from analysis.probe_datasets import get_default_probes

    # Use a known checkpoint
    checkpoint = Path("/path/to/test/checkpoint")
    if not checkpoint.exists():
        pytest.skip("Test checkpoint not available")

    result = run_layer_stats_analysis(
        checkpoint_path=str(checkpoint),
        campaign_id="test",
        hero_id="test",
        model_ref="qwen3-0.6b",
        probe_sequences=get_default_probes(),
        max_probe_tokens=256,
    )

    assert result.checkpoint_step > 0
    assert len(result.weight_stats) > 0
    assert result.compute_duration_sec > 0
```

---

## 13. Phased Rollout

### Phase 1: Core Infrastructure (Day 1-2)
- [ ] Add `layer_stats` job type to registry
- [ ] Create `analysis/` module with schemas
- [ ] Implement `compute_weight_stats()`
- [ ] Add worker handler for `layer_stats`
- [ ] Test on a single checkpoint manually

### Phase 2: Drift & Activations (Day 3-4)
- [ ] Implement `compute_weight_drift()`
- [ ] Implement `compute_activation_stats()`
- [ ] Add probe dataset management
- [ ] Test drift between two checkpoints

### Phase 3: API & Storage (Day 5)
- [ ] Add analysis API endpoints
- [ ] Define storage paths
- [ ] Implement `drift_timeline` aggregation

### Phase 4: Tavern UI (Day 6-7)
- [ ] Create analysis.html page
- [ ] Add summary cards
- [ ] Implement drift heatmap/chart
- [ ] Add manual trigger form

### Phase 5: Integration (Day 8)
- [ ] Add Battle Log events
- [ ] Wire up automatic triggering
- [ ] Add CLI script
- [ ] Documentation

### Phase 6: Polish (Day 9-10)
- [ ] Performance optimization
- [ ] Error handling
- [ ] Unit tests
- [ ] Integration tests

---

## 14. File Changes Checklist

| File | Action | Description |
|------|--------|-------------|
| `jobs/registry.py` | EDIT | Add `layer_stats`, `layer_drift` job types |
| `config/devices.json` | EDIT | Add `analytics` role to inference3090 |
| `analysis/__init__.py` | CREATE | Module exports |
| `analysis/schemas.py` | CREATE | Data schemas |
| `analysis/layer_stats.py` | CREATE | Core computation |
| `analysis/layer_drift.py` | CREATE | Drift computation |
| `analysis/probe_datasets.py` | CREATE | Probe management |
| `analysis/model_loader.py` | CREATE | Efficient loading |
| `workers/claiming_worker.py` | EDIT | Add `_execute_layer_stats` |
| `vault/server.py` OR `tavern/server.py` | EDIT | Add analysis API endpoints |
| `tavern/templates/analysis.html` | CREATE | Analysis UI |
| `tavern/static/js/analysis.js` | CREATE | UI JavaScript |
| `events/types.py` | EDIT | Add analysis events |
| `trainer/monitoring/callbacks.py` | EDIT | Add auto-trigger (optional) |
| `scripts/run_layer_stats.py` | CREATE | Manual CLI trigger |
| `tests/test_layer_stats.py` | CREATE | Unit tests |

---

## 15. Open Questions

1. **Reference checkpoint selection**: Should we compare to:
   - Base model (step 0)?
   - Previous analyzed checkpoint?
   - Best checkpoint so far?
   - User-specified reference?

   **Recommendation**: Make configurable, default to previous analyzed.

2. **Activation probes**: Should we:
   - Use fixed probes (consistent across all analysis)?
   - Use skill-specific probes (BIN problems, SY puzzles)?
   - Both?

   **Recommendation**: Start with fixed probes, add skill-specific later.

3. **Storage zone**: Where should analysis results live?
   - HOT zone (local fast storage)?
   - WARM zone (NAS)?

   **Recommendation**: HOT for recent, archive to WARM after 7 days.

4. **Frequency**: How often to analyze?
   - Every checkpoint?
   - Every N checkpoints?
   - Only on skill level-up?

   **Recommendation**: Configurable, default every 5 checkpoints or on demand.

---

## Summary

Model Archaeology turns the 3090 into an interpretability specialist that:

1. **Computes structural metrics** - Weight norms, activation stats
2. **Tracks drift** - Which layers change most over training
3. **Stores results** - JSON artifacts in `campaigns/.../analysis/`
4. **Visualizes in Tavern** - Heatmaps, time series, top movers
5. **Integrates with Job System V2** - Proper queuing, leases, error handling
6. **Emits Battle Log events** - Real-time visibility

The design is modular, testable, and integrates cleanly with the existing RPG framework. The Archaeologist digs through checkpoints so you can understand *how* DIO is learning, not just *that* DIO is learning.
