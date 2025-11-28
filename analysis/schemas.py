"""
Data schemas for Model Archaeology.

These dataclasses define the structure of analysis results,
ensuring consistent JSON output across all analysis jobs.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import json


@dataclass
class LayerWeightStats:
    """Weight statistics for a single layer."""

    name: str
    layer_type: str  # 'attention', 'mlp', 'embedding', 'norm', 'output'

    # Weight norms by submodule
    # e.g., {'q_proj': 12.34, 'k_proj': 11.2, 'v_proj': 10.8, 'o_proj': 9.5}
    weight_norms: Dict[str, float] = field(default_factory=dict)

    # Bias norms (if present)
    bias_norms: Dict[str, float] = field(default_factory=dict)

    # Total parameter count in this layer
    param_count: int = 0

    # Aggregated norm for this layer
    total_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerActivationStats:
    """Activation statistics for a single layer over probe dataset."""

    name: str
    mean: float
    std: float
    max: float
    min: float

    # Distribution info
    percentile_1: float = 0.0
    percentile_99: float = 0.0

    # Health indicators
    zero_fraction: float = 0.0  # % of activations that are exactly 0
    nan_fraction: float = 0.0   # % of activations that are NaN
    inf_fraction: float = 0.0   # % of activations that are Inf

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerDriftStats:
    """Drift statistics for a single layer vs reference."""

    name: str

    # Per-submodule drift
    # e.g., {'q_proj': 0.023, 'k_proj': 0.018, ...}
    weight_l2: Dict[str, float] = field(default_factory=dict)

    # Cosine similarity (1.0 = identical, 0.0 = orthogonal)
    # e.g., {'q_proj': 0.9991, 'k_proj': 0.9993, ...}
    weight_cosine: Dict[str, float] = field(default_factory=dict)

    # Aggregated metrics for this layer
    total_l2: float = 0.0
    avg_cosine: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalStats:
    """Aggregated statistics across all layers."""

    # Weight norm stats
    avg_weight_norm: float = 0.0
    max_weight_norm: float = 0.0
    min_weight_norm: float = 0.0
    total_params: int = 0

    # Drift stats (if computed)
    avg_drift_l2: Optional[float] = None
    max_drift_l2: Optional[float] = None
    min_drift_l2: Optional[float] = None
    avg_cosine_similarity: Optional[float] = None

    # Most/least changed layers
    most_changed_layer: Optional[str] = None
    least_changed_layer: Optional[str] = None
    most_changed_drift: Optional[float] = None
    least_changed_drift: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeInfo:
    """Information about the probe dataset used for activation analysis."""

    dataset_id: str
    num_sequences: int
    total_tokens: int
    source_file: str
    avg_sequence_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerStatsResult:
    """
    Complete result from a layer_stats analysis job.

    This is the main output schema that gets written to JSON.
    """

    # Schema version for future compatibility
    version: int = 1

    # Identifiers
    campaign_id: str = ""
    hero_id: str = ""
    checkpoint_path: str = ""
    checkpoint_step: int = 0
    model_ref: str = ""

    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Weight statistics (always computed)
    weight_stats: Dict[str, Any] = field(default_factory=dict)
    global_weight_stats: Optional[Dict[str, Any]] = None

    # Activation statistics (optional, requires probes)
    activation_stats: Dict[str, Any] = field(default_factory=dict)
    probe_info: Optional[Dict[str, Any]] = None

    # Drift statistics (optional, requires reference checkpoint)
    drift_stats: Dict[str, Any] = field(default_factory=dict)
    reference_checkpoint_path: Optional[str] = None
    reference_checkpoint_step: Optional[int] = None
    global_drift_stats: Optional[Dict[str, Any]] = None

    # Execution metadata
    compute_duration_sec: float = 0.0
    device: str = ""
    dtype: str = "bfloat16"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerStatsResult":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> "LayerStatsResult":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def summary(self) -> Dict[str, Any]:
        """Get a compact summary for listings."""
        return {
            "checkpoint_step": self.checkpoint_step,
            "created_at": self.created_at,
            "has_drift": bool(self.drift_stats),
            "has_activations": bool(self.activation_stats),
            "num_layers": len(self.weight_stats),
            "most_changed_layer": (
                self.global_drift_stats.get("most_changed_layer")
                if self.global_drift_stats else None
            ),
            "avg_weight_norm": (
                self.global_weight_stats.get("avg_weight_norm")
                if self.global_weight_stats else None
            ),
            "compute_duration_sec": self.compute_duration_sec,
        }
