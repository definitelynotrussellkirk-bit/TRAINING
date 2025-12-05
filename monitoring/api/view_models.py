#!/usr/bin/env python3
"""
View Models for Monitoring API

Clean, focused data models for the UI to consume.
Each view model corresponds to an API endpoint.

Design principles:
- Cheap to compute (< 10ms)
- Small payload (< 100KB)
- Single responsibility
- No heavy aggregations (precompute those)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal
from datetime import datetime


@dataclass
class LiveStatusView:
    """
    GET /api/status/live

    Polled every 1-2 seconds. Core training state + hardware.
    Target: < 50KB, < 5ms compute time
    """
    # Training state
    status: Literal["idle", "training", "paused", "completed", "crashed"]
    current_step: int
    total_steps: int
    epoch: float

    # Loss metrics
    loss: float
    streaming_ce: Optional[float]
    loss_trend: Literal["improving", "stable", "rising", "unknown"]
    loss_variance: float
    learning_rate: float

    # Progress
    batch_step: int
    batch_total_steps: int
    current_file: Optional[str]
    batch_number: int
    batch_queue_size: int
    eta_current_file: Optional[str]  # "2h 15m"
    eta_overall: Optional[str]  # "12h 30m"

    # Throughput
    tokens_per_sec: float
    tokens_per_sec_avg: float
    tokens_per_sec_baseline: float
    throughput_trend: Literal["normal", "degraded", "excellent"]

    # Hardware - 4090 (training)
    gpu_4090: Dict = field(default_factory=lambda: {
        "temp_c": 0,
        "util_pct": 0,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "vram_pct": 0,
        "power_w": 0,
        "power_limit_w": 0,
        "fan_pct": 0,
    })

    # Hardware - 3090 (inference)
    gpu_3090: Dict = field(default_factory=lambda: {
        "online": False,
        "temp_c": 0,
        "util_pct": 0,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "power_w": 0,
        "power_profile": "unknown",  # quiet/normal/max
    })

    # RAM
    ram: Dict = field(default_factory=lambda: {
        "used_gb": 0.0,
        "total_gb": 0.0,
        "pct": 0,
    })

    # Snapshot refs
    current_checkpoint_id: Optional[str] = None
    current_model_name: Optional[str] = None
    latest_snapshot_date: Optional[str] = None  # YYYY-MM-DD

    # Health
    health: Literal["healthy", "warning", "critical"] = "healthy"
    health_issues: List[str] = field(default_factory=list)

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PreviewResultView:
    """
    Single preview result from inference on 3090
    """
    step: int
    checkpoint_id: str
    timestamp: str

    # Example metadata
    example_id: str
    dataset_id: str
    regime: str  # emoji_think, regime3, plain_sft
    source_pool: Literal["fixed_eval", "train", "failures"]

    # Content (truncated for UI)
    prompt: str  # Max 500 chars for UI
    golden: str
    model_answer: str

    # Outcome
    exact_match: bool
    normalized_match: bool
    failure_mode: Optional[str]  # wrong_content, format_error, truncated, etc.

    # Metrics
    ce: Optional[float]  # Cross-entropy loss on this example
    prompt_tokens: int
    golden_tokens: int
    model_tokens: int
    generation_time_ms: int

    # Regime-specific contract checks
    contract_status: Dict = field(default_factory=dict)  # e.g., {"emoji_prefix": true, "stop_emoji": true}

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PreviewStatusView:
    """
    GET /api/status/preview

    Polled every 2-5 seconds. Latest preview + aggregated stats.
    Target: < 20KB
    """
    # Latest preview
    latest_preview: Optional[PreviewResultView] = None

    # Aggregate stats (last N previews)
    preview_em_last_20: float = 0.0
    preview_em_last_50: float = 0.0
    preview_em_last_100: float = 0.0

    # Per-domain EM (if tagged)
    domain_stats: Dict[str, Dict] = field(default_factory=dict)
    # e.g., {"syllogism": {"em": 0.85, "count": 45}, "math": {"em": 0.72, "count": 28}}

    # Per-regime stats
    regime_stats: Dict[str, Dict] = field(default_factory=dict)
    # e.g., {"emoji_think": {"em": 0.83, "count": 60}, "regime3": {"em": 0.79, "count": 40}}

    # Preview job state
    pending_jobs: int = 0
    last_job_latency_ms: int = 0

    # Trend (simple sparkline data: last 20 preview EMs)
    em_trend: List[float] = field(default_factory=list)

    # Pattern heatmap (precomputed)
    pattern_heatmap: Optional[Dict] = None  # {rows, cols, data}

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvalStatusView:
    """
    GET /api/status/evals

    Polled every 10-30 seconds. Eval metrics from fixed eval set.
    Target: < 10KB
    """
    # Latest fixed eval run
    fixed_eval_em: Optional[float] = None
    fixed_eval_ce: Optional[float] = None
    fixed_eval_ece: Optional[float] = None  # Expected calibration error
    fixed_eval_trend: Literal["improving", "stable", "declining", "unknown"] = "unknown"
    fixed_eval_step: Optional[int] = None
    fixed_eval_timestamp: Optional[str] = None

    # Per-domain eval (if available)
    domain_evals: Dict[str, Dict] = field(default_factory=dict)
    # e.g., {"syllogism": {"em": 0.88, "ce": 0.35}, "math": {"em": 0.75, "ce": 0.52}}

    # Micro-eval (quick eval during training)
    micro_eval_loss: Optional[float] = None
    micro_eval_step: Optional[int] = None

    # Val/train gap
    val_loss: Optional[float] = None
    train_loss: Optional[float] = None
    val_train_gap: Optional[float] = None
    gap_status: Literal["good", "warning", "overfitting"] = "good"

    # Recent snapshots with metrics
    snapshots: List[Dict] = field(default_factory=list)
    # e.g., [{"date": "2025-11-22", "model_id": "checkpoint-15000",
    #         "fixed_eval_em": 0.83, "tags": ["best_so_far"]}]

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SystemStatusView:
    """
    GET /api/status/system

    Polled every 5-10 seconds. System resources + job queues.
    Target: < 5KB
    """
    # 4090 system
    system_4090: Dict = field(default_factory=lambda: {
        "cpu_pct": 0,
        "ram_gb": 0.0,
        "ram_total_gb": 0.0,
        "disk_used_gb": 0.0,
        "disk_total_gb": 0.0,
        "disk_pct": 0,
        "swap_used_gb": 0.0,
        "swap_total_gb": 0.0,
    })

    # 3090 system
    system_3090: Dict = field(default_factory=lambda: {
        "online": False,
        "cpu_pct": 0,
        "ram_gb": 0.0,
        "ram_total_gb": 0.0,
        "disk_used_gb": 0.0,
        "disk_total_gb": 0.0,
    })

    # Job queues
    queues: Dict = field(default_factory=lambda: {
        "preview_jobs_pending": 0,
        "eval_jobs_pending": 0,
        "data_gen_jobs_pending": 0,
        "training_queue_high": 0,
        "training_queue_normal": 0,
        "training_queue_low": 0,
    })

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PreviewHistoryEntry:
    """
    Single entry in preview history log
    Stored in logs/preview_history.jsonl
    """
    ts: str
    step: int
    checkpoint_id: str
    example_id: str
    dataset_id: str
    regime: str
    source_pool: str

    # Content (full, not truncated)
    prompt: str
    golden: str
    model_answer: str

    # Outcome
    exact_match: bool
    normalized_match: bool
    failure_mode: Optional[str]

    # Metrics
    ce: Optional[float]
    prompt_tokens: int
    golden_tokens: int
    model_tokens: int
    generation_time_ms: int

    # Optional: full contract status
    contract_status: Dict = field(default_factory=dict)

    # Optional: per-token details (for advanced analysis)
    token_log_probs: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ThroughputSample:
    """
    Single sample of throughput + VRAM for correlation analysis
    """
    timestamp: str
    step: int
    tokens_per_sec: float
    vram_used_gb: float
    vram_pct: float
    batch_size: int
    sequence_length: int

    def to_dict(self) -> Dict:
        return asdict(self)


# Utility functions for view model creation

def _get_gpu_3090_stats(status_dict: Dict) -> Dict:
    """Extract GPU 3090 stats from status_dict or return defaults."""
    gpu_3090 = status_dict.get("gpu_3090", {})
    return {
        "online": gpu_3090.get("online", False),
        "temp_c": gpu_3090.get("temp_c", 0),
        "util_pct": gpu_3090.get("util_pct", 0),
        "vram_used_gb": gpu_3090.get("vram_used_gb", 0.0),
        "vram_total_gb": gpu_3090.get("vram_total_gb", 24.0),
        "vram_pct": gpu_3090.get("vram_pct", 0),
        "power_w": gpu_3090.get("power_w", 0),
        "power_profile": gpu_3090.get("power_profile", "unknown"),
    }


def create_live_status_from_training_status(status_dict: Dict) -> LiveStatusView:
    """Convert old training_status.json format to LiveStatusView"""

    # Determine status
    if status_dict.get("status") == "crashed":
        state = "crashed"
    elif status_dict.get("status") == "completed":
        state = "completed"
    elif status_dict.get("is_paused"):
        state = "paused"
    elif status_dict.get("is_training"):
        state = "training"
    else:
        state = "idle"

    # Determine loss trend
    loss_trend = "unknown"
    if "loss_trend" in status_dict:
        trend_val = status_dict["loss_trend"]
        if trend_val < -0.01:
            loss_trend = "improving"
        elif trend_val > 0.01:
            loss_trend = "rising"
        else:
            loss_trend = "stable"

    # Determine throughput trend
    tps = status_dict.get("tokens_per_sec", 0)
    tps_avg = status_dict.get("tokens_per_sec_avg", 0)
    if tps > tps_avg * 1.1:
        throughput_trend = "excellent"
    elif tps < tps_avg * 0.8:
        throughput_trend = "degraded"
    else:
        throughput_trend = "normal"

    # Extract GPU stats
    gpu_stats = status_dict.get("gpu_stats", {})

    # Determine health
    health = "healthy"
    health_issues = []

    if gpu_stats.get("temperature", 0) > 80:
        health = "warning"
        health_issues.append("GPU temperature high")

    if gpu_stats.get("vram_pct", 0) > 95:
        health = "warning"
        health_issues.append("VRAM near capacity")

    if loss_trend == "rising":
        health = "warning"
        health_issues.append("Loss increasing")

    if state == "crashed":
        health = "critical"
        health_issues.append("Training crashed")

    return LiveStatusView(
        status=state,
        current_step=status_dict.get("current_step", 0),
        total_steps=status_dict.get("total_steps", 0),
        epoch=status_dict.get("epoch", 0.0),

        loss=status_dict.get("loss", 0.0),
        streaming_ce=status_dict.get("streaming_ce"),
        loss_trend=loss_trend,
        loss_variance=status_dict.get("loss_variance", 0.0),
        learning_rate=status_dict.get("learning_rate", 0.0),

        batch_step=status_dict.get("batch_step", 0),
        batch_total_steps=status_dict.get("batch_total_steps", 0),
        current_file=status_dict.get("current_file"),
        batch_number=status_dict.get("batch_number", 0),
        batch_queue_size=status_dict.get("batch_queue_size", 0),
        eta_current_file=status_dict.get("eta_current_file"),
        eta_overall=status_dict.get("eta_overall"),

        tokens_per_sec=tps,
        tokens_per_sec_avg=tps_avg,
        tokens_per_sec_baseline=status_dict.get("tokens_per_sec_baseline", 0.0),
        throughput_trend=throughput_trend,

        gpu_4090={
            "temp_c": gpu_stats.get("temperature", 0),
            "util_pct": gpu_stats.get("utilization", 0),
            "vram_used_gb": gpu_stats.get("vram_used", 0) / 1024,
            "vram_total_gb": gpu_stats.get("vram_total", 0) / 1024,
            "vram_pct": gpu_stats.get("vram_pct", 0),
            "power_w": gpu_stats.get("power_draw", 0),
            "power_limit_w": gpu_stats.get("power_limit", 0),
            "fan_pct": gpu_stats.get("fan_speed", 0),
        },

        gpu_3090=_get_gpu_3090_stats(status_dict),

        ram={
            "used_gb": status_dict.get("ram_used_gb", 0.0),
            "total_gb": status_dict.get("ram_total_gb", 0.0),
            "pct": status_dict.get("ram_pct", 0),
        },

        current_checkpoint_id=status_dict.get("current_checkpoint"),
        current_model_name=status_dict.get("model_name"),
        latest_snapshot_date=status_dict.get("latest_snapshot_date"),

        health=health,
        health_issues=health_issues,
    )


if __name__ == '__main__':
    # Test view model creation
    test_status = {
        "status": "training",
        "is_training": True,
        "current_step": 1500,
        "total_steps": 10000,
        "loss": 0.45,
        "tokens_per_sec": 2500,
        "tokens_per_sec_avg": 2400,
        "gpu_stats": {
            "temperature": 75,
            "utilization": 98,
            "vram_used": 20000,
            "vram_total": 24000,
            "vram_pct": 83,
        }
    }

    live = create_live_status_from_training_status(test_status)
    print("LiveStatusView:")
    print(json.dumps(live.to_dict(), indent=2))
