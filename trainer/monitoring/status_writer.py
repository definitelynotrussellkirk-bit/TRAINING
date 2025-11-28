#!/usr/bin/env python3
"""
Training Status Tracker

Writes JSON status files that the UI can read to show:
- Current training step
- Latest inference results (prompt, golden answer, model answer)
- Training metrics
- Crash information

This creates a communication bridge between the training process and the UI.
"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime


class TrainingHistoryLogger:
    """
    Append-only training history logger for time-series metrics.

    Writes training metrics to daily JSONL files for historical analysis:
    - Loss, validation loss, val/train gap over time
    - Throughput and GPU utilization
    - Alerts and accuracy

    Usage:
        logger = TrainingHistoryLogger(logs_dir)
        logger.append(step=1000, loss=0.45, val_loss=0.52, ...)
    """

    def __init__(self, logs_dir: Path, history_interval: int = 50):
        """
        Initialize training history logger.

        Args:
            logs_dir: Directory for log files (e.g., base_dir/logs/training)
            history_interval: Append to history every N steps
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.history_interval = history_interval
        self.last_logged_step = -1

    def _get_history_file(self) -> Path:
        """Get path to today's history file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.logs_dir / f"training_history_{date_str}.jsonl"

    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        if step <= self.last_logged_step:
            return False
        if step % self.history_interval == 0:
            return True
        return False

    def append(
        self,
        step: int,
        epoch: float,
        loss: float,
        val_loss: Optional[float] = None,
        val_train_gap: Optional[float] = None,
        accuracy_percent: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        tokens_per_sec_avg: Optional[float] = None,
        gpu_vram_gb: Optional[float] = None,
        gpu_util_percent: Optional[float] = None,
        loss_trend: Optional[str] = None,
        streaming_ce: Optional[float] = None,
        active_alerts: Optional[List[Dict]] = None,
        current_file: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> None:
        """
        Append a training metrics record to the history file.

        Args:
            step: Current training step
            epoch: Current epoch (can be fractional)
            loss: Training loss
            val_loss: Validation loss (if available)
            val_train_gap: val_loss - loss
            accuracy_percent: Current accuracy
            tokens_per_sec: Current throughput
            tokens_per_sec_avg: Average throughput
            gpu_vram_gb: GPU VRAM used
            gpu_util_percent: GPU utilization
            loss_trend: "improving", "stable", "degrading"
            streaming_ce: EMA-smoothed cross-entropy
            active_alerts: List of active alerts
            current_file: Current training file
            job_id: Job ID for linking to job history
        """
        record = {
            "ts": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "streaming_ce": streaming_ce,
            "loss_trend": loss_trend,
            "val_loss": val_loss,
            "val_train_gap": val_train_gap,
            "accuracy_percent": accuracy_percent,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_sec_avg": tokens_per_sec_avg,
            "gpu_vram_gb": gpu_vram_gb,
            "gpu_util_percent": gpu_util_percent,
            "active_alerts": [a.get("type") for a in (active_alerts or [])] if active_alerts else [],
            "current_file": current_file,
            "job_id": job_id
        }

        # Remove None values to keep file compact
        record = {k: v for k, v in record.items() if v is not None}

        try:
            history_file = self._get_history_file()
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            self.last_logged_step = step
        except Exception:
            pass  # Don't break training for logging errors

    def get_recent(self, limit: int = 1000) -> List[Dict]:
        """
        Get recent history records from today's file.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of history records, most recent first
        """
        history_file = self._get_history_file()
        if not history_file.exists():
            return []

        records = []
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            return []

        # Return most recent first
        return records[-limit:][::-1]

THINKING_EMOJI = "🤔"
THINKING_PREFIX = THINKING_EMOJI * 4 + "\n"
THINKING_INSTRUCTION = f"For this task, think with {THINKING_EMOJI} /four/ times."


@dataclass
class TrainingStatus:
    """Current state of training."""
    status: str  # "idle", "training", "crashed", "completed"
    current_step: int
    total_steps: int
    epoch: int
    loss: float
    learning_rate: float
    timestamp: str

    # Display label for UI
    model_name: Optional[str] = None

    # Model configuration
    max_output_tokens: int = 2048  # Max tokens model can generate
    context_window: int = 2048     # Training context window size

    # File/batch progress (NEW: for better progress tracking)
    batch_step: Optional[int] = None         # Step within current file (0-118)
    batch_total_steps: Optional[int] = None  # Total steps for current file (118)
    batch_number: Optional[int] = None       # Current batch number (e.g., 6)
    batch_queue_size: Optional[int] = None   # Total batches in queue (e.g., 22)
    current_file: Optional[str] = None       # Filename being trained on

    # Latest inference results
    current_system_prompt: Optional[str] = None  # System prompt for current example
    current_prompt: Optional[str] = None
    golden_answer: Optional[str] = None
    model_answer: Optional[str] = None
    answer_matches: Optional[bool] = None

    # Recent examples (last 5)
    recent_examples: Optional[List[Dict]] = None

    # Running accuracy
    total_evals: int = 0
    total_correct: int = 0
    accuracy_percent: float = 0.0

    # Metrics
    train_loss_history: List[float] = None
    accuracy_history: List[float] = None

    # NEW: Fixed evaluation metrics
    fixed_eval_em: Optional[float] = None          # Exact Match on fixed set
    fixed_eval_ce: Optional[float] = None          # Cross-Entropy on fixed set
    fixed_eval_ece: Optional[float] = None         # Expected Calibration Error
    fixed_eval_trend: Optional[str] = None         # "improving", "stable", "degrading"

    # NEW: Extended accuracy tracking
    accuracy_last_20: Optional[float] = None       # Accuracy over last 20 evals
    accuracy_last_50: Optional[float] = None       # Accuracy over last 50 evals
    accuracy_trend: Optional[str] = None           # "improving", "stable", "degrading"

    # NEW: Pattern analysis
    pattern_heatmap: Optional[Dict] = None         # Pattern×length heatmap data
    pattern_loss_trend: Optional[Dict] = None      # Avg/recent loss per pattern
    pattern_layer_correlation: Optional[Dict] = None  # Layer influence counts per pattern
    length_bin_staleness: Optional[Dict] = None    # Seconds since each length bucket observed

    # NEW: Layer activity tracking
    layer_activity_summary: Optional[Dict] = None
    layer_stability_summary: Optional[Dict] = None  # Derived stability ranking

    # NEW: LoRA layer monitoring
    lora_stats: Optional[Dict] = None              # Per-layer gradient norms and updates
    lora_summary: Optional[Dict] = None            # High-level LoRA activity summary

    # NEW: Streaming metrics
    streaming_ce: Optional[float] = None           # EMA-smoothed cross-entropy
    loss_variance: Optional[float] = None          # Loss noise level
    token_entropy: Optional[float] = None          # Per-token prediction uncertainty
    loss_trend: Optional[str] = None               # "improving", "stable", "degrading"

    # NEW: Smart alerts
    active_alerts: Optional[List[Dict]] = None     # List of active training alerts
    alert_summary: Optional[Dict] = None           # Counts by severity

    # NEW: Throughput tracking
    tokens_per_sec: Optional[float] = None         # Current throughput
    tokens_per_sec_avg: Optional[float] = None     # Average throughput
    tokens_per_sec_baseline: Optional[float] = None  # Baseline throughput after warmup
    throughput_trend: Optional[str] = None         # "improving", "stable", "degrading"
    throughput_vram_samples: Optional[List[Dict]] = None  # Recent throughput+VRAM samples
    queue_velocity: Optional[Dict] = None          # Estimated samples/sec/hour

    # NEW: Validation loss (PHASE 4 - 2025-11-16)
    validation_loss: Optional[float] = None        # Loss on fixed validation set
    val_train_gap: Optional[float] = None          # validation_loss - training_loss

    # NEW: Think tag tracking (PHASE 4 - 2025-11-16)
    think_tag_count: int = 0                       # Count of outputs with <think> tags
    think_tag_percent: float = 0.0                 # Percentage of outputs with <think>
    logit_penalty_stats: Optional[List[Dict]] = None  # Penalty hit counters
    penalty_heatmap: Optional[Dict] = None            # Aggregated penalty distribution

    # NEW: Output length tracking
    max_golden_output_length: Optional[int] = None  # Max golden output length seen (tokens)
    max_model_output_length: Optional[int] = None   # Max model output length seen (tokens)
    current_golden_output_length: Optional[int] = None  # Current golden output length
    current_model_output_length: Optional[int] = None   # Current model output length

    # Error info (if crashed)
    error_message: Optional[str] = None
    error_type: Optional[str] = None


    def __post_init__(self):
        if self.train_loss_history is None:
            self.train_loss_history = []
        if self.accuracy_history is None:
            self.accuracy_history = []


class TrainingStatusWriter:
    """Writes training status to disk for UI consumption."""

    def __init__(
        self,
        status_file: Path,
        max_output_tokens: int = 2048,
        context_window: int = 2048,
        model_name: Optional[str] = None,
        history_interval: int = 50
    ):
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_inference_data = {}  # Cache last inference data
        self.recent_examples = []  # Last 5 examples
        self.total_evals = 0
        self.total_correct = 0
        self.think_tag_count = 0  # NEW: Track <think> tags
        self.max_output_tokens = max_output_tokens
        self.context_window = context_window
        self.model_name = model_name
        # NEW: Track max output lengths seen during training
        self.max_golden_output_length = 0
        self.max_model_output_length = 0
        # Pattern heatmap cache
        self.pattern_heatmap = None
        self.penalty_heatmap = None
        self.pattern_layer_correlation = {}
        self.length_bin_last_seen = {}
        self.pattern_loss_history = {}
        self.pattern_loss_summary = {}
        self.queue_velocity_snapshot = None
        self.vram_samples = []
        self.penalty_stats_snapshot = None
        self.layer_stability_summary = None
        # Append-only log for inference previews
        self.inference_log_dir = self.status_file.parent / "logs"
        self.inference_log_dir.mkdir(parents=True, exist_ok=True)
        self.pattern_layer_log = self.inference_log_dir / "pattern_layer_history.jsonl"

        # Training history logger - time-series metrics for charting
        self.history_logger = TrainingHistoryLogger(
            logs_dir=self.status_file.parent.parent / "logs" / "training",
            history_interval=history_interval
        )
        self.current_job_id: Optional[str] = None  # Set by daemon for job linking

        # Loss history for tracking trends
        self.loss_history: List[float] = []
        self.loss_history_max = 200
    def _ensure_thinking_instruction(self, content: Optional[str]) -> Optional[str]:
        """Append the four-emoji instruction if it's missing."""
        if content is None:
            return None
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            return content
        # Check for ANY "think with" instruction (any emoji variant), not just our specific one
        if "think with" in content.lower():
            return content
        return content.rstrip() + "\n\n" + THINKING_INSTRUCTION

    def _ensure_thinking_prefix(self, content: Optional[str]) -> Optional[str]:
        """Ensure assistant outputs start with thinking emoji prefix."""
        if content is None:
            return None
        if not isinstance(content, str):
            content = str(content)
        stripped = content.lstrip()
        # Check if starts with ANY thinking emoji (not just our specific one)
        THINKING_EMOJIS = ["🤔", "💭", "🧠", "💡", "🎯", "🔍", "🤨", "🧐", "⚡", "✨"]
        if any(stripped.startswith(e) for e in THINKING_EMOJIS):
            return stripped
        # If no thinking emoji found, add our default prefix
        return THINKING_PREFIX + stripped

    def write(self, status: TrainingStatus):
        """Write status to file (atomic write)."""
        temp_file = self.status_file.with_suffix('.tmp')

        with open(temp_file, 'w') as f:
            json.dump(asdict(status), f, indent=2)

        # Atomic rename
        temp_file.rename(self.status_file)

    def _dedup_recents(self, recent_list, max_items: int = 5):
        """Deduplicate recent examples by (step, current_file, model_output) preserving order."""
        deduped = []
        seen = set()
        for ex in recent_list:
            key = (
                ex.get('step'),
                ex.get('current_file') or "unknown",
                ex.get('model_output') or ""
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ex)
        return deduped[-max_items:]

    def _compute_length_staleness(self) -> Optional[Dict[str, float]]:
        if not self.length_bin_last_seen:
            return None
        now = time.time()
        return {
            bin_name: max(0.0, now - timestamp)
            for bin_name, timestamp in self.length_bin_last_seen.items()
        }

    def _update_length_bin(self, metadata: Dict[str, Any]):
        bin_name = metadata.get("length_bin")
        if not bin_name:
            return
        self.length_bin_last_seen[bin_name] = time.time()

    def _update_pattern_loss(self, metadata: Dict[str, Any]):
        pattern_id = metadata.get("pattern_id")
        loss = metadata.get("loss")
        if pattern_id is None or loss is None:
            return
        history = self.pattern_loss_history.setdefault(pattern_id, [])
        history.append(float(loss))
        if len(history) > 50:
            history.pop(0)
        avg_loss = sum(history) / len(history)
        recent_window = history[-5:] if len(history) >= 5 else history
        recent_avg = sum(recent_window) / len(recent_window)
        self.pattern_loss_summary[pattern_id] = {
            "avg_loss": avg_loss,
            "recent_loss": recent_avg,
            "samples": len(history),
        }

    def _update_pattern_layer_correlation(self, metadata: Dict[str, Any], layer_summary: Optional[Dict]):
        if not layer_summary or not layer_summary.get("top_changes"):
            return
        pattern_id = metadata.get("pattern_id") or "unknown"
        bucket = self.pattern_layer_correlation.setdefault(pattern_id, {})
        for entry in layer_summary["top_changes"][:3]:
            name = entry.get("name")
            if not name:
                continue
            stats = bucket.setdefault(name, {"count": 0, "cumulative_delta": 0.0})
            stats["count"] += 1
            delta = entry.get("delta")
            if isinstance(delta, (int, float)):
                stats["cumulative_delta"] += abs(delta)
        self._append_pattern_layer_history(pattern_id, layer_summary)

    def _append_pattern_layer_history(self, pattern_id: str, layer_summary: Dict):
        if not layer_summary:
            return
        entry = {
            "timestamp": datetime.now().isoformat(),
            "pattern": pattern_id,
            "file": self.last_inference_data.get('current_file'),
            "top_layers": layer_summary.get("top_changes", [])[:5],
        }
        try:
            with open(self.pattern_layer_log, 'a') as f:
                json.dump(entry, f)
                f.write("\n")
        except Exception:
            pass

    def update_progress(
        self,
        step: int,
        total_steps: int,
        epoch: int,
        loss: float,
        lr: float,
        batch_step: Optional[int] = None,
        batch_total_steps: Optional[int] = None,
        batch_number: Optional[int] = None,
        batch_queue_size: Optional[int] = None,
        current_file: Optional[str] = None,
        val_loss: Optional[float] = None,  # NEW: Validation loss
        tokens_per_sec: Optional[float] = None,
        tokens_per_sec_avg: Optional[float] = None,
        tokens_per_sec_baseline: Optional[float] = None,
        loss_variance: Optional[float] = None,
        loss_trend: Optional[str] = None,
        active_alerts: Optional[List[Dict]] = None,
        alert_summary: Optional[Dict] = None,
        throughput_vram_samples: Optional[List[Dict]] = None,
        queue_velocity: Optional[Dict] = None,
        logit_penalty_stats: Optional[List[Dict]] = None,
        penalty_heatmap: Optional[Dict] = None,
    ):
        """Update basic training progress (preserves last inference data)."""
        accuracy_pct = (self.total_correct / self.total_evals * 100) if self.total_evals > 0 else 0.0
        think_pct = (self.think_tag_count / self.total_evals * 100) if self.total_evals > 0 else 0.0

        # Calculate validation gap
        val_train_gap = None
        if val_loss is not None:
            val_train_gap = val_loss - loss

        if throughput_vram_samples is not None:
            self.vram_samples = list(throughput_vram_samples)[-60:]
        if queue_velocity is not None:
            self.queue_velocity_snapshot = queue_velocity
        if logit_penalty_stats is not None:
            self.penalty_stats_snapshot = logit_penalty_stats
        if penalty_heatmap is not None:
            self.penalty_heatmap = penalty_heatmap

        # Append to training history (time-series for charting)
        if self.history_logger.should_log(step):
            # Get GPU stats from vram_samples if available
            gpu_vram_gb = None
            gpu_util_percent = None
            if self.vram_samples:
                latest = self.vram_samples[-1]
                gpu_vram_gb = latest.get("vram_used_gb")
                gpu_util_percent = latest.get("gpu_util")

            self.history_logger.append(
                step=step,
                epoch=epoch,
                loss=loss,
                val_loss=val_loss,
                val_train_gap=val_train_gap,
                accuracy_percent=accuracy_pct,
                tokens_per_sec=tokens_per_sec,
                tokens_per_sec_avg=tokens_per_sec_avg,
                gpu_vram_gb=gpu_vram_gb,
                gpu_util_percent=gpu_util_percent,
                loss_trend=loss_trend,
                streaming_ce=None,  # Streaming CE added later if available
                active_alerts=active_alerts,
                current_file=current_file,
                job_id=self.current_job_id
            )

        status = TrainingStatus(
            status="training",
            current_step=step,
            total_steps=total_steps,
            epoch=epoch,
            loss=loss,
            learning_rate=lr,
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            max_output_tokens=self.max_output_tokens,
            context_window=self.context_window,
            # Batch/file progress (NEW)
            batch_step=batch_step,
            batch_total_steps=batch_total_steps,
            batch_number=batch_number,
            batch_queue_size=batch_queue_size,
            current_file=current_file,
            # Preserve last inference data
            current_system_prompt=self.last_inference_data.get('system_prompt'),
            current_prompt=self.last_inference_data.get('prompt'),
            golden_answer=self.last_inference_data.get('golden'),
            model_answer=self.last_inference_data.get('model_output'),
            answer_matches=self.last_inference_data.get('matches'),
            recent_examples=self.recent_examples[-5:],  # Last 5
            total_evals=self.total_evals,
            total_correct=self.total_correct,
            accuracy_percent=accuracy_pct,
            # NEW: Validation loss (PHASE 4)
            validation_loss=val_loss,
            val_train_gap=val_train_gap,
            # NEW: Think tag tracking
            think_tag_count=self.think_tag_count,
            think_tag_percent=think_pct,
            # NEW: Throughput + stability
            tokens_per_sec=tokens_per_sec,
            tokens_per_sec_avg=tokens_per_sec_avg,
            tokens_per_sec_baseline=tokens_per_sec_baseline,
            loss_variance=loss_variance,
            loss_trend=loss_trend,
            active_alerts=active_alerts,
            alert_summary=alert_summary,
            pattern_heatmap=self.pattern_heatmap,
            penalty_heatmap=self.penalty_heatmap,
            pattern_loss_trend=self.pattern_loss_summary or None,
            pattern_layer_correlation=self.pattern_layer_correlation or None,
            length_bin_staleness=self._compute_length_staleness(),
            layer_activity_summary=self.last_inference_data.get('layer_activity_summary'),
            layer_stability_summary=self.layer_stability_summary,
            throughput_vram_samples=self.vram_samples or None,
            queue_velocity=self.queue_velocity_snapshot,
            logit_penalty_stats=self.penalty_stats_snapshot,
        )
        self.write(status)

    def update_prompt_snapshot(
        self,
        prompt: str,
        golden: str,
        system_prompt: Optional[str] = None
    ):
        """Snapshot prompt/golden without inference; does not change model output."""
        prompt = self._ensure_thinking_instruction(prompt)
        golden = self._ensure_thinking_prefix(golden)
        self.last_inference_data.update({
            'system_prompt': system_prompt,
            'prompt': prompt,
            'golden': golden,
            'model_output': None,
            'matches': None
        })

    def update_inference(
        self,
        step: int,
        total_steps: int,
        epoch: int,
        loss: float,
        lr: float,
        prompt: str,
        golden: str,
        model_output: str,
        matches: bool,
        system_prompt: Optional[str] = None,
        batch_step: Optional[int] = None,
        batch_total_steps: Optional[int] = None,
        batch_number: Optional[int] = None,
        batch_queue_size: Optional[int] = None,
        current_file: Optional[str] = None,
        golden_output_length: Optional[int] = None,
        model_output_length: Optional[int] = None,
        pattern_heatmap: Optional[Dict] = None,
        layer_activity_summary: Optional[Dict] = None,
        pattern_metadata: Optional[Dict] = None
    ):
        """Update with latest inference results."""
        prompt = self._ensure_thinking_instruction(prompt) if prompt else prompt
        golden = self._ensure_thinking_prefix(golden) if golden else golden
        model_output = self._ensure_thinking_prefix(model_output) if model_output else model_output

        # Update running accuracy
        self.total_evals += 1
        if matches:
            self.total_correct += 1

        # NEW: Track <think> tags
        if model_output is not None and '<think>' in model_output:
            self.think_tag_count += 1

        # NEW: Track max output lengths
        if golden_output_length is not None:
            self.max_golden_output_length = max(self.max_golden_output_length, golden_output_length)
        if model_output_length is not None:
            self.max_model_output_length = max(self.max_model_output_length, model_output_length)

        accuracy_pct = (self.total_correct / self.total_evals * 100) if self.total_evals > 0 else 0.0
        think_pct = (self.think_tag_count / self.total_evals * 100) if self.total_evals > 0 else 0.0

        # Cache inference data so update_progress can preserve it
        cf = current_file or "unknown"
        self.last_inference_data = {
            'system_prompt': system_prompt,
            'prompt': prompt,
            'golden': golden,
            'model_output': model_output,
            'matches': matches,
            'current_file': cf,
            'layer_activity_summary': layer_activity_summary
        }

        if pattern_heatmap is not None:
            self.pattern_heatmap = pattern_heatmap
        if pattern_metadata:
            self._update_length_bin(pattern_metadata)
            self._update_pattern_loss(pattern_metadata)
            self._update_pattern_layer_correlation(pattern_metadata, layer_activity_summary)

        snapshot_layer_summary = layer_activity_summary or self.last_inference_data.get('layer_activity_summary')
        if layer_activity_summary and layer_activity_summary.get("stability"):
            self.layer_stability_summary = layer_activity_summary.get("stability")

        # Keep last 5 recent examples, dedup by step+file, keep latest
        entry = {
            'step': step,
            'current_file': cf,
            'system_prompt': system_prompt,
            'prompt': prompt,
            'golden': golden,  # full
            'model_output': model_output,  # full
            'matches': matches,
            'loss': loss
        }
        try:
            self.recent_examples.append(entry)
            self.recent_examples = self._dedup_recents(self.recent_examples, max_items=5)
        except Exception:
            pass

        status = TrainingStatus(
            status="training",
            current_step=step,
            total_steps=total_steps,
            epoch=epoch,
            loss=loss,
            learning_rate=lr,
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            max_output_tokens=self.max_output_tokens,
            context_window=self.context_window,
            # Batch/file progress (NEW)
            batch_step=batch_step,
            batch_total_steps=batch_total_steps,
            batch_number=batch_number,
            batch_queue_size=batch_queue_size,
            current_file=current_file,
            current_system_prompt=system_prompt or self.last_inference_data.get('system_prompt'),
            current_prompt=prompt,
            golden_answer=golden,
            model_answer=model_output,
            answer_matches=matches,
            recent_examples=self._dedup_recents(self.recent_examples, max_items=5),
            total_evals=self.total_evals,
            total_correct=self.total_correct,
            accuracy_percent=accuracy_pct,
            # NEW: Think tag tracking
            think_tag_count=self.think_tag_count,
            think_tag_percent=think_pct,
            # NEW: Output length tracking
            max_golden_output_length=self.max_golden_output_length,
            max_model_output_length=self.max_model_output_length,
            current_golden_output_length=golden_output_length,
            current_model_output_length=model_output_length,
            pattern_heatmap=self.pattern_heatmap,
            pattern_loss_trend=self.pattern_loss_summary or None,
            pattern_layer_correlation=self.pattern_layer_correlation or None,
            length_bin_staleness=self._compute_length_staleness(),
            layer_activity_summary=snapshot_layer_summary,
            layer_stability_summary=self.layer_stability_summary,
            throughput_vram_samples=self.vram_samples or None,
            queue_velocity=self.queue_velocity_snapshot,
            logit_penalty_stats=self.penalty_stats_snapshot,
        )
        self.write(status)

        # Append to daily inference log
        try:
            day_log = self.inference_log_dir / f"inference_{datetime.now().strftime('%Y%m%d')}.log"
            with open(day_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.now().isoformat(),
                    "step": step,
                    "total_steps": total_steps,
                    "epoch": epoch,
                    "loss": loss,
                    "lr": lr,
                    "batch_step": batch_step,
                    "batch_total_steps": batch_total_steps,
                    "batch_number": batch_number,
                    "batch_queue_size": batch_queue_size,
                    "current_file": current_file,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "golden": golden,
                    "model_output": model_output,
                    "matches": matches,
                    "golden_output_length": golden_output_length,
                    "model_output_length": model_output_length,
                    "layer_activity_summary": snapshot_layer_summary
                }) + "\n")
        except Exception:
            pass  # Logging errors should not break training

    def mark_crashed(self, error: str, error_type: str = "Unknown"):
        """Mark training as crashed."""
        status = TrainingStatus(
            status="crashed",
            current_step=0,
            total_steps=0,
            epoch=0,
            loss=0.0,
            learning_rate=0.0,
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            error_message=error,
            error_type=error_type
        )
        self.write(status)

    def update_advanced_metrics(
        self,
        step: int,
        total_steps: int,
        epoch: int,
        loss: float,
        lr: float,
        # Fixed eval metrics
        fixed_eval_em: Optional[float] = None,
        fixed_eval_ce: Optional[float] = None,
        fixed_eval_ece: Optional[float] = None,
        fixed_eval_trend: Optional[str] = None,
        # Extended accuracy
        accuracy_last_20: Optional[float] = None,
        accuracy_last_50: Optional[float] = None,
        accuracy_trend: Optional[str] = None,
        # Pattern analysis
        pattern_heatmap: Optional[Dict] = None,
        # LoRA monitoring
        lora_stats: Optional[Dict] = None,
        lora_summary: Optional[Dict] = None,
        # Streaming metrics
        streaming_ce: Optional[float] = None,
        loss_variance: Optional[float] = None,
        token_entropy: Optional[float] = None,
        loss_trend: Optional[str] = None,
        # Smart alerts
        active_alerts: Optional[List[Dict]] = None,
        alert_summary: Optional[Dict] = None,
        # Throughput
        tokens_per_sec: Optional[float] = None,
        tokens_per_sec_avg: Optional[float] = None,
        throughput_trend: Optional[str] = None,
        # Batch progress
        batch_step: Optional[int] = None,
        batch_total_steps: Optional[int] = None,
        batch_number: Optional[int] = None,
        batch_queue_size: Optional[int] = None,
        current_file: Optional[str] = None
    ):
        """Update status with advanced metrics from all collectors."""
        accuracy_pct = (self.total_correct / self.total_evals * 100) if self.total_evals > 0 else 0.0
        if pattern_heatmap is None:
            pattern_heatmap = self.pattern_heatmap

        status = TrainingStatus(
            status="training",
            current_step=step,
            total_steps=total_steps,
            epoch=epoch,
            loss=loss,
            learning_rate=lr,
            timestamp=datetime.now().isoformat(),
            max_output_tokens=self.max_output_tokens,
            context_window=self.context_window,
            # Batch progress
            batch_step=batch_step,
            batch_total_steps=batch_total_steps,
            batch_number=batch_number,
            batch_queue_size=batch_queue_size,
            current_file=current_file,
            # Preserve inference data
            current_system_prompt=self.last_inference_data.get('system_prompt'),
            current_prompt=self.last_inference_data.get('prompt'),
            golden_answer=self.last_inference_data.get('golden'),
            model_answer=self.last_inference_data.get('model_output'),
            answer_matches=self.last_inference_data.get('matches'),
            recent_examples=self.recent_examples[-5:],
            total_evals=self.total_evals,
            total_correct=self.total_correct,
            accuracy_percent=accuracy_pct,
            # NEW: Fixed eval metrics
            fixed_eval_em=fixed_eval_em,
            fixed_eval_ce=fixed_eval_ce,
            fixed_eval_ece=fixed_eval_ece,
            fixed_eval_trend=fixed_eval_trend,
            # NEW: Extended accuracy
            accuracy_last_20=accuracy_last_20,
            accuracy_last_50=accuracy_last_50,
            accuracy_trend=accuracy_trend,
            # NEW: Pattern analysis
            pattern_heatmap=pattern_heatmap,
            # NEW: LoRA monitoring
            lora_stats=lora_stats,
            lora_summary=lora_summary,
            # NEW: Streaming metrics
            streaming_ce=streaming_ce,
            loss_variance=loss_variance,
            token_entropy=token_entropy,
            loss_trend=loss_trend,
            # NEW: Smart alerts
            active_alerts=active_alerts,
            alert_summary=alert_summary,
            # NEW: Throughput
            tokens_per_sec=tokens_per_sec,
            tokens_per_sec_avg=tokens_per_sec_avg,
            throughput_trend=throughput_trend
        )
        self.write(status)

    def mark_completed(self, final_step: int, total_steps: int):
        """Mark training as successfully completed."""
        status = TrainingStatus(
            status="completed",
            current_step=final_step,
            total_steps=total_steps,
            epoch=0,
            loss=0.0,
            learning_rate=0.0,
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name
        )
        self.write(status)


class TrainingStatusReader:
    """Reads training status from disk."""

    def __init__(self, status_file: Path):
        self.status_file = Path(status_file)

    def read(self) -> Optional[TrainingStatus]:
        """Read current status from file."""
        if not self.status_file.exists():
            return None

        try:
            with open(self.status_file) as f:
                data = json.load(f)
            return TrainingStatus(**data)
        except Exception as e:
            print(f"Failed to read status: {e}")
            return None

    def is_training(self) -> bool:
        """Check if training is currently active."""
        status = self.read()
        return status is not None and status.status == "training"

    def has_crashed(self) -> bool:
        """Check if last training crashed."""
        status = self.read()
        return status is not None and status.status == "crashed"


# Default status file location
def get_default_status_file() -> Path:
    """Get default status file path using core.paths."""
    try:
        from core.paths import get_status_dir
        return get_status_dir() / "training_status.json"
    except ImportError:
        # Fallback to auto-detection
        base = Path(__file__).parent.parent.parent
        return base / "status" / "training_status.json"

DEFAULT_STATUS_FILE = get_default_status_file()


def example_usage():
    """Example of how to use the status tracker."""
    writer = TrainingStatusWriter(DEFAULT_STATUS_FILE, model_name="demo-model")

    # During training loop
    for step in range(100):
        writer.update_progress(
            step=step,
            total_steps=100,
            epoch=1,
            loss=0.5 - (step * 0.001),
            lr=2e-4
        )
        time.sleep(0.1)

        # Every N steps, update with inference
        if step % 10 == 0:
            writer.update_inference(
                step=step,
                total_steps=100,
                epoch=1,
                loss=0.5 - (step * 0.001),
                lr=2e-4,
                prompt="What is 2+2?",
                golden="4",
                model_output="4",
                matches=True
            )

    # On completion
    writer.mark_completed(final_step=100, total_steps=100)

    # Or on crash
    # writer.mark_crashed(error="CUDA out of memory", error_type="OOM")


if __name__ == "__main__":
    example_usage()
