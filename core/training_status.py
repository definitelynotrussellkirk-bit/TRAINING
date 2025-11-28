#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
MODULE: core/training_status.py
═══════════════════════════════════════════════════════════════════════════════

Communication bridge between training process and monitoring UI via JSON status files.

OVERVIEW
--------
Provides structured read/write of training progress to status/training_status.json.
Training daemon writes status updates, web UI/monitoring tools read for real-time display.
All writes are atomic (temp file + rename) to prevent corruption during concurrent reads.

KEY COMPONENTS
--------------
1. TrainingStatus - Dataclass representing complete training state (~50 fields)
2. TrainingStatusWriter - Writes status updates to disk (atomic writes + caching)
3. TrainingStatusReader - Reads status from disk with error handling

ARCHITECTURE
------------

    Training Process          JSON File                Monitoring UI
    ================          =========                =============

    update_progress()    →    training_status.json  →  Read + display
    update_inference()   →    (atomic write)        →  Charts/metrics
    mark_crashed()       →                          →  Alerts

Atomic Writes:
    1. Write to training_status.tmp
    2. Rename to training_status.json (atomic operation)
    3. No partial reads, no corruption

DATA FLOW
---------

Training Loop → StatusWriter → JSON File → StatusReader → Web UI

1. Training loop calls update_progress() every step:
   - step, loss, learning_rate, epoch
   - Preserves last inference data (cached)

2. Periodic inference calls update_inference():
   - prompt, golden answer, model output, matches
   - Appends to recent_examples (last 5)
   - Updates running accuracy stats

3. Advanced collectors call update_advanced_metrics():
   - Pattern heatmaps, layer activity, LoRA stats
   - Throughput tracking, alerts, penalties
   - Validation metrics, confidence calibration

4. Status written atomically to training_status.json

5. Inference details appended to logs/inference_YYYYMMDD.log (daily rotation)

CACHING STRATEGY
----------------
Writer maintains internal state between updates to avoid data loss:

1. last_inference_data: Most recent inference result
   - Preserved across update_progress() calls
   - Prevents loss of inference data during frequent progress updates

2. recent_examples: Last 5 inference examples
   - Deduped by (step, current_file, model_output)
   - Ordered by recency (newest last)

3. pattern_heatmap: Aggregated pattern×length distribution
   - Updated incrementally as new patterns observed
   - Persisted across all updates

4. penalty_heatmap: Aggregated logit penalty hits
   - Tracks penalty system effectiveness
   - Updated from logit_penalty_stats

5. vram_samples: Last 60 throughput+VRAM measurements
   - Rolling window for trend analysis
   - Used by throughput charts

6. pattern_loss_history: Per-pattern loss tracking
   - Last 50 losses per pattern_id
   - Enables pattern-specific analysis

TRAINING_STATUS.JSON SCHEMA
training_status.json Format
---------------------------
Complete JSON schema for the status file:

{
    // Training state
    "status": "training",  // "idle" | "training" | "crashed" | "completed"
    "current_step": 1000,  // Current training step (int)
    "total_steps": 10000,  // Total steps to train (int)
    "epoch": 2,            // Current epoch (int)
    "loss": 0.45,          // Current training loss (float)
    "learning_rate": 0.0002, // Current learning rate (float)
    "timestamp": "2025-11-24T10:30:00",  // ISO timestamp (str)

    // Model information
    "model_name": "Qwen3-0.6B",  // Display name for UI (str or null)
    "max_output_tokens": 2048,   // Max generation length (int)
    "context_window": 4096,      // Training context size (int)

    // File/batch progress
    "batch_step": 50,            // Step within current file (int or null)
    "batch_total_steps": 118,    // Total steps for file (int or null)
    "batch_number": 6,           // Current batch number (int or null)
    "batch_queue_size": 22,      // Total batches queued (int or null)
    "current_file": "data.jsonl", // Filename being trained (str or null)

    // Latest inference results
    "current_system_prompt": "...",  // System prompt (str or null)
    "current_prompt": "What is 2+2?", // Input prompt (str or null)
    "golden_answer": "4",            // Expected answer (str or null)
    "model_answer": "4",             // Model's answer (str or null)
    "answer_matches": true,          // Correct? (bool or null)

    // Recent inference examples (last 5)
    "recent_examples": [
        {
            "step": 950,
            "current_file": "data.jsonl",
            "system_prompt": "...",
            "prompt": "What is 2+2?",
            "golden": "4",
            "model_output": "4",
            "matches": true,
            "loss": 0.45
        }
    ],

    // Running accuracy
    "total_evals": 100,       // Total inference evaluations (int)
    "total_correct": 85,      // Correct answers (int)
    "accuracy_percent": 85.0, // Accuracy (float)

    // Validation metrics
    "validation_loss": 0.52,  // Loss on fixed val set (float or null)
    "val_train_gap": 0.07,    // val_loss - train_loss (float or null)

    // Think tag tracking
    "think_tag_count": 5,     // Count with <think> tags (int)
    "think_tag_percent": 5.0, // Percentage with <think> (float)

    // Output length tracking
    "max_golden_output_length": 512,   // Max golden tokens (int or null)
    "max_model_output_length": 1024,   // Max model tokens (int or null)
    "current_golden_output_length": 128, // Current golden (int or null)
    "current_model_output_length": 130,  // Current model (int or null)

    // Fixed evaluation metrics
    "fixed_eval_em": 0.82,    // Exact match on fixed set (float or null)
    "fixed_eval_ce": 0.45,    // Cross-entropy (float or null)
    "fixed_eval_ece": 0.08,   // Expected calibration error (float or null)
    "fixed_eval_trend": "improving", // "improving"|"stable"|"degrading" (str or null)

    // Extended accuracy tracking
    "accuracy_last_20": 88.0,     // Last 20 evals (float or null)
    "accuracy_last_50": 85.5,     // Last 50 evals (float or null)
    "accuracy_trend": "improving", // Trend (str or null)

    // Pattern analysis
    "pattern_heatmap": {...},  // Pattern×length heatmap (dict or null)
    "pattern_loss_trend": {...}, // Loss per pattern (dict or null)
    "pattern_layer_correlation": {...}, // Layer influence (dict or null)
    "length_bin_staleness": {...}, // Seconds since observation (dict or null)

    // Layer activity
    "layer_activity_summary": {...},  // Layer statistics (dict or null)
    "layer_stability_summary": {...}, // Stability ranking (dict or null)

    // LoRA monitoring
    "lora_stats": {...},    // Per-layer gradient norms (dict or null)
    "lora_summary": {...},  // High-level LoRA activity (dict or null)

    // Streaming metrics
    "streaming_ce": 0.44,       // EMA-smoothed cross-entropy (float or null)
    "loss_variance": 0.02,      // Loss noise level (float or null)
    "token_entropy": 1.25,      // Prediction uncertainty (float or null)
    "loss_trend": "improving",  // Loss trend (str or null)

    // Alerts
    "active_alerts": [],     // List of active alerts (list or null)
    "alert_summary": {...},  // Counts by severity (dict or null)

    // Throughput
    "tokens_per_sec": 1250.5,         // Current throughput (float or null)
    "tokens_per_sec_avg": 1200.0,     // Average throughput (float or null)
    "tokens_per_sec_baseline": 1220.0, // Baseline after warmup (float or null)
    "throughput_trend": "stable",      // Trend (str or null)
    "throughput_vram_samples": [...],  // Recent samples (list or null)
    "queue_velocity": {...},           // Est. samples/sec/hour (dict or null)

    // Logit penalties
    "logit_penalty_stats": [...],  // Penalty hit counters (list or null)
    "penalty_heatmap": {...},      // Penalty distribution (dict or null)

    // Error info (if crashed)
    "error_message": "CUDA OOM",   // Error description (str or null)
    "error_type": "OOM"            // Error category (str or null)
}

Notes:
    - All writes are atomic (write to temp file, then rename)
    - Null values indicate the metric is not currently tracked
    - Timestamps are ISO 8601 format (YYYY-MM-DDTHH:MM:SS)
    - Dict/list fields have variable structure (see specific collectors)

USAGE EXAMPLES
--------------

Basic Training Loop:

    from pathlib import Path
    from training_status import TrainingStatusWriter

    writer = TrainingStatusWriter(
        status_file=Path("status/training_status.json"),
        max_output_tokens=2048,
        context_window=4096,
        model_name="Qwen3-0.6B"
    )

    # During training
    for step in range(total_steps):
        # Update progress every step
        writer.update_progress(
            step=step,
            total_steps=total_steps,
            epoch=epoch,
            loss=loss,
            lr=learning_rate
        )

        # Periodic inference (every 50 steps)
        if step % 50 == 0:
            prompt = validation_set[step % len(validation_set)]
            model_output = model.generate(prompt)

            writer.update_inference(
                step=step,
                total_steps=total_steps,
                epoch=epoch,
                loss=loss,
                lr=learning_rate,
                prompt=prompt['input'],
                golden=prompt['expected'],
                model_output=model_output,
                matches=(model_output.strip() == prompt['expected'].strip())
            )

    # On completion
    writer.mark_completed(final_step=total_steps, total_steps=total_steps)

With Advanced Metrics:

    # Training loop with advanced monitoring
    for step in range(total_steps):
        writer.update_progress(
            step=step,
            total_steps=total_steps,
            epoch=epoch,
            loss=loss,
            lr=learning_rate,
            # NEW: Validation metrics
            val_loss=validation_loss,
            # NEW: Throughput tracking
            tokens_per_sec=throughput_tracker.current(),
            tokens_per_sec_avg=throughput_tracker.average(),
            tokens_per_sec_baseline=throughput_tracker.baseline,
            # NEW: Loss stability
            loss_variance=loss_tracker.variance(),
            loss_trend=loss_tracker.trend(),  # "improving" | "stable" | "degrading"
            # NEW: Alerts
            active_alerts=alert_system.get_active(),
            alert_summary=alert_system.summary(),
            # NEW: Logit penalties
            logit_penalty_stats=penalty_tracker.stats(),
            penalty_heatmap=penalty_tracker.heatmap()
        )

Error Handling:

    try:
        # Training loop
        train_model()
    except torch.cuda.OutOfMemoryError as e:
        writer.mark_crashed(
            error="CUDA out of memory",
            error_type="OOM"
        )
        raise
    except Exception as e:
        writer.mark_crashed(
            error=str(e),
            error_type=type(e).__name__
        )
        raise

Reading Status:

    from training_status import TrainingStatusReader

    reader = TrainingStatusReader(Path("status/training_status.json"))

    # Check if training is active
    if reader.is_training():
        print("Training in progress")

    # Check for crashes
    if reader.has_crashed():
        status = reader.read()
        print(f"Training crashed: {status.error_message}")

    # Get full status
    status = reader.read()
    if status:
        print(f"Step {status.current_step}/{status.total_steps}")
        print(f"Loss: {status.loss:.4f}")
        print(f"Accuracy: {status.accuracy_percent:.1f}%")

KEY METHODS
-----------

TrainingStatusWriter Methods:

1. update_progress(step, total_steps, epoch, loss, lr, ...)
   - Called every training step
   - Updates basic progress metrics
   - Preserves last inference data (cached)
   - Fast (~1ms write time)

2. update_inference(step, ..., prompt, golden, model_output, matches)
   - Called during periodic inference (every N steps)
   - Updates inference results and accuracy
   - Appends to recent_examples (last 5)
   - Appends to daily inference log
   - Slower (~5ms due to log append)

3. update_advanced_metrics(step, ..., pattern_heatmap, lora_stats, ...)
   - Called by advanced monitoring collectors
   - Updates detailed analytics (patterns, LoRA, alerts, etc.)
   - Optional (training works without this)

4. mark_crashed(error, error_type)
   - Call on exception to mark training as crashed
   - Sets status="crashed" with error details
   - UI displays crash alert

5. mark_completed(final_step, total_steps)
   - Call when training finishes successfully
   - Sets status="completed"
   - UI shows completion message

TrainingStatusReader Methods:

1. read() -> TrainingStatus | None
   - Read current status from file
   - Returns None if file doesn't exist
   - Returns None on JSON parse error

2. is_training() -> bool
   - Check if training is currently active
   - Returns status.status == "training"

3. has_crashed() -> bool
   - Check if last training crashed
   - Returns status.status == "crashed"

INTEGRATION WITH TRAINING DAEMON
---------------------------------

In core/train.py (main training script):

    # Initialize status writer
    status_writer = TrainingStatusWriter(
        status_file=Path(base_dir) / "status" / "training_status.json",
        max_output_tokens=2048,
        context_window=config.max_length,
        model_name=model_config.name_or_path
    )

    # Training loop with HuggingFace Trainer
    class StatusCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # Called every logging step
            status_writer.update_progress(
                step=state.global_step,
                total_steps=state.max_steps,
                epoch=state.epoch,
                loss=logs.get('loss', 0),
                lr=logs.get('learning_rate', 0)
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            # Called after validation
            status_writer.update_progress(
                step=state.global_step,
                total_steps=state.max_steps,
                epoch=state.epoch,
                loss=logs.get('loss', 0),
                lr=logs.get('learning_rate', 0),
                val_loss=metrics.get('eval_loss')
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[StatusCallback()]
    )

    try:
        trainer.train()
        status_writer.mark_completed(
            final_step=trainer.state.global_step,
            total_steps=trainer.state.max_steps
        )
    except Exception as e:
        status_writer.mark_crashed(
            error=str(e),
            error_type=type(e).__name__
        )
        raise

SIDE EFFECTS & FILE OUTPUTS
----------------------------

1. status/training_status.json
   - Main status file (atomic writes)
   - Updated every training step (~1-5 seconds)
   - Read by web UI for real-time display
   - Size: ~10-50KB depending on metrics

2. logs/inference_YYYYMMDD.log
   - Daily rotating inference log
   - One JSON line per inference
   - Appended by update_inference()
   - Used for historical analysis
   - Size: ~1-10MB per day

3. logs/pattern_layer_history.jsonl
   - Pattern×layer correlation log
   - One JSON line per inference with layer activity
   - Appended by update_inference() if pattern_metadata provided
   - Used for pattern analysis
   - Size: ~1-5MB per day

Directory Creation:
    - Creates status/ if missing
    - Creates logs/ if missing
    - No error if already exists

File Permissions:
    - Files created with default umask
    - Readable by all processes (monitoring, UI)
    - Writable only by training process

THREAD SAFETY
-------------

Atomic Writes:
    ✓ Write to .tmp file, then rename (atomic operation)
    ✓ No partial reads during concurrent access
    ✓ Safe for multiple readers

Single Writer Assumption:
    ✗ NOT safe for multiple writers
    ✗ Design assumption: Only training daemon writes
    ✗ Multiple writers would cause race conditions

Caching:
    ✗ Internal caches (last_inference_data, recent_examples) not thread-safe
    ✗ Must use from single thread (main training thread)

ERROR HANDLING
--------------

Missing Directories:
    - status/ and logs/ created automatically
    - No error if already exists (mkdir_p=True)

Write Errors:
    - Atomic write prevents corruption
    - Temp file left behind on failure
    - Training continues (logging is non-critical)

Read Errors:
    - Reader returns None on missing file
    - Reader returns None on JSON parse error
    - UI handles None gracefully (shows "No training active")

Disk Full:
    - Write fails silently (try/except in append logs)
    - Status file write will fail
    - Training process should handle (not this module's responsibility)

Corrupt JSON:
    - Reader catches json.JSONDecodeError
    - Returns None (UI shows error)
    - Next write creates fresh file

GOTCHAS & EDGE CASES
--------------------

1. Thinking Emoji Auto-Injection:
   - _ensure_thinking_instruction() adds emoji instruction to prompts
   - _ensure_thinking_prefix() adds emoji prefix to outputs
   - Only applies if not already present
   - Can be disabled by removing these calls

2. Recent Examples Deduplication:
   - Deduped by (step, current_file, model_output)
   - Prevents duplicate entries in UI
   - Keeps last 5 unique examples

3. Pattern Loss History Window:
   - Last 50 losses per pattern_id
   - Older losses automatically dropped
   - Prevents unbounded memory growth

4. VRAM Samples Window:
   - Last 60 samples kept
   - Rolling window for trend analysis
   - ~1 minute of data at 1 sample/sec

5. Null vs Empty:
   - null: Metric not tracked (missing data)
   - empty list/dict: Tracked but no data yet
   - UI handles both gracefully

6. Status vs State:
   - training_status.json: Training progress (this module)
   - control/state.json: Control signals (training_controller.py)
   - Different files, different purposes

HISTORY
-------
Created: 2025-11-05 - Initial implementation
Updated: 2025-11-10 - Added atomic writes
Updated: 2025-11-15 - Added validation metrics
Updated: 2025-11-16 - Added think tag tracking
Updated: 2025-11-20 - Added advanced metrics

RELATED MODULES
---------------
- core/train.py - Main training script (writes status)
- core/training_controller.py - Control signals (different from status)
- monitoring/servers/live_monitor.py - Web UI (reads status)
- monitoring/servers/memory_stats_api.py - Memory API (reads status)

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

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
    train_loss_history: List[Dict[str, Any]] = None  # [{step, loss, lr}, ...]
    accuracy_history: List[float] = None
    steps_per_second: Optional[float] = None  # Training speed
    eta_seconds: Optional[int] = None  # Estimated time to finish current file

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

    # NEW: Protocol conformance stats (50% emoji mode)
    protocol_stats: Optional[Dict] = None             # Emoji/direct mode validation stats

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
    """
    Writes training status updates to disk for UI and monitoring consumption.

    Responsibilities:
        - Maintain current training state (step, loss, lr, etc.)
        - Track running inference accuracy and recent examples
        - Write status updates to JSON file (atomic writes)
        - Append detailed inference logs to daily log files
        - Cache advanced metrics (pattern heatmaps, layer activity, etc.)

    Data Flow:
        1. Training loop calls update_progress() every step
        2. Inference callbacks call update_inference() periodically
        3. Advanced collectors call update_advanced_metrics() for detailed stats
        4. Status written atomically to training_status.json
        5. Inference details appended to logs/inference_YYYYMMDD.log

    Key Methods:
        - update_progress(): Update basic training progress (step, loss, lr)
        - update_inference(): Update with inference results (prompt, answer, matches)
        - update_advanced_metrics(): Update with advanced metrics (patterns, LoRA, etc.)
        - mark_crashed(): Mark training as crashed with error info
        - mark_completed(): Mark training as successfully completed

    Caching Strategy:
        - last_inference_data: Most recent inference (preserved across progress updates)
        - recent_examples: Last 5 inference examples (deduped by step+file+output)
        - pattern_heatmap: Aggregated pattern×length distribution
        - penalty_heatmap: Aggregated logit penalty hits
        - layer_stability_summary: Layer activity stability ranking
        - vram_samples: Last 60 throughput+VRAM measurements

    Side Effects:
        - Writes training_status.json atomically (temp file + rename)
        - Appends to logs/inference_YYYYMMDD.log (daily rotation)
        - Appends to logs/pattern_layer_history.jsonl (pattern tracking)
        - Creates status/ and logs/ directories if missing

    Thread Safety:
        - Write operations are atomic (temp file + rename)
        - Not thread-safe for concurrent writes (use from single training thread)

    Example:
        writer = TrainingStatusWriter(
            status_file=Path("status/training_status.json"),
            max_output_tokens=2048,
            context_window=4096,
            model_name="Qwen3-0.6B"
        )

        # During training loop
        for step in range(total_steps):
            writer.update_progress(
                step=step,
                total_steps=total_steps,
                epoch=epoch,
                loss=loss,
                lr=lr
            )

            # Periodic inference
            if step % 50 == 0:
                writer.update_inference(
                    step=step,
                    prompt="What is 2+2?",
                    golden="4",
                    model_output="4",
                    matches=True
                )
    """

    def __init__(
        self,
        status_file: Path,
        max_output_tokens: int = 2048,
        context_window: int = 2048,
        model_name: Optional[str] = None
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
        self.loss_history = []  # Track last 200 loss values for graph
        self.loss_history_max = 200  # Keep last N losses
        # Steps/second tracking for ETA
        self.last_step = 0
        self.last_step_time = None
        self.steps_per_second = 0.0
        self.steps_history = []  # Track recent step times for smoothing
        self.steps_history_max = 10
        self.queue_velocity_snapshot = None
        self.vram_samples = []
        self.penalty_stats_snapshot = None
        self.layer_stability_summary = None
        self.protocol_stats_snapshot = None  # Protocol conformance (50% emoji mode)
        # Append-only log for inference previews
        self.inference_log_dir = self.status_file.parent / "logs"
        self.inference_log_dir.mkdir(parents=True, exist_ok=True)
        self.pattern_layer_log = self.inference_log_dir / "pattern_layer_history.jsonl"
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
        protocol_stats: Optional[Dict] = None,
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
        if protocol_stats is not None:
            self.protocol_stats_snapshot = protocol_stats

        # Track loss history for graphing
        self.loss_history.append({"step": step, "loss": loss, "lr": lr})
        if len(self.loss_history) > self.loss_history_max:
            self.loss_history = self.loss_history[-self.loss_history_max:]

        # Calculate steps/second and ETA
        now = datetime.now()
        if self.last_step_time and step > self.last_step:
            elapsed = (now - self.last_step_time).total_seconds()
            if elapsed > 0:
                instant_rate = (step - self.last_step) / elapsed
                self.steps_history.append(instant_rate)
                if len(self.steps_history) > self.steps_history_max:
                    self.steps_history = self.steps_history[-self.steps_history_max:]
                # Smoothed average
                self.steps_per_second = sum(self.steps_history) / len(self.steps_history)
        self.last_step = step
        self.last_step_time = now

        # Calculate ETA for current file
        eta_seconds = None
        if self.steps_per_second > 0 and batch_total_steps and batch_step:
            remaining_steps = batch_total_steps - batch_step
            if remaining_steps > 0:
                eta_seconds = int(remaining_steps / self.steps_per_second)

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
            protocol_stats=self.protocol_stats_snapshot,
            train_loss_history=self.loss_history[-200:] if self.loss_history else None,
            steps_per_second=round(self.steps_per_second, 2) if self.steps_per_second else None,
            eta_seconds=eta_seconds,
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
            protocol_stats=self.protocol_stats_snapshot,
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
DEFAULT_STATUS_FILE = Path("/path/to/training/status/training_status.json")


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
