#!/usr/bin/env python3
"""
Training Callbacks for Monitoring

Extracted from core/train.py to enable better modularity.
Contains LiveMonitorCallback for real-time training monitoring.
"""

import os
import time
import torch
import math
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import OrderedDict
from transformers import TrainerCallback

from trainer.monitoring.status_writer import TrainingStatusWriter
from core.logit_penalty import reset_processor_states, collect_penalty_stats
from monitoring.servers.pattern_tracker import PatternTracker, get_default_patterns
from monitoring.servers.layer_monitor import LayerMonitor

# World State integration (heartbeats)
try:
    from core.heartbeat import HeartbeatWriter
    from core.events import emit_event
    WORLD_STATE_AVAILABLE = True
except ImportError:
    WORLD_STATE_AVAILABLE = False
    HeartbeatWriter = None

# Realm Store - single source of truth for UI state
try:
    from core.realm_store import (
        update_training as realm_update_training,
        update_worker as realm_update_worker,
        emit_event as realm_emit_event,
    )
    REALM_STORE_AVAILABLE = True
except ImportError:
    REALM_STORE_AVAILABLE = False
    def realm_update_training(**kwargs): pass
    def realm_update_worker(worker_id, **kwargs): pass
    def realm_emit_event(kind, message, **kwargs): pass

# Optional imports for remote evaluation
try:
    from data_manager.remote_evaluator import RemoteEvaluator
    REMOTE_EVAL_AVAILABLE = True
except ImportError:
    REMOTE_EVAL_AVAILABLE = False


class LiveMonitorCallback(TrainerCallback):
    """
    Real-time training monitor callback.

    Provides:
    - Progress tracking (steps, loss, learning rate)
    - Validation loss tracking (micro-eval)
    - Throughput monitoring (tokens/sec, VRAM usage)
    - Pattern tracking (heatmaps)
    - Layer monitoring
    - Control signal handling (pause/stop)
    - Smart alerts (throughput drops, val gaps, etc.)
    - Checkpoint ledger recording (on_save)
    - Remote evaluation sync (on_save)
    """

    def __init__(
        self,
        monitor,
        status_writer: TrainingStatusWriter,
        eval_steps: int,
        total_steps: int,
        raw_train_examples: List[Dict],
        tokenizer,
        model,
        batch_total_steps: int,
        current_global_step: int,
        evolution_tracker=None,
        current_file: Optional[str] = None,
        batch_number: Optional[int] = None,
        batch_queue_size: Optional[int] = None,
        controller=None,
        fixed_val_dataset=None,
        avg_seq_len: float = 0.0,
        effective_batch: int = 1,
        micro_eval_inputs=None,
        micro_eval_interval: int = 500,
        logits_processor=None,
        layer_monitor: Optional[LayerMonitor] = None,
        remote_eval_config: Optional[Dict] = None,
        skill_id: Optional[str] = None,
        skill_level: Optional[int] = None,
    ):
        """
        Initialize LiveMonitorCallback.

        Args:
            monitor: Live inference monitor
            status_writer: Training status writer
            eval_steps: Evaluation interval
            total_steps: Total training steps
            raw_train_examples: Raw training examples (pre-tokenization)
            tokenizer: Model tokenizer
            model: Training model
            batch_total_steps: Steps in current batch/file
            current_global_step: Starting global step for this batch
            evolution_tracker: Optional evolution tracker
            current_file: Current file being trained
            batch_number: Current batch number
            batch_queue_size: Total batches in queue
            controller: Training controller (pause/stop)
            fixed_val_dataset: Fixed validation dataset
            avg_seq_len: Average sequence length
            effective_batch: Effective batch size
            micro_eval_inputs: Tokenized micro eval inputs
            micro_eval_interval: Micro eval interval
            logits_processor: Logits processor for generation
            layer_monitor: Optional layer monitor
            remote_eval_config: Remote evaluation configuration
        """
        self.monitor = monitor
        self.status_writer = status_writer
        self.eval_steps = eval_steps
        self.total_steps = total_steps
        self.raw_train_examples = raw_train_examples
        self.tokenizer = tokenizer
        self.model_ref = model
        self.evolution_tracker = evolution_tracker
        self.controller = controller  # Training control system
        self.fixed_val_dataset = fixed_val_dataset  # Fixed validation set

        # Skill context (for skill-based training)
        self.skill_id = skill_id
        self.skill_level = skill_level
        self.skill_context = None
        if skill_id:
            try:
                from core.training_status import build_skill_context
                self.skill_context = build_skill_context(skill_id, skill_level)
            except Exception as e:
                print(f"⚠️  Failed to build skill context: {e}")

        # Remote evaluation setup
        self.remote_eval_config = remote_eval_config or {}
        self.remote_evaluator = None
        if REMOTE_EVAL_AVAILABLE and self.remote_eval_config.get("enabled", False):
            try:
                # Get defaults from hosts.json if not in config
                from core.hosts import get_host
                inference = get_host("3090")
                default_host = inference.host if inference else "localhost"
                default_port = inference.services.get("inference", {}).port if inference else 8765
                self.remote_evaluator = RemoteEvaluator(
                    host=self.remote_eval_config.get("host", default_host),
                    port=self.remote_eval_config.get("port", default_port)
                )
                print(f"✅ Remote eval enabled: {self.remote_eval_config.get('host', default_host)}:{self.remote_eval_config.get('port', default_port)}")
            except Exception as e:
                print(f"⚠️  Remote eval setup failed: {e}")
                self.remote_evaluator = None
        self.last_remote_eval_step = 0

        self.last_update_time = time.time()
        self.last_prompt_snapshot_time = time.time()
        self.update_interval = 1  # Status JSON refresh cadence (seconds)

        # Micro-eval settings
        self.micro_eval_inputs = micro_eval_inputs
        self.micro_eval_interval = micro_eval_interval
        self.last_micro_eval_step = 0
        self.control_check_interval = 10  # Check control signals every 10 steps
        self.last_control_check_step = 0
        self.last_val_loss = None  # Track validation loss

        # Throughput tracking
        self.prev_step_time = time.time()
        self.steps_per_sec_ema = None
        self.tokens_per_sec_ema = None
        self.avg_seq_len = avg_seq_len
        self.effective_batch = max(1, effective_batch)

        # Loss stability tracking
        self.loss_window = []
        self.loss_window_size = 50
        self.loss_trend = None

        # Alert tracking baseline
        self.throughput_baseline = None

        # Batch progress tracking
        self.batch_total_steps = batch_total_steps
        self.current_global_step = current_global_step  # Starting global_step for this batch
        self.current_file = current_file
        self.batch_number = batch_number
        self.batch_queue_size = batch_queue_size
        self.logits_processor = logits_processor
        self.pattern_tracker = PatternTracker(get_default_patterns())
        self.layer_monitor = layer_monitor
        self.total_vram = None
        self.low_vram_counter = 0
        self.last_vram_ratio = None

        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                self.total_vram = props.total_memory / (1024 ** 3)
            except Exception:
                self.total_vram = None

        self.penalty_last_hits: Dict[str, int] = {}
        self.penalty_totals: Dict[str, int] = {}
        self.penalty_files: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
        self.latest_penalty_delta: Dict[str, int] = {}
        self.vram_samples = []
        self.queue_velocity = None
        self.enable_penalty_metrics = os.environ.get("ENABLE_PENALTY_MONITORING", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # World State heartbeat integration
        self.heartbeat_writer = None
        if WORLD_STATE_AVAILABLE and HeartbeatWriter:
            try:
                # Auto-detect GPU device
                device_name = "GPU0"
                if torch.cuda.is_available():
                    device_name = f"GPU{torch.cuda.current_device()}"
                self.heartbeat_writer = HeartbeatWriter(
                    worker_id="training_daemon",
                    role="training",
                    device=device_name,
                )
            except Exception as e:
                print(f"⚠️  Heartbeat setup failed: {e}")

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step."""
        current_time = time.time()
        current_loss = state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0
        # Use configured LR for constant scheduler
        # Don't rely on log_history which has stale values from checkpoint resume
        current_lr = args.learning_rate
        current_epoch = state.epoch if state.epoch else 0

        # Track loss stability
        self.loss_window.append(current_loss)
        if len(self.loss_window) > self.loss_window_size:
            self.loss_window.pop(0)

        loss_variance = None
        if len(self.loss_window) > 1:
            mean_loss = sum(self.loss_window) / len(self.loss_window)
            loss_variance = sum((x - mean_loss) ** 2 for x in self.loss_window) / len(self.loss_window)
            if len(self.loss_window) >= 20:
                recent = self.loss_window[-10:]
                earlier = self.loss_window[:10]
                if earlier:
                    delta = (sum(recent) / len(recent)) - (sum(earlier) / len(earlier))
                    if delta < -0.001:
                        self.loss_trend = "improving"
                    elif delta > 0.001:
                        self.loss_trend = "rising"
                    else:
                        self.loss_trend = "stable"

        # CRITICAL FIX #5: NaN detection
        if math.isnan(current_loss) or math.isinf(current_loss):
            print(f"\n❌ CRITICAL: NaN/Inf loss detected at step {state.global_step}!")
            print(f"   Loss value: {current_loss}")
            print(f"   This indicates model corruption - training will be stopped")
            print(f"   The model will need to be restored from last good checkpoint")
            # Stop training immediately
            try:
                self.status_writer.mark_crashed(error="NaN/Inf loss detected", error_type="NaN")
            except Exception:
                pass
            control.should_training_stop = True
            return control

        # Check for pause/stop signals (every N steps)
        if self.controller and (state.global_step - self.last_control_check_step) >= self.control_check_interval:
            self.last_control_check_step = state.global_step

            # Check for stop signal
            if self.controller.should_stop_after_batch():
                print(f"\n🛑 STOP signal detected at step {state.global_step}")
                print(f"   Stopping training gracefully after current batch...")
                control.should_training_stop = True
                return control

            # Check for pause signal
            if self.controller.should_pause_after_batch():
                print(f"\n⏸️  PAUSE signal detected at step {state.global_step}")
                print(f"   Pausing training gracefully after current batch...")
                control.should_training_stop = True
                return control

        # Calculate batch-relative step (step within current file)
        # state.global_step is cumulative, so subtract the starting point
        batch_step = state.global_step - self.current_global_step

        # Throughput tracking (steps/sec and rough tokens/sec)
        step_time = current_time - self.prev_step_time if self.prev_step_time else None
        tokens_per_sec = None
        steps_per_sec = None
        if step_time and step_time > 0:
            steps_per_sec = 1.0 / step_time
            tokens_per_step = self.avg_seq_len * self.effective_batch if self.avg_seq_len else None
            tokens_per_sec = steps_per_sec * tokens_per_step if tokens_per_step else None
            self.steps_per_sec_ema = steps_per_sec if self.steps_per_sec_ema is None else 0.1 * steps_per_sec + 0.9 * self.steps_per_sec_ema
            if tokens_per_sec is not None:
                self.tokens_per_sec_ema = tokens_per_sec if self.tokens_per_sec_ema is None else 0.1 * tokens_per_sec + 0.9 * self.tokens_per_sec_ema
                if self.throughput_baseline is None and state.global_step > 20:
                    self.throughput_baseline = self.tokens_per_sec_ema

        current_vram = None
        if torch.cuda.is_available():
            try:
                current_vram = max(torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) / (1024 ** 3)
            except Exception:
                current_vram = None
        ratio = None
        if current_vram is not None and self.total_vram:
            ratio = current_vram / self.total_vram
            self.last_vram_ratio = ratio
            if ratio < 0.6:
                self.low_vram_counter += 1
            else:
                self.low_vram_counter = 0
        else:
            self.low_vram_counter = 0
        if steps_per_sec:
            samples_per_sec = steps_per_sec * self.effective_batch
            self.queue_velocity = {
                "samples_per_sec": samples_per_sec,
                "samples_per_hour": samples_per_sec * 3600,
                "effective_batch": self.effective_batch,
            }
        if tokens_per_sec is not None and current_vram is not None:
            self.vram_samples.append({
                "step": state.global_step,
                "tokens_per_sec": tokens_per_sec,
                "vram_gb": round(current_vram, 3),
                "penalty": None,
            })
            if len(self.vram_samples) > 120:
                self.vram_samples.pop(0)

        self.prev_step_time = current_time

        # Micro-eval DISABLED - use 3090 for all evaluation
        val_loss = self.last_val_loss

        # Build simple alerts within the 10% overhead budget
        alerts = []
        alert_summary = None
        if self.throughput_baseline and self.tokens_per_sec_ema:
            if self.tokens_per_sec_ema < 0.6 * self.throughput_baseline:
                alerts.append({"severity": "warn", "type": "throughput_drop", "detail": "Throughput <60% of baseline"})
        if val_loss is not None and current_loss is not None:
            gap = val_loss - current_loss
            if gap > 0.3:
                alerts.append({"severity": "warn", "type": "val_gap", "detail": f"val-train gap {gap:.3f}"})
        if loss_variance is not None and self.loss_window:
            mean_loss = sum(self.loss_window) / len(self.loss_window)
            if mean_loss > 0 and loss_variance > (mean_loss ** 2) * 0.25:
                alerts.append({"severity": "info", "type": "loss_variance", "detail": "High loss variance"})
        if self.last_vram_ratio is not None and self.low_vram_counter >= 15:
            severity = "warn" if self.last_vram_ratio < 0.5 else "info"
            detail = f"VRAM usage at {self.last_vram_ratio*100:.1f}% capacity"
            alerts.append({"severity": severity, "type": "low_vram_utilization", "detail": detail})
        if alerts:
            summary = {}
            for a in alerts:
                summary[a["severity"]] = summary.get(a["severity"], 0) + 1
            alert_summary = summary

        # Time-based status updates (every ~2 seconds)
        if current_time - self.last_update_time >= self.update_interval:
            penalty_stats = None
            penalty_heatmap_payload = None
            penalty_deltas: Dict[str, int] = {}
            if self.enable_penalty_metrics and self.logits_processor:
                penalty_stats = collect_penalty_stats(self.logits_processor)
                if penalty_stats:
                    for stat in penalty_stats:
                        label = stat.get("label", "penalty")
                        hits = stat.get("hits", 0)
                        previous = self.penalty_last_hits.get(label, 0)
                        delta = max(0, hits - previous)
                        self.penalty_last_hits[label] = hits
                        if delta > 0:
                            penalty_deltas[label] = delta
                            self.penalty_totals[label] = self.penalty_totals.get(label, 0) + delta
                            file_key = self.current_file or "unknown"
                            file_stats = self.penalty_files.setdefault(file_key, {})
                            file_stats[label] = file_stats.get(label, 0) + delta
                            while len(self.penalty_files) > 8:
                                self.penalty_files.popitem(last=False)
                    if self.penalty_files:
                        penalty_heatmap_payload = {
                            "totals": dict(self.penalty_totals),
                            "per_file": {k: dict(v) for k, v in self.penalty_files.items()}
                        }

            if self.vram_samples:
                if penalty_deltas:
                    self.vram_samples[-1]["penalty"] = penalty_deltas
                else:
                    self.vram_samples[-1]["penalty"] = None

            serialized_samples = []
            for sample in self.vram_samples:
                serialized_samples.append(sample)
            penalty_stats_payload = penalty_stats

            self.status_writer.update_progress(
                step=state.global_step,
                total_steps=self.total_steps,
                epoch=int(current_epoch),
                loss=current_loss,
                lr=current_lr,
                val_loss=self.last_val_loss,  # Use validation loss from on_evaluate callback
                batch_step=batch_step,
                batch_total_steps=self.batch_total_steps,
                batch_number=self.batch_number,
                batch_queue_size=self.batch_queue_size,
                current_file=self.current_file,
                tokens_per_sec=tokens_per_sec,
                tokens_per_sec_avg=self.tokens_per_sec_ema,
                tokens_per_sec_baseline=self.throughput_baseline,
                loss_variance=loss_variance,
                loss_trend=self.loss_trend,
                active_alerts=alerts if alerts else None,
                alert_summary=alert_summary,
                throughput_vram_samples=serialized_samples,
                queue_velocity=self.queue_velocity,
                logit_penalty_stats=penalty_stats_payload,
                penalty_heatmap=penalty_heatmap_payload,
                skill_context=self.skill_context,
            )

            # NEW: Update RealmStore (single source of truth for UI)
            if REALM_STORE_AVAILABLE:
                # Calculate ETA
                eta_seconds = None
                if self.steps_per_sec_ema and self.steps_per_sec_ema > 0:
                    remaining_steps = self.total_steps - state.global_step
                    if remaining_steps > 0:
                        eta_seconds = int(remaining_steps / self.steps_per_sec_ema)

                realm_update_training(
                    status="training",
                    step=state.global_step,
                    total_steps=self.total_steps,
                    loss=round(current_loss, 4) if current_loss else None,
                    learning_rate=current_lr,
                    file=self.current_file,
                    speed=round(self.steps_per_sec_ema, 3) if self.steps_per_sec_ema else None,
                    eta_seconds=eta_seconds,
                )

                # Also update worker state
                realm_update_worker(
                    worker_id="training_daemon",
                    role="training",
                    status="running",
                    current_job=self.current_file,
                    step=state.global_step,
                    total_steps=self.total_steps,
                    it_per_sec=round(self.steps_per_sec_ema, 3) if self.steps_per_sec_ema else None,
                )

            # World State heartbeat - emit during training with live stats
            if self.heartbeat_writer:
                try:
                    self.heartbeat_writer.beat(
                        status="running",
                        current_job_id=self.current_file,
                        current_job_type="train",
                        extra={
                            "step": state.global_step,
                            "total_steps": self.total_steps,
                            "loss": round(current_loss, 4) if current_loss else None,
                            "it_per_sec": round(self.steps_per_sec_ema, 3) if self.steps_per_sec_ema else None,
                            "current_file": self.current_file,
                        },
                        min_interval=5.0,  # Only write every 5 seconds max
                    )
                except Exception:
                    pass  # Don't let heartbeat failures affect training

            self.last_update_time = current_time

        # NOTE: Inference preview removed - use 3090 for inference, not 4090 during training

        # Evolution tracking (at special steps only)
        # DISABLED: Takes 60-300s per snapshot (too slow)
        if False and self.evolution_tracker and self.raw_train_examples:
            try:
                snapshot_id = self.evolution_tracker.capture_snapshot(
                    model=self.model_ref,
                    tokenizer=self.tokenizer,
                    examples=self.raw_train_examples,
                    current_step=state.global_step,
                    model_version="training",
                    max_examples=100  # Limit for performance
                )
                if snapshot_id:
                    print(f"📊 Evolution snapshot saved: {snapshot_id}")
            except Exception as e:
                print(f"Warning: Evolution snapshot failed at step {state.global_step}: {e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Capture validation loss after evaluation runs."""
        if metrics and 'eval_loss' in metrics:
            self.last_val_loss = metrics['eval_loss']
            print(f"\n📊 Validation Loss: {self.last_val_loss:.4f}")

            # Calculate train/val gap if we have recent training loss
            if state.log_history:
                recent_train_loss = state.log_history[-1].get('loss', None)
                if recent_train_loss:
                    gap = self.last_val_loss - recent_train_loss
                    print(f"   Train Loss: {recent_train_loss:.4f}")
                    print(f"   Val-Train Gap: {gap:+.4f}")
                    if gap > 0.5:
                        print(f"   ⚠️  Large gap detected - possible overfitting!")

    def on_save(self, args, state, control, **kwargs):
        """Handle checkpoint save - record to ledger, rename, sync to remote."""
        # =========================================================
        # CHECKPOINT LEDGER - Record stats at exact moment of save
        # =========================================================
        try:
            from core.checkpoint_ledger import record_checkpoint

            # Find the checkpoint that was just saved
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            if checkpoint_dir.exists():
                # Extract stats from training state
                train_loss = None
                val_loss = None
                learning_rate = None
                for entry in reversed(state.log_history or []):
                    if train_loss is None and "loss" in entry:
                        train_loss = entry["loss"]
                    if val_loss is None and "eval_loss" in entry:
                        val_loss = entry["eval_loss"]
                    if learning_rate is None and "learning_rate" in entry:
                        learning_rate = entry["learning_rate"]
                    if train_loss and learning_rate:
                        break

                # Record to ledger (also renames to canonical name)
                record = record_checkpoint(
                    step=state.global_step,
                    path=str(checkpoint_dir),
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=learning_rate,
                    epoch=state.epoch,
                    training_file=getattr(self, 'current_file', None),
                    rename=True,  # Rename to canonical: checkpoint-{step}-{date}-{time}
                )
                print(f"📖 Ledger: {record.canonical_name} (loss={train_loss:.4f})" if train_loss else f"📖 Ledger: {record.canonical_name}")

                # Battle Log - checkpoint saved event
                try:
                    from core.battle_log import log_training
                    loss_str = f" (loss: {train_loss:.4f})" if train_loss else ""
                    log_training(
                        f"Checkpoint {state.global_step:,} saved{loss_str}",
                        severity="success",
                        source="training.callback",
                        hero_id="DIO",
                        details={
                            "step": state.global_step,
                            "loss": train_loss,
                            "val_loss": val_loss,
                            "learning_rate": learning_rate,
                            "canonical_name": record.canonical_name,
                        },
                    )
                except Exception:
                    pass  # Don't let battle log errors affect training

                # World State - emit checkpoint_saved event
                if WORLD_STATE_AVAILABLE:
                    try:
                        from core.run_context import get_run_context
                        ctx = get_run_context()
                        emit_event(
                            "checkpoint_saved",
                            hero_id=ctx.hero_id or "unknown",
                            campaign_id=ctx.campaign_id or "unknown",
                            step=state.global_step,
                            path=str(record.path) if record else str(checkpoint_dir),
                            loss=train_loss,
                        )
                    except Exception:
                        pass

                # NEW: RealmStore - emit checkpoint_saved to battle log
                if REALM_STORE_AVAILABLE:
                    try:
                        loss_str = f"loss: {train_loss:.4f}" if train_loss else ""
                        realm_emit_event(
                            kind="checkpoint_saved",
                            message=f"💾 Saved checkpoint {state.global_step:,} ({loss_str})",
                            channel="training",
                            severity="success",
                            details={
                                "step": state.global_step,
                                "loss": train_loss,
                                "val_loss": val_loss,
                                "canonical_name": record.canonical_name if record else None,
                            },
                        )
                    except Exception:
                        pass

                # =========================================================
                # QUEUE EVALUATIONS - Quick eval + LITE passives
                # Also queue FULL eval every 5000 steps
                # =========================================================
                try:
                    from core.evaluation_ledger import queue_evaluation, queue_full_evaluation
                    from core.passives import queue_passive_lite
                    from core.paths import get_base_dir

                    # Get current curriculum level from canonical state (status/ symlink)
                    base_dir = get_base_dir()
                    curriculum_state_file = base_dir / "status" / "curriculum_state.json"
                    if curriculum_state_file.exists():
                        import json
                        with open(curriculum_state_file) as f:
                            curriculum = json.load(f)

                        # Queue QUICK eval for each active skill at current level
                        for skill_id, skill_state in curriculum.get("skills", {}).items():
                            current_level = skill_state.get("current_level", 1)
                            queue_evaluation(
                                checkpoint_step=state.global_step,
                                skill=skill_id,
                                level=current_level,
                                eval_type="quick",
                                priority=10,
                            )
                            print(f"📋 Queued quick eval: {skill_id} L{current_level}")

                        # Queue FULL eval every 5000 steps (all levels)
                        if state.global_step % 5000 == 0:
                            for skill_id in curriculum.get("skills", {}).keys():
                                queue_full_evaluation(
                                    checkpoint_step=state.global_step,
                                    skill=skill_id,
                                    priority=6,  # Medium priority
                                )
                                print(f"📋 Queued FULL eval: {skill_id} (all levels)")

                    # Queue LITE passives
                    queue_passive_lite(state.global_step)
                    print(f"📋 Queued LITE passives")

                except Exception as e:
                    print(f"⚠️  Eval queue failed: {e}")

                # =========================================================
                # EVAL SUITES - Policy-driven suite scheduling (P0/P1/P2)
                # =========================================================
                try:
                    from core.eval_scheduler import get_scheduler
                    from core.run_context import get_run_context

                    run_ctx = get_run_context()
                    scheduler = get_scheduler()

                    # Trigger P0 (gatekeeping) and eligible P1 (coverage) suites
                    suite_runs = scheduler.on_checkpoint_saved(
                        run_ctx=run_ctx,
                        checkpoint_step=state.global_step,
                        checkpoint_path=str(record.path) if record else str(checkpoint_dir),
                    )

                    if suite_runs:
                        suite_names = ", ".join(r.suite_id for r in suite_runs)
                        total_jobs = sum(r.jobs_submitted for r in suite_runs)
                        print(f"📊 Eval suites triggered: {suite_names} ({total_jobs} jobs)")

                except Exception as e:
                    print(f"⚠️  Eval scheduler failed: {e}")

                # =========================================================
                # LAYER STATS JOB - Model Archaeology analysis
                # Submit every N steps (configurable, default: every 5000 steps)
                # =========================================================
                try:
                    layer_stats_interval = int(os.environ.get("LAYER_STATS_INTERVAL", "5000"))
                    if state.global_step % layer_stats_interval == 0:
                        self._submit_layer_stats_job(
                            checkpoint_dir=checkpoint_dir,
                            step=state.global_step,
                            output_dir=args.output_dir,
                        )
                except Exception as e:
                    print(f"⚠️  Layer stats job submission failed: {e}")

        except Exception as e:
            print(f"⚠️  Ledger recording failed: {e}")

        # =========================================================
        # REMOTE EVAL - Sync and evaluate on 3090
        # =========================================================
        if not self.remote_evaluator or not self.remote_eval_config.get("enabled", False):
            return

        eval_interval = self.remote_eval_config.get("eval_interval_steps", 5000)

        # Check if it's time for remote eval
        if state.global_step - self.last_remote_eval_step < eval_interval:
            return

        print(f"\n🌐 Remote Eval: Checkpoint at step {state.global_step}")

        try:
            # Find the checkpoint directory (may have been renamed)
            checkpoint_dir = None
            for candidate in Path(args.output_dir).glob(f"checkpoint-{state.global_step}*"):
                if candidate.is_dir():
                    checkpoint_dir = candidate
                    break

            if not checkpoint_dir:
                checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            if not checkpoint_dir.exists():
                print(f"⚠️  Checkpoint not found: {checkpoint_dir}")
                return

            # Get remote config from hosts.json
            from core.hosts import get_host
            inference = get_host("3090")
            remote_user = self.remote_eval_config.get("remote_user", inference.ssh_user if inference else "")
            remote_host = self.remote_eval_config.get("host", inference.host if inference else "localhost")
            # Use paths.py constant for remote models dir fallback
            try:
                from paths import REMOTE_MODELS_DIR
                remote_dir = self.remote_eval_config.get("remote_models_dir", str(REMOTE_MODELS_DIR))
            except ImportError:
                remote_dir = self.remote_eval_config.get("remote_models_dir", "~/llm/models")
            model_id = f"qwen3-step-{state.global_step}"
            remote_path = f"{remote_dir}/{model_id}"

            # Async copy function
            def async_sync_and_eval():
                try:
                    if self.remote_eval_config.get("sync_checkpoints", True):
                        print(f"   📦 Syncing to {remote_host}:{remote_path}...")

                        # Use rsync for efficient transfer
                        rsync_cmd = [
                            "rsync", "-az", "--delete",
                            str(checkpoint_dir) + "/",
                            f"{remote_user}@{remote_host}:{remote_path}/"
                        ]

                        result = subprocess.run(
                            rsync_cmd,
                            capture_output=True,
                            text=True,
                            timeout=300
                        )

                        if result.returncode != 0:
                            print(f"   ⚠️  Sync failed: {result.stderr}")
                            return

                        print(f"   ✅ Sync complete")

                    # Register checkpoint with remote server
                    print(f"   📝 Registering model: {model_id}")
                    success = self.remote_evaluator.register_checkpoint(
                        checkpoint_path=remote_path,
                        model_id=model_id,
                        tags=f"step{state.global_step},training"
                    )

                    if not success:
                        print(f"   ⚠️  Registration failed")
                        return

                    # Submit eval job
                    dataset_ref = self.remote_eval_config.get("eval_dataset", "eval_dataset.jsonl")
                    metrics = self.remote_eval_config.get("metrics", ["accuracy", "loss"])

                    print(f"   🎯 Submitting eval job...")
                    job_id = self.remote_evaluator.submit_eval_job(
                        model_id=model_id,
                        dataset_ref=dataset_ref,
                        metrics=metrics
                    )

                    print(f"   ✅ Eval job submitted: {job_id}")
                    print(f"      Check status: python3 data_manager/remote_evaluator.py status {job_id}")

                except subprocess.TimeoutExpired:
                    print(f"   ⚠️  Sync timed out")
                except Exception as e:
                    print(f"   ⚠️  Remote eval error: {e}")

            # Start async sync and eval (don't wait)
            thread = threading.Thread(target=async_sync_and_eval, daemon=True)
            thread.start()

            self.last_remote_eval_step = state.global_step

        except Exception as e:
            print(f"⚠️  Remote eval setup failed: {e}")

    def _submit_layer_stats_job(
        self,
        checkpoint_dir: Path,
        step: int,
        output_dir: str,
    ):
        """
        Submit a layer_stats job for Model Archaeology analysis.

        Submits to the job server (VaultKeeper API) for async execution.
        Finds the previous checkpoint for drift comparison.
        """
        import requests

        # Get campaign info
        try:
            from core.hero import get_active_campaign
            campaign = get_active_campaign()
            campaign_id = campaign.get("campaign_id", "campaign-001")
            hero_id = campaign.get("hero_id", "dio-qwen3-0.6b")
        except Exception:
            campaign_id = "campaign-001"
            hero_id = "dio-qwen3-0.6b"

        # Find the checkpoint path (may have been renamed)
        checkpoint_path = checkpoint_dir
        if not checkpoint_path.exists():
            for candidate in Path(output_dir).glob(f"checkpoint-{step}*"):
                if candidate.is_dir():
                    checkpoint_path = candidate
                    break

        if not checkpoint_path.exists():
            print(f"⚠️  Layer stats: checkpoint not found at {checkpoint_path}")
            return

        # Find reference checkpoint (previous one for drift comparison)
        reference_path = None
        try:
            checkpoints = sorted(
                Path(output_dir).glob("checkpoint-*"),
                key=lambda x: self._extract_step_from_checkpoint(x.name),
            )
            # Find checkpoint with step < current step
            for ckpt in reversed(checkpoints):
                ckpt_step = self._extract_step_from_checkpoint(ckpt.name)
                if ckpt_step > 0 and ckpt_step < step:
                    reference_path = str(ckpt)
                    break
        except Exception as e:
            print(f"⚠️  Could not find reference checkpoint: {e}")

        # Build job payload
        payload = {
            "job_type": "layer_stats",
            "payload": {
                "campaign_id": campaign_id,
                "hero_id": hero_id,
                "checkpoint_path": str(checkpoint_path),
                "model_ref": "qwen3-0.6b",
                "compute_activations": True,
            },
            "priority": "normal",
        }

        if reference_path:
            payload["payload"]["reference_checkpoint_path"] = reference_path

        # Submit to job server
        job_server = os.environ.get("JOB_SERVER_URL", "http://localhost:8767")

        try:
            response = requests.post(
                f"{job_server}/api/jobs",
                json=payload,
                timeout=10,
            )

            result = response.json()

            if result.get("accepted"):
                print(f"🔬 Queued layer_stats job: {result['job_id']}")
                if reference_path:
                    ref_step = self._extract_step_from_checkpoint(Path(reference_path).name)
                    print(f"   Reference: checkpoint-{ref_step}")
            else:
                print(f"⚠️  Layer stats job rejected: {result.get('message', 'unknown')}")

        except requests.RequestException as e:
            print(f"⚠️  Could not submit layer_stats job: {e}")

    def _extract_step_from_checkpoint(self, name: str) -> int:
        """Extract step number from checkpoint directory name."""
        try:
            # Handle formats like:
            # - checkpoint-183000
            # - checkpoint-183000-20251128-1430
            parts = name.replace("checkpoint-", "").split("-")
            return int(parts[0])
        except (ValueError, IndexError):
            return 0


__all__ = ["LiveMonitorCallback"]
