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
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import OrderedDict
from transformers import TrainerCallback

from trainer.monitoring.status_writer import TrainingStatusWriter
from core.logit_penalty import reset_processor_states, collect_penalty_stats
from monitoring.servers.pattern_tracker import PatternTracker, get_default_patterns
from monitoring.servers.layer_monitor import LayerMonitor


class LiveMonitorCallback(TrainerCallback):
    """
    Real-time training monitor callback.

    Provides:
    - Progress tracking (steps, loss, learning rate)
    - Live inference previews
    - Validation loss tracking (micro-eval)
    - Throughput monitoring (tokens/sec, VRAM usage)
    - Pattern tracking (heatmaps)
    - Layer monitoring
    - Control signal handling (pause/stop)
    - Smart alerts (throughput drops, val gaps, etc.)
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
        layer_monitor: Optional[LayerMonitor] = None
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
        self.last_update_time = time.time()
        self.last_prompt_snapshot_time = time.time()
        self.update_interval = 2  # Status JSON refresh cadence (seconds)
        self.prompt_snapshot_interval = 20  # Update prompt/golden cache without inference

        # Keep inference lightweight: short outputs, frequent previews
        self.inference_interval_steps = max(10, (self.eval_steps or 200) // 10)
        self.last_inference_step = 0

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

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step."""
        current_time = time.time()
        current_loss = state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0
        current_lr = state.log_history[-1].get('learning_rate', args.learning_rate) if state.log_history else args.learning_rate
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

        # Lightweight micro-eval on tiny fixed set
        val_loss = self.last_val_loss
        if (
            self.micro_eval_inputs is not None
            and state.global_step > 0
            and state.global_step % self.micro_eval_interval == 0
            and state.global_step != self.last_micro_eval_step
        ):
            try:
                self.model_ref.eval()
                with torch.no_grad():
                    micro_inputs = {k: v.to(self.model_ref.device) for k, v in self.micro_eval_inputs.items()}
                    outputs = self.model_ref(**micro_inputs, labels=micro_inputs["input_ids"])
                    val_loss = outputs.loss.item()
                    self.last_val_loss = val_loss
                    self.last_micro_eval_step = state.global_step
            except Exception as e:
                print(f"Warning: Micro-eval failed at step {state.global_step}: {e}")
            finally:
                self.model_ref.train()

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
                val_loss=self.last_val_loss,
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
            )
            self.last_update_time = current_time

        # Inference updates (periodic) - show a live example with golden/model output
        if (
            self.raw_train_examples
            and state.global_step > 0
            and (state.global_step - self.last_inference_step) >= self.inference_interval_steps
            and state.global_step != self.last_inference_step
        ):
            try:
                print(f"[InferencePreview] step={state.global_step} interval={self.inference_interval_steps} last={self.last_inference_step}")
                # Get the raw example from ORIGINAL dataset (before tokenization)
                if self.raw_train_examples and len(self.raw_train_examples) > 0:
                    # Get example based on current step (with wraparound)
                    dataset_idx = state.global_step % len(self.raw_train_examples)
                    current_example = self.raw_train_examples[dataset_idx]

                    # Extract messages
                    if 'messages' in current_example:
                        # Extract from messages
                        messages = current_example['messages']
                        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
                        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), None)
                        golden_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), None)

                        if user_msg and golden_msg:
                            # Run inference on this exact example
                            self.model_ref.eval()
                            with torch.no_grad():
                                # Format prompt
                                prompt_messages = [{"role": "user", "content": user_msg}]
                                text = self.tokenizer.apply_chat_template(
                                    prompt_messages,
                                    tokenize=False,
                                    add_generation_prompt=True
                                )

                                # Generate
                                inputs = self.tokenizer(text, return_tensors="pt").to(self.model_ref.device)
                                reset_processor_states(self.logits_processor)
                                outputs = self.model_ref.generate(
                                    **inputs,
                                    max_new_tokens=2048,  # Full output for reasoning tasks
                                    temperature=0.1,
                                    do_sample=False,
                                    pad_token_id=self.tokenizer.eos_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    logits_processor=self.logits_processor,
                                    min_new_tokens=1
                                )

                                # Decode model output
                                model_output = self.tokenizer.decode(
                                    outputs[0][inputs['input_ids'].shape[1]:],
                                    skip_special_tokens=True
                                ).strip()

                            # Calculate loss on this specific example
                            example_loss = None
                            try:
                                # Tokenize the full conversation (user + golden assistant)
                                full_messages = [
                                    {"role": "user", "content": user_msg},
                                    {"role": "assistant", "content": golden_msg}
                                ]
                                full_text = self.tokenizer.apply_chat_template(
                                    full_messages,
                                    tokenize=False,
                                    add_generation_prompt=False
                                )

                                # Tokenize
                                full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model_ref.device)

                                # Get model logits
                                with torch.no_grad():
                                    outputs = self.model_ref(**full_inputs, labels=full_inputs['input_ids'])
                                    example_loss = outputs.loss.item()
                            except Exception as e:
                                print(f"Warning: Could not calculate loss for this example: {e}")

                            self.model_ref.train()

                            # Clear GPU cache after inference to prevent OOM
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Estimate token lengths for metadata/analysis
                            golden_token_len = None
                            model_token_len = None
                            try:
                                golden_token_len = len(self.tokenizer.encode(golden_msg, add_special_tokens=False))
                            except Exception:
                                pass
                            try:
                                model_token_len = len(self.tokenizer.encode(model_output, add_special_tokens=False))
                            except Exception:
                                pass

                            # Check if match
                            matches = golden_msg.strip() == model_output.strip()

                            # Update pattern tracker for heatmap coverage
                            pattern_matrix = None
                            pattern_id = None
                            bin_name = None
                            if self.pattern_tracker:
                                try:
                                    response_tokens = model_token_len if model_token_len is not None else 0
                                    pattern_id, bin_name = self.pattern_tracker.classify(
                                        user_msg or "",
                                        response_tokens
                                    )
                                    self.pattern_tracker.record(pattern_id, bin_name, matches)
                                    pattern_matrix = self.pattern_tracker.get_matrix()
                                except Exception as e:
                                    print(f"Warning: Pattern tracker update failed: {e}")

                            pattern_metadata = None
                            if pattern_id:
                                pattern_metadata = {
                                    "pattern_id": pattern_id,
                                    "length_bin": bin_name,
                                    "timestamp": datetime.now().isoformat(),
                                    "loss": example_loss,
                                }

                            layer_summary = None
                            if self.layer_monitor:
                                try:
                                    layer_summary = self.layer_monitor.snapshot()
                                except Exception as e:
                                    print(f"Warning: Layer monitor snapshot failed: {e}")

                            # Display in terminal
                            print("\n" + "=" * 80)
                            print(f"🔍 CURRENT TRAINING EXAMPLE - Step {state.global_step:,}")
                            print("=" * 80)
                            print(f"📝 PROMPT:\n{user_msg[:500]}...")
                            print(f"\n✅ GOLDEN:\n{golden_msg[:200]}...")
                            print(f"\n🤖 MODEL:\n{model_output[:200]}...")
                            status = "✅ MATCH" if matches else "❌ NO MATCH"
                            print(f"\n{status}")
                            if example_loss is not None:
                                print(f"📉 LOSS ON THIS EXAMPLE: {example_loss:.4f}")
                            print("=" * 80 + "\n")

                            # Calculate batch-relative step
                            batch_step = state.global_step - self.current_global_step

                            # Update status JSON (use example-specific loss if available)
                            display_loss = example_loss if example_loss is not None else current_loss
                            self.status_writer.update_inference(
                                step=state.global_step,
                                total_steps=self.total_steps,
                                epoch=int(current_epoch),
                                loss=display_loss,
                                lr=current_lr,
                                prompt=user_msg,
                                golden=golden_msg,
                                model_output=model_output,
                                matches=matches,
                                system_prompt=system_msg,
                                batch_step=batch_step,
                                batch_total_steps=self.batch_total_steps,
                                batch_number=self.batch_number,
                                batch_queue_size=self.batch_queue_size,
                                current_file=self.current_file,
                                golden_output_length=golden_token_len,
                                model_output_length=model_token_len,
                                pattern_heatmap=pattern_matrix,
                                layer_activity_summary=layer_summary,
                                pattern_metadata=pattern_metadata
                            )
                            self.last_update_time = current_time
                            self.last_inference_step = state.global_step
            except Exception as e:
                print(f"Warning: Could not display current training example at step {state.global_step}: {e}")
                import traceback
                traceback.print_exc()

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


__all__ = ["LiveMonitorCallback"]
