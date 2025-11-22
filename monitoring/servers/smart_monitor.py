#!/usr/bin/env python3
"""
Smart Training Monitor - Automated Anomaly Detection & Snapshot System

Watches training in real-time and automatically saves checkpoints when:
- Loss spikes unexpectedly
- Accuracy drops significantly
- New best model achieved (lowest loss)
- Training appears to be diverging
- Other configurable triggers

Snapshots saved to: snapshots/anomaly_YYYYMMDD_HHMMSS_reason/
"""

import json
import time
import shutil
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any
import argparse
import subprocess

class SmartMonitor:
    def __init__(self, base_dir: str, config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.status_file = self.base_dir / "status" / "training_status.json"
        self.model_dir = self.base_dir / "current_model"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.log_file = self.base_dir / "logs" / f"smart_monitor_{datetime.now().strftime('%Y%m%d')}.log"

        # Configuration
        self.config = config
        self.poll_interval = config.get('poll_interval', 10)  # seconds

        # Thresholds
        self.loss_spike_threshold = config.get('loss_spike_threshold', 0.3)  # 30% increase
        self.loss_window = config.get('loss_window', 10)  # Compare last N values
        self.accuracy_drop_threshold = config.get('accuracy_drop_threshold', 10)  # 10% drop
        self.min_steps_between_saves = config.get('min_steps_between_saves', 500)

        # State tracking
        self.loss_history = deque(maxlen=self.loss_window)
        self.accuracy_history = deque(maxlen=self.loss_window)
        self.learning_rate_history = deque(maxlen=self.loss_window)
        self.best_loss = float('inf')
        self.best_loss_step = 0
        self.last_save_step = 0
        self.last_step = 0

        # Advanced tracking for anomaly detection
        self.recent_examples = deque(maxlen=50)  # Track loss/match pairs

        # Statistics
        self.snapshots_created = 0
        self.anomalies_detected = 0

        # Desktop notifications
        self.notifications_enabled = config.get('notifications', True)

        self.log(f"Smart Monitor initialized with config: {config}")

    def log(self, message: str, level: str = "INFO"):
        """Log message to file and optionally console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"

        # Console
        if level in ["WARNING", "ERROR", "ALERT"]:
            print(log_line)

        # File
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")

    def send_notification(self, title: str, message: str, urgency: str = "normal"):
        """Send desktop notification"""
        if not self.notifications_enabled:
            return

        try:
            # Use notify-send if available (Linux)
            subprocess.run([
                'notify-send',
                '-u', urgency,
                '-i', 'dialog-information',
                f'Training Monitor: {title}',
                message
            ], check=False, capture_output=True)
        except:
            pass  # Silently fail if notifications not available

    def read_status(self) -> Optional[Dict]:
        """Read current training status"""
        try:
            if not self.status_file.exists():
                return None
            with open(self.status_file) as f:
                return json.load(f)
        except Exception as e:
            self.log(f"Error reading status: {e}", "ERROR")
            return None

    def create_snapshot(self, reason: str, metadata: Dict[str, Any]):
        """Create anomaly snapshot with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"anomaly_{timestamp}_{reason.replace(' ', '_')}"
        snapshot_path = self.snapshots_dir / snapshot_name

        try:
            # Create snapshot directory
            snapshot_path.mkdir(parents=True, exist_ok=True)

            # Copy model
            if self.model_dir.exists():
                shutil.copytree(
                    self.model_dir,
                    snapshot_path / "model",
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('checkpoint-*')  # Don't copy all checkpoints
                )

                # Just copy the latest checkpoint
                checkpoints = sorted(self.model_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[1]))
                if checkpoints:
                    latest = checkpoints[-1]
                    shutil.copytree(latest, snapshot_path / "model" / latest.name)

            # Save metadata
            metadata.update({
                'timestamp': timestamp,
                'reason': reason,
                'snapshot_name': snapshot_name,
                'monitor_config': self.config
            })

            with open(snapshot_path / "metadata.json", 'w') as f:
                json.dump(metadata, indent=2, fp=f)

            # Save current status
            status = self.read_status()
            if status:
                with open(snapshot_path / "training_status.json", 'w') as f:
                    json.dump(status, indent=2, fp=f)

            self.snapshots_created += 1
            self.log(f"✓ Created snapshot: {snapshot_name}", "ALERT")

            # Send desktop notification
            self.notify_snapshot(reason, metadata)

            return snapshot_path

        except Exception as e:
            self.log(f"Error creating snapshot: {e}", "ERROR")
            return None

    def notify_snapshot(self, reason: str, metadata: Dict):
        """Send notification about snapshot creation"""
        step = metadata.get('step', 0)
        loss = metadata.get('loss', 0)
        triggers = metadata.get('triggers', [])

        # Determine notification urgency and message
        if 'divergence' in reason or 'inverted' in reason:
            urgency = "critical"
            title = "🚨 CRITICAL: Training Issue!"
            message = f"Step {step:,}: {', '.join(triggers[:2])}\nLoss: {loss:.4f}"
            self.send_notification(title, message, urgency)

        elif 'best_model' in reason:
            urgency = "normal"
            title = "🏆 New Best Model!"
            message = f"Step {step:,}: Loss {loss:.4f}\nSnapshot saved"
            self.send_notification(title, message, urgency)

        elif 'zscore' in reason or 'spike' in reason:
            urgency = "critical"
            title = "⚠️ Anomaly Detected!"
            message = f"Step {step:,}: {triggers[0]}\nLoss: {loss:.4f}"
            self.send_notification(title, message, urgency)

        elif 'accuracy_drop' in reason:
            urgency = "normal"
            title = "📉 Accuracy Drop"
            message = f"Step {step:,}: Accuracy decreased\nReview recommended"
            self.send_notification(title, message, urgency)

        else:
            # Generic anomaly
            urgency = "normal"
            title = "🔍 Anomaly Detected"
            message = f"Step {step:,}: {reason[:50]}"
            self.send_notification(title, message, urgency)

    def check_loss_spike(self, current_loss: float, step: int) -> Optional[str]:
        """Detect sudden loss increases"""
        if len(self.loss_history) < 3:
            return None

        recent_avg = sum(list(self.loss_history)[-5:]) / min(5, len(self.loss_history))

        # Check for spike
        if current_loss > recent_avg * (1 + self.loss_spike_threshold):
            spike_pct = ((current_loss / recent_avg) - 1) * 100
            return f"loss_spike_{spike_pct:.1f}pct"

        return None

    def check_accuracy_drop(self, current_acc: float, step: int) -> Optional[str]:
        """Detect sudden accuracy drops"""
        if len(self.accuracy_history) < 3:
            return None

        recent_avg = sum(list(self.accuracy_history)[-5:]) / min(5, len(self.accuracy_history))

        # Check for drop
        if current_acc < recent_avg - self.accuracy_drop_threshold:
            drop = recent_avg - current_acc
            return f"accuracy_drop_{drop:.1f}pct"

        return None

    def check_best_model(self, current_loss: float, step: int) -> Optional[str]:
        """Check if this is the best model so far"""
        if current_loss < self.best_loss:
            # New best!
            improvement = ((self.best_loss - current_loss) / self.best_loss * 100) if self.best_loss != float('inf') else 0
            self.best_loss = current_loss
            self.best_loss_step = step

            # Only save if significant improvement or first time
            if improvement > 2 or self.best_loss == current_loss:
                return f"best_model_loss_{current_loss:.4f}"

        return None

    def check_divergence(self, current_loss: float, step: int) -> Optional[str]:
        """Detect if training is diverging (loss increasing steadily)"""
        if len(self.loss_history) < self.loss_window:
            return None

        # Check if loss has been increasing for entire window
        losses = list(self.loss_history)
        increasing_count = sum(1 for i in range(len(losses)-1) if losses[i+1] > losses[i])

        if increasing_count >= len(losses) * 0.8:  # 80% of values increasing
            return f"divergence_detected"

        return None

    def calculate_z_score(self, value: float, history: deque) -> float:
        """Calculate z-score for a value given historical data"""
        if len(history) < 3:
            return 0.0

        values = list(history)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return 0.0

        return (value - mean) / std_dev

    def check_statistical_anomaly(self, current_loss: float, current_lr: float, step: int) -> Optional[str]:
        """Detect statistical anomalies using z-scores"""
        anomalies = []

        # Loss z-score (detect unusual loss values)
        if len(self.loss_history) >= 5:
            loss_z = self.calculate_z_score(current_loss, self.loss_history)
            if abs(loss_z) > 3.0:  # 3 standard deviations
                anomalies.append(f"loss_zscore_{loss_z:.2f}")

        # Learning rate z-score (detect LR schedule anomalies)
        if len(self.learning_rate_history) >= 5 and current_lr > 0:
            lr_z = self.calculate_z_score(current_lr, self.learning_rate_history)
            if abs(lr_z) > 3.0:
                anomalies.append(f"lr_zscore_{lr_z:.2f}")

        if anomalies:
            return "_".join(anomalies)
        return None

    def check_prediction_anomaly(self, status: Dict) -> Optional[str]:
        """Detect mismatched loss/accuracy patterns"""
        # Get current example if available
        if not status.get('current_prompt'):
            return None

        current_loss = status.get('loss', 0)
        matches = status.get('answer_matches', None)

        if matches is None or current_loss == 0:
            return None

        # Track this example
        self.recent_examples.append({
            'loss': current_loss,
            'matches': matches,
            'step': status.get('current_step', 0)
        })

        # Anomaly 1: Perfect answer but HIGH loss (>1.5)
        if matches and current_loss > 1.5:
            return f"perfect_answer_high_loss_{current_loss:.2f}"

        # Anomaly 2: Wrong answer but LOW loss (<0.3)
        if not matches and current_loss < 0.3:
            return f"wrong_answer_low_loss_{current_loss:.2f}"

        # Anomaly 3: Statistical mismatch
        # If we have enough examples, check if this is unusual
        if len(self.recent_examples) >= 10:
            correct_losses = [ex['loss'] for ex in self.recent_examples if ex['matches']]
            incorrect_losses = [ex['loss'] for ex in self.recent_examples if not ex['matches']]

            if correct_losses and incorrect_losses:
                avg_correct = sum(correct_losses) / len(correct_losses)
                avg_incorrect = sum(incorrect_losses) / len(incorrect_losses)

                # Inverted pattern: correct answers have HIGHER loss than incorrect
                if matches and current_loss > avg_incorrect * 1.5:
                    return f"inverted_loss_pattern_correct_high"
                if not matches and current_loss < avg_correct * 0.5:
                    return f"inverted_loss_pattern_incorrect_low"

        return None

    def should_save(self, step: int) -> bool:
        """Check if enough steps passed since last save"""
        return (step - self.last_save_step) >= self.min_steps_between_saves

    def monitor_loop(self):
        """Main monitoring loop"""
        self.log("Starting smart monitoring...")
        print("🔍 Smart Monitor Active - Watching for anomalies...")
        print(f"Thresholds: Loss spike >{self.loss_spike_threshold*100}%, Accuracy drop >{self.accuracy_drop_threshold}%")
        print(f"Saving best models and anomalies to: {self.snapshots_dir}/anomaly_*")
        print("Press Ctrl+C to stop\n")

        consecutive_errors = 0
        max_errors = 10

        while True:
            try:
                status = self.read_status()

                if not status:
                    time.sleep(self.poll_interval)
                    continue

                # Reset error count on successful read
                consecutive_errors = 0

                # Extract metrics
                current_step = status.get('current_step', 0)
                current_loss = status.get('loss', 0)
                current_accuracy = status.get('accuracy_percent', 0)
                current_lr = status.get('learning_rate', 0)
                training_status = status.get('status', 'unknown')

                # Skip if not training
                if training_status != 'training':
                    time.sleep(self.poll_interval)
                    continue

                # Skip if same step as before
                if current_step == self.last_step:
                    time.sleep(self.poll_interval)
                    continue

                self.last_step = current_step

                # Update history
                if current_loss > 0:
                    self.loss_history.append(current_loss)
                if current_accuracy > 0:
                    self.accuracy_history.append(current_accuracy)
                if current_lr > 0:
                    self.learning_rate_history.append(current_lr)

                # Run checks
                triggers = []

                # Loss spike
                if spike_reason := self.check_loss_spike(current_loss, current_step):
                    triggers.append(spike_reason)

                # Accuracy drop
                if current_accuracy > 0:
                    if drop_reason := self.check_accuracy_drop(current_accuracy, current_step):
                        triggers.append(drop_reason)

                # Best model
                if best_reason := self.check_best_model(current_loss, current_step):
                    triggers.append(best_reason)

                # Divergence
                if div_reason := self.check_divergence(current_loss, current_step):
                    triggers.append(div_reason)

                # Statistical anomalies (z-scores)
                if stat_reason := self.check_statistical_anomaly(current_loss, current_lr, current_step):
                    triggers.append(stat_reason)

                # Prediction anomalies (loss/accuracy mismatch)
                if pred_reason := self.check_prediction_anomaly(status):
                    triggers.append(pred_reason)

                # Save if triggered and enough time passed
                if triggers and self.should_save(current_step):
                    reason = "_".join(triggers)

                    metadata = {
                        'step': current_step,
                        'loss': current_loss,
                        'accuracy': current_accuracy,
                        'learning_rate': current_lr,
                        'triggers': triggers,
                        'loss_history': list(self.loss_history)[-10:],
                        'accuracy_history': list(self.accuracy_history)[-10:] if self.accuracy_history else [],
                        'lr_history': list(self.learning_rate_history)[-10:] if self.learning_rate_history else [],
                        'recent_examples': list(self.recent_examples)[-10:] if self.recent_examples else [],
                        'best_loss': self.best_loss,
                        'best_loss_step': self.best_loss_step,
                        'z_scores': {
                            'loss': self.calculate_z_score(current_loss, self.loss_history) if len(self.loss_history) >= 3 else 0,
                            'lr': self.calculate_z_score(current_lr, self.learning_rate_history) if len(self.learning_rate_history) >= 3 and current_lr > 0 else 0
                        }
                    }

                    if self.create_snapshot(reason, metadata):
                        self.last_save_step = current_step
                        self.anomalies_detected += 1

                # Periodic status
                if current_step % 500 == 0:
                    self.log(f"Monitoring: Step {current_step}, Loss {current_loss:.4f}, "
                           f"Best {self.best_loss:.4f} @ {self.best_loss_step}, "
                           f"Snapshots {self.snapshots_created}")

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                self.log("Monitoring stopped by user")
                print(f"\n✓ Stopped. Created {self.snapshots_created} snapshots, detected {self.anomalies_detected} anomalies")
                break
            except Exception as e:
                consecutive_errors += 1
                self.log(f"Error in monitoring loop: {e}", "ERROR")

                if consecutive_errors >= max_errors:
                    self.log(f"Too many consecutive errors ({max_errors}), stopping", "ERROR")
                    break

                time.sleep(self.poll_interval * 2)

def main():
    parser = argparse.ArgumentParser(description="Smart training monitor with anomaly detection")
    parser.add_argument('--base-dir', default='/path/to/training', help='Base training directory')
    parser.add_argument('--loss-spike-threshold', type=float, default=0.3, help='Loss spike threshold (0.3 = 30%)')
    parser.add_argument('--accuracy-drop-threshold', type=float, default=10, help='Accuracy drop threshold (10 = 10%%)')
    parser.add_argument('--poll-interval', type=int, default=10, help='Polling interval in seconds')
    parser.add_argument('--min-steps-between-saves', type=int, default=500, help='Minimum steps between saves')
    parser.add_argument('--loss-window', type=int, default=10, help='Window size for loss averaging')
    parser.add_argument('--no-notifications', action='store_true', help='Disable desktop notifications')

    args = parser.parse_args()

    config = {
        'loss_spike_threshold': args.loss_spike_threshold,
        'accuracy_drop_threshold': args.accuracy_drop_threshold,
        'poll_interval': args.poll_interval,
        'min_steps_between_saves': args.min_steps_between_saves,
        'loss_window': args.loss_window,
        'notifications': not args.no_notifications
    }

    monitor = SmartMonitor(args.base_dir, config)
    monitor.monitor_loop()

if __name__ == '__main__':
    main()
