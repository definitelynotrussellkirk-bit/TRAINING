#!/usr/bin/env python3
"""
Centralized Alerts Daemon

Polls monitoring APIs and sends notifications when issues are detected:
- Training stuck (no progress for 15 minutes)
- Regression detected
- System status critical
- Queue health poor (too many pending, empty for too long)

Notification channels:
- Console logging (always)
- Discord webhook (if DISCORD_WEBHOOK_URL set)
- Slack webhook (if SLACK_WEBHOOK_URL set)

Usage:
    python3 monitoring/alerts_daemon.py

    # With Discord notifications
    DISCORD_WEBHOOK_URL=https://... python3 monitoring/alerts_daemon.py

Configuration:
    Environment variables:
        DISCORD_WEBHOOK_URL - Discord webhook URL for notifications
        SLACK_WEBHOOK_URL - Slack webhook URL for notifications
        ALERTS_CHECK_INTERVAL - Check interval in seconds (default: 60)
        STUCK_TIMEOUT_MINUTES - Minutes without progress = stuck (default: 15)
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.paths import get_base_dir
    BASE_DIR = get_base_dir()
except ImportError:
    BASE_DIR = Path(__file__).parent.parent

# Configuration from environment
UNIFIED_API_URL = os.getenv("UNIFIED_API_URL", "http://localhost:8081/api/unified")
JOBS_API_URL = os.getenv("JOBS_API_URL", "http://localhost:8081/api/jobs")
QUEUE_API_URL = os.getenv("QUEUE_API_URL", "http://localhost:8081/api/queue")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
CHECK_INTERVAL = int(os.getenv("ALERTS_CHECK_INTERVAL", "60"))
STUCK_TIMEOUT_MINUTES = int(os.getenv("STUCK_TIMEOUT_MINUTES", "15"))

# Logging
LOG_FILE = BASE_DIR / "logs" / "alerts_daemon.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents an alert to be sent."""
    type: str           # stuck, regression, critical, queue_empty, etc.
    level: str          # info, warning, critical
    message: str        # Human-readable message
    details: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AlertState:
    """Track alert state to avoid duplicate notifications."""
    last_step: Optional[int] = None
    last_step_time: Optional[datetime] = None
    last_alerts: Dict[str, datetime] = field(default_factory=dict)
    alert_cooldown_minutes: int = 30  # Don't repeat same alert within this window

    def should_alert(self, alert_type: str) -> bool:
        """Check if we should send this alert (not in cooldown)."""
        if alert_type not in self.last_alerts:
            return True
        elapsed = (datetime.now() - self.last_alerts[alert_type]).total_seconds() / 60
        return elapsed > self.alert_cooldown_minutes

    def record_alert(self, alert_type: str):
        """Record that we sent an alert."""
        self.last_alerts[alert_type] = datetime.now()

    def update_step(self, step: int):
        """Update progress tracking."""
        if step != self.last_step:
            self.last_step = step
            self.last_step_time = datetime.now()

    def is_stuck(self) -> bool:
        """Check if training appears stuck."""
        if self.last_step_time is None:
            return False
        elapsed = (datetime.now() - self.last_step_time).total_seconds() / 60
        return elapsed > STUCK_TIMEOUT_MINUTES


class AlertsDaemon:
    """
    Centralized alerts daemon that polls monitoring APIs and sends notifications.
    """

    def __init__(self):
        self.state = AlertState()
        self.running = True

    def send_alert(self, alert: Alert):
        """Send an alert through configured channels."""
        # Always log to console/file
        log_method = {
            "info": logger.info,
            "warning": logger.warning,
            "critical": logger.error
        }.get(alert.level, logger.info)

        log_method(f"[{alert.type.upper()}] {alert.message}")

        # Send to Discord if configured
        if DISCORD_WEBHOOK_URL:
            self._send_discord(alert)

        # Send to Slack if configured
        if SLACK_WEBHOOK_URL:
            self._send_slack(alert)

        # Record that we sent this alert
        self.state.record_alert(alert.type)

        # Also write to alerts log file
        self._write_alert_log(alert)

    def _send_discord(self, alert: Alert):
        """Send alert to Discord webhook."""
        try:
            color = {
                "info": 0x3498db,      # Blue
                "warning": 0xf39c12,   # Orange
                "critical": 0xe74c3c   # Red
            }.get(alert.level, 0x95a5a6)

            embed = {
                "title": f"{alert.type.upper()}",
                "description": alert.message,
                "color": color,
                "timestamp": alert.timestamp,
                "fields": [
                    {"name": k, "value": str(v)[:1024], "inline": True}
                    for k, v in alert.details.items()
                ][:10]  # Max 10 fields
            }

            requests.post(
                DISCORD_WEBHOOK_URL,
                json={"embeds": [embed]},
                timeout=10
            )
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    def _send_slack(self, alert: Alert):
        """Send alert to Slack webhook."""
        try:
            emoji = {
                "info": ":information_source:",
                "warning": ":warning:",
                "critical": ":rotating_light:"
            }.get(alert.level, ":bell:")

            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{emoji} *{alert.type.upper()}*\n{alert.message}"
                    }
                }
            ]

            if alert.details:
                fields = [
                    {
                        "type": "mrkdwn",
                        "text": f"*{k}:* {v}"
                    }
                    for k, v in list(alert.details.items())[:8]
                ]
                blocks.append({
                    "type": "section",
                    "fields": fields
                })

            requests.post(
                SLACK_WEBHOOK_URL,
                json={"blocks": blocks},
                timeout=10
            )
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _write_alert_log(self, alert: Alert):
        """Write alert to JSONL log file."""
        try:
            alerts_log = BASE_DIR / "logs" / "alerts_history.jsonl"
            record = {
                "type": alert.type,
                "level": alert.level,
                "message": alert.message,
                "details": alert.details,
                "timestamp": alert.timestamp
            }
            with open(alerts_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def check_unified_api(self) -> Optional[Dict]:
        """Fetch unified API data."""
        try:
            resp = requests.get(UNIFIED_API_URL, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"Failed to fetch unified API: {e}")
        return None

    def check_jobs_api(self) -> Optional[Dict]:
        """Fetch jobs API data."""
        try:
            resp = requests.get(JOBS_API_URL, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"Failed to fetch jobs API: {e}")
        return None

    def check_queue_api(self) -> Optional[Dict]:
        """Fetch queue API data."""
        try:
            resp = requests.get(QUEUE_API_URL, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"Failed to fetch queue API: {e}")
        return None

    def process_unified_data(self, data: Dict) -> List[Alert]:
        """Process unified API data and generate alerts."""
        alerts = []
        sources = data.get("sources", {})
        summary = data.get("summary", {})

        # 1. Check system status
        system_status = summary.get("system_status")
        if system_status == "critical":
            if self.state.should_alert("system_critical"):
                alerts.append(Alert(
                    type="system_critical",
                    level="critical",
                    message="System status is CRITICAL",
                    details={"status": system_status}
                ))

        # 2. Check training stuck
        training_status = sources.get("training_status", {}).get("data", {})
        current_step = training_status.get("current_step")
        if current_step is not None:
            self.state.update_step(current_step)
            if self.state.is_stuck():
                if self.state.should_alert("training_stuck"):
                    alerts.append(Alert(
                        type="training_stuck",
                        level="critical",
                        message=f"Training appears stuck at step {current_step} for >{STUCK_TIMEOUT_MINUTES}min",
                        details={
                            "step": current_step,
                            "last_progress": self.state.last_step_time.isoformat() if self.state.last_step_time else None,
                            "current_file": training_status.get("current_file")
                        }
                    ))

        # 3. Check for active alerts from training status
        active_alerts = training_status.get("active_alerts", [])
        for alert_info in active_alerts:
            alert_type = alert_info.get("type", "unknown")
            if alert_info.get("severity") == "critical":
                if self.state.should_alert(f"training_{alert_type}"):
                    alerts.append(Alert(
                        type=f"training_{alert_type}",
                        level="critical",
                        message=alert_info.get("message", f"Training alert: {alert_type}"),
                        details=alert_info
                    ))

        # 4. Check regression
        regression = sources.get("regression_monitor", {}).get("data", {})
        if regression.get("regression_detected"):
            if self.state.should_alert("regression"):
                alerts.append(Alert(
                    type="regression",
                    level="critical",
                    message=f"Regression detected: loss increased by {regression.get('increase_percent', '?')}%",
                    details={
                        "checkpoint": regression.get("checkpoint"),
                        "increase_percent": regression.get("increase_percent"),
                        "baseline_loss": regression.get("baseline_loss"),
                        "current_loss": regression.get("current_loss")
                    }
                ))

        # 5. Check val/train gap
        val_train_gap = training_status.get("val_train_gap")
        if val_train_gap is not None and val_train_gap > 0.3:
            if self.state.should_alert("overfitting"):
                alerts.append(Alert(
                    type="overfitting",
                    level="warning",
                    message=f"Possible overfitting: val/train gap is {val_train_gap:.3f}",
                    details={
                        "val_train_gap": val_train_gap,
                        "train_loss": training_status.get("loss"),
                        "val_loss": training_status.get("validation_loss")
                    }
                ))

        return alerts

    def process_queue_data(self, data: Dict) -> List[Alert]:
        """Process queue API data and generate alerts."""
        alerts = []

        # Check queue health
        queue_healthy = data.get("queue_healthy", True)
        total_files = data.get("total_files", 0)

        if not queue_healthy and total_files == 0:
            if self.state.should_alert("queue_empty"):
                alerts.append(Alert(
                    type="queue_empty",
                    level="warning",
                    message="Training queue is empty",
                    details={
                        "total_files": total_files,
                        "inbox_count": data.get("inbox_count", 0)
                    }
                ))

        return alerts

    def process_jobs_data(self, data: Dict) -> List[Alert]:
        """Process jobs API data and generate alerts."""
        alerts = []

        # Check for recent failures
        jobs = data.get("jobs", [])
        recent_failures = [
            j for j in jobs
            if j.get("status") == "failed"
            and j.get("finished_at")
        ]

        # Alert on new failures (finished in last hour)
        for job in recent_failures[:3]:  # Max 3 alerts
            finished_at = job.get("finished_at")
            if finished_at:
                try:
                    finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                    if (datetime.now() - finished.replace(tzinfo=None)).total_seconds() < 3600:
                        alert_key = f"job_failed_{job.get('job_id')}"
                        if self.state.should_alert(alert_key):
                            alerts.append(Alert(
                                type="job_failed",
                                level="warning",
                                message=f"Job failed: {job.get('file_name')}",
                                details={
                                    "job_id": job.get("job_id"),
                                    "file_name": job.get("file_name"),
                                    "reason": job.get("reason"),
                                    "final_step": job.get("final_step")
                                }
                            ))
                except Exception:
                    pass

        return alerts

    def run_check(self):
        """Run one monitoring check cycle."""
        all_alerts = []

        # Check unified API
        unified_data = self.check_unified_api()
        if unified_data:
            all_alerts.extend(self.process_unified_data(unified_data))
        else:
            # Can't reach API - that's a problem
            if self.state.should_alert("api_unreachable"):
                all_alerts.append(Alert(
                    type="api_unreachable",
                    level="warning",
                    message="Cannot reach monitoring API",
                    details={"url": UNIFIED_API_URL}
                ))

        # Check queue API
        queue_data = self.check_queue_api()
        if queue_data:
            all_alerts.extend(self.process_queue_data(queue_data))

        # Check jobs API
        jobs_data = self.check_jobs_api()
        if jobs_data:
            all_alerts.extend(self.process_jobs_data(jobs_data))

        # Send all alerts
        for alert in all_alerts:
            self.send_alert(alert)

    def run(self):
        """Main daemon loop."""
        logger.info("=" * 60)
        logger.info("Alerts Daemon Starting")
        logger.info(f"Check interval: {CHECK_INTERVAL}s")
        logger.info(f"Stuck timeout: {STUCK_TIMEOUT_MINUTES}min")
        logger.info(f"Discord webhook: {'Configured' if DISCORD_WEBHOOK_URL else 'Not configured'}")
        logger.info(f"Slack webhook: {'Configured' if SLACK_WEBHOOK_URL else 'Not configured'}")
        logger.info("=" * 60)

        while self.running:
            try:
                self.run_check()
            except Exception as e:
                logger.error(f"Error in check cycle: {e}")

            time.sleep(CHECK_INTERVAL)


def main():
    daemon = AlertsDaemon()

    # Handle graceful shutdown
    import signal
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        daemon.running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    daemon.run()
    logger.info("Alerts daemon stopped")


if __name__ == "__main__":
    main()
