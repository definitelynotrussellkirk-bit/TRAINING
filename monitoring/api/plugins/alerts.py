#!/usr/bin/env python3
"""
Alerts Plugin - Provides alerts data for unified API

Reads from logs/alerts_history.jsonl to provide:
- Recent alerts
- Alert counts by type/level
- Active alerts summary
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .base import BasePlugin, PluginError


class AlertsPlugin(BasePlugin):
    """
    Plugin that provides alerts data.

    Reads from alerts_history.jsonl and provides:
    - Recent alerts (last 24h)
    - Counts by type and level
    - Active alerts summary
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.cache_duration = config.get('cache_duration', 30)  # 30 sec cache

    def get_name(self) -> str:
        return "alerts"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "description": "System alerts and notifications",
            "machine": "4090",
            "location": "logs/alerts_history.jsonl",
            "refresh_interval": 30,
            "critical": False,
            "data_type": "file"
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch alerts data."""
        try:
            from core.paths import get_base_dir
            alerts_file = get_base_dir() / "logs" / "alerts_history.jsonl"
        except ImportError:
            alerts_file = Path(__file__).parent.parent.parent.parent / "logs" / "alerts_history.jsonl"

        if not alerts_file.exists():
            return {
                "recent_alerts": [],
                "total_alerts_24h": 0,
                "by_type": {},
                "by_level": {},
                "active_issues": []
            }

        # Read recent alerts (last 24h)
        alerts: List[Dict] = []
        cutoff = datetime.now() - timedelta(hours=24)

        try:
            with open(alerts_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            # Check if within 24h
                            ts = record.get("timestamp")
                            if ts:
                                try:
                                    alert_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                    if alert_time.replace(tzinfo=None) > cutoff:
                                        alerts.append(record)
                                except Exception:
                                    alerts.append(record)  # Include if can't parse time
                            else:
                                alerts.append(record)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise PluginError(f"Failed to read alerts history: {e}")

        # Compute statistics
        by_type = {}
        by_level = {}

        for alert in alerts:
            alert_type = alert.get("type", "unknown")
            level = alert.get("level", "info")

            by_type[alert_type] = by_type.get(alert_type, 0) + 1
            by_level[level] = by_level.get(level, 0) + 1

        # Most recent alerts first
        alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)

        # Identify active issues (critical alerts in last hour)
        active_issues = []
        one_hour_ago = datetime.now() - timedelta(hours=1)

        for alert in alerts[:20]:  # Check recent 20
            if alert.get("level") == "critical":
                ts = alert.get("timestamp")
                if ts:
                    try:
                        alert_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if alert_time.replace(tzinfo=None) > one_hour_ago:
                            active_issues.append({
                                "type": alert.get("type"),
                                "message": alert.get("message"),
                                "timestamp": ts
                            })
                    except Exception:
                        pass

        return {
            "recent_alerts": alerts[:20],
            "total_alerts_24h": len(alerts),
            "by_type": by_type,
            "by_level": by_level,
            "active_issues": active_issues,
            "critical_count": by_level.get("critical", 0),
            "warning_count": by_level.get("warning", 0)
        }
