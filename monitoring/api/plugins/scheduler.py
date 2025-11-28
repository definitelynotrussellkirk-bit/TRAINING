#!/usr/bin/env python3
"""
GPU Task Scheduler Plugin

Fetches GPU scheduler metrics from the 3090 scheduler API (port 8766).
Surfaces queue status, active tasks, and utilization band warnings.
"""

from typing import Dict, Any, Optional
import logging
import requests
from datetime import datetime
import sys
from pathlib import Path

# Add core to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "core"))
from hosts import get_host

from .base import BasePlugin, PluginError

logger = logging.getLogger(__name__)


class SchedulerPlugin(BasePlugin):
    """
    Plugin for GPU Task Scheduler metrics.

    Connects to the scheduler API on the 3090 to fetch:
    - Queue length and breakdown by priority
    - Active task type and status
    - GPU utilization (target: 20-80%)
    - Task completion stats

    Data source: http://192.168.x.x:8766/api/metrics, /api/health
    Refresh: Every 30 seconds
    Critical: Yes (scheduler down = no GPU task execution)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.scheduler_host = self.config.get('scheduler_host', get_host('3090').host)
        self.scheduler_port = self.config.get('scheduler_port', 8766)
        self.base_url = f"http://{self.scheduler_host}:{self.scheduler_port}"
        self.timeout = self.config.get('timeout', 5)

    def get_name(self) -> str:
        return 'scheduler'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': 'GPU Task Scheduler status and queue metrics',
            'refresh_interval': 30,
            'critical': True,
            'data_type': 'scheduler',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'status', 'queue_length', 'active_task', 'gpu_utilization',
                'in_target_band', 'tasks_completed', 'queue_by_priority', 'warnings'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch scheduler metrics from the 3090 API."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'available': False,
            'status': 'unknown',
            'queue_length': 0,
            'active_task': None,
            'gpu_utilization': 0,
            'gpu_memory_pct': 0,
            'in_target_band': False,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'uptime_sec': 0,
            'queue_by_priority': {
                'critical': 0,
                'high': 0,
                'normal': 0,
                'low': 0,
                'idle': 0
            },
            'warnings': [],
            'health': 'unknown'
        }

        try:
            # Fetch health status
            health_resp = requests.get(
                f"{self.base_url}/api/health",
                timeout=self.timeout
            )
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                result['status'] = health_data.get('scheduler', 'unknown')
                result['available'] = True

            # Fetch metrics
            metrics_resp = requests.get(
                f"{self.base_url}/api/metrics",
                timeout=self.timeout
            )
            if metrics_resp.status_code == 200:
                metrics = metrics_resp.json()
                result['queue_length'] = metrics.get('queue_length', 0)
                result['active_task'] = metrics.get('active_task')
                result['gpu_utilization'] = metrics.get('gpu_utilization_pct', 0)
                result['gpu_memory_pct'] = metrics.get('gpu_memory_pct', 0)
                result['tasks_completed'] = metrics.get('tasks_completed', 0)
                result['tasks_failed'] = metrics.get('tasks_failed', 0)
                result['uptime_sec'] = metrics.get('uptime_sec', 0)
                result['idle_tasks_dispatched'] = metrics.get('idle_tasks_dispatched', 0)

                # Check if in target utilization band (20-80%)
                util = result['gpu_utilization']
                result['in_target_band'] = 20 <= util <= 80

            # Assess health and generate warnings
            result['health'], result['warnings'] = self._assess_health(result)

        except requests.exceptions.Timeout:
            result['warnings'].append('Scheduler API timeout')
            result['health'] = 'critical'
        except requests.exceptions.ConnectionError:
            result['warnings'].append('Cannot connect to scheduler')
            result['health'] = 'critical'
        except Exception as e:
            logger.warning(f"Scheduler plugin error: {e}")
            result['warnings'].append(f'Error: {str(e)[:50]}')
            result['health'] = 'warning'

        return result

    def _assess_health(self, data: Dict) -> tuple:
        """Assess scheduler health and generate warnings."""
        warnings = []
        health = 'good'

        if not data.get('available'):
            return 'critical', ['Scheduler not responding']

        if data['status'] != 'running':
            warnings.append(f"Scheduler status: {data['status']}")
            health = 'critical'

        # Check utilization band
        util = data.get('gpu_utilization', 0)
        queue_len = data.get('queue_length', 0)

        if util < 20 and queue_len > 0:
            warnings.append(f'GPU underutilized ({util:.0f}%) with {queue_len} tasks queued')
            health = 'warning'
        elif util > 90:
            warnings.append(f'GPU saturated ({util:.0f}%)')
            health = 'warning'

        # Check for failed tasks
        if data.get('tasks_failed', 0) > 0:
            warnings.append(f"{data['tasks_failed']} failed tasks")
            if health == 'good':
                health = 'warning'

        return health, warnings
