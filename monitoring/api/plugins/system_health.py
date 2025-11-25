#!/usr/bin/env python3
"""
System Health Plugin
Monitors overall system health across all machines and daemons
"""

from typing import Dict, Any, Optional
from .base import LocalFilePlugin
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from core.paths import get_status_dir
except ImportError:
    def get_status_dir():
        return Path(__file__).parent.parent.parent.parent / "status"


class SystemHealthPlugin(LocalFilePlugin):
    """
    Monitors overall system health from status/system_health.json

    Data source: Local file written by system_health_aggregator.py
    Refresh: Every 60 seconds
    Critical: Yes (meta-health of the whole system)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        status_dir = get_status_dir()
        file_path = str(status_dir / "system_health.json")

        # Override with config if provided
        if config and 'file_path' in config:
            file_path = config['file_path']

        # Cache for 60 seconds
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 60)

        super().__init__(file_path, config)

    def get_name(self) -> str:
        return 'system_health'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': 'Overall system health across all machines',
            'refresh_interval': 60,
            'critical': True,
            'data_type': 'meta',
            'machine': 'all',
            'location': 'local',
            'fields': [
                'overall_status', 'machines', 'processes', 'summary'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch and enhance system health data"""
        data = super().fetch()

        # Add convenience fields
        summary = data.get('summary', {})

        data['health_score'] = self._calculate_health_score(data)
        data['status_icon'] = {
            'healthy': 'green',
            'degraded': 'yellow',
            'critical': 'red'
        }.get(data.get('overall_status'), 'gray')

        # Process summaries by machine
        machine_summaries = {}
        for machine_name, machine_data in data.get('machines', {}).items():
            machine_summaries[machine_name] = {
                'reachable': machine_data.get('reachable', False),
                'processes_ok': machine_data.get('processes_running', 0) == machine_data.get('processes_expected', 0),
                'gpu_ok': machine_data.get('gpu_available', False),
                'missing': machine_data.get('processes_missing', [])
            }

        data['machine_summaries'] = machine_summaries

        return data

    def _calculate_health_score(self, data: Dict) -> float:
        """Calculate a 0-100 health score"""
        summary = data.get('summary', {})
        total = summary.get('total_expected', 1)
        running = summary.get('total_running', 0)

        # Base score from process count
        process_score = (running / total) * 70 if total > 0 else 0

        # GPU score (15 points each)
        gpus_available = summary.get('gpus_available', 0)
        gpu_score = gpus_available * 15

        # Subtract for critical failures
        critical_missing = len(summary.get('critical_missing', []))
        critical_penalty = critical_missing * 20

        return max(0, min(100, process_score + gpu_score - critical_penalty))
