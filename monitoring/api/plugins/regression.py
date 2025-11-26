#!/usr/bin/env python3
"""
Regression Monitoring Plugin
Phase 4: Regression detection from 3090 intelligence machine
"""

from typing import Dict, Any, Optional
from .base import RemoteFilePlugin


class RegressionPlugin(RemoteFilePlugin):
    """
    Fetches regression monitoring results from the 3090 intelligence machine.

    Data source: ssh://192.168.x.x/~/TRAINING/status/regression_monitoring.json
    Refresh: Every 5 minutes
    Critical: Yes (regressions must be caught immediately)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        ssh_host = (config or {}).get('ssh_host', '192.168.x.x')
        remote_path = (config or {}).get(
            'remote_path',
            '/path/to/training/status/regression_monitoring.json'
        )

        # Cache for 5 minutes (data updates every 5 min)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 300)

        super().__init__(ssh_host, remote_path, config)

    def get_name(self) -> str:
        return 'regression_monitoring'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': '3090 Intelligence Machine - Regression detection',
            'refresh_interval': 300,
            'critical': True,
            'data_type': 'intelligence',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'regressions_detected', 'latest_check',
                'baseline_accuracy', 'current_accuracy'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch and extract key regression monitoring metrics"""
        data = super().fetch()

        # Extract latest check summary
        if 'checks' in data and len(data['checks']) > 0:
            latest = data['checks'][-1]

            summary = {
                'timestamp': latest.get('timestamp'),
                'checkpoint': latest.get('checkpoint'),
                'regression_detected': latest.get('regression_detected', False),
                'loss_increase': latest.get('loss_increase_percent', 0.0),
                'accuracy_drop': latest.get('accuracy_drop_percent', 0.0)
            }

            data['latest_summary'] = summary

        # Count total regressions
        total_regressions = sum(
            1 for check in data.get('checks', [])
            if check.get('regression_detected', False)
        )
        data['total_regressions'] = total_regressions

        return data
