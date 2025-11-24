#!/usr/bin/env python3
"""
Curriculum Optimization Plugin
Phase 2, Task 2.2: Curriculum strategy results from 3090 machine
"""

from typing import Dict, Any, Optional
from .base import RemoteFilePlugin


class CurriculumPlugin(RemoteFilePlugin):
    """
    Fetches curriculum optimization results from the 3090 intelligence machine.

    Data source: ssh://192.168.x.x/~/TRAINING/status/curriculum_optimization.json
    Refresh: Every 5 minutes
    Critical: No
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        ssh_host = (config or {}).get('ssh_host', '192.168.x.x')
        remote_path = (config or {}).get(
            'remote_path',
            '/home/user/TRAINING/status/curriculum_optimization.json'
        )

        # Cache for 5 minutes (data updates every 5 min)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 300)

        super().__init__(ssh_host, remote_path, config)

    def get_name(self) -> str:
        return 'curriculum_optimization'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': '3090 Intelligence Machine - Curriculum strategy optimization',
            'refresh_interval': 300,
            'critical': False,
            'data_type': 'intelligence',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'evaluations', 'difficulties.easy.accuracy',
                'difficulties.medium.accuracy', 'difficulties.hard.accuracy'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch and extract key curriculum metrics"""
        data = super().fetch()

        # Extract latest evaluation summary
        if 'evaluations' in data and len(data['evaluations']) > 0:
            latest = data['evaluations'][-1]

            summary = {
                'step': latest.get('step'),
                'checkpoint': latest.get('checkpoint'),
                'timestamp': latest.get('timestamp'),
                'accuracies': {}
            }

            for difficulty in ['easy', 'medium', 'hard']:
                if difficulty in latest.get('difficulties', {}):
                    diff_data = latest['difficulties'][difficulty]
                    summary['accuracies'][difficulty] = diff_data.get('accuracy', 0.0)

            data['latest_summary'] = summary

        return data
