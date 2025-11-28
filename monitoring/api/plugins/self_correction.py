#!/usr/bin/env python3
"""
Self-Correction Loop Plugin
Phase 4: Self-correction loop results from 3090 intelligence machine
"""

from typing import Dict, Any, Optional
from .base import RemoteFilePlugin
import sys
from pathlib import Path

# Add core to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "core"))
from hosts import get_host
from paths import get_status_dir


class SelfCorrectionPlugin(RemoteFilePlugin):
    """
    Fetches self-correction loop results from the 3090 intelligence machine.

    Data source: ssh://inference.local/~/TRAINING/status/self_correction.json
    Refresh: Every 5 minutes
    Critical: No
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        ssh_host = (config or {}).get('ssh_host', get_host('3090').host)
        remote_path = (config or {}).get(
            'remote_path',
            str(get_status_dir() / 'self_correction.json')
        )

        # Cache for 5 minutes (data updates every 5 min)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 300)

        super().__init__(ssh_host, remote_path, config)

    def get_name(self) -> str:
        return 'self_correction'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': '3090 Intelligence Machine - Self-correction loop',
            'refresh_interval': 300,
            'critical': False,
            'data_type': 'intelligence',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'correction_runs', 'errors_captured',
                'patterns_identified', 'correction_examples_generated'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch and extract key self-correction metrics"""
        data = super().fetch()

        # Extract latest correction run summary
        if 'correction_runs' in data and len(data['correction_runs']) > 0:
            latest = data['correction_runs'][-1]

            summary = {
                'timestamp': latest.get('timestamp'),
                'errors_captured': latest.get('errors_captured', 0),
                'patterns_found': len(latest.get('error_patterns', [])),
                'corrections_generated': latest.get('corrections_generated', 0),
                'top_error_patterns': []
            }

            # Get top 5 error patterns
            patterns = latest.get('error_patterns', [])
            for pattern in patterns[:5]:
                summary['top_error_patterns'].append({
                    'type': pattern.get('type'),
                    'count': pattern.get('count', 0),
                    'description': pattern.get('description', 'Unknown')
                })

            data['latest_summary'] = summary

        # Total corrections generated
        total_corrections = sum(
            run.get('corrections_generated', 0)
            for run in data.get('correction_runs', [])
        )
        data['total_corrections'] = total_corrections

        return data
