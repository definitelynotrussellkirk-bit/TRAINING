#!/usr/bin/env python3
"""
Confidence Calibration Plugin
Phase 4: Confidence calibration from 3090 intelligence machine
"""

from .base import RemoteFilePlugin


class ConfidencePlugin(RemoteFilePlugin):
    """
    Fetches confidence calibration results from the 3090 intelligence machine.

    Data source: ssh://192.168.x.x/~/TRAINING/status/confidence_calibration.json
    Refresh: Every 10 minutes
    Critical: No
    """

    def __init__(self, config=None):
        ssh_host = (config or {}).get('ssh_host', '192.168.x.x')
        remote_path = (config or {}).get(
            'remote_path',
            '/home/user/TRAINING/status/confidence_calibration.json'
        )

        # Cache for 10 minutes (data updates every 10 min)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 600)

        super().__init__(ssh_host, remote_path, config)

    def get_name(self):
        return 'confidence_calibration'

    def get_metadata(self):
        return {
            'description': '3090 Intelligence Machine - Confidence calibration',
            'refresh_interval': 600,
            'critical': False,
            'data_type': 'intelligence',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'calibration_bins', 'expected_calibration_error',
                'confidence_accuracy_gap', 'overconfidence_ratio'
            ]
        }

    def fetch(self):
        """Fetch and extract key confidence calibration metrics"""
        data = super().fetch()

        # Extract latest calibration summary
        if 'calibrations' in data and len(data['calibrations']) > 0:
            latest = data['calibrations'][-1]

            summary = {
                'timestamp': latest.get('timestamp'),
                'checkpoint': latest.get('checkpoint'),
                'ece': latest.get('expected_calibration_error', 0.0),
                'num_bins': len(latest.get('bins', [])),
                'overconfident': latest.get('overconfidence_ratio', 0.0)
            }

            # Extract bin info
            bins = []
            for bin_data in latest.get('bins', []):
                bins.append({
                    'range': bin_data.get('confidence_range'),
                    'accuracy': bin_data.get('accuracy', 0.0),
                    'count': bin_data.get('count', 0),
                    'gap': bin_data.get('confidence_accuracy_gap', 0.0)
                })

            summary['bins'] = bins
            data['latest_summary'] = summary

        return data
