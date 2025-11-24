#!/usr/bin/env python3
"""
Automated Testing Plugin
Phase 4: Automated testing results from 3090 intelligence machine
"""

from .base import RemoteFilePlugin


class TestingPlugin(RemoteFilePlugin):
    """
    Fetches automated testing results from the 3090 intelligence machine.

    Data source: ssh://192.168.x.x/~/TRAINING/status/automated_testing.json
    Refresh: Every 10 minutes
    Critical: Yes (test failures indicate problems)
    """

    def __init__(self, config=None):
        ssh_host = (config or {}).get('ssh_host', '192.168.x.x')
        remote_path = (config or {}).get(
            'remote_path',
            '/home/user/TRAINING/status/automated_testing.json'
        )

        # Cache for 10 minutes (data updates every 10 min)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 600)

        super().__init__(ssh_host, remote_path, config)

    def get_name(self):
        return 'automated_testing'

    def get_metadata(self):
        return {
            'description': '3090 Intelligence Machine - Automated test suite',
            'refresh_interval': 600,
            'critical': True,
            'data_type': 'intelligence',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'test_runs', 'pass_rate',
                'accuracy_by_difficulty', 'failing_tests'
            ]
        }

    def fetch(self):
        """Fetch and extract key automated testing metrics"""
        data = super().fetch()

        # Extract latest test run summary
        if 'test_runs' in data and len(data['test_runs']) > 0:
            latest = data['test_runs'][-1]

            summary = {
                'timestamp': latest.get('timestamp'),
                'checkpoint': latest.get('checkpoint'),
                'total_tests': latest.get('total_tests', 0),
                'passed': latest.get('passed', 0),
                'failed': latest.get('failed', 0),
                'pass_rate': latest.get('pass_rate', 0.0),
                'accuracy_by_difficulty': latest.get('accuracy_by_difficulty', {})
            }

            # Calculate trend (last 5 runs)
            recent_runs = data['test_runs'][-5:]
            avg_pass_rate = sum(r.get('pass_rate', 0.0) for r in recent_runs) / len(recent_runs)
            summary['recent_avg_pass_rate'] = avg_pass_rate

            data['latest_summary'] = summary

        return data
