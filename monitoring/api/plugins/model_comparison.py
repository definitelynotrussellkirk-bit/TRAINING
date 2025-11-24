#!/usr/bin/env python3
"""
Model Comparison Plugin
Phase 4: Model/checkpoint rankings from 3090 intelligence machine
"""

from typing import Dict, Any, Optional
from .base import RemoteFilePlugin


class ModelComparisonPlugin(RemoteFilePlugin):
    """
    Fetches model comparison/ranking results from the 3090 intelligence machine.

    Data source: ssh://192.168.x.x/~/TRAINING/status/model_comparisons.json
    Refresh: Every 10 minutes
    Critical: No
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        ssh_host = (config or {}).get('ssh_host', '192.168.x.x')
        remote_path = (config or {}).get(
            'remote_path',
            '/home/user/TRAINING/status/model_comparisons.json'
        )

        # Cache for 10 minutes (data updates every 10 min)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 600)

        super().__init__(ssh_host, remote_path, config)

    def get_name(self) -> str:
        return 'model_comparison'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': '3090 Intelligence Machine - Model/checkpoint rankings',
            'refresh_interval': 600,
            'critical': False,
            'data_type': 'intelligence',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'comparisons', 'best_checkpoint',
                'ranking_method', 'composite_scores'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch and extract key model comparison metrics"""
        data = super().fetch()

        # Extract latest comparison summary
        if 'comparisons' in data and len(data['comparisons']) > 0:
            latest = data['comparisons'][-1]

            # Get top 3 models
            ranked = latest.get('ranked_checkpoints', [])[:3]

            summary = {
                'timestamp': latest.get('timestamp'),
                'total_compared': len(latest.get('ranked_checkpoints', [])),
                'best_checkpoint': ranked[0].get('checkpoint') if ranked else None,
                'best_score': ranked[0].get('composite_score') if ranked else 0.0,
                'top_3': [
                    {
                        'checkpoint': r.get('checkpoint'),
                        'score': r.get('composite_score'),
                        'rank': i + 1
                    }
                    for i, r in enumerate(ranked)
                ]
            }

            data['latest_summary'] = summary

        return data
