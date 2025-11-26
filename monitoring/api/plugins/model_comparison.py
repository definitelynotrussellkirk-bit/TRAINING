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
            '/path/to/training/status/model_comparisons.json'
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

            # Get ranking (actual key in JSON is 'ranking', not 'ranked_checkpoints')
            ranked = latest.get('ranking', [])[:3]
            best = latest.get('best_checkpoint', {})

            summary = {
                'timestamp': latest.get('timestamp'),
                'total_compared': latest.get('num_checkpoints', len(ranked)),
                'best_checkpoint': f"checkpoint-{best.get('step')}" if best.get('step') else None,
                'best_score': best.get('score', 0.0),
                'top_3': [
                    {
                        'checkpoint': f"checkpoint-{r.get('step')}",
                        'score': r.get('score', 0.0),
                        'rank': r.get('rank', i + 1)
                    }
                    for i, r in enumerate(ranked)
                ]
            }

            data['latest_summary'] = summary

        return data
