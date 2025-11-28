#!/usr/bin/env python3
"""
Adversarial Mining Plugin
Phase 4: Adversarial examples from 3090 intelligence machine
"""

from typing import Dict, Any, Optional
from .base import RemoteFilePlugin
import sys
from pathlib import Path

# Add core to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "core"))
from hosts import get_host
from paths import get_status_dir


class AdversarialPlugin(RemoteFilePlugin):
    """
    Fetches adversarial mining results from the 3090 intelligence machine.

    Data source: ssh://192.168.x.x/~/TRAINING/status/adversarial_mining.json
    Refresh: Every 5 minutes
    Critical: No
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        ssh_host = (config or {}).get('ssh_host', get_host('3090').host)
        remote_path = (config or {}).get(
            'remote_path',
            str(get_status_dir() / 'adversarial_mining.json')
        )

        # Cache for 5 minutes (data updates every 5 min)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 300)

        super().__init__(ssh_host, remote_path, config)

    def get_name(self) -> str:
        return 'adversarial_mining'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': '3090 Intelligence Machine - Adversarial example mining',
            'refresh_interval': 300,
            'critical': False,
            'data_type': 'intelligence',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'total_mined', 'high_loss_examples',
                'low_confidence_examples', 'pattern_failures'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch and extract key adversarial mining metrics"""
        data = super().fetch()

        # Extract summary if available
        if 'mining_runs' in data and len(data['mining_runs']) > 0:
            latest = data['mining_runs'][-1]

            summary = {
                'timestamp': latest.get('timestamp'),
                'total_examples': latest.get('total_examples_mined', 0),
                'categories': {}
            }

            # Categorize adversarial examples
            # Note: backend writes 'avg_confidence', we normalize to 'avg_loss' for frontend
            categories = latest.get('categories', {})
            for cat_name, cat_data in categories.items():
                summary['categories'][cat_name] = {
                    'count': cat_data.get('count', 0),
                    'avg_loss': cat_data.get('avg_confidence', cat_data.get('average_loss', 0.0))
                }

            data['latest_summary'] = summary

        return data
