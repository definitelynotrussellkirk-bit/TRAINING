#!/usr/bin/env python3
"""
Training Status Plugin
Phase 2, Task 2.2: Real-time training status from 4090 machine
"""

from typing import Dict, Any, Optional
from .base import LocalFilePlugin
import sys
from pathlib import Path

# Add core to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "core"))
from paths import get_status_dir


class TrainingStatusPlugin(LocalFilePlugin):
    """
    Fetches real-time training status from the 4090 training machine.

    Data source: status/training_status.json (local file)
    Refresh: Real-time (< 5s)
    Critical: Yes
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Default path, can be overridden in config
        file_path = (config or {}).get(
            'file_path',
            str(get_status_dir() / 'training_status.json')
        )

        # Cache for 5 seconds (very fresh data)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 5)

        super().__init__(file_path, config)

    def get_name(self) -> str:
        return 'training_status'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': '4090 Training Machine - Real-time training progress',
            'refresh_interval': 5,
            'critical': True,
            'data_type': 'training',
            'machine': '4090',
            'location': 'local',
            'fields': [
                'status', 'current_step', 'total_steps', 'loss',
                'validation_loss', 'learning_rate', 'accuracy_percent',
                'tokens_per_sec', 'batch_queue_size'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch and optionally transform training status data"""
        data = super().fetch()

        # Add computed fields
        if 'current_step' in data and 'total_steps' in data:
            data['progress_percent'] = (data['current_step'] / data['total_steps'] * 100)

        return data
