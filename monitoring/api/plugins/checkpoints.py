#!/usr/bin/env python3
"""
Checkpoint Sync Plugin
Phase 4: Checkpoint synchronization status from 3090 intelligence machine
"""

from typing import Dict, Any, Optional
from .base import RemoteFilePlugin


class CheckpointSyncPlugin(RemoteFilePlugin):
    """
    Fetches checkpoint sync status from the 3090 intelligence machine.

    Data source: ssh://192.168.x.x/~/TRAINING/status/checkpoint_sync.json
    Refresh: Every 5 minutes
    Critical: Yes (checkpoint availability is critical)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        ssh_host = (config or {}).get('ssh_host', '192.168.x.x')
        remote_path = (config or {}).get(
            'remote_path',
            '/home/user/TRAINING/status/checkpoint_sync.json'
        )

        # Cache for 5 minutes (data updates every 5 min)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 300)

        super().__init__(ssh_host, remote_path, config)

    def get_name(self) -> str:
        return 'checkpoint_sync'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': '3090 Intelligence Machine - Checkpoint sync status',
            'refresh_interval': 300,
            'critical': True,
            'data_type': 'infrastructure',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'last_sync_time', 'sync_status',
                'latest_checkpoint', 'sync_failures'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch and extract key checkpoint sync metrics"""
        data = super().fetch()

        # Extract summary
        summary = {
            'last_sync': data.get('last_sync_time'),
            'status': data.get('sync_status', 'unknown'),
            'latest_checkpoint': data.get('latest_checkpoint'),
            'total_synced': data.get('total_checkpoints_synced', 0),
            'failures': data.get('sync_failures', 0)
        }

        data['summary'] = summary

        return data
