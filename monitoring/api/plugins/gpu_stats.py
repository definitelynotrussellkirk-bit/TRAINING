#!/usr/bin/env python3
"""
GPU Statistics Plugins
Phase 2, Task 2.2: GPU VRAM and utilization from both machines
"""

from .base import CommandPlugin
import subprocess


class GPU4090Plugin(CommandPlugin):
    """
    Fetches GPU stats from the local 4090 training machine.

    Data source: nvidia-smi command (local)
    Refresh: Every 5 seconds
    Critical: Yes
    """

    def __init__(self, config=None):
        command = 'nvidia-smi --query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,name --format=csv,noheader,nounits'

        # Cache for 5 seconds
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 5)

        super().__init__(command, config)

    def get_name(self):
        return 'gpu_4090'

    def get_metadata(self):
        return {
            'description': '4090 Training Machine - GPU Statistics',
            'refresh_interval': 5,
            'critical': True,
            'data_type': 'hardware',
            'machine': '4090',
            'location': 'local',
            'fields': [
                'vram_used_mb', 'vram_total_mb', 'vram_free_mb',
                'utilization_percent', 'temperature_c', 'name'
            ]
        }

    def parse_output(self, stdout: str) -> dict:
        """Parse nvidia-smi CSV output"""
        lines = stdout.strip().split('\n')

        if not lines:
            return {}

        # Parse first GPU (should only be one)
        parts = [p.strip() for p in lines[0].split(',')]

        if len(parts) < 6:
            return {}

        try:
            return {
                'vram_used_mb': int(parts[0]),
                'vram_total_mb': int(parts[1]),
                'vram_free_mb': int(parts[2]),
                'utilization_percent': int(parts[3]),
                'temperature_c': int(parts[4]),
                'name': parts[5],
                'vram_used_gb': round(int(parts[0]) / 1024, 2),
                'vram_total_gb': round(int(parts[1]) / 1024, 2),
                'vram_percent': round(int(parts[0]) / int(parts[1]) * 100, 1)
            }
        except (ValueError, ZeroDivisionError):
            return {}


class GPU3090Plugin(CommandPlugin):
    """
    Fetches GPU stats from the remote 3090 intelligence machine.

    Data source: nvidia-smi via SSH (remote)
    Refresh: Every 10 seconds
    Critical: No
    """

    def __init__(self, config=None):
        ssh_host = (config or {}).get('ssh_host', '192.168.x.x')
        command = f'ssh {ssh_host} "nvidia-smi --query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,name --format=csv,noheader,nounits"'

        # Cache for 10 seconds
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 10)

        super().__init__(command, config)
        self.ssh_host = ssh_host

    def get_name(self):
        return 'gpu_3090'

    def get_metadata(self):
        return {
            'description': '3090 Intelligence Machine - GPU Statistics',
            'refresh_interval': 10,
            'critical': False,
            'data_type': 'hardware',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'vram_used_mb', 'vram_total_mb', 'vram_free_mb',
                'utilization_percent', 'temperature_c', 'name'
            ]
        }

    def parse_output(self, stdout: str) -> dict:
        """Parse nvidia-smi CSV output (same as 4090)"""
        lines = stdout.strip().split('\n')

        if not lines:
            return {}

        parts = [p.strip() for p in lines[0].split(',')]

        if len(parts) < 6:
            return {}

        try:
            return {
                'vram_used_mb': int(parts[0]),
                'vram_total_mb': int(parts[1]),
                'vram_free_mb': int(parts[2]),
                'utilization_percent': int(parts[3]),
                'temperature_c': int(parts[4]),
                'name': parts[5],
                'vram_used_gb': round(int(parts[0]) / 1024, 2),
                'vram_total_gb': round(int(parts[1]) / 1024, 2),
                'vram_percent': round(int(parts[0]) / int(parts[1]) * 100, 1)
            }
        except (ValueError, ZeroDivisionError):
            return {}
