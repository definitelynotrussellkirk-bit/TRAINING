#!/usr/bin/env python3
"""
Retention Status Plugin
Monitors checkpoint and snapshot retention status
"""

from typing import Dict, Any, Optional
from .base import BasePlugin, PluginError
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from core.paths import get_base_dir
except ImportError:
    def get_base_dir():
        return Path(__file__).parent.parent.parent.parent


class RetentionPlugin(BasePlugin):
    """
    Monitors checkpoint and snapshot retention status.

    Uses RetentionManager to get live retention data.

    Data source: RetentionManager.get_status()
    Refresh: Every 5 minutes
    Critical: No
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

        self.base_dir = Path(config.get('base_dir')) if config and 'base_dir' in config else get_base_dir()
        self.output_dir = self.base_dir / "models" / "current_model"

        # Override with config if provided
        if config and 'output_dir' in config:
            self.output_dir = Path(config['output_dir'])

        # Cache for 5 minutes
        self.cache_duration = config.get('cache_duration', 300) if config else 300

    def get_name(self) -> str:
        return 'retention'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': 'Checkpoint and snapshot retention status',
            'refresh_interval': 300,
            'critical': False,
            'data_type': 'storage',
            'machine': '4090',
            'location': 'local',
            'fields': [
                'total_size_gb', 'limit_gb', 'usage_pct',
                'checkpoints', 'snapshots', 'latest_checkpoint', 'best_checkpoint'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch retention status"""
        try:
            # Try to import RetentionManager
            from management.retention_manager import RetentionManager

            manager = RetentionManager(self.output_dir)
            status = manager.get_status()

            # Add computed fields
            status['health'] = self._assess_retention_health(status)
            status['warnings'] = self._get_warnings(status)

            return status

        except ImportError as e:
            # RetentionManager not available, try reading index file directly
            logger.warning(f"RetentionManager not available: {e}")
            return self._read_index_fallback()

        except Exception as e:
            raise PluginError(f"Failed to get retention status: {e}")

    def _read_index_fallback(self) -> Dict[str, Any]:
        """Fallback: read retention_index.json directly"""
        import json

        index_file = self.output_dir / "retention_index.json"
        if not index_file.exists():
            return {
                'error': 'No retention index found',
                'output_dir': str(self.output_dir),
                'health': 'unknown'
            }

        try:
            with open(index_file) as f:
                data = json.load(f)

            checkpoints = data.get('checkpoints', [])
            snapshots = data.get('snapshots', [])

            checkpoint_size = sum(c.get('size_bytes', 0) for c in checkpoints)
            snapshot_size = sum(s.get('size_bytes', 0) for s in snapshots)
            total_size = checkpoint_size + snapshot_size

            GB = 1024 ** 3
            LIMIT_GB = 150

            return {
                'total_size_gb': total_size / GB,
                'limit_gb': LIMIT_GB,
                'usage_pct': (total_size / (LIMIT_GB * GB)) * 100,
                'checkpoints': {
                    'count': len(checkpoints),
                    'size_gb': checkpoint_size / GB
                },
                'snapshots': {
                    'count': len(snapshots),
                    'size_gb': snapshot_size / GB
                },
                'latest_checkpoint': data.get('latest_checkpoint'),
                'best_checkpoint': data.get('best_checkpoint'),
                'last_updated': data.get('last_updated'),
                'health': 'good' if (total_size / (LIMIT_GB * GB)) < 0.8 else 'warning'
            }

        except Exception as e:
            raise PluginError(f"Failed to read retention index: {e}")

    def _assess_retention_health(self, status: Dict) -> str:
        """Assess retention health: good, warning, critical"""
        usage_pct = status.get('usage_pct', 0)

        if usage_pct >= 95:
            return 'critical'
        elif usage_pct >= 80:
            return 'warning'
        else:
            return 'good'

    def _get_warnings(self, status: Dict) -> list:
        """Generate warnings based on status"""
        warnings = []

        usage_pct = status.get('usage_pct', 0)
        if usage_pct >= 90:
            warnings.append(f"Storage critically high: {usage_pct:.1f}% used")
        elif usage_pct >= 80:
            warnings.append(f"Storage getting full: {usage_pct:.1f}% used")

        checkpoints = status.get('checkpoints', {})
        if checkpoints.get('count', 0) == 0:
            warnings.append("No checkpoints found")

        if checkpoints.get('oldest_age_hours', 0) > 72:
            warnings.append(f"Old checkpoints present: {checkpoints['oldest_age_hours']:.0f}h oldest")

        snapshots = status.get('snapshots', {})
        if snapshots.get('count', 0) == 0:
            warnings.append("No daily snapshots found")

        return warnings
