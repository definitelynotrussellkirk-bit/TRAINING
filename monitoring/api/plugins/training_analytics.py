#!/usr/bin/env python3
"""
Training Analytics API Plugin

Exposes training analytics data through the unified API:
- Layer drift (which layers are changing)
- Parameter stability (weight norm health)
- Data file impact (per-file training impact)

Data sources (on 3090 remote):
- status/layer_drift.json
- status/parameter_stability.json
- status/data_file_impact.jsonl
- status/data_file_summary.json

The analytics daemons run continuously on the 3090, updating every 10 minutes
when new checkpoints appear. This plugin fetches that data via SSH.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging
import subprocess
import sys

# Add core to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "core"))
from hosts import get_host
from paths import get_status_dir

from .base import BasePlugin

logger = logging.getLogger(__name__)


class TrainingAnalyticsPlugin(BasePlugin):
    """
    Plugin for training analytics data from the 3090 intelligence machine.

    Aggregates data from multiple analytics modules:
    - Layer Drift Monitor: Tracks which transformer layers are changing
    - Parameter Stability Monitor: Detects weight explosion/vanishing
    - Data File Impact Analyzer: Per-file training impact

    Data is read from status/*.json files on the 3090 via SSH.
    The analytics daemons run as background processes on the 3090,
    updating every 10 minutes when new checkpoints are detected.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ssh_host = self.config.get('ssh_host', get_host('3090').host)
        self.remote_status_dir = self.config.get(
            'remote_status_dir',
            '~/TRAINING/status'
        )

    def get_name(self) -> str:
        return 'training_analytics'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': 'Training analytics (layer drift, stability, file impact)',
            'refresh_interval': 60,  # Fetch every 60s (data updates every 10 min)
            'critical': False,
            'data_type': 'analytics',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'layer_drift', 'parameter_stability', 'data_file_impact'
            ]
        }

    def _ssh_read_json(self, filename: str) -> Optional[Dict]:
        """Read a JSON file from remote status directory via SSH."""
        try:
            result = subprocess.run(
                ['ssh', self.ssh_host, f'cat {self.remote_status_dir}/{filename}'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning(f"SSH timeout reading {filename}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for {filename}: {e}")
        except Exception as e:
            logger.warning(f"Could not SSH read {filename}: {e}")
        return None

    def _ssh_read_jsonl(self, filename: str, limit: int = 50) -> List[Dict]:
        """Read last N entries from a remote JSONL file via SSH."""
        try:
            # Use tail to get last N lines
            result = subprocess.run(
                ['ssh', self.ssh_host, f'tail -n {limit * 2} {self.remote_status_dir}/{filename}'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                entries = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return entries[-limit:]
        except subprocess.TimeoutExpired:
            logger.warning(f"SSH timeout reading {filename}")
        except Exception as e:
            logger.warning(f"Could not SSH read {filename}: {e}")
        return []

    def fetch(self) -> Dict[str, Any]:
        """Fetch all training analytics data from 3090 via SSH."""
        from datetime import datetime

        result = {
            'timestamp': datetime.now().isoformat(),
            'ssh_host': self.ssh_host,
            'layer_drift': {
                'available': False,
                'data': None,
                'summary': None
            },
            'parameter_stability': {
                'available': False,
                'data': None,
                'summary': None,
                'alerts': []
            },
            'data_file_impact': {
                'available': False,
                'summary': None,
                'recent_impacts': []
            }
        }

        # Load layer drift from 3090
        layer_drift = self._ssh_read_json('layer_drift.json')
        if layer_drift:
            # Extract full layer data for visualization
            layers_full = layer_drift.get('layers', [])
            transformer_layers = [l for l in layers_full if l.get('layer_idx', -1) >= 0]
            embedding_layer = next((l for l in layers_full if l.get('layer_idx', -1) == -1), None)

            result['layer_drift'] = {
                'available': True,
                'reference': layer_drift.get('reference_checkpoint'),
                'current': layer_drift.get('current_checkpoint'),
                'current_step': layer_drift.get('current_step'),
                'timestamp': layer_drift.get('timestamp'),
                'total_relative_change': layer_drift.get('total_relative_change'),
                'total_params': layer_drift.get('total_params'),
                'summary': layer_drift.get('summary'),
                'layers': self._summarize_layers(transformer_layers),
                'embedding_change': round(embedding_layer.get('relative_change', 0), 4) if embedding_layer else None
            }

        # Load parameter stability from 3090
        stability = self._ssh_read_json('parameter_stability.json')
        if stability:
            # Extract layer norms for visualization
            layers = stability.get('layers', [])
            transformer_layers = [l for l in layers if l.get('layer_idx', -1) >= 0]

            result['parameter_stability'] = {
                'available': True,
                'checkpoint': stability.get('checkpoint'),
                'step': stability.get('step'),
                'timestamp': stability.get('timestamp'),
                'summary': stability.get('summary'),
                'alerts': stability.get('alerts', []),
                'health_status': stability.get('summary', {}).get('health_status', 'unknown'),
                'layer_norms': self._summarize_norms(transformer_layers)
            }

        # Load data file impact from LOCAL files (generated by daemon's DataFileImpactTracker)
        # Try local first (4090 training machine), then remote (3090) as fallback
        local_status_dir = get_status_dir()
        impact_summary = None
        recent_impacts = []

        # Try local files first
        local_summary = local_status_dir / 'data_file_summary.json'
        local_impacts = local_status_dir / 'data_file_impact.jsonl'

        if local_summary.exists():
            try:
                with open(local_summary) as f:
                    impact_summary = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read local impact summary: {e}")

        if local_impacts.exists():
            try:
                with open(local_impacts) as f:
                    for line in f:
                        if line.strip():
                            recent_impacts.append(json.loads(line))
                recent_impacts = recent_impacts[-20:]  # Keep last 20
            except Exception as e:
                logger.warning(f"Could not read local impact file: {e}")

        # Fallback to remote if local not available
        if not impact_summary and not recent_impacts:
            impact_summary = self._ssh_read_json('data_file_summary.json')
            recent_impacts = self._ssh_read_jsonl('data_file_impact.jsonl', limit=20)

        if impact_summary or recent_impacts:
            result['data_file_impact'] = {
                'available': True,
                'summary': impact_summary,
                'recent_impacts': recent_impacts[-10:],
                'total_analyzed': impact_summary.get('total_files_analyzed', len(recent_impacts)) if impact_summary else len(recent_impacts),
                'top_positive': impact_summary.get('top_positive', []) if impact_summary else [],
                'top_negative': impact_summary.get('top_negative', []) if impact_summary else []
            }

        return result

    def _summarize_layers(self, layers: List[Dict]) -> List[Dict]:
        """
        Summarize layer drift data for API response.

        Keeps transformer layers and key drift metrics for visualization.
        """
        summarized = []
        for layer in layers:
            idx = layer.get('layer_idx', -1)
            if idx >= 0:
                summarized.append({
                    'layer': idx,
                    'relative_change': round(layer.get('relative_change', 0), 6),
                    'delta_norm': round(layer.get('delta_norm', 0), 4),
                    'reference_norm': round(layer.get('reference_norm', 0), 2)
                })
        return sorted(summarized, key=lambda x: x['layer'])

    def _summarize_norms(self, layers: List[Dict]) -> List[Dict]:
        """
        Summarize parameter stability data for API response.

        Returns weight norms for each layer for visualization.
        """
        summarized = []
        for layer in layers:
            idx = layer.get('layer_idx', -1)
            if idx >= 0:
                summarized.append({
                    'layer': idx,
                    'weight_norm': round(layer.get('weight_norm', 0), 2),
                    'max_abs_weight': round(layer.get('max_abs_weight', 0), 4),
                    'min_abs_weight': layer.get('min_abs_weight', 0)
                })
        return sorted(summarized, key=lambda x: x['layer'])


class LayerDriftPlugin(BasePlugin):
    """Standalone plugin for layer drift data (more detailed)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.status_dir = Path(self.config.get(
            'status_dir',
            str(get_status_dir())
        ))

    def get_name(self) -> str:
        return 'layer_drift'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': 'Per-layer weight drift analysis',
            'refresh_interval': 600,
            'critical': False,
            'data_type': 'weights'
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch layer drift data."""
        from datetime import datetime

        filepath = self.status_dir / 'layer_drift.json'
        if not filepath.exists():
            return {
                'available': False,
                'timestamp': datetime.now().isoformat()
            }

        try:
            with open(filepath) as f:
                data = json.load(f)
            data['available'] = True
            return data
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class ParameterStabilityPlugin(BasePlugin):
    """Standalone plugin for parameter stability data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.status_dir = Path(self.config.get(
            'status_dir',
            str(get_status_dir())
        ))

    def get_name(self) -> str:
        return 'parameter_stability'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': 'Weight norm stability and health alerts',
            'refresh_interval': 600,
            'critical': True,  # Health alerts are important
            'data_type': 'health'
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch parameter stability data."""
        from datetime import datetime

        filepath = self.status_dir / 'parameter_stability.json'
        if not filepath.exists():
            return {
                'available': False,
                'health_status': 'unknown',
                'timestamp': datetime.now().isoformat()
            }

        try:
            with open(filepath) as f:
                data = json.load(f)
            data['available'] = True
            return data
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'health_status': 'unknown',
                'timestamp': datetime.now().isoformat()
            }


class DataFileImpactPlugin(BasePlugin):
    """Standalone plugin for data file impact analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.status_dir = Path(self.config.get(
            'status_dir',
            str(get_status_dir())
        ))

    def get_name(self) -> str:
        return 'data_file_impact'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': 'Per-training-file impact analysis',
            'refresh_interval': 300,
            'critical': False,
            'data_type': 'impact'
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch data file impact data."""
        from datetime import datetime

        result = {
            'timestamp': datetime.now().isoformat(),
            'available': False
        }

        # Load summary
        summary_path = self.status_dir / 'data_file_summary.json'
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    result['summary'] = json.load(f)
                result['available'] = True
            except Exception as e:
                result['summary_error'] = str(e)

        # Load recent impacts
        impacts_path = self.status_dir / 'data_file_impact.jsonl'
        if impacts_path.exists():
            try:
                impacts = []
                with open(impacts_path) as f:
                    for line in f:
                        if line.strip():
                            impacts.append(json.loads(line))
                result['recent_impacts'] = impacts[-20:]
                result['available'] = True
            except Exception as e:
                result['impacts_error'] = str(e)

        return result
