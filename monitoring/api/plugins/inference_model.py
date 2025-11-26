#!/usr/bin/env python3
"""
Inference Model Plugin

Fetches currently loaded model info from the 3090 inference server.
Shows which checkpoint is actively serving inference requests.
"""

from typing import Dict, Any, Optional
import logging
import os
import requests
from datetime import datetime

from .base import BasePlugin, PluginError

logger = logging.getLogger(__name__)


class InferenceModelPlugin(BasePlugin):
    """
    Plugin for 3090 inference model status.

    Connects to the inference server API to fetch:
    - Currently loaded model/checkpoint
    - Checkpoint step number
    - VRAM usage
    - Load timestamp

    Data source: http://192.168.x.x:8765/models/info
    Refresh: Every 30 seconds
    Critical: Yes (shows what model is actually serving)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.inference_host = self.config.get('inference_host', '192.168.x.x')
        self.inference_port = self.config.get('inference_port', 8765)
        self.base_url = f"http://{self.inference_host}:{self.inference_port}"
        self.timeout = self.config.get('timeout', 5)
        # API key from environment or config
        self.api_key = self.config.get('api_key') or os.environ.get('INFERENCE_ADMIN_KEY', '')

    def get_name(self) -> str:
        return 'inference_model'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': 'Currently loaded model on 3090 inference server',
            'refresh_interval': 30,
            'critical': True,
            'data_type': 'inference_model',
            'machine': '3090',
            'location': 'remote',
            'fields': [
                'model_id', 'checkpoint_step', 'loaded_from', 'loaded_at',
                'vram_usage_gb', 'server_status'
            ]
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch current model info from the 3090 inference server."""
        result = {
            'model_id': None,
            'checkpoint_step': None,
            'loaded_from': None,
            'loaded_at': None,
            'vram_usage_gb': None,
            'server_status': 'unknown',
            'warnings': []
        }

        try:
            # Build headers
            headers = {}
            if self.api_key:
                headers['X-API-Key'] = self.api_key

            # Fetch /models/info
            response = requests.get(
                f"{self.base_url}/models/info",
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            info = response.json()

            result['model_id'] = info.get('model_id')
            result['checkpoint_step'] = info.get('checkpoint_step')
            result['loaded_from'] = info.get('loaded_from')
            result['loaded_at'] = info.get('loaded_at')
            result['vram_usage_gb'] = info.get('vram_usage_gb')
            result['server_status'] = 'ok' if info.get('loaded') else 'no_model'

            # Warning if base model loaded instead of checkpoint
            model_id = result['model_id'] or ''
            if 'Qwen3-0.6B' in model_id and 'checkpoint' not in model_id.lower():
                result['warnings'].append({
                    'level': 'warning',
                    'message': 'Base model loaded, not a trained checkpoint'
                })

            # Warning if model_id doesn't match checkpoint pattern
            if result['model_id'] and not result['model_id'].startswith('checkpoint-'):
                if 'Qwen3' not in result['model_id']:
                    result['warnings'].append({
                        'level': 'info',
                        'message': f'Non-standard model name: {result["model_id"]}'
                    })

        except requests.exceptions.ConnectionError:
            result['server_status'] = 'offline'
            result['warnings'].append({
                'level': 'critical',
                'message': 'Inference server unreachable'
            })
        except requests.exceptions.Timeout:
            result['server_status'] = 'timeout'
            result['warnings'].append({
                'level': 'warning',
                'message': 'Inference server timeout'
            })
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                result['server_status'] = 'auth_error'
                result['warnings'].append({
                    'level': 'warning',
                    'message': 'API key required - set INFERENCE_ADMIN_KEY'
                })
            else:
                result['server_status'] = 'error'
                result['warnings'].append({
                    'level': 'warning',
                    'message': f'HTTP error: {e.response.status_code}'
                })
        except Exception as e:
            logger.exception("Error fetching inference model info")
            result['server_status'] = 'error'
            result['warnings'].append({
                'level': 'warning',
                'message': str(e)
            })

        return result
