#!/usr/bin/env python3
"""
Data Aggregator - Unified API
Phase 2, Task 2.3: Combine all plugin data into single endpoint

This module provides a unified view of all data sources through
a single aggregated response.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

from .plugins import PluginRegistry
from .plugins.training_status import TrainingStatusPlugin
from .plugins.curriculum import CurriculumPlugin
from .plugins.gpu_stats import GPU4090Plugin, GPU3090Plugin
from .plugins.adversarial import AdversarialPlugin
from .plugins.checkpoints import CheckpointSyncPlugin
from .plugins.regression import RegressionPlugin
from .plugins.model_comparison import ModelComparisonPlugin
from .plugins.confidence import ConfidencePlugin
from .plugins.testing import TestingPlugin
from .plugins.self_correction import SelfCorrectionPlugin
from .plugins.skill_metrics import SkillMetricsPlugin
from .plugins.training_analytics import TrainingAnalyticsPlugin
from .plugins.system_health import SystemHealthPlugin
from .plugins.retention import RetentionPlugin
from .plugins.scheduler import SchedulerPlugin

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Aggregates data from multiple plugins into a unified response.

    Provides:
    - /api/unified - All plugin data in one response
    - /api/health - Health status of all plugins
    - Caching - 5-minute cache across all plugins
    - Error handling - Graceful degradation when plugins fail
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize aggregator with plugins.

        Args:
            config: Configuration dict with plugin-specific configs
        """
        self.config = config or {}
        self.registry = PluginRegistry()

        # Register default plugins
        self._register_default_plugins()

    def _register_default_plugins(self):
        """Register the default set of plugins"""
        try:
            # Training status (4090, local, critical)
            self.registry.register(TrainingStatusPlugin(
                config=self.config.get('training_status', {})
            ))

            # Curriculum optimization (3090, remote)
            self.registry.register(CurriculumPlugin(
                config=self.config.get('curriculum', {})
            ))

            # GPU stats (both machines)
            self.registry.register(GPU4090Plugin(
                config=self.config.get('gpu_4090', {})
            ))

            self.registry.register(GPU3090Plugin(
                config=self.config.get('gpu_3090', {})
            ))

            # Phase 4: Intelligence system plugins (all 3090 remote)
            self.registry.register(AdversarialPlugin(
                config=self.config.get('adversarial', {})
            ))

            self.registry.register(CheckpointSyncPlugin(
                config=self.config.get('checkpoint_sync', {})
            ))

            self.registry.register(RegressionPlugin(
                config=self.config.get('regression', {})
            ))

            self.registry.register(ModelComparisonPlugin(
                config=self.config.get('model_comparison', {})
            ))

            self.registry.register(ConfidencePlugin(
                config=self.config.get('confidence', {})
            ))

            self.registry.register(TestingPlugin(
                config=self.config.get('testing', {})
            ))

            self.registry.register(SelfCorrectionPlugin(
                config=self.config.get('self_correction', {})
            ))

            # Skill metrics (local + remote baselines)
            self.registry.register(SkillMetricsPlugin(
                config=self.config.get('skill_metrics', {})
            ))

            # Training analytics (layer drift, parameter stability) - from 3090
            self.registry.register(TrainingAnalyticsPlugin(
                config=self.config.get('training_analytics', {})
            ))

            # System health (meta-monitor across all machines)
            self.registry.register(SystemHealthPlugin(
                config=self.config.get('system_health', {})
            ))

            # Retention status (checkpoint/snapshot storage)
            self.registry.register(RetentionPlugin(
                config=self.config.get('retention', {})
            ))

            # GPU Task Scheduler (3090 port 8766)
            self.registry.register(SchedulerPlugin(
                config=self.config.get('scheduler', {})
            ))

            logger.info(f"Registered {len(self.registry.plugins)} default plugins")

        except Exception as e:
            logger.error(f"Error registering default plugins: {e}")

    def get_unified_data(self) -> Dict[str, Any]:
        """
        Get unified data from all plugins.

        Returns:
            Dict with keys:
            - timestamp: When this response was generated
            - sources: Dict mapping plugin names to their data/errors
            - summary: High-level summary of system state
        """
        # Fetch from all plugins
        plugin_results = self.registry.fetch_all()

        # Build response
        response = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'summary': self._build_summary(plugin_results)
        }

        # Include each plugin's data
        for name, result in plugin_results.items():
            if result['success']:
                response['sources'][name] = {
                    'status': 'ok',
                    'data': result['data'],
                    'cached': result.get('cached', False),
                    'fetched_at': result['timestamp']
                }
            else:
                response['sources'][name] = {
                    'status': 'error',
                    'error': result.get('error', 'Unknown error'),
                    'stale_data': result.get('data') if result.get('stale') else None,
                    'cached': result.get('cached', False),
                    'fetched_at': result.get('timestamp')
                }

        return response

    def _build_summary(self, plugin_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Build high-level summary from plugin results.

        Args:
            plugin_results: Raw plugin fetch results

        Returns:
            Summary dict with key metrics
        """
        summary = {
            'system_status': 'unknown',
            'plugins_total': len(plugin_results),
            'plugins_healthy': 0,
            'plugins_failed': 0,
            'training': {},
            'intelligence': {},
            'hardware': {}
        }

        # Count healthy plugins
        for result in plugin_results.values():
            if result['success']:
                summary['plugins_healthy'] += 1
            else:
                summary['plugins_failed'] += 1

        # Extract training summary
        if 'training_status' in plugin_results:
            ts_result = plugin_results['training_status']
            if ts_result['success']:
                data = ts_result['data']
                summary['training'] = {
                    'status': data.get('status'),
                    'progress_percent': data.get('progress_percent', 0),
                    'current_step': data.get('current_step'),
                    'total_steps': data.get('total_steps'),
                    'loss': data.get('loss'),
                    'accuracy_percent': data.get('accuracy_percent', 0)
                }

        # Extract curriculum summary
        if 'curriculum_optimization' in plugin_results:
            curr_result = plugin_results['curriculum_optimization']
            if curr_result['success'] and 'latest_summary' in curr_result['data']:
                latest = curr_result['data']['latest_summary']
                summary['intelligence']['curriculum'] = {
                    'step': latest.get('step'),
                    'accuracies': latest.get('accuracies', {})
                }

        # Extract GPU summary
        for gpu_name in ['gpu_4090', 'gpu_3090']:
            if gpu_name in plugin_results:
                gpu_result = plugin_results[gpu_name]
                if gpu_result['success']:
                    data = gpu_result['data']
                    machine = '4090' if '4090' in gpu_name else '3090'
                    summary['hardware'][machine] = {
                        'vram_used_gb': data.get('vram_used_gb'),
                        'vram_total_gb': data.get('vram_total_gb'),
                        'vram_percent': data.get('vram_percent'),
                        'utilization_percent': data.get('utilization_percent'),
                        'temperature_c': data.get('temperature_c')
                    }

        # Determine overall system status
        critical_plugins = ['training_status', 'gpu_4090']
        critical_healthy = all(
            plugin_results.get(name, {}).get('success', False)
            for name in critical_plugins
        )

        if critical_healthy:
            if summary['plugins_failed'] == 0:
                summary['system_status'] = 'healthy'
            else:
                summary['system_status'] = 'degraded'
        else:
            summary['system_status'] = 'critical'

        return summary

    def get_health(self) -> Dict[str, Any]:
        """
        Get health status of all plugins.

        Returns:
            Dict with health information for each plugin
        """
        health_data = self.registry.get_health_all()

        # Add overall summary
        total = len(health_data)
        healthy = sum(1 for h in health_data.values() if h['healthy'])

        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy' if healthy == total else 'degraded',
            'plugins_total': total,
            'plugins_healthy': healthy,
            'plugins_unhealthy': total - healthy,
            'plugins': health_data
        }

    def clear_caches(self):
        """Clear all plugin caches"""
        self.registry.clear_all_caches()
        logger.info("Cleared all caches via aggregator")

    def register_plugin(self, plugin):
        """Add a new plugin to the aggregator"""
        self.registry.register(plugin)
