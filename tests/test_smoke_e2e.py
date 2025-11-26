#!/usr/bin/env python3
"""
End-to-End Smoke Test
=====================

Verifies that key system components work together:
1. Path configuration loads correctly
2. System health aggregator runs
3. Monitoring API plugins load
4. Status files are readable
5. Key daemons are importable

Run:
    pytest tests/test_smoke_e2e.py -v
    python tests/test_smoke_e2e.py  # Direct execution
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import paths module
from core.paths import (
    get_base_dir,
    get_status_dir,
    get_models_dir,
    get_queue_dir,
    get_logs_dir,
    get_remote_api_url,
    get_scheduler_api_url,
    REMOTE_HOST,
)


class TestPathConfiguration:
    """Test that path configuration works correctly"""

    def test_base_dir_exists(self):
        """Base directory should exist"""
        base = get_base_dir()
        assert base.exists(), f"Base dir does not exist: {base}"

    def test_claude_md_exists(self):
        """CLAUDE.md should exist in base dir"""
        base = get_base_dir()
        claude_md = base / "CLAUDE.md"
        assert claude_md.exists(), "CLAUDE.md not found"

    def test_status_dir_exists(self):
        """Status directory should exist or be creatable"""
        status = get_status_dir()
        status.mkdir(parents=True, exist_ok=True)
        assert status.exists()

    def test_models_dir_exists(self):
        """Models directory should exist"""
        models = get_models_dir()
        assert models.exists(), f"Models dir does not exist: {models}"

    def test_remote_api_url_format(self):
        """Remote API URL should be valid format"""
        url = get_remote_api_url()
        assert url.startswith("http://")
        assert ":" in url

    def test_scheduler_api_url_format(self):
        """Scheduler API URL should be valid format"""
        url = get_scheduler_api_url()
        assert url.startswith("http://")
        assert "8766" in url or ":" in url


class TestSystemHealthAggregator:
    """Test system health aggregator functionality"""

    def test_import_aggregator(self):
        """Should be able to import health aggregator"""
        from monitoring.system_health_aggregator import SystemHealthAggregator
        assert SystemHealthAggregator is not None

    def test_aggregator_instantiation(self):
        """Should be able to instantiate aggregator"""
        from monitoring.system_health_aggregator import SystemHealthAggregator
        aggregator = SystemHealthAggregator()
        assert aggregator.base_dir.exists()

    def test_local_process_check_mock(self):
        """Should check local processes (mocked)"""
        from monitoring.system_health_aggregator import SystemHealthAggregator

        aggregator = SystemHealthAggregator()

        # Should return None for non-existent process
        result = aggregator.check_local_process("definitely_not_a_real_process_xyz")
        assert result is None

    def test_health_collection_local_only(self):
        """Should be able to collect health for local machine"""
        from monitoring.system_health_aggregator import SystemHealthAggregator

        aggregator = SystemHealthAggregator()

        # Mock remote checks to avoid network
        with patch.object(aggregator, 'check_machine_reachable', return_value=False):
            health = aggregator.collect_health()

        assert health.timestamp is not None
        assert health.overall_status in ['healthy', 'degraded', 'critical']
        assert '4090' in health.machines


class TestMonitoringPlugins:
    """Test monitoring API plugins load correctly"""

    def test_import_system_health_plugin(self):
        """Should import system health plugin"""
        from monitoring.api.plugins.system_health import SystemHealthPlugin
        assert SystemHealthPlugin is not None

    def test_import_retention_plugin(self):
        """Should import retention plugin"""
        from monitoring.api.plugins.retention import RetentionPlugin
        assert RetentionPlugin is not None

    def test_plugin_instantiation(self):
        """Should instantiate plugins without error"""
        from monitoring.api.plugins.system_health import SystemHealthPlugin
        from monitoring.api.plugins.retention import RetentionPlugin

        # These should instantiate without errors
        health_plugin = SystemHealthPlugin()
        retention_plugin = RetentionPlugin()

        assert health_plugin.get_name() == 'system_health'
        assert retention_plugin.get_name() == 'retention'

    def test_plugin_metadata(self):
        """Plugins should provide valid metadata"""
        from monitoring.api.plugins.system_health import SystemHealthPlugin
        from monitoring.api.plugins.retention import RetentionPlugin

        for plugin_class in [SystemHealthPlugin, RetentionPlugin]:
            plugin = plugin_class()
            metadata = plugin.get_metadata()

            assert 'description' in metadata
            assert 'refresh_interval' in metadata
            assert 'critical' in metadata
            assert isinstance(metadata['refresh_interval'], int)


class TestSelfCorrectionImpact:
    """Test self-correction impact monitor"""

    def test_import_impact_monitor(self):
        """Should import impact monitor"""
        from monitoring.self_correction_impact import SelfCorrectionImpactMonitor
        assert SelfCorrectionImpactMonitor is not None

    def test_impact_monitor_instantiation(self):
        """Should instantiate monitor using path helpers"""
        from monitoring.self_correction_impact import SelfCorrectionImpactMonitor

        monitor = SelfCorrectionImpactMonitor()

        # Should use auto-detected paths
        assert monitor.base_dir.exists()
        assert "192.168.x.x" in monitor.api_url or "localhost" in monitor.api_url

    def test_effectiveness_summary_empty(self):
        """Should handle empty effectiveness data"""
        from monitoring.self_correction_impact import SelfCorrectionImpactMonitor

        monitor = SelfCorrectionImpactMonitor()
        summary = monitor.get_effectiveness_summary()

        assert 'total_batches_assessed' in summary
        assert 'verdict' in summary


class TestStatusFileParsing:
    """Test that status files are parseable"""

    def test_training_status_parseable(self):
        """Training status file should be valid JSON if it exists"""
        status_file = get_status_dir() / "training_status.json"
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_system_health_parseable(self):
        """System health file should be valid JSON if it exists"""
        status_file = get_status_dir() / "system_health.json"
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
            assert 'overall_status' in data
            assert 'machines' in data

    def test_model_comparisons_parseable(self):
        """Model comparisons file should be valid JSON if it exists"""
        status_file = get_status_dir() / "model_comparisons.json"
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
            assert isinstance(data, dict)


class TestKeyModulesImportable:
    """Test that key modules can be imported without errors"""

    def test_import_training_daemon(self):
        """Training daemon should be importable"""
        try:
            from core.training_daemon import TrainingDaemon
            assert TrainingDaemon is not None
        except ImportError as e:
            # May fail due to optional dependencies
            pytest.skip(f"Optional dependency missing: {e}")

    def test_import_model_comparison(self):
        """Model comparison engine should be importable"""
        try:
            from monitoring.model_comparison_engine import ModelComparisonEngine
            assert ModelComparisonEngine is not None
        except ImportError as e:
            pytest.skip(f"Optional dependency missing: {e}")

    def test_import_self_correction_loop(self):
        """Self-correction loop should be importable"""
        from monitoring.self_correction_loop import SelfCorrectionLoop
        assert SelfCorrectionLoop is not None

    def test_import_retention_manager(self):
        """Retention manager should be importable"""
        from management.retention_manager import RetentionManager
        assert RetentionManager is not None


class TestIntegration:
    """Integration tests that verify components work together"""

    def test_aggregator_writes_status(self):
        """Health aggregator should write valid status file"""
        from monitoring.system_health_aggregator import SystemHealthAggregator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal directory structure
            status_dir = Path(tmpdir) / "status"
            status_dir.mkdir()

            aggregator = SystemHealthAggregator(base_dir=tmpdir)

            # Mock remote checks
            with patch.object(aggregator, 'check_machine_reachable', return_value=False):
                with patch.object(aggregator, 'check_local_gpu', return_value={'available': False}):
                    health = aggregator.collect_health()
                    aggregator.save_health(health)

            # Verify file was written and is valid
            status_file = Path(tmpdir) / "status" / "system_health.json"
            assert status_file.exists()

            with open(status_file) as f:
                data = json.load(f)

            assert 'overall_status' in data
            assert 'machines' in data
            assert 'processes' in data

    def test_paths_consistent(self):
        """All path helpers should return paths under base_dir"""
        base = get_base_dir()

        paths_to_check = [
            get_status_dir(),
            get_models_dir(),
            get_queue_dir(),
            get_logs_dir(),
        ]

        for path in paths_to_check:
            # Path should be relative to or under base
            try:
                path.relative_to(base)
            except ValueError:
                pytest.fail(f"Path {path} is not under base {base}")


def run_smoke_tests():
    """Run all smoke tests and report results"""
    print("=" * 70)
    print("END-TO-END SMOKE TEST")
    print("=" * 70)

    results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }

    test_classes = [
        TestPathConfiguration,
        TestSystemHealthAggregator,
        TestMonitoringPlugins,
        TestSelfCorrectionImpact,
        TestStatusFileParsing,
        TestKeyModulesImportable,
        TestIntegration,
    ]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                try:
                    method()
                    results['passed'] += 1
                    print(f"  \u2705 {method_name}")
                except pytest.skip.Exception as e:
                    results['skipped'] += 1
                    print(f"  \u23ed\ufe0f  {method_name} (skipped: {e})")
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append((method_name, str(e)))
                    print(f"  \u274c {method_name}: {e}")

    print("\n" + "=" * 70)
    print(f"RESULTS: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped")
    print("=" * 70)

    if results['failed'] > 0:
        print("\nFailed tests:")
        for name, error in results['errors']:
            print(f"  - {name}: {error}")

    return results['failed'] == 0


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
