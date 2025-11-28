#!/usr/bin/env python3
"""
Test script for plugin system
Phase 2, Task 2.1: Verify base plugin system works
"""

import sys
sys.path.insert(0, '{BASE_DIR}')

from monitoring.api.plugins.base import BasePlugin, LocalFilePlugin, PluginError
from monitoring.api.plugins import PluginRegistry


class DummyPlugin(BasePlugin):
    """Test plugin that returns fixed data"""

    def __init__(self, config=None):
        super().__init__(config)
        self.call_count = 0

    def fetch(self):
        self.call_count += 1
        return {'message': 'Hello from dummy plugin', 'call_count': self.call_count}

    def get_name(self):
        return 'dummy'

    def get_metadata(self):
        return {
            'description': 'Test dummy plugin',
            'refresh_interval': 60,
            'critical': False,
            'data_type': 'test'
        }


def test_basic_plugin():
    """Test basic plugin functionality"""
    print("Test 1: Basic Plugin...")

    plugin = DummyPlugin()
    assert plugin.get_name() == 'dummy', "Plugin name incorrect"

    # Fetch data
    data = plugin.fetch()
    assert data['message'] == 'Hello from dummy plugin', "Data mismatch"
    assert data['call_count'] == 1, "Call count wrong"

    print("✓ Basic plugin works")


def test_caching():
    """Test plugin caching"""
    print("\nTest 2: Caching...")

    plugin = DummyPlugin(config={'cache_duration': 5})

    # First fetch (cache miss)
    result1 = plugin.fetch_with_cache()
    assert result1['success'], "First fetch failed"
    assert not result1['cached'], "First fetch should not be cached"
    assert result1['data']['call_count'] == 1, "Call count wrong"

    # Second fetch (cache hit)
    result2 = plugin.fetch_with_cache()
    assert result2['success'], "Second fetch failed"
    assert result2['cached'], "Second fetch should be cached"
    assert result2['data']['call_count'] == 1, "Call count shouldn't increase (cached)"

    # Clear cache
    plugin.clear_cache()

    # Third fetch (cache miss again)
    result3 = plugin.fetch_with_cache()
    assert not result3['cached'], "Third fetch should not be cached"
    assert result3['data']['call_count'] == 2, "Call count should increase after cache clear"

    print("✓ Caching works")


def test_error_handling():
    """Test plugin error handling"""
    print("\nTest 3: Error Handling...")

    class FailingPlugin(DummyPlugin):
        def fetch(self):
            raise PluginError("Intentional failure")

    plugin = FailingPlugin()

    # Fetch should fail
    result = plugin.fetch_with_cache()
    assert not result['success'], "Should fail"
    assert 'error' in result, "Should have error message"

    # Check health
    health = plugin.get_health()
    assert not health['healthy'], "Should be unhealthy"
    assert health['error_count'] > 0, "Should have error count"

    print("✓ Error handling works")


def test_registry():
    """Test plugin registry"""
    print("\nTest 4: Plugin Registry...")

    registry = PluginRegistry()

    # Register plugins
    plugin1 = DummyPlugin()
    plugin2 = DummyPlugin()
    plugin2.get_name = lambda: 'dummy2'  # Override name

    registry.register(plugin1)
    registry.register(plugin2)

    # Get plugin
    fetched = registry.get('dummy')
    assert fetched is plugin1, "Retrieved wrong plugin"

    # Get all
    all_plugins = registry.get_all()
    assert len(all_plugins) == 2, f"Should have 2 plugins, got {len(all_plugins)}"

    # Fetch all
    results = registry.fetch_all()
    assert len(results) == 2, "Should fetch from 2 plugins"
    assert results['dummy']['success'], "dummy should succeed"
    assert results['dummy2']['success'], "dummy2 should succeed"

    # Health check all
    health = registry.get_health_all()
    assert len(health) == 2, "Should have 2 health reports"

    print("✓ Registry works")


def test_local_file_plugin():
    """Test LocalFilePlugin with actual training_status.json"""
    print("\nTest 5: LocalFilePlugin...")

    class TrainingStatusPlugin(LocalFilePlugin):
        def __init__(self):
            super().__init__('{BASE_DIR}/status/training_status.json')

        def get_name(self):
            return 'training_status'

        def get_metadata(self):
            return {
                'description': 'Real training status file',
                'refresh_interval': 5,
                'critical': True,
                'data_type': 'training'
            }

    plugin = TrainingStatusPlugin()

    # Fetch
    result = plugin.fetch_with_cache()
    assert result['success'], f"Fetch failed: {result.get('error')}"
    assert 'data' in result, "Should have data"
    assert 'status' in result['data'], "Should have status field"

    print(f"  Status: {result['data'].get('status')}")
    print(f"  Step: {result['data'].get('current_step')}")
    print("✓ LocalFilePlugin works with real data")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("PLUGIN SYSTEM TESTS - Phase 2, Task 2.1")
    print("=" * 60)

    try:
        test_basic_plugin()
        test_caching()
        test_error_handling()
        test_registry()
        test_local_file_plugin()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
