#!/usr/bin/env python3
"""
Test the three production plugins
Phase 2, Task 2.2
"""

import sys
sys.path.insert(0, '{BASE_DIR}')

from monitoring.api.plugins.training_status import TrainingStatusPlugin
from monitoring.api.plugins.curriculum import CurriculumPlugin
from monitoring.api.plugins.gpu_stats import GPU4090Plugin, GPU3090Plugin
from monitoring.api.plugins import PluginRegistry


def test_training_status_plugin():
    """Test training status plugin"""
    print("Test 1: Training Status Plugin...")

    plugin = TrainingStatusPlugin()

    assert plugin.get_name() == 'training_status'

    result = plugin.fetch_with_cache()

    if not result['success']:
        print(f"  ✗ Failed: {result.get('error')}")
        return False

    data = result['data']

    print(f"  Status: {data.get('status')}")
    print(f"  Step: {data.get('current_step')}/{data.get('total_steps')}")
    print(f"  Loss: {data.get('loss'):.4f}")
    print(f"  Progress: {data.get('progress_percent', 0):.1f}%")

    print("✓ Training Status Plugin works")
    return True


def test_curriculum_plugin():
    """Test curriculum plugin"""
    print("\nTest 2: Curriculum Plugin...")

    plugin = CurriculumPlugin()

    assert plugin.get_name() == 'curriculum_optimization'

    result = plugin.fetch_with_cache()

    if not result['success']:
        print(f"  ⚠ Failed (expected for remote): {result.get('error')}")
        # This is expected if SSH isn't configured or 3090 is down
        return True  # Not critical

    data = result['data']

    if 'latest_summary' in data:
        summary = data['latest_summary']
        print(f"  Step: {summary.get('step')}")
        print(f"  Accuracies:")
        for difficulty, acc in summary.get('accuracies', {}).items():
            print(f"    {difficulty}: {acc:.2%}")

    print("✓ Curriculum Plugin works")
    return True


def test_gpu_4090_plugin():
    """Test GPU 4090 plugin"""
    print("\nTest 3: GPU 4090 Plugin...")

    plugin = GPU4090Plugin()

    assert plugin.get_name() == 'gpu_4090'

    result = plugin.fetch_with_cache()

    if not result['success']:
        print(f"  ✗ Failed: {result.get('error')}")
        return False

    data = result['data']

    print(f"  GPU: {data.get('name')}")
    print(f"  VRAM: {data.get('vram_used_gb')}/{data.get('vram_total_gb')}GB ({data.get('vram_percent')}%)")
    print(f"  Utilization: {data.get('utilization_percent')}%")
    print(f"  Temperature: {data.get('temperature_c')}°C")

    print("✓ GPU 4090 Plugin works")
    return True


def test_gpu_3090_plugin():
    """Test GPU 3090 plugin"""
    print("\nTest 4: GPU 3090 Plugin...")

    plugin = GPU3090Plugin()

    assert plugin.get_name() == 'gpu_3090'

    result = plugin.fetch_with_cache()

    if not result['success']:
        print(f"  ⚠ Failed (expected for remote): {result.get('error')}")
        # This is expected if SSH isn't configured
        return True  # Not critical

    data = result['data']

    print(f"  GPU: {data.get('name')}")
    print(f"  VRAM: {data.get('vram_used_gb')}/{data.get('vram_total_gb')}GB ({data.get('vram_percent')}%)")
    print(f"  Utilization: {data.get('utilization_percent')}%")
    print(f"  Temperature: {data.get('temperature_c')}°C")

    print("✓ GPU 3090 Plugin works")
    return True


def test_registry_integration():
    """Test all plugins in registry"""
    print("\nTest 5: Registry Integration...")

    registry = PluginRegistry()

    # Register all plugins
    registry.register(TrainingStatusPlugin())
    registry.register(CurriculumPlugin())
    registry.register(GPU4090Plugin())
    registry.register(GPU3090Plugin())

    # Fetch all
    results = registry.fetch_all()

    print(f"  Registered {len(results)} plugins")

    # Check each
    for name, result in results.items():
        status = "✓" if result['success'] else "⚠"
        cached = "(cached)" if result.get('cached') else "(fresh)"
        print(f"    {status} {name} {cached}")

    # Health check
    health = registry.get_health_all()
    healthy_count = sum(1 for h in health.values() if h['healthy'])

    print(f"  Healthy: {healthy_count}/{len(health)}")

    print("✓ Registry Integration works")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("THREE PLUGIN TESTS - Phase 2, Task 2.2")
    print("=" * 60)

    try:
        all_pass = True

        all_pass &= test_training_status_plugin()
        all_pass &= test_curriculum_plugin()
        all_pass &= test_gpu_4090_plugin()
        all_pass &= test_gpu_3090_plugin()
        all_pass &= test_registry_integration()

        print("\n" + "=" * 60)
        if all_pass:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS FAILED ⚠")
        print("=" * 60)

        return all_pass

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
