"""Tests for facility resolution."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import tempfile
import os

from guild.facilities.resolver import PathResolver, init_resolver, reset_resolver
from guild.facilities.types import Facility, FacilityType


class TestPathResolver:
    @pytest.fixture
    def temp_config(self):
        config = """
facilities:
  test_arena:
    id: test_arena
    name: Test Arena
    type: battlefield
    base_path: /tmp/test_training
    paths:
      checkpoints: checkpoints/
      logs: logs/

  test_hub:
    id: test_hub
    name: Test Hub
    type: hub
    base_path: /tmp/test_hub
    paths:
      status: status/

default_facility: test_arena
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            f.flush()  # Ensure content is written to disk
            yield f.name
        os.unlink(f.name)

    def test_resolve_facility_path(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("facility:test_arena:checkpoints")
        assert path == Path("/tmp/test_training/checkpoints/")

    def test_resolve_facility_with_subpath(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("facility:test_arena:checkpoints/step-1000")
        assert path == Path("/tmp/test_training/checkpoints/step-1000")

    def test_resolve_shorthand(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("@checkpoints")
        assert path == Path("/tmp/test_training/checkpoints/")

    def test_resolve_shorthand_with_subpath(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("@checkpoints/step-1000")
        assert path == Path("/tmp/test_training/checkpoints/step-1000")

    def test_resolve_with_current_facility(self, temp_config):
        resolver = PathResolver(temp_config)
        resolver.set_current_facility("test_hub")

        path = resolver.resolve("@status")
        assert path == Path("/tmp/test_hub/status/")

    def test_resolve_regular_path(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("/absolute/path")
        assert path == Path("/absolute/path")

    def test_resolve_home_path(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("~/some/path")
        assert str(path).startswith(os.path.expanduser("~"))

    def test_resolve_env_var(self):
        os.environ["TEST_BASE"] = "/custom/path"

        config = """
facilities:
  env_test:
    id: env_test
    name: Env Test
    type: battlefield
    base_path: ${TEST_BASE}
    paths:
      data: data/
default_facility: env_test
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            f.flush()  # Ensure content is written to disk
            config_path = f.name

        try:
            resolver = PathResolver(config_path)
            path = resolver.resolve("@data")
            assert path == Path("/custom/path/data/")
        finally:
            os.unlink(config_path)

    def test_list_facilities(self, temp_config):
        resolver = PathResolver(temp_config)

        all_facilities = resolver.list_facilities()
        assert "test_arena" in all_facilities
        assert "test_hub" in all_facilities

        battlefields = resolver.list_facilities(FacilityType.BATTLEFIELD)
        assert battlefields == ["test_arena"]

    def test_get_facility(self, temp_config):
        resolver = PathResolver(temp_config)

        facility = resolver.get_facility("test_arena")
        assert facility.name == "Test Arena"
        assert facility.type == FacilityType.BATTLEFIELD

    def test_unknown_facility_raises(self, temp_config):
        resolver = PathResolver(temp_config)

        with pytest.raises(ValueError, match="Unknown facility"):
            resolver.resolve("facility:nonexistent:path")

    def test_set_unknown_facility_raises(self, temp_config):
        resolver = PathResolver(temp_config)

        with pytest.raises(ValueError, match="Unknown facility"):
            resolver.set_current_facility("nonexistent")

    def test_invalid_facility_path_format_raises(self, temp_config):
        resolver = PathResolver(temp_config)

        with pytest.raises(ValueError, match="Invalid facility path"):
            resolver.resolve("facility:only_two_parts")

    def test_shorthand_without_facility_raises(self):
        resolver = PathResolver()  # No config, no default facility

        with pytest.raises(ValueError, match="No current or default facility"):
            resolver.resolve("@checkpoints")

    def test_add_facility_directly(self):
        resolver = PathResolver()

        facility = Facility(
            id="direct",
            name="Direct Facility",
            type=FacilityType.BATTLEFIELD,
            base_path="/direct/path",
            paths={"data": "data/"}
        )
        resolver.add_facility(facility)
        resolver.set_current_facility("direct")

        path = resolver.resolve("@data")
        assert path == Path("/direct/path/data/")

    def test_resolve_unmapped_path(self, temp_config):
        """Paths not in facility.paths should resolve relative to base."""
        resolver = PathResolver(temp_config)

        path = resolver.resolve("facility:test_arena:custom_dir")
        assert path == Path("/tmp/test_training/custom_dir")

    def test_current_facility_id_property(self, temp_config):
        resolver = PathResolver(temp_config)

        # Default facility from config
        assert resolver.current_facility_id == "test_arena"

        # After setting current
        resolver.set_current_facility("test_hub")
        assert resolver.current_facility_id == "test_hub"


class TestGlobalResolver:
    @pytest.fixture(autouse=True)
    def reset_global_resolver(self):
        """Reset global resolver before and after each test."""
        reset_resolver()
        yield
        reset_resolver()

    def test_init_resolver_with_example_config(self):
        """Test that init_resolver works with the example config."""
        from guild.facilities.resolver import init_resolver, get_resolver

        # Use the example config
        config_path = project_root / "configs" / "facilities" / "example.yaml"
        if not config_path.exists():
            pytest.skip("example.yaml not found")

        resolver = init_resolver(config_path)
        assert resolver is not None
        assert "main_hub" in resolver.list_facilities()

    def test_get_resolver_auto_initializes(self):
        """Test that get_resolver initializes when needed."""
        from guild.facilities.resolver import get_resolver

        # This should auto-initialize using example.yaml
        # (since local.yaml is gitignored)
        try:
            resolver = get_resolver()
            assert resolver is not None
        except FileNotFoundError:
            pytest.skip("No facility config available")

    def test_resolve_function(self):
        """Test the convenience resolve function."""
        from guild.facilities.resolver import resolve, init_resolver

        config_path = project_root / "configs" / "facilities" / "example.yaml"
        if not config_path.exists():
            pytest.skip("example.yaml not found")

        init_resolver(config_path)

        # Should resolve using the default facility (training_gpu)
        path = resolve("@checkpoints")
        assert "current_model" in str(path)
