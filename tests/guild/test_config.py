"""Tests for configuration loading."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import tempfile
import os

from guild.config.loader import (
    load_yaml, expand_env_vars, dict_to_dataclass,
    load_config, ConfigLoader, set_config_dir, get_config_dir
)
from guild.skills.types import SkillConfig, SkillCategory


class TestEnvVarExpansion:
    def test_simple_var(self):
        os.environ["TEST_VAR"] = "hello"
        result = expand_env_vars("${TEST_VAR}")
        assert result == "hello"

    def test_var_with_default(self):
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]
        result = expand_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"

    def test_nested_dict(self):
        os.environ["TEST_PATH"] = "/custom/path"
        data = {
            "base": "${TEST_PATH}",
            "nested": {"path": "${TEST_PATH}/subdir"}
        }
        result = expand_env_vars(data)
        assert result["base"] == "/custom/path"
        assert result["nested"]["path"] == "/custom/path/subdir"

    def test_list_expansion(self):
        os.environ["TEST_ITEM"] = "expanded"
        data = ["${TEST_ITEM}", "static"]
        result = expand_env_vars(data)
        assert result == ["expanded", "static"]

    def test_unset_var_preserved(self):
        if "DEFINITELY_UNSET_VAR" in os.environ:
            del os.environ["DEFINITELY_UNSET_VAR"]
        result = expand_env_vars("${DEFINITELY_UNSET_VAR}")
        assert result == "${DEFINITELY_UNSET_VAR}"

    def test_non_string_passthrough(self):
        assert expand_env_vars(42) == 42
        assert expand_env_vars(3.14) == 3.14
        assert expand_env_vars(None) is None


class TestDictToDataclass:
    def test_simple_conversion(self):
        data = {
            "id": "test_skill",
            "name": "Test Skill",
            "description": "A test",
            "category": "reasoning"
        }
        result = dict_to_dataclass(data, SkillConfig)
        assert result.id == "test_skill"
        assert result.category == SkillCategory.REASONING

    def test_extra_fields_ignored(self):
        data = {
            "id": "test",
            "name": "Test",
            "description": "Test",
            "category": "reasoning",
            "unknown_field": "ignored"
        }
        result = dict_to_dataclass(data, SkillConfig)
        assert result.id == "test"

    def test_not_dataclass_raises(self):
        with pytest.raises(TypeError):
            dict_to_dataclass({"key": "value"}, str)


class TestConfigLoader:
    @pytest.fixture
    def temp_config_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            skill_yaml = """
id: test_skill
name: Test Skill
description: A test skill
category: reasoning
tags:
  - test
metrics:
  - accuracy
primary_metric: accuracy
accuracy_thresholds:
  1: 0.6
  2: 0.7
"""
            (skills_dir / "test_skill.yaml").write_text(skill_yaml)

            old_dir = get_config_dir()
            set_config_dir(tmpdir)
            yield tmpdir
            set_config_dir(old_dir)

    def test_load_skill_config(self, temp_config_dir):
        config = load_config("skills", "test_skill", SkillConfig)
        assert config.id == "test_skill"
        assert config.name == "Test Skill"
        assert config.category == SkillCategory.REASONING
        assert config.get_threshold(1) == 0.6

    def test_config_loader_caching(self, temp_config_dir):
        loader = ConfigLoader(temp_config_dir)

        config1 = loader.load("skills", "test_skill", SkillConfig)
        config2 = loader.load("skills", "test_skill", SkillConfig)

        assert config1 is config2  # Same object due to caching

        loader.clear_cache()
        config3 = loader.load("skills", "test_skill", SkillConfig)
        assert config3 is not config1

    def test_load_raw_dict(self, temp_config_dir):
        data = load_config("skills", "test_skill")
        assert isinstance(data, dict)
        assert data["id"] == "test_skill"

    def test_load_nonexistent_raises(self, temp_config_dir):
        with pytest.raises(FileNotFoundError):
            load_config("skills", "nonexistent")


class TestLoadYaml:
    @pytest.fixture
    def temp_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key: value\nlist:\n  - a\n  - b\n")
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_load_yaml_file(self, temp_yaml_file):
        data = load_yaml(temp_yaml_file)
        assert data["key"] == "value"
        assert data["list"] == ["a", "b"]

    def test_load_yaml_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/path.yaml")


class TestRealConfigs:
    """Test loading actual config files from the project."""

    def test_load_logic_weaving_skill(self):
        # Use the actual project configs directory
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "configs"

        if not (config_dir / "skills" / "logic_weaving.yaml").exists():
            pytest.skip("logic_weaving.yaml not found")

        loader = ConfigLoader(config_dir)
        config = loader.load("skills", "logic_weaving", SkillConfig)

        assert config.id == "logic_weaving"
        assert config.category == SkillCategory.REASONING
        assert "syllo" in config.tags

    def test_load_facilities_example(self):
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "configs"

        if not (config_dir / "facilities" / "example.yaml").exists():
            pytest.skip("example.yaml not found")

        loader = ConfigLoader(config_dir)
        data = loader.load("facilities", "example")

        assert "facilities" in data
        assert "main_hub" in data["facilities"]
        assert "training_gpu" in data["facilities"]
