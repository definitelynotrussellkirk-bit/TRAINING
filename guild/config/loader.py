"""YAML configuration loader with environment variable support."""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Optional, TypeVar, Type
from dataclasses import fields, is_dataclass
from enum import Enum

T = TypeVar('T')

_config_dir: Optional[Path] = None


def set_config_dir(path: str | Path):
    """Set the global config directory."""
    global _config_dir
    _config_dir = Path(path)


def get_config_dir() -> Path:
    """Get the config directory."""
    global _config_dir
    if _config_dir is None:
        env_path = os.environ.get("GUILD_CONFIG_DIR")
        if env_path:
            _config_dir = Path(env_path)
        else:
            # Default: configs/ relative to project root
            _config_dir = Path(__file__).parent.parent.parent / "configs"
    return _config_dir


def get_config_path(category: str, name: str, ext: str = ".yaml") -> Path:
    """Get path to a specific config file."""
    return get_config_dir() / category / f"{name}{ext}"


def expand_env_vars(value: Any) -> Any:
    """
    Recursively expand environment variables in strings.

    Supports:
    - ${VAR} - required variable
    - ${VAR:-default} - variable with default
    """
    if isinstance(value, str):
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            # Return original if no value and no default
            return match.group(0)

        return re.sub(pattern, replacer, value)

    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [expand_env_vars(v) for v in value]

    return value


def load_yaml(path: Path | str) -> dict:
    """Load a YAML file with environment variable expansion."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return expand_env_vars(data)


def dict_to_dataclass(data: dict, cls: Type[T]) -> T:
    """Convert a dict to a dataclass instance."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    field_info = {f.name: f for f in fields(cls)}
    filtered = {}

    for key, value in data.items():
        if key not in field_info:
            continue

        field_type = field_info[key].type

        # Handle Enum conversion
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            if isinstance(value, str):
                value = field_type(value)

        # Handle nested dataclass
        elif is_dataclass(field_type) and isinstance(value, dict):
            value = dict_to_dataclass(value, field_type)

        filtered[key] = value

    return cls(**filtered)


def load_config(category: str, name: str, cls: Optional[Type[T]] = None) -> T | dict:
    """Load a config file and optionally convert to dataclass."""
    path = get_config_path(category, name)
    data = load_yaml(path)

    if cls is not None:
        return dict_to_dataclass(data, cls)
    return data


def load_all_configs(category: str, cls: Optional[Type[T]] = None,
                     pattern: str = "*.yaml") -> dict[str, T | dict]:
    """Load all config files in a category."""
    category_dir = get_config_dir() / category
    if not category_dir.exists():
        return {}

    configs = {}
    for path in category_dir.glob(pattern):
        if path.name.startswith("_"):
            continue
        name = path.stem
        try:
            configs[name] = load_config(category, name, cls)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    return configs


class ConfigLoader:
    """Manages loading and caching of configurations."""

    def __init__(self, config_dir: Optional[Path | str] = None):
        self.config_dir = Path(config_dir) if config_dir else get_config_dir()
        self._cache: dict[str, Any] = {}

    def load(self, category: str, name: str, cls: Optional[Type[T]] = None,
             use_cache: bool = True) -> T | dict:
        """Load a config with optional caching."""
        cache_key = f"{category}/{name}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        path = self.config_dir / category / f"{name}.yaml"
        data = load_yaml(path)

        result = dict_to_dataclass(data, cls) if cls else data

        if use_cache:
            self._cache[cache_key] = result

        return result

    def load_all(self, category: str, cls: Optional[Type[T]] = None) -> dict[str, T | dict]:
        """Load all configs in a category."""
        category_dir = self.config_dir / category
        if not category_dir.exists():
            return {}

        configs = {}
        for path in category_dir.glob("*.yaml"):
            if path.name.startswith("_"):
                continue
            name = path.stem
            try:
                configs[name] = self.load(category, name, cls)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        return configs

    def clear_cache(self):
        """Clear the config cache."""
        self._cache.clear()
