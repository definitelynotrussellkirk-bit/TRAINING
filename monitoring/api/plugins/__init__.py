#!/usr/bin/env python3
"""
Plugin Loader and Registry
Phase 2, Task 2.1: Dynamic plugin discovery and management
"""

from typing import Dict, List, Type
import importlib
import logging
import os
from pathlib import Path

from .base import BasePlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for all data source plugins.

    Discovers plugins dynamically by scanning the plugins/ directory
    for Python files that define Plugin classes.
    """

    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self._plugin_classes: Dict[str, Type[BasePlugin]] = {}

    def register(self, plugin: BasePlugin):
        """
        Register a plugin instance.

        Args:
            plugin: Initialized plugin instance
        """
        name = plugin.get_name()
        if name in self.plugins:
            logger.warning(f"Plugin '{name}' already registered, replacing")

        self.plugins[name] = plugin
        logger.info(f"Registered plugin: {name}")

    def unregister(self, name: str):
        """Unregister a plugin by name"""
        if name in self.plugins:
            del self.plugins[name]
            logger.info(f"Unregistered plugin: {name}")

    def get(self, name: str) -> BasePlugin:
        """
        Get plugin by name.

        Args:
            name: Plugin identifier

        Returns:
            Plugin instance

        Raises:
            KeyError: If plugin not found
        """
        if name not in self.plugins:
            raise KeyError(f"Plugin '{name}' not found")
        return self.plugins[name]

    def get_all(self) -> Dict[str, BasePlugin]:
        """Get all registered plugins"""
        return self.plugins.copy()

    def fetch_all(self) -> Dict[str, Dict]:
        """
        Fetch data from all plugins.

        Returns:
            Dict mapping plugin names to fetch results
        """
        results = {}

        for name, plugin in self.plugins.items():
            try:
                results[name] = plugin.fetch_with_cache()
            except Exception as e:
                logger.error(f"Error fetching from plugin '{name}': {e}")
                results[name] = {
                    'success': False,
                    'error': str(e),
                    'plugin': name
                }

        return results

    def get_health_all(self) -> Dict[str, Dict]:
        """
        Get health status of all plugins.

        Returns:
            Dict mapping plugin names to health info
        """
        return {
            name: plugin.get_health()
            for name, plugin in self.plugins.items()
        }

    def clear_all_caches(self):
        """Clear cache for all plugins"""
        for plugin in self.plugins.values():
            plugin.clear_cache()
        logger.info("Cleared all plugin caches")

    def discover_plugins(self, plugins_dir: str = None):
        """
        Auto-discover plugins by scanning plugins/ directory.

        Looks for Python files containing classes that inherit from BasePlugin.

        Args:
            plugins_dir: Directory to scan (defaults to this file's directory)
        """
        if plugins_dir is None:
            plugins_dir = os.path.dirname(__file__)

        plugins_path = Path(plugins_dir)
        logger.info(f"Discovering plugins in: {plugins_path}")

        discovered_count = 0

        # Scan for .py files
        for py_file in plugins_path.glob("*.py"):
            if py_file.name.startswith('_'):
                continue  # Skip __init__.py, etc.

            module_name = py_file.stem

            try:
                # Import module
                module = importlib.import_module(f'.{module_name}', package='monitoring.api.plugins')

                # Look for Plugin classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if it's a Plugin class (not BasePlugin itself)
                    if (isinstance(attr, type) and
                        issubclass(attr, BasePlugin) and
                        attr is not BasePlugin and
                        not attr_name.startswith('_')):

                        self._plugin_classes[attr_name] = attr
                        discovered_count += 1
                        logger.info(f"Discovered plugin class: {attr_name} in {module_name}.py")

            except Exception as e:
                logger.error(f"Error loading plugin module {module_name}: {e}")

        logger.info(f"Discovered {discovered_count} plugin classes")

    def auto_instantiate(self, config: Dict = None):
        """
        Auto-instantiate all discovered plugin classes.

        Args:
            config: Configuration dict with plugin-specific configs
        """
        config = config or {}

        for class_name, plugin_class in self._plugin_classes.items():
            try:
                # Get plugin-specific config
                plugin_config = config.get(class_name, {})

                # Instantiate
                plugin = plugin_class(config=plugin_config)

                # Register
                self.register(plugin)

            except Exception as e:
                logger.error(f"Error instantiating plugin {class_name}: {e}")


# Global registry instance
_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry"""
    return _registry


def register_plugin(plugin: BasePlugin):
    """Convenience function to register a plugin"""
    _registry.register(plugin)


def get_plugin(name: str) -> BasePlugin:
    """Convenience function to get a plugin by name"""
    return _registry.get(name)


def fetch_all_plugins() -> Dict[str, Dict]:
    """Convenience function to fetch from all plugins"""
    return _registry.fetch_all()


def get_health_all() -> Dict[str, Dict]:
    """Convenience function to get health of all plugins"""
    return _registry.get_health_all()


__all__ = [
    'BasePlugin',
    'PluginRegistry',
    'get_registry',
    'register_plugin',
    'get_plugin',
    'fetch_all_plugins',
    'get_health_all'
]
