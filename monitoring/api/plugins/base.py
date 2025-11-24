#!/usr/bin/env python3
"""
Base Plugin System for Master Monitoring Dashboard
Phase 2, Task 2.1: Abstract base class and plugin interface

Each plugin represents a single data source (local file, remote file, API, etc.)
and is responsible for fetching, caching, and transforming that data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PluginError(Exception):
    """Raised when a plugin encounters an error"""
    pass


class BasePlugin(ABC):
    """
    Abstract base class for all data source plugins.

    Each plugin must implement:
    - fetch(): Retrieve data from source
    - get_name(): Return unique plugin identifier
    - get_metadata(): Return plugin info (refresh rate, criticality, etc.)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin with optional configuration.

        Args:
            config: Plugin-specific configuration dict
        """
        self.config = config or {}
        self.cache_duration = self.config.get('cache_duration', 300)  # 5 min default
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._error_count = 0

    @abstractmethod
    def fetch(self) -> Dict[str, Any]:
        """
        Fetch data from source (must be implemented by subclass).

        Returns:
            Dict containing the fetched data

        Raises:
            PluginError: If fetch fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return unique plugin identifier.

        Returns:
            String identifier (e.g., "training_status", "gpu_stats_4090")
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return plugin metadata.

        Returns:
            Dict with keys:
            - description: Human-readable description
            - refresh_interval: How often data updates (seconds)
            - critical: Whether system needs this data to function
            - data_type: Type of data provided
        """
        pass

    def get_cached(self) -> Optional[Dict[str, Any]]:
        """
        Get cached data if available and not expired.

        Returns:
            Cached data dict or None if expired/unavailable
        """
        if self._cache is None or self._cache_timestamp is None:
            return None

        age = (datetime.now() - self._cache_timestamp).total_seconds()

        if age > self.cache_duration:
            logger.debug(f"{self.get_name()}: Cache expired (age={age:.1f}s)")
            return None

        logger.debug(f"{self.get_name()}: Cache hit (age={age:.1f}s)")
        return self._cache

    def fetch_with_cache(self) -> Dict[str, Any]:
        """
        Fetch data with caching and error handling.

        Returns:
            Data dict with keys:
            - success: True if fetch succeeded
            - data: The fetched data (if success=True)
            - error: Error message (if success=False)
            - cached: True if data is from cache
            - timestamp: When data was fetched
            - plugin: Plugin name
        """
        # Try cache first
        cached_data = self.get_cached()
        if cached_data is not None:
            return {
                'success': True,
                'data': cached_data,
                'cached': True,
                'timestamp': self._cache_timestamp.isoformat(),
                'plugin': self.get_name()
            }

        # Fetch fresh data
        try:
            logger.info(f"{self.get_name()}: Fetching fresh data")
            data = self.fetch()

            # Update cache
            self._cache = data
            self._cache_timestamp = datetime.now()
            self._last_error = None
            self._error_count = 0

            return {
                'success': True,
                'data': data,
                'cached': False,
                'timestamp': self._cache_timestamp.isoformat(),
                'plugin': self.get_name()
            }

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error(f"{self.get_name()}: Fetch failed: {e}", exc_info=True)

            # Return cached data if available (even if expired) as fallback
            if self._cache is not None:
                logger.warning(f"{self.get_name()}: Returning stale cache due to error")
                return {
                    'success': False,
                    'data': self._cache,
                    'cached': True,
                    'stale': True,
                    'error': str(e),
                    'timestamp': self._cache_timestamp.isoformat(),
                    'plugin': self.get_name()
                }

            # No cache available
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'plugin': self.get_name()
            }

    def get_health(self) -> Dict[str, Any]:
        """
        Get plugin health status.

        Returns:
            Dict with health information
        """
        has_cache = self._cache is not None
        cache_age = None
        if self._cache_timestamp:
            cache_age = (datetime.now() - self._cache_timestamp).total_seconds()

        return {
            'name': self.get_name(),
            'healthy': self._error_count == 0,
            'has_cache': has_cache,
            'cache_age_seconds': cache_age,
            'last_error': self._last_error,
            'error_count': self._error_count,
            'metadata': self.get_metadata()
        }

    def clear_cache(self):
        """Clear cached data"""
        self._cache = None
        self._cache_timestamp = None
        logger.info(f"{self.get_name()}: Cache cleared")


class LocalFilePlugin(BasePlugin):
    """
    Base class for plugins that read local JSON files.
    Subclasses just need to specify the file path.
    """

    def __init__(self, file_path: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.file_path = file_path

    def fetch(self) -> Dict[str, Any]:
        """Read JSON file from disk"""
        import json
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise PluginError(f"File not found: {self.file_path}")
        except json.JSONDecodeError as e:
            raise PluginError(f"Invalid JSON in {self.file_path}: {e}")
        except Exception as e:
            raise PluginError(f"Error reading {self.file_path}: {e}")


class RemoteFilePlugin(BasePlugin):
    """
    Base class for plugins that read remote JSON files via SSH.
    Subclasses specify the SSH host and file path.
    """

    def __init__(self, ssh_host: str, remote_path: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ssh_host = ssh_host
        self.remote_path = remote_path

    def fetch(self) -> Dict[str, Any]:
        """Fetch JSON file via SSH"""
        import subprocess
        import json

        cmd = f'ssh {self.ssh_host} "cat {self.remote_path}"'

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                raise PluginError(f"SSH command failed: {result.stderr}")

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            raise PluginError(f"SSH timeout accessing {self.ssh_host}:{self.remote_path}")
        except json.JSONDecodeError as e:
            raise PluginError(f"Invalid JSON from {self.ssh_host}:{self.remote_path}: {e}")
        except Exception as e:
            raise PluginError(f"Error fetching {self.ssh_host}:{self.remote_path}: {e}")


class CommandPlugin(BasePlugin):
    """
    Base class for plugins that execute commands and parse output.
    Useful for nvidia-smi, ps aux, etc.
    """

    def __init__(self, command: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.command = command

    def fetch(self) -> Dict[str, Any]:
        """Execute command and parse output (must be overridden for parsing)"""
        import subprocess

        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise PluginError(f"Command failed: {result.stderr}")

            # Subclass should override to parse stdout
            return self.parse_output(result.stdout)

        except subprocess.TimeoutExpired:
            raise PluginError(f"Command timeout: {self.command}")
        except Exception as e:
            raise PluginError(f"Command error: {e}")

    def parse_output(self, stdout: str) -> Dict[str, Any]:
        """
        Parse command output (must be overridden).

        Args:
            stdout: Command output

        Returns:
            Parsed data dict
        """
        raise NotImplementedError("Subclass must implement parse_output()")
