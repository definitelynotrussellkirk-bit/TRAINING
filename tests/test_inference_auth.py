#!/usr/bin/env python3
"""
Tests for inference server authentication.

Tests the auth middleware logic without requiring the full server to be running.
Uses direct function calls to test the authentication logic.

Run with: pytest tests/test_inference_auth.py -v
"""

import os
import pytest
from unittest.mock import patch


class TestAuthFunctions:
    """Test auth helper functions directly."""

    def test_check_auth_configured_no_keys(self):
        """When no keys configured, should report not configured."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing keys
            os.environ.pop("INFERENCE_ADMIN_KEY", None)
            os.environ.pop("INFERENCE_READ_KEY", None)

            # Reimport to get fresh values
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            result = auth_module.check_auth_configured()
            assert result["admin_key_configured"] is False
            assert result["read_key_configured"] is False
            assert result["any_key_configured"] is False

    def test_check_auth_configured_with_admin(self):
        """When admin key set, should report configured."""
        with patch.dict(os.environ, {"INFERENCE_ADMIN_KEY": "test-admin-key"}, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            result = auth_module.check_auth_configured()
            assert result["admin_key_configured"] is True
            assert result["any_key_configured"] is True

    def test_check_auth_configured_with_read(self):
        """When read key set, should report configured."""
        with patch.dict(os.environ, {"INFERENCE_READ_KEY": "test-read-key"}, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            result = auth_module.check_auth_configured()
            assert result["read_key_configured"] is True
            assert result["any_key_configured"] is True


class TestRequireAdmin:
    """Test require_admin dependency."""

    def test_require_admin_no_key_configured(self):
        """Should raise 500 if admin key not configured."""
        from fastapi import HTTPException

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("INFERENCE_ADMIN_KEY", None)

            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            with pytest.raises(HTTPException) as exc_info:
                auth_module.require_admin(api_key=None)

            assert exc_info.value.status_code == 500
            assert "not configured" in exc_info.value.detail.lower()

    def test_require_admin_no_key_provided(self):
        """Should raise 401 if no API key header provided."""
        from fastapi import HTTPException

        with patch.dict(os.environ, {"INFERENCE_ADMIN_KEY": "test-admin"}, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            with pytest.raises(HTTPException) as exc_info:
                auth_module.require_admin(api_key=None)

            assert exc_info.value.status_code == 401
            assert "missing" in exc_info.value.detail.lower()

    def test_require_admin_wrong_key(self):
        """Should raise 401 if wrong API key provided."""
        from fastapi import HTTPException

        with patch.dict(os.environ, {"INFERENCE_ADMIN_KEY": "correct-key"}, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            with pytest.raises(HTTPException) as exc_info:
                auth_module.require_admin(api_key="wrong-key")

            assert exc_info.value.status_code == 401
            assert "invalid" in exc_info.value.detail.lower()

    def test_require_admin_correct_key(self):
        """Should return key if correct API key provided."""
        with patch.dict(os.environ, {"INFERENCE_ADMIN_KEY": "correct-key"}, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            result = auth_module.require_admin(api_key="correct-key")
            assert result == "correct-key"


class TestRequireRead:
    """Test require_read dependency."""

    def test_require_read_no_keys_configured(self):
        """Should raise 500 if no keys configured."""
        from fastapi import HTTPException

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("INFERENCE_ADMIN_KEY", None)
            os.environ.pop("INFERENCE_READ_KEY", None)

            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            with pytest.raises(HTTPException) as exc_info:
                auth_module.require_read(api_key=None)

            assert exc_info.value.status_code == 500

    def test_require_read_no_key_provided(self):
        """Should raise 401 if no API key header provided."""
        from fastapi import HTTPException

        with patch.dict(os.environ, {"INFERENCE_READ_KEY": "test-read"}, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            with pytest.raises(HTTPException) as exc_info:
                auth_module.require_read(api_key=None)

            assert exc_info.value.status_code == 401

    def test_require_read_with_read_key(self):
        """Should accept valid read key."""
        with patch.dict(os.environ, {"INFERENCE_READ_KEY": "read-key"}, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            result = auth_module.require_read(api_key="read-key")
            assert result == "read-key"

    def test_require_read_with_admin_key(self):
        """Admin key should grant read access."""
        with patch.dict(os.environ, {
            "INFERENCE_ADMIN_KEY": "admin-key",
            "INFERENCE_READ_KEY": "read-key"
        }, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            # Admin key should work for read endpoints
            result = auth_module.require_read(api_key="admin-key")
            assert result == "admin-key"

    def test_require_read_wrong_key(self):
        """Should reject invalid key."""
        from fastapi import HTTPException

        with patch.dict(os.environ, {"INFERENCE_READ_KEY": "correct-key"}, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            with pytest.raises(HTTPException) as exc_info:
                auth_module.require_read(api_key="wrong-key")

            assert exc_info.value.status_code == 401


class TestAuthSecurityProperties:
    """Test important security properties of the auth system."""

    def test_keys_not_in_logs(self):
        """Keys should not appear in error messages."""
        from fastapi import HTTPException

        secret_key = "super-secret-admin-key-12345"
        with patch.dict(os.environ, {"INFERENCE_ADMIN_KEY": secret_key}, clear=True):
            import importlib
            import inference.auth as auth_module
            importlib.reload(auth_module)

            with pytest.raises(HTTPException) as exc_info:
                auth_module.require_admin(api_key="wrong")

            # The actual key should NOT appear in the error detail
            assert secret_key not in str(exc_info.value.detail)
            assert secret_key not in str(exc_info.value)

    def test_timing_attack_resistance_placeholder(self):
        """
        Placeholder for timing attack resistance test.

        Note: The current implementation uses simple string comparison,
        which could be vulnerable to timing attacks. For production,
        consider using secrets.compare_digest().

        This test documents the concern but doesn't enforce it yet.
        """
        # TODO: If this becomes a concern, implement constant-time comparison
        pass
