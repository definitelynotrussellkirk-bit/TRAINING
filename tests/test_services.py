"""
Tests for core/services.py - ServiceClient abstraction.

Tests cover:
- Success path (JSON response)
- Timeout/connection errors -> ServiceUnavailable
- HTTP 5xx with retries -> ServiceUnavailable
- HTTP 4xx -> ServiceHttpError (no retry)
- Invalid JSON -> ServiceDecodeError
- Auth errors (401/403) -> ServiceAuthError
"""

import unittest
from unittest.mock import patch, Mock, MagicMock
import json

from core.services import (
    ServiceClient,
    ServiceConfig,
    ServiceId,
    ServiceError,
    ServiceUnavailable,
    ServiceHttpError,
    ServiceDecodeError,
    ServiceAuthError,
    get_service_config,
    get_service_client,
    service_health,
)


class TestServiceConfig(unittest.TestCase):
    """Test ServiceConfig dataclass."""

    def test_basic_config(self):
        config = ServiceConfig(
            id=ServiceId(name="test"),
            base_url="http://localhost:8080",
        )
        self.assertEqual(config.id.name, "test")
        self.assertEqual(config.base_url, "http://localhost:8080")
        self.assertEqual(config.timeout_s, 30.0)  # default
        self.assertEqual(config.max_retries, 3)  # default

    def test_with_auth(self):
        config = ServiceConfig(
            id=ServiceId(name="test"),
            base_url="http://localhost:8080",
        )
        new_config = config.with_auth(api_key="secret", admin_key="admin_secret")
        self.assertEqual(new_config.api_key, "secret")
        self.assertEqual(new_config.admin_key, "admin_secret")
        # Original unchanged
        self.assertIsNone(config.api_key)


class TestServiceClient(unittest.TestCase):
    """Test ServiceClient HTTP methods."""

    def setUp(self):
        self.config = ServiceConfig(
            id=ServiceId(name="test"),
            base_url="http://test.local:8080",
            timeout_s=5.0,
            max_retries=2,
            backoff_factor=0.1,  # Fast backoff for tests
        )
        self.client = ServiceClient(self.config)

    def test_full_url_construction(self):
        self.assertEqual(
            self.client._full_url("/health"),
            "http://test.local:8080/health"
        )
        self.assertEqual(
            self.client._full_url("health"),
            "http://test.local:8080/health"
        )

    @patch('core.services.requests.Session')
    def test_get_json_success(self, mock_session_class):
        """Test successful GET request."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_session.request.return_value = mock_response

        client = ServiceClient(self.config)
        result = client.get_json("/health")

        self.assertEqual(result, {"status": "ok"})
        mock_session.request.assert_called_once()

    @patch('core.services.requests.Session')
    def test_post_json_success(self, mock_session_class):
        """Test successful POST request."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123"}
        mock_session.request.return_value = mock_response

        client = ServiceClient(self.config)
        result = client.post_json("/create", json={"name": "test"})

        self.assertEqual(result, {"id": "123"})

    @patch('core.services.requests.Session')
    def test_connection_error_retries(self, mock_session_class):
        """Test that connection errors trigger retries."""
        import requests as real_requests

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.request.side_effect = real_requests.ConnectionError("Connection refused")

        client = ServiceClient(self.config)

        with self.assertRaises(ServiceUnavailable) as ctx:
            client.get_json("/health")

        self.assertEqual(ctx.exception.service, "test")
        # Should have tried max_retries times
        self.assertEqual(mock_session.request.call_count, self.config.max_retries)

    @patch('core.services.requests.Session')
    def test_timeout_retries(self, mock_session_class):
        """Test that timeouts trigger retries."""
        import requests as real_requests

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.request.side_effect = real_requests.Timeout("Request timed out")

        client = ServiceClient(self.config)

        with self.assertRaises(ServiceUnavailable):
            client.get_json("/health")

        self.assertEqual(mock_session.request.call_count, self.config.max_retries)

    @patch('core.services.requests.Session')
    def test_http_500_retries(self, mock_session_class):
        """Test that HTTP 5xx errors trigger retries."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_session.request.return_value = mock_response

        client = ServiceClient(self.config)

        with self.assertRaises(ServiceUnavailable):
            client.get_json("/health")

        self.assertEqual(mock_session.request.call_count, self.config.max_retries)

    @patch('core.services.requests.Session')
    def test_http_400_no_retry(self, mock_session_class):
        """Test that HTTP 4xx errors do NOT retry."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_session.request.return_value = mock_response

        client = ServiceClient(self.config)

        with self.assertRaises(ServiceHttpError) as ctx:
            client.get_json("/bad")

        self.assertEqual(ctx.exception.status, 400)
        # Should NOT have retried
        self.assertEqual(mock_session.request.call_count, 1)

    @patch('core.services.requests.Session')
    def test_http_404_no_retry(self, mock_session_class):
        """Test that HTTP 404 does NOT retry."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_session.request.return_value = mock_response

        client = ServiceClient(self.config)

        with self.assertRaises(ServiceHttpError) as ctx:
            client.get_json("/notfound")

        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(mock_session.request.call_count, 1)

    @patch('core.services.requests.Session')
    def test_http_401_auth_error(self, mock_session_class):
        """Test that HTTP 401 raises ServiceAuthError."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_session.request.return_value = mock_response

        client = ServiceClient(self.config)

        with self.assertRaises(ServiceAuthError) as ctx:
            client.get_json("/protected")

        self.assertEqual(ctx.exception.service, "test")
        self.assertEqual(mock_session.request.call_count, 1)

    @patch('core.services.requests.Session')
    def test_invalid_json_response(self, mock_session_class):
        """Test that invalid JSON raises ServiceDecodeError."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_session.request.return_value = mock_response

        client = ServiceClient(self.config)

        with self.assertRaises(ServiceDecodeError):
            client.get_json("/health")

    @patch('core.services.requests.Session')
    def test_health_check_success(self, mock_session_class):
        """Test health_check returns True on 200."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        client = ServiceClient(self.config)
        self.assertTrue(client.health_check())

    @patch('core.services.requests.Session')
    def test_health_check_failure(self, mock_session_class):
        """Test health_check returns False on error."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.get.side_effect = Exception("Connection refused")

        client = ServiceClient(self.config)
        self.assertFalse(client.health_check())

    @patch('core.services.requests.Session')
    def test_api_key_header(self, mock_session_class):
        """Test that API key is sent in headers."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_session.request.return_value = mock_response

        config = self.config.with_auth(api_key="secret123")
        client = ServiceClient(config)
        client.get_json("/protected")

        # Check headers were passed
        call_kwargs = mock_session.request.call_args[1]
        self.assertEqual(call_kwargs["headers"]["X-API-Key"], "secret123")


class TestServiceFactory(unittest.TestCase):
    """Test get_service_config and get_service_client."""

    @patch('core.hosts.get_service_url')
    def test_get_service_config_from_registry(self, mock_get_url):
        """Test config is built from host registry."""
        mock_get_url.return_value = "http://inference.local:8765"

        config = get_service_config("inference")

        self.assertEqual(config.base_url, "http://inference.local:8765")
        self.assertEqual(config.id.name, "inference")
        mock_get_url.assert_called_once_with("inference")

    def test_get_service_config_fallback(self):
        """Test fallback when host registry unavailable."""
        # This will use fallback since we're not mocking get_service_url
        # and the real one may not be available in test environment
        config = get_service_config("test_service")

        # Should get a valid config even without registry
        self.assertIsNotNone(config.base_url)
        self.assertEqual(config.id.name, "test_service")

    @patch('core.hosts.get_service_url')
    def test_get_service_client(self, mock_get_url):
        """Test get_service_client returns a ServiceClient."""
        mock_get_url.return_value = "http://test.local:8080"

        client = get_service_client("test")

        self.assertIsInstance(client, ServiceClient)
        self.assertEqual(client.service_name, "test")


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""

    @patch('core.services.ServiceClient')
    @patch('core.services.get_service_config')
    def test_service_health(self, mock_get_config, mock_client_class):
        """Test service_health convenience function."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_client.health_check.return_value = True
        mock_client_class.return_value = mock_client

        result = service_health("test")

        self.assertTrue(result)
        mock_client.health_check.assert_called_once()


if __name__ == "__main__":
    unittest.main()
