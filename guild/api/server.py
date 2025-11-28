"""
Simple HTTP server for Guild API.

Provides a standalone server using Python's built-in http.server.
For production, use with a proper WSGI server or integrate with Flask/FastAPI.

Usage:
    # Start server
    python -m guild.api.server --port 8082

    # Or programmatically
    from guild.api.server import GuildServer
    server = GuildServer(port=8082)
    server.start()
"""

import json
import logging
import re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

from guild.api.endpoints import (
    APIRequest,
    APIResponse,
    GuildEndpoints,
    get_endpoints,
)

logger = logging.getLogger(__name__)


class GuildRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Guild API."""

    # Reference to endpoints (set by server)
    endpoints: Optional[GuildEndpoints] = None

    def log_message(self, format, *args):
        """Override to use logging module."""
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_response(self, response: APIResponse):
        """Send API response."""
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        body = json.dumps(response.to_dict(), indent=2)
        self.wfile.write(body.encode())

    def _parse_request(self) -> APIRequest:
        """Parse incoming request into APIRequest."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Parse query params
        query_params = parse_qs(parsed.query)
        params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}

        # Parse body for POST requests
        body = None
        if self.command == "POST":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                raw_body = self.rfile.read(content_length)
                try:
                    body = json.loads(raw_body.decode())
                except json.JSONDecodeError:
                    body = {}

        # Extract path params (e.g., {skill_id} from /guild/skills/{skill_id})
        path_params = self._extract_path_params(path)
        params.update(path_params)

        return APIRequest(
            method=self.command,
            path=path,
            params=params,
            body=body,
            headers=dict(self.headers),
        )

    def _extract_path_params(self, path: str) -> Dict[str, str]:
        """Extract path parameters like {skill_id}."""
        params = {}

        # Define patterns for parameterized routes
        patterns = [
            (r"/guild/skills/([^/]+)/state", "skill_id"),
            (r"/guild/skills/([^/]+)", "skill_id"),
            (r"/guild/quests/([^/]+)", "quest_id"),
            (r"/guild/runs/([^/]+)/(start|pause|complete)", "run_id"),
            (r"/guild/runs/([^/]+)", "run_id"),
        ]

        for pattern, param_name in patterns:
            match = re.match(pattern, path)
            if match:
                params[param_name] = match.group(1)
                break

        return params

    def _normalize_path(self, path: str) -> str:
        """Normalize path to match registered routes."""
        # Replace actual IDs with placeholders
        replacements = [
            (r"/guild/skills/[^/]+/state", "/guild/skills/{skill_id}/state"),
            (r"/guild/skills/[^/]+", "/guild/skills/{skill_id}"),
            (r"/guild/quests/[^/]+", "/guild/quests/{quest_id}"),
            (r"/guild/runs/[^/]+/(start|pause|complete)", r"/guild/runs/{run_id}/\1"),
            (r"/guild/runs/[^/]+", "/guild/runs/{run_id}"),
        ]

        for pattern, replacement in replacements:
            if re.match(pattern, path):
                return re.sub(pattern, replacement, path)

        return path

    def _handle_request(self):
        """Handle any request method."""
        if self.endpoints is None:
            self._send_response(APIResponse.internal_error("Endpoints not configured"))
            return

        request = self._parse_request()

        # Normalize path for route matching
        normalized_path = self._normalize_path(request.path)

        # Find handler
        handler = self.endpoints.get_handler(request.method, normalized_path)

        if handler is None:
            # Check if path exists but method not allowed
            for method in ["GET", "POST", "PUT", "DELETE"]:
                if self.endpoints.get_handler(method, normalized_path):
                    self._send_response(APIResponse(
                        status=405,
                        error=f"Method {request.method} not allowed",
                    ))
                    return

            self._send_response(APIResponse.not_found(f"Endpoint not found: {request.path}"))
            return

        try:
            response = handler(request)
            self._send_response(response)
        except Exception as e:
            logger.error(f"Handler error: {e}")
            self._send_response(APIResponse.internal_error(str(e)))

    def do_GET(self):
        """Handle GET requests."""
        self._handle_request()

    def do_POST(self):
        """Handle POST requests."""
        self._handle_request()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


class GuildServer:
    """
    Simple HTTP server for Guild API.

    Example:
        server = GuildServer(port=8082)
        server.start()  # Blocking

        # Or in background
        import threading
        thread = threading.Thread(target=server.start, daemon=True)
        thread.start()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8082,
        endpoints: Optional[GuildEndpoints] = None,
    ):
        self.host = host
        self.port = port
        self.endpoints = endpoints or get_endpoints()
        self._server: Optional[ThreadingHTTPServer] = None

    def start(self):
        """Start the server (blocking)."""
        # Configure handler with endpoints
        GuildRequestHandler.endpoints = self.endpoints

        # Use ThreadingHTTPServer to handle concurrent requests without blocking
        self._server = ThreadingHTTPServer((self.host, self.port), GuildRequestHandler)
        logger.info(f"Guild API server starting on http://{self.host}:{self.port}")

        print(f"Guild API Server")
        print(f"================")
        print(f"Listening on: http://{self.host}:{self.port}")
        print(f"")
        print(f"Endpoints:")
        for route in self.endpoints.list_routes()[:10]:
            print(f"  {route['method']:6} {route['path']}")
        print(f"  ... and {len(self.endpoints.list_routes()) - 10} more")
        print(f"")
        print(f"Press Ctrl+C to stop")

        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop()

    def stop(self):
        """Stop the server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("Guild API server stopped")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Guild API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Start server
    server = GuildServer(host=args.host, port=args.port)
    server.start()


if __name__ == "__main__":
    main()
