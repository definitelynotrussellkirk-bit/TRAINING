"""
Guild API endpoints and server.

The API module provides REST-style endpoints for guild operations:
- Status and health checks
- Hero state management
- Skill registry and state
- Quest management
- Run execution
- Combat evaluation
- Consistency checks

Example (standalone server):
    from guild.api import GuildServer

    server = GuildServer(port=8082)
    server.start()

Example (Flask integration):
    from flask import Flask, jsonify, request
    from guild.api import get_endpoints, APIRequest

    app = Flask(__name__)
    endpoints = get_endpoints()

    @app.route('/guild/status')
    def status():
        req = APIRequest(method='GET', path='/guild/status')
        response = endpoints.handle_status(req)
        return jsonify(response.to_dict()), response.status

Example (FastAPI integration):
    from fastapi import FastAPI
    from guild.api import get_endpoints, APIRequest

    app = FastAPI()
    endpoints = get_endpoints()

    @app.get('/guild/status')
    def status():
        req = APIRequest(method='GET', path='/guild/status')
        response = endpoints.handle_status(req)
        return response.to_dict()

CLI:
    python -m guild.api.server --port 8082
"""

from guild.api.endpoints import (
    APIRequest,
    APIResponse,
    EndpointHandler,
    GuildEndpoints,
    init_endpoints,
    get_endpoints,
    reset_endpoints,
)

from guild.api.server import (
    GuildRequestHandler,
    GuildServer,
)

__all__ = [
    # Request/Response
    "APIRequest",
    "APIResponse",
    "EndpointHandler",
    # Endpoints
    "GuildEndpoints",
    "init_endpoints",
    "get_endpoints",
    "reset_endpoints",
    # Server
    "GuildRequestHandler",
    "GuildServer",
]
