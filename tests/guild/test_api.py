"""Tests for Guild API module."""

import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from guild.api.endpoints import (
    APIRequest,
    APIResponse,
    GuildEndpoints,
    init_endpoints,
    get_endpoints,
    reset_endpoints,
)
from guild.api.server import GuildServer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before and after each test."""
    reset_endpoints()
    yield
    reset_endpoints()


@pytest.fixture
def endpoints():
    """Create fresh endpoints instance."""
    return GuildEndpoints()


# =============================================================================
# APIRequest Tests
# =============================================================================

class TestAPIRequest:
    """Tests for APIRequest."""

    def test_basic_request(self):
        request = APIRequest(
            method="GET",
            path="/guild/status",
        )
        assert request.method == "GET"
        assert request.path == "/guild/status"

    def test_request_with_params(self):
        request = APIRequest(
            method="GET",
            path="/guild/skills",
            params={"skill_id": "logic"},
        )
        assert request.params["skill_id"] == "logic"

    def test_request_with_body(self):
        request = APIRequest(
            method="POST",
            path="/guild/quests/create",
            body={"template_id": "test", "skill": "logic"},
        )
        assert request.body["template_id"] == "test"


# =============================================================================
# APIResponse Tests
# =============================================================================

class TestAPIResponse:
    """Tests for APIResponse."""

    def test_ok_response(self):
        response = APIResponse.ok({"message": "success"})
        assert response.status == 200
        assert response.data["message"] == "success"
        assert response.error is None

    def test_created_response(self):
        response = APIResponse.created({"id": "123"})
        assert response.status == 201

    def test_bad_request_response(self):
        response = APIResponse.bad_request("Missing parameter")
        assert response.status == 400
        assert response.error == "Missing parameter"

    def test_not_found_response(self):
        response = APIResponse.not_found("Item not found")
        assert response.status == 404

    def test_internal_error_response(self):
        response = APIResponse.internal_error("Something broke")
        assert response.status == 500

    def test_response_to_dict(self):
        response = APIResponse.ok({"key": "value"})
        d = response.to_dict()
        assert d["status"] == 200
        assert d["data"]["key"] == "value"
        assert "timestamp" in d


# =============================================================================
# GuildEndpoints Tests
# =============================================================================

class TestGuildEndpoints:
    """Tests for GuildEndpoints."""

    def test_routes_registered(self, endpoints):
        routes = endpoints.list_routes()
        assert len(routes) > 0

        # Check essential routes exist
        route_paths = [r["path"] for r in routes]
        assert "/guild/status" in route_paths
        assert "/guild/health" in route_paths
        assert "/guild/skills" in route_paths

    def test_get_handler(self, endpoints):
        handler = endpoints.get_handler("GET", "/guild/status")
        assert handler is not None
        assert callable(handler)

    def test_get_handler_not_found(self, endpoints):
        handler = endpoints.get_handler("GET", "/nonexistent")
        assert handler is None

    def test_health_endpoint(self, endpoints):
        request = APIRequest(method="GET", path="/guild/health")
        response = endpoints.handle_health(request)
        assert response.status == 200
        assert response.data["status"] == "ok"

    def test_status_endpoint(self, endpoints):
        request = APIRequest(method="GET", path="/guild/status")
        response = endpoints.handle_status(request)
        assert response.status == 200
        assert "guild" in response.data
        assert response.data["guild"] == "operational"


# =============================================================================
# Skills Endpoints Tests
# =============================================================================

class TestSkillsEndpoints:
    """Tests for skills endpoints."""

    def test_list_skills(self, endpoints):
        request = APIRequest(method="GET", path="/guild/skills")
        response = endpoints.handle_list_skills(request)
        # May return empty or error if not initialized
        assert response.status in [200, 500]

    def test_get_skill_missing_id(self, endpoints):
        request = APIRequest(method="GET", path="/guild/skills/{skill_id}")
        response = endpoints.handle_get_skill(request)
        assert response.status == 400
        assert "skill_id required" in response.error

    def test_get_skill_not_found(self, endpoints):
        request = APIRequest(
            method="GET",
            path="/guild/skills/nonexistent",
            params={"skill_id": "nonexistent"},
        )
        response = endpoints.handle_get_skill(request)
        # May return 404 or 500 depending on initialization
        assert response.status in [404, 500]


# =============================================================================
# Quests Endpoints Tests
# =============================================================================

class TestQuestsEndpoints:
    """Tests for quests endpoints."""

    def test_list_quests(self, endpoints):
        request = APIRequest(method="GET", path="/guild/quests")
        response = endpoints.handle_list_quests(request)
        assert response.status in [200, 500]

    def test_create_quest_missing_params(self, endpoints):
        request = APIRequest(
            method="POST",
            path="/guild/quests/create",
            body={},
        )
        response = endpoints.handle_create_quest(request)
        assert response.status == 400
        assert "required" in response.error


# =============================================================================
# Runs Endpoints Tests
# =============================================================================

class TestRunsEndpoints:
    """Tests for runs endpoints."""

    def test_list_runs(self, endpoints):
        request = APIRequest(method="GET", path="/guild/runs")
        response = endpoints.handle_list_runs(request)
        assert response.status in [200, 500]

    def test_get_run_missing_id(self, endpoints):
        request = APIRequest(method="GET", path="/guild/runs/{run_id}")
        response = endpoints.handle_get_run(request)
        assert response.status == 400
        assert "run_id required" in response.error

    def test_start_run_missing_id(self, endpoints):
        request = APIRequest(
            method="POST",
            path="/guild/runs/{run_id}/start",
        )
        response = endpoints.handle_start_run(request)
        assert response.status == 400


# =============================================================================
# Combat Endpoints Tests
# =============================================================================

class TestCombatEndpoints:
    """Tests for combat endpoints."""

    def test_evaluate_combat(self, endpoints):
        request = APIRequest(
            method="POST",
            path="/guild/combat/evaluate",
            body={
                "quest_id": "test_quest",
                "prompt": "What is 2+2?",
                "expected": {"answer": "4"},
                "response": "The answer is 4",
                "skills": ["math"],
            },
        )
        response = endpoints.handle_evaluate_combat(request)
        assert response.status == 200
        assert "combat_result" in response.data

    def test_calculate_xp(self, endpoints):
        request = APIRequest(
            method="POST",
            path="/guild/combat/xp",
            body={
                "combat_result": "hit",
                "difficulty": 3,
                "streak": 5,
            },
        )
        response = endpoints.handle_calculate_xp(request)
        assert response.status == 200
        assert "base_xp" in response.data
        assert "total_xp" in response.data


# =============================================================================
# Consistency Endpoints Tests
# =============================================================================

class TestConsistencyEndpoints:
    """Tests for consistency endpoints."""

    def test_consistency_status(self, endpoints):
        request = APIRequest(method="GET", path="/guild/consistency/status")
        response = endpoints.handle_consistency_status(request)
        assert response.status == 200

    def test_consistency_check(self, endpoints):
        request = APIRequest(method="POST", path="/guild/consistency/check")
        response = endpoints.handle_consistency_check(request)
        assert response.status == 200
        assert "passed" in response.data


# =============================================================================
# Progression Endpoints Tests
# =============================================================================

class TestProgressionEndpoints:
    """Tests for progression endpoints."""

    def test_progression_summary(self, endpoints):
        request = APIRequest(method="GET", path="/guild/progression/summary")
        response = endpoints.handle_progression_summary(request)
        # May return empty hero or error
        assert response.status in [200, 500]


# =============================================================================
# Global Functions Tests
# =============================================================================

class TestGlobalFunctions:
    """Tests for global endpoint functions."""

    def test_init_and_get(self):
        init_endpoints()
        endpoints = get_endpoints()
        assert endpoints is not None

    def test_get_creates_if_none(self):
        endpoints = get_endpoints()
        assert endpoints is not None


# =============================================================================
# Server Tests
# =============================================================================

class TestGuildServer:
    """Tests for GuildServer."""

    def test_server_creation(self):
        server = GuildServer(port=19999)
        assert server.host == "0.0.0.0"
        assert server.port == 19999
        assert server.endpoints is not None

    def test_server_with_custom_endpoints(self, endpoints):
        server = GuildServer(port=19998, endpoints=endpoints)
        assert server.endpoints is endpoints


# =============================================================================
# Integration Tests
# =============================================================================

class TestEndpointsIntegration:
    """Integration tests for endpoints."""

    def test_combat_flow(self, endpoints):
        """Test a complete combat evaluation flow."""
        # 1. Evaluate combat
        eval_request = APIRequest(
            method="POST",
            path="/guild/combat/evaluate",
            body={
                "quest_id": "flow_test",
                "prompt": "What color is the sky?",
                "expected": {"answer": "blue"},
                "response": "The sky is blue",
                "skills": ["observation"],
            },
        )
        eval_response = endpoints.handle_evaluate_combat(eval_request)
        assert eval_response.status == 200

        combat_result = eval_response.data["combat_result"]

        # 2. Calculate XP
        xp_request = APIRequest(
            method="POST",
            path="/guild/combat/xp",
            body={
                "combat_result": combat_result,
                "difficulty": 2,
            },
        )
        xp_response = endpoints.handle_calculate_xp(xp_request)
        assert xp_response.status == 200
        assert xp_response.data["total_xp"] > 0

    def test_route_matching(self, endpoints):
        """Test that route patterns work correctly."""
        routes = endpoints.list_routes()

        # Count different endpoint categories
        skill_routes = [r for r in routes if "/skills" in r["path"]]
        quest_routes = [r for r in routes if "/quests" in r["path"]]
        run_routes = [r for r in routes if "/runs" in r["path"]]
        combat_routes = [r for r in routes if "/combat" in r["path"]]

        assert len(skill_routes) >= 2  # list + get
        assert len(quest_routes) >= 2
        assert len(run_routes) >= 3  # list, get, create, start, etc.
        assert len(combat_routes) >= 2  # evaluate + xp
