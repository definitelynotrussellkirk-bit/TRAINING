"""
Guild API endpoint definitions.

Provides REST-style endpoint handlers for guild operations.
These can be used with any web framework (Flask, FastAPI, etc.)

Endpoints:
- /guild/status - Overall guild status
- /guild/hero - Hero state management
- /guild/skills - Skill registry and state
- /guild/quests - Quest management
- /guild/runs - Run execution
- /guild/combat - Combat evaluation
- /guild/consistency - Consistency checks
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)


@dataclass
class APIRequest:
    """Incoming API request."""
    method: str
    path: str
    params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class APIResponse:
    """Outgoing API response."""
    status: int = 200
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        result = {
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result

    @classmethod
    def ok(cls, data: Dict[str, Any]) -> "APIResponse":
        return cls(status=200, data=data)

    @classmethod
    def created(cls, data: Dict[str, Any]) -> "APIResponse":
        return cls(status=201, data=data)

    @classmethod
    def bad_request(cls, error: str) -> "APIResponse":
        return cls(status=400, error=error)

    @classmethod
    def not_found(cls, error: str = "Not found") -> "APIResponse":
        return cls(status=404, error=error)

    @classmethod
    def internal_error(cls, error: str) -> "APIResponse":
        return cls(status=500, error=error)


# Type for endpoint handlers
EndpointHandler = Callable[[APIRequest], APIResponse]


class GuildEndpoints:
    """
    Guild API endpoint handlers.

    Provides handlers for all guild API endpoints.
    Can be integrated with any web framework.

    Example (Flask):
        endpoints = GuildEndpoints()

        @app.route('/guild/status')
        def status():
            request = APIRequest(method='GET', path='/guild/status')
            response = endpoints.handle_status(request)
            return jsonify(response.to_dict()), response.status

    Example (FastAPI):
        endpoints = GuildEndpoints()

        @app.get('/guild/status')
        def status():
            request = APIRequest(method='GET', path='/guild/status')
            response = endpoints.handle_status(request)
            return response.to_dict()
    """

    def __init__(self):
        self._routes: Dict[str, Dict[str, EndpointHandler]] = {}
        self._register_routes()

    def _register_routes(self):
        """Register all endpoint routes."""
        # Status endpoints
        self.register("GET", "/guild/status", self.handle_status)
        self.register("GET", "/guild/health", self.handle_health)

        # Hero endpoints
        self.register("GET", "/guild/hero", self.handle_get_hero)
        self.register("GET", "/guild/hero/status", self.handle_hero_status)

        # Skills endpoints
        self.register("GET", "/guild/skills", self.handle_list_skills)
        self.register("GET", "/guild/skills/{skill_id}", self.handle_get_skill)
        self.register("GET", "/guild/skills/{skill_id}/state", self.handle_get_skill_state)

        # Quests endpoints
        self.register("GET", "/guild/quests", self.handle_list_quests)
        self.register("GET", "/guild/quests/{quest_id}", self.handle_get_quest)
        self.register("POST", "/guild/quests/create", self.handle_create_quest)

        # Runs endpoints
        self.register("GET", "/guild/runs", self.handle_list_runs)
        self.register("GET", "/guild/runs/{run_id}", self.handle_get_run)
        self.register("POST", "/guild/runs/create", self.handle_create_run)
        self.register("POST", "/guild/runs/{run_id}/start", self.handle_start_run)
        self.register("POST", "/guild/runs/{run_id}/pause", self.handle_pause_run)
        self.register("POST", "/guild/runs/{run_id}/complete", self.handle_complete_run)

        # Combat endpoints
        self.register("POST", "/guild/combat/evaluate", self.handle_evaluate_combat)
        self.register("POST", "/guild/combat/xp", self.handle_calculate_xp)

        # Consistency endpoints
        self.register("GET", "/guild/consistency/status", self.handle_consistency_status)
        self.register("POST", "/guild/consistency/check", self.handle_consistency_check)

        # Progression endpoints
        self.register("GET", "/guild/progression/summary", self.handle_progression_summary)

    def register(self, method: str, path: str, handler: EndpointHandler):
        """Register an endpoint handler."""
        if path not in self._routes:
            self._routes[path] = {}
        self._routes[path][method] = handler

    def get_handler(self, method: str, path: str) -> Optional[EndpointHandler]:
        """Get handler for a method/path combination."""
        if path in self._routes:
            return self._routes[path].get(method)
        return None

    def list_routes(self) -> List[Dict[str, str]]:
        """List all registered routes."""
        routes = []
        for path, methods in self._routes.items():
            for method in methods:
                routes.append({"method": method, "path": path})
        return routes

    # =========================================================================
    # Status Endpoints
    # =========================================================================

    def handle_status(self, request: APIRequest) -> APIResponse:
        """GET /guild/status - Get overall guild status."""
        try:
            from guild import (
                get_hero_manager,
                get_skill_registry,
                get_quest_registry,
                get_run_state_manager,
            )

            status = {
                "guild": "operational",
                "timestamp": datetime.now().isoformat(),
            }

            # Try to get component statuses
            try:
                hero_mgr = get_hero_manager()
                status["hero"] = {"available": True}
            except Exception:
                status["hero"] = {"available": False}

            try:
                skill_reg = get_skill_registry()
                status["skills"] = {
                    "available": True,
                    "count": len(skill_reg.list_skills()),
                }
            except Exception:
                status["skills"] = {"available": False}

            try:
                quest_reg = get_quest_registry()
                status["quests"] = {
                    "available": True,
                    "count": len(quest_reg.list_quests()),
                }
            except Exception:
                status["quests"] = {"available": False}

            try:
                run_mgr = get_run_state_manager()
                current = run_mgr.get_current_run()
                status["runs"] = {
                    "available": True,
                    "current_run": current.run_id if current else None,
                }
            except Exception:
                status["runs"] = {"available": False}

            return APIResponse.ok(status)

        except Exception as e:
            logger.error(f"Status endpoint error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_health(self, request: APIRequest) -> APIResponse:
        """GET /guild/health - Simple health check."""
        return APIResponse.ok({
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        })

    # =========================================================================
    # Hero Endpoints
    # =========================================================================

    def handle_get_hero(self, request: APIRequest) -> APIResponse:
        """GET /guild/hero - Get hero state."""
        try:
            from guild import get_hero

            hero_id = request.params.get("hero_id", "default")
            hero = get_hero(hero_id)

            if hero is None:
                return APIResponse.not_found(f"Hero '{hero_id}' not found")

            return APIResponse.ok({
                "hero_id": hero_id,
                "level": hero.level,
                "xp": hero.xp,
                "health": hero.health,
                "skills": list(hero.skills.keys()),
            })

        except Exception as e:
            logger.error(f"Get hero error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_hero_status(self, request: APIRequest) -> APIResponse:
        """GET /guild/hero/status - Get hero status summary."""
        try:
            from guild import get_hero_status

            hero_id = request.params.get("hero_id", "default")
            status = get_hero_status(hero_id)

            return APIResponse.ok(status)

        except Exception as e:
            logger.error(f"Hero status error: {e}")
            return APIResponse.internal_error(str(e))

    # =========================================================================
    # Skills Endpoints
    # =========================================================================

    def handle_list_skills(self, request: APIRequest) -> APIResponse:
        """GET /guild/skills - List all skills."""
        try:
            from guild import list_skills, get_skill

            skill_ids = list_skills()
            skills = []

            for skill_id in skill_ids:
                skill = get_skill(skill_id)
                if skill:
                    skills.append({
                        "id": skill.id,
                        "name": skill.name,
                        "category": skill.category.value if hasattr(skill.category, 'value') else str(skill.category),
                    })

            return APIResponse.ok({"skills": skills, "count": len(skills)})

        except Exception as e:
            logger.error(f"List skills error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_get_skill(self, request: APIRequest) -> APIResponse:
        """GET /guild/skills/{skill_id} - Get skill config."""
        try:
            from guild import get_skill

            skill_id = request.params.get("skill_id")
            if not skill_id:
                return APIResponse.bad_request("skill_id required")

            skill = get_skill(skill_id)
            if skill is None:
                return APIResponse.not_found(f"Skill '{skill_id}' not found")

            return APIResponse.ok({
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "category": skill.category.value if hasattr(skill.category, 'value') else str(skill.category),
                "tags": skill.tags,
            })

        except Exception as e:
            logger.error(f"Get skill error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_get_skill_state(self, request: APIRequest) -> APIResponse:
        """GET /guild/skills/{skill_id}/state - Get skill state."""
        try:
            from guild import get_skill_state_manager

            skill_id = request.params.get("skill_id")
            if not skill_id:
                return APIResponse.bad_request("skill_id required")

            state_mgr = get_skill_state_manager()
            status = state_mgr.get_status(skill_id)

            return APIResponse.ok(status)

        except Exception as e:
            logger.error(f"Get skill state error: {e}")
            return APIResponse.internal_error(str(e))

    # =========================================================================
    # Quests Endpoints
    # =========================================================================

    def handle_list_quests(self, request: APIRequest) -> APIResponse:
        """GET /guild/quests - List quest templates."""
        try:
            from guild import list_quests, get_quest

            quest_ids = list_quests()
            limit = int(request.params.get("limit", 100))

            quests = []
            for quest_id in quest_ids[:limit]:
                quest = get_quest(quest_id)
                if quest:
                    quests.append({
                        "id": quest.id,
                        "name": quest.name,
                        "skills": quest.skills,
                    })

            return APIResponse.ok({"quests": quests, "count": len(quests)})

        except Exception as e:
            logger.error(f"List quests error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_get_quest(self, request: APIRequest) -> APIResponse:
        """GET /guild/quests/{quest_id} - Get quest template."""
        try:
            from guild import get_quest

            quest_id = request.params.get("quest_id")
            if not quest_id:
                return APIResponse.bad_request("quest_id required")

            quest = get_quest(quest_id)
            if quest is None:
                return APIResponse.not_found(f"Quest '{quest_id}' not found")

            return APIResponse.ok({
                "id": quest.id,
                "name": quest.name,
                "description": quest.description,
                "skills": quest.skills,
                "difficulty": quest.difficulty.value if hasattr(quest.difficulty, 'value') else str(quest.difficulty),
            })

        except Exception as e:
            logger.error(f"Get quest error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_create_quest(self, request: APIRequest) -> APIResponse:
        """POST /guild/quests/create - Create quest instance."""
        try:
            from guild import create_quest

            body = request.body or {}
            template_id = body.get("template_id")
            skill = body.get("skill")

            if not template_id and not skill:
                return APIResponse.bad_request("template_id or skill required")

            quest = create_quest(template_id=template_id, skill=skill)
            if quest is None:
                return APIResponse.not_found("Could not create quest")

            return APIResponse.created({
                "quest_id": quest.id,
                "prompt": quest.prompt,
                "skills": quest.skills,
                "difficulty": quest.difficulty_level,
            })

        except Exception as e:
            logger.error(f"Create quest error: {e}")
            return APIResponse.internal_error(str(e))

    # =========================================================================
    # Runs Endpoints
    # =========================================================================

    def handle_list_runs(self, request: APIRequest) -> APIResponse:
        """GET /guild/runs - List runs."""
        try:
            from guild import get_run_state_manager

            run_mgr = get_run_state_manager()
            runs = run_mgr.list_runs()

            return APIResponse.ok({
                "runs": [{"run_id": r.run_id, "status": r.status} for r in runs],
                "count": len(runs),
            })

        except Exception as e:
            logger.error(f"List runs error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_get_run(self, request: APIRequest) -> APIResponse:
        """GET /guild/runs/{run_id} - Get run state."""
        try:
            from guild import get_run

            run_id = request.params.get("run_id")
            if not run_id:
                return APIResponse.bad_request("run_id required")

            run = get_run(run_id)
            if run is None:
                return APIResponse.not_found(f"Run '{run_id}' not found")

            return APIResponse.ok({
                "run_id": run.run_id,
                "status": run.status,
                "progress": run.progress,
                "total_steps": run.total_steps,
                "completed_steps": run.completed_steps,
            })

        except Exception as e:
            logger.error(f"Get run error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_create_run(self, request: APIRequest) -> APIResponse:
        """POST /guild/runs/create - Create a new run."""
        try:
            from guild import create_run, RunConfig, RunType

            body = request.body or {}
            run_type = body.get("type", "training")
            run_id = body.get("id")

            config = RunConfig(
                id=run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=RunType(run_type),
            )

            run = create_run(config)

            return APIResponse.created({
                "run_id": run.run_id,
                "status": run.status,
            })

        except Exception as e:
            logger.error(f"Create run error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_start_run(self, request: APIRequest) -> APIResponse:
        """POST /guild/runs/{run_id}/start - Start a run."""
        try:
            from guild import start_run

            run_id = request.params.get("run_id")
            if not run_id:
                return APIResponse.bad_request("run_id required")

            run = start_run(run_id)
            if run is None:
                return APIResponse.not_found(f"Run '{run_id}' not found")

            return APIResponse.ok({
                "run_id": run.run_id,
                "status": run.status,
            })

        except Exception as e:
            logger.error(f"Start run error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_pause_run(self, request: APIRequest) -> APIResponse:
        """POST /guild/runs/{run_id}/pause - Pause a run."""
        try:
            from guild import pause_run

            run_id = request.params.get("run_id")
            if not run_id:
                return APIResponse.bad_request("run_id required")

            run = pause_run(run_id)
            if run is None:
                return APIResponse.not_found(f"Run '{run_id}' not found")

            return APIResponse.ok({
                "run_id": run.run_id,
                "status": run.status,
            })

        except Exception as e:
            logger.error(f"Pause run error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_complete_run(self, request: APIRequest) -> APIResponse:
        """POST /guild/runs/{run_id}/complete - Complete a run."""
        try:
            from guild import complete_run

            run_id = request.params.get("run_id")
            if not run_id:
                return APIResponse.bad_request("run_id required")

            run = complete_run(run_id)
            if run is None:
                return APIResponse.not_found(f"Run '{run_id}' not found")

            return APIResponse.ok({
                "run_id": run.run_id,
                "status": run.status,
            })

        except Exception as e:
            logger.error(f"Complete run error: {e}")
            return APIResponse.internal_error(str(e))

    # =========================================================================
    # Combat Endpoints
    # =========================================================================

    def handle_evaluate_combat(self, request: APIRequest) -> APIResponse:
        """POST /guild/combat/evaluate - Evaluate combat result."""
        try:
            from guild import evaluate_combat
            from guild.quests.types import QuestInstance, QuestDifficulty

            body = request.body or {}

            # Build quest instance from body
            quest = QuestInstance(
                id=body.get("quest_id", "api_quest"),
                template_id=body.get("template_id", "api"),
                skills=body.get("skills", ["default"]),
                prompt=body.get("prompt", ""),
                expected=body.get("expected", {}),
                difficulty=QuestDifficulty.BRONZE,
                difficulty_level=body.get("difficulty", 1),
            )

            response = body.get("response", "")

            result = evaluate_combat(quest, response)

            return APIResponse.ok({
                "combat_result": result.combat_result.value,
                "match_quality": result.match_quality.value,
                "success": result.success,
            })

        except Exception as e:
            logger.error(f"Evaluate combat error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_calculate_xp(self, request: APIRequest) -> APIResponse:
        """POST /guild/combat/xp - Calculate XP from combat."""
        try:
            from guild import calculate_combat_xp
            from guild.quests.types import CombatResult

            body = request.body or {}

            combat_result = CombatResult(body.get("combat_result", "miss"))
            difficulty = body.get("difficulty", 1)
            streak = body.get("streak", 0)

            xp = calculate_combat_xp(
                combat_result=combat_result,
                difficulty=difficulty,
                streak=streak,
            )

            return APIResponse.ok({
                "base_xp": xp.base_xp,
                "total_xp": xp.total_xp,
                "multipliers": xp.multipliers,
            })

        except Exception as e:
            logger.error(f"Calculate XP error: {e}")
            return APIResponse.internal_error(str(e))

    # =========================================================================
    # Consistency Endpoints
    # =========================================================================

    def handle_consistency_status(self, request: APIRequest) -> APIResponse:
        """GET /guild/consistency/status - Get last check status."""
        try:
            from guild import get_consistency_checker

            checker = get_consistency_checker()
            result = checker.get_last_result()

            if result is None:
                return APIResponse.ok({
                    "last_check": None,
                    "message": "No consistency check has been run",
                })

            return APIResponse.ok({
                "last_check": result.summary(),
            })

        except Exception as e:
            logger.error(f"Consistency status error: {e}")
            return APIResponse.internal_error(str(e))

    def handle_consistency_check(self, request: APIRequest) -> APIResponse:
        """POST /guild/consistency/check - Run consistency check."""
        try:
            from guild import check_all

            result = check_all()

            return APIResponse.ok({
                "passed": result.passed,
                "summary": result.summary(),
                "violations": [v.to_dict() for v in result.violations[:10]],  # Limit
            })

        except Exception as e:
            logger.error(f"Consistency check error: {e}")
            return APIResponse.internal_error(str(e))

    # =========================================================================
    # Progression Endpoints
    # =========================================================================

    def handle_progression_summary(self, request: APIRequest) -> APIResponse:
        """GET /guild/progression/summary - Get progression summary."""
        try:
            from guild import get_hero, get_xp_calculator

            hero_id = request.params.get("hero_id", "default")
            hero = get_hero(hero_id)

            if hero is None:
                return APIResponse.ok({
                    "hero_id": hero_id,
                    "exists": False,
                })

            calc = get_xp_calculator()

            return APIResponse.ok({
                "hero_id": hero_id,
                "exists": True,
                "level": hero.level,
                "xp": hero.xp,
                "xp_to_next": calc.xp_for_level(hero.level + 1) - hero.xp,
                "skills": {
                    skill_id: {"level": state.level, "accuracy": state.accuracy}
                    for skill_id, state in hero.skills.items()
                },
            })

        except Exception as e:
            logger.error(f"Progression summary error: {e}")
            return APIResponse.internal_error(str(e))


# Global endpoints instance
_endpoints: Optional[GuildEndpoints] = None


def init_endpoints() -> GuildEndpoints:
    """Initialize global endpoints."""
    global _endpoints
    _endpoints = GuildEndpoints()
    return _endpoints


def get_endpoints() -> GuildEndpoints:
    """Get global endpoints."""
    global _endpoints
    if _endpoints is None:
        _endpoints = GuildEndpoints()
    return _endpoints


def reset_endpoints() -> None:
    """Reset global endpoints (for testing)."""
    global _endpoints
    _endpoints = None
