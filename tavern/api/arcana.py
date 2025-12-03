"""
Arcana API - Endpoints for the Arcana DSL panel.

Provides:
- World state serialization (compact and meta-aware)
- Skill progress with trends
- Plan history and outcomes
- Verb reference
- REPL execution
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent


def get_world_state(meta_aware: bool = True) -> Dict[str, Any]:
    """Get world state for display."""
    try:
        from arcana import create_engine, serialize_world_compact
        from arcana.meta import get_meta_context, serialize_world_meta

        engine = create_engine(BASE_DIR)

        if meta_aware:
            sexpr = serialize_world_meta(engine.world, BASE_DIR)
        else:
            engine.world.load_training_status()
            sexpr = serialize_world_compact(engine.world)

        return {
            "sexpr": sexpr,
            "meta_aware": meta_aware,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get world state: {e}")
        return {
            "sexpr": f"; Error loading world state: {e}",
            "meta_aware": meta_aware,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def get_skill_progress() -> Dict[str, Any]:
    """Get detailed skill progress for visualization."""
    try:
        from arcana.meta import get_meta_context

        ctx = get_meta_context(BASE_DIR)

        skills = []
        for skill_id, progress in ctx.skill_progress.items():
            skills.append({
                "id": skill_id,
                "level": progress.training_level,
                "max_level": progress.max_level,
                "mastered": progress.mastered_level,
                "accuracy": progress.recent_accuracy,
                "best_accuracy": progress.best_accuracy,
                "trend": progress.trend,
                "evals_at_level": progress.evals_at_level,
                "history_count": len(progress.accuracy_history),
            })

        return {
            "skills": skills,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get skill progress: {e}")
        return {"skills": [], "error": str(e)}


def get_eval_history(limit: int = 20) -> Dict[str, Any]:
    """Get recent evaluation history."""
    try:
        from arcana.meta import get_meta_context

        ctx = get_meta_context(BASE_DIR)

        evals = []
        for ev in ctx.eval_history[-limit:]:
            evals.append({
                "skill": ev.skill,
                "level": ev.level,
                "accuracy": ev.accuracy,
                "timestamp": ev.timestamp.isoformat(),
                "step": ev.step,
            })

        return {
            "evals": evals,
            "count": len(evals),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get eval history: {e}")
        return {"evals": [], "error": str(e)}


def get_plan_history(limit: int = 10) -> Dict[str, Any]:
    """Get recent plan history with outcomes."""
    try:
        from arcana.meta import get_meta_context

        ctx = get_meta_context(BASE_DIR)

        plans = []
        for plan in ctx.plan_history[-limit:]:
            plans.append({
                "id": plan.plan_id,
                "goal": plan.goal,
                "forms_count": plan.forms_count,
                "executed_at": plan.executed_at.isoformat(),
                "outcome": plan.outcome,
                "accuracy_delta": plan.accuracy_delta,
            })

        return {
            "plans": plans,
            "count": len(plans),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get plan history: {e}")
        return {"plans": [], "error": str(e)}


def get_verb_reference() -> Dict[str, Any]:
    """Get the available verb reference for the UI."""
    verbs = {
        "training": [
            {"name": "train", "syntax": "(train :quest QUEST_ID :steps N)", "desc": "Queue training on a quest"},
            {"name": "train-file", "syntax": "(train-file :path \"...\" :steps N)", "desc": "Train on specific file"},
            {"name": "pause", "syntax": "(pause)", "desc": "Pause training"},
            {"name": "resume", "syntax": "(resume)", "desc": "Resume training"},
            {"name": "checkpoint", "syntax": "(checkpoint :name \"...\")", "desc": "Save checkpoint"},
        ],
        "curriculum": [
            {"name": "level-up", "syntax": "(level-up :skill SKILL_ID)", "desc": "Advance skill to next level"},
            {"name": "level-down", "syntax": "(level-down :skill SKILL_ID)", "desc": "Regress skill to previous level"},
            {"name": "set-level", "syntax": "(set-level :skill SKILL_ID :level N)", "desc": "Set skill to specific level"},
            {"name": "run-eval", "syntax": "(run-eval :skill SKILL_ID :samples N)", "desc": "Trigger evaluation"},
            {"name": "generate-data", "syntax": "(generate-data :skill SKILL_ID :level N :count N)", "desc": "Generate training data"},
        ],
        "queries": [
            {"name": "metric", "syntax": "(metric :name METRIC)", "desc": "Get metric value"},
            {"name": "skill-status", "syntax": "(skill-status :skill SKILL_ID)", "desc": "Get detailed skill status"},
            {"name": "compare-skills", "syntax": "(compare-skills)", "desc": "Compare all skills"},
            {"name": "suggest-action", "syntax": "(suggest-action)", "desc": "Get AI suggestion"},
            {"name": "status", "syntax": "(status)", "desc": "Get full status"},
            {"name": "queue-status", "syntax": "(queue-status)", "desc": "Check training queue"},
        ],
        "control_flow": [
            {"name": "if", "syntax": "(if COND THEN ELSE)", "desc": "Conditional"},
            {"name": "when", "syntax": "(when COND BODY...)", "desc": "One-armed conditional"},
            {"name": "do", "syntax": "(do FORM1 FORM2...)", "desc": "Sequence"},
        ],
        "comparisons": [
            {"name": ">", "syntax": "(> A B)", "desc": "Greater than"},
            {"name": "<", "syntax": "(< A B)", "desc": "Less than"},
            {"name": "=", "syntax": "(= A B)", "desc": "Equal"},
            {"name": ">=", "syntax": "(>= A B)", "desc": "Greater or equal"},
            {"name": "<=", "syntax": "(<= A B)", "desc": "Less or equal"},
        ],
    }
    return {"verbs": verbs}


def execute_repl(code: str, dry_run: bool = False) -> Dict[str, Any]:
    """Execute Arcana code and return results."""
    try:
        from arcana import create_engine, parse, to_sexpr

        engine = create_engine(BASE_DIR)
        forms = parse(code)

        results = []
        errors = []

        for form in forms:
            try:
                if dry_run:
                    results.append({
                        "form": to_sexpr(form),
                        "result": "[DRY RUN - not executed]",
                        "success": True,
                    })
                else:
                    result = engine.eval(form)
                    results.append({
                        "form": to_sexpr(form),
                        "result": _serialize_result(result),
                        "success": True,
                    })
            except Exception as e:
                errors.append({
                    "form": to_sexpr(form) if isinstance(form, list) else str(form),
                    "error": str(e),
                })

        return {
            "results": results,
            "errors": errors,
            "dry_run": dry_run,
            "forms_count": len(forms),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"REPL execution failed: {e}")
        return {
            "results": [],
            "errors": [{"form": code, "error": str(e)}],
            "timestamp": datetime.now().isoformat(),
        }


def _serialize_result(result: Any) -> Any:
    """Serialize a result for JSON output."""
    if result is None:
        return None
    if isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, dict):
        return {k: _serialize_result(v) for k, v in result.items()}
    if isinstance(result, list):
        return [_serialize_result(v) for v in result]
    if hasattr(result, 'to_dict'):
        return result.to_dict()
    return str(result)


def get_training_status() -> Dict[str, Any]:
    """Get current training status for the panel."""
    try:
        status_path = BASE_DIR / "status" / "training_status.json"
        if status_path.is_symlink():
            status_path = status_path.resolve()

        if not status_path.exists():
            return {"status": "unknown", "error": "No training status file"}

        with open(status_path) as f:
            data = json.load(f)

        return {
            "status": data.get("status", "unknown"),
            "step": data.get("current_step", 0),
            "total_steps": data.get("total_steps", 0),
            "loss": data.get("loss"),
            "throughput": data.get("tokens_per_sec"),
            "model": data.get("model_name", "unknown"),
            "file": data.get("current_file"),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        return {"status": "error", "error": str(e)}


def get_analysis_hints() -> Dict[str, Any]:
    """Get analysis hints for decision making."""
    try:
        from arcana.meta import get_meta_context

        ctx = get_meta_context(BASE_DIR)
        hints = ctx._generate_hints()

        return {
            "hints": hints,
            "count": len(hints),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to generate hints: {e}")
        return {"hints": [], "error": str(e)}


def propose_plan(goal: str) -> Dict[str, Any]:
    """Generate a plan proposal (for display, not execution)."""
    try:
        from arcana import create_engine
        from arcana.planner import Planner

        engine = create_engine(BASE_DIR)
        planner = Planner(engine=engine)

        system_prompt, user_prompt = planner.build_prompt(goal)
        world_state = planner.get_world_state()

        return {
            "goal": goal,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "world_state": world_state,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to propose plan: {e}")
        return {"error": str(e)}


# Handler for HTTP requests
def handle_arcana_api(path: str, method: str = "GET", body: Optional[Dict] = None) -> Dict[str, Any]:
    """Handle Arcana API requests."""

    if path == "/api/arcana/world-state":
        meta = body.get("meta_aware", True) if body else True
        return get_world_state(meta_aware=meta)

    elif path == "/api/arcana/skills":
        return get_skill_progress()

    elif path == "/api/arcana/evals":
        limit = body.get("limit", 20) if body else 20
        return get_eval_history(limit=limit)

    elif path == "/api/arcana/plans":
        limit = body.get("limit", 10) if body else 10
        return get_plan_history(limit=limit)

    elif path == "/api/arcana/verbs":
        return get_verb_reference()

    elif path == "/api/arcana/training":
        return get_training_status()

    elif path == "/api/arcana/hints":
        return get_analysis_hints()

    elif path == "/api/arcana/repl" and method == "POST":
        code = body.get("code", "") if body else ""
        dry_run = body.get("dry_run", False) if body else False
        return execute_repl(code, dry_run=dry_run)

    elif path == "/api/arcana/propose" and method == "POST":
        goal = body.get("goal", "maintain") if body else "maintain"
        return propose_plan(goal)

    elif path == "/api/arcana/summary":
        # Combined summary for initial load
        return {
            "world_state": get_world_state(meta_aware=True),
            "skills": get_skill_progress(),
            "training": get_training_status(),
            "hints": get_analysis_hints(),
            "verbs": get_verb_reference(),
            "plans": get_plan_history(limit=5),
            "timestamp": datetime.now().isoformat(),
        }

    return {"error": f"Unknown arcana endpoint: {path}"}
