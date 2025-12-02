"""
Skills API

Extracted from tavern/server.py for better organization.
Handles:
- /api/skills - List all skills with levels/progress
- /api/skill/{skill_id} - Full skill data with config + state
- /api/titles - Hero titles based on progress
- /api/curriculum - Curriculum state
- /api/passives/* - Passive modules
- /api/engine/* - SkillEngine endpoints
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core import paths

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def _get_skills_data():
    """Load skill data from CurriculumManager (single source of truth)."""
    try:
        from guild.skills.loader import load_skill_config
        from data_manager.curriculum_manager import CurriculumManager

        base_dir = paths.get_base_dir()
        cm = CurriculumManager(base_dir, {})

        # Get eval counts from ledger (persisted truth) - stable across restarts
        eval_counts = {}
        try:
            from core.evaluation_ledger import get_eval_ledger
            ledger = get_eval_ledger(base_dir)
            summary = ledger.summary()
            for skill, info in summary.get("by_skill", {}).items():
                eval_counts[skill] = info.get("count", 0)
        except Exception as e:
            logger.debug(f"Could not load eval counts from ledger: {e}")

        skills = []
        # Iterate over skills in curriculum state
        for skill_id in cm.state.get("skills", {}).keys():
            try:
                config = load_skill_config(skill_id)

                # Get curriculum state for this skill
                mastered = cm.get_mastered_level(skill_id)
                training = cm.get_training_level(skill_id)

                # Get accuracy history
                skill_state = cm.state["skills"][skill_id]
                history = skill_state.get("accuracy_history", [])

                # Calculate recent accuracy (last 3 evals)
                recent_acc = 0
                last_eval_time = None
                last_single_acc = 0
                if history:
                    recent = history[-3:]
                    recent_acc = (sum(r.get("accuracy", 0) for r in recent) / len(recent)) * 100
                    # Get most recent eval timestamp and accuracy
                    last_entry = history[-1]
                    last_eval_time = last_entry.get("timestamp")
                    last_single_acc = last_entry.get("accuracy", 0) * 100

                # Use ledger count for total evals (persisted, stable)
                ledger_count = eval_counts.get(skill_id, 0)

                skills.append({
                    "id": config.id,
                    "name": config.name,
                    "rpg_name": config.rpg_name or config.name,
                    "rpg_description": config.rpg_description or config.description or "",
                    "icon": config.display.icon if config.display else "⚔️",
                    "color": config.display.color if config.display else "#888",
                    "short_name": config.display.short_name if config.display else config.id.upper(),
                    "max_level": config.max_level,
                    "mastered_level": mastered,
                    "training_level": training,
                    "accuracy": round(recent_acc, 1),
                    "last_accuracy": round(last_single_acc, 1),
                    "last_eval_time": last_eval_time,
                    "eval_count": ledger_count,
                    "category": config.category.value if hasattr(config.category, 'value') else str(config.category),
                    "description": config.description or "",
                })
            except Exception as e:
                logger.warning(f"Failed to load skill {skill_id}: {e}")
                continue

        # Sort by name
        skills.sort(key=lambda s: s["name"])

        return {
            "skills": skills,
            "active_skill": cm.state.get("active_skill", ""),
            "total_mastered": sum(s["mastered_level"] for s in skills),
        }

    except Exception as e:
        logger.error(f"Failed to load skills from curriculum: {e}")
        return {"skills": [], "error": str(e)}


def serve_skills(handler: "TavernHandler"):
    """
    GET /api/skills - List all skills with levels and progress.
    """
    try:
        data = _get_skills_data()
        handler._send_json(data)
    except Exception as e:
        logger.error(f"Skills error: {e}")
        handler._send_json({"skills": [], "error": str(e)})


def serve_skill_data(handler: "TavernHandler", skill_id: str):
    """
    GET /api/skill/{skill_id} - Full skill data with config + state.
    """
    try:
        import yaml

        base_dir = paths.get_base_dir()
        skills_dir = base_dir / "configs" / "skills"
        yaml_file = skills_dir / f"{skill_id}.yaml"

        if not yaml_file.exists():
            handler._send_json({"error": f"Skill '{skill_id}' not found"}, 404)
            return

        # Load full YAML config
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        # Load curriculum state using CurriculumManager (single source of truth)
        from data_manager.curriculum_manager import CurriculumManager

        cm = CurriculumManager(base_dir, {})
        skill_state = cm.state.get("skills", {}).get(skill_id, {})

        # Build response
        max_level = config.get("max_level", 30)
        mastered = cm.get_mastered_level(skill_id)
        training = cm.get_training_level(skill_id)

        # Fill in missing levels (extrapolate)
        level_prog = config.get("level_progression", {})
        if level_prog and len(level_prog) < max_level:
            defined_levels = [int(k) for k in level_prog.keys() if str(k).isdigit()]
            if defined_levels:
                last_level = max(defined_levels)
                last_data = level_prog.get(last_level, level_prog.get(str(last_level), {}))
                for lvl in range(last_level + 1, max_level + 1):
                    level_prog[str(lvl)] = {
                        "name": f"Level {lvl}",
                        "desc": f"Beyond L{last_level} - extrapolated",
                        **{k: v for k, v in last_data.items() if k != "name"}
                    }
                config["level_progression"] = level_prog

        # Get accuracy history
        history = skill_state.get("accuracy_history", [])

        # Recent accuracy
        recent_acc = 0
        if history:
            recent = history[-3:]
            recent_acc = sum(r.get("accuracy", 0) for r in recent) / len(recent) * 100

        # Count evals at training level
        at_level = [r for r in history if r.get("training_level", r.get("level")) == training]

        response = {
            "id": skill_id,
            "config": config,
            "state": {
                "mastered_level": mastered,
                "training_level": training,
                "accuracy": round(recent_acc, 1),
                "eval_count": len(at_level),
                "total_evals": len(history),
                "accuracy_history": history[-20:],
            },
            "is_active": cm.state.get("active_skill") == skill_id,
        }

        handler._send_json(response)

    except Exception as e:
        logger.error(f"Skill data error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_titles(handler: "TavernHandler"):
    """
    GET /api/titles - Hero titles based on current training state.
    """
    try:
        from guild.titles import get_titles

        base_dir = paths.get_base_dir()

        # Get training status for total steps
        status_file = base_dir / "status" / "training_status.json"
        total_steps = 0
        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)
                total_steps = status.get("global_step", 0)

        # Get skill states from curriculum
        skill_states = {}
        skills_data = _get_skills_data()
        total_level = 0
        for skill in skills_data.get("skills", []):
            skill_id = skill.get("id")
            if skill_id:
                mastered = skill.get("mastered_level", 0)
                total_level += mastered
                skill_states[skill_id] = {
                    "level": skill.get("training_level", 1),
                    "accuracy": skill.get("accuracy", 0) / 100.0,
                    "primitive_accuracy": {},
                }

        # Get titles
        result = get_titles(total_steps, total_level, skill_states)

        # Format response
        response = {
            "primary": {
                "id": result.primary.id,
                "name": result.primary.name,
                "description": result.primary.description,
                "category": result.primary.category,
            } if result.primary else None,
            "skill_titles": {
                k: {"id": v.id, "name": v.name, "description": v.description}
                for k, v in result.skill_titles.items()
            },
            "warnings": [
                {"id": w.id, "name": w.name, "description": w.description, "icon": w.icon}
                for w in result.warnings
            ],
            "achievements": [
                {"id": a.id, "name": a.name, "icon": a.icon}
                for a in result.achievements
            ],
            "total_count": len(result.all_titles),
            "total_steps": total_steps,
            "total_level": total_level,
        }
        handler._send_json(response)

    except Exception as e:
        logger.error(f"Titles error: {e}")
        handler._send_json({
            "primary": {"id": "unknown", "name": "Unknown", "description": ""},
            "skill_titles": {},
            "warnings": [],
            "achievements": [],
            "error": str(e)
        })


def serve_curriculum(handler: "TavernHandler"):
    """
    GET /api/curriculum - Curriculum state.
    """
    try:
        base_dir = paths.get_base_dir()
        curriculum_file = base_dir / "data_manager" / "curriculum_state.json"

        if not curriculum_file.exists():
            handler._send_json({
                "skills": {},
                "active_skill": None,
                "message": "No curriculum state yet"
            })
            return

        with open(curriculum_file) as f:
            data = json.load(f)

        handler._send_json(data)

    except Exception as e:
        logger.error(f"Curriculum error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_engine_health(handler: "TavernHandler"):
    """
    GET /api/engine/health - SkillEngine health check.
    """
    try:
        from guild.skills import get_engine

        engine = get_engine()
        health = engine.health_check()
        handler._send_json(health)

    except Exception as e:
        logger.error(f"Engine health error: {e}")
        handler._send_json({
            "total_skills": 0,
            "healthy": 0,
            "unhealthy": [],
            "error": str(e)
        }, 500)


def serve_primitive_summary(handler: "TavernHandler"):
    """
    GET /api/engine/primitive-summary - Per-primitive accuracy across skills.
    """
    try:
        from guild.skills import get_engine

        engine = get_engine()
        summary = engine.get_primitive_summary()
        handler._send_json(summary)

    except Exception as e:
        logger.error(f"Primitive summary error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_skill_state(handler: "TavernHandler", skill_id: str):
    """
    GET /api/engine/skill/{skill_id}/state - Skill state from engine.
    """
    try:
        from guild.skills import get_engine

        engine = get_engine()
        state = engine.get_state(skill_id)
        handler._send_json(state.to_dict())

    except Exception as e:
        logger.error(f"Skill state error for {skill_id}: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_skill_primitives(handler: "TavernHandler", skill_id: str):
    """
    GET /api/engine/skill/{skill_id}/primitives - Primitives for a skill.
    """
    try:
        from guild.skills import get_engine

        engine = get_engine()
        skill = engine.get(skill_id)

        primitives = []
        if hasattr(skill, 'primitives'):
            for prim in skill.primitives:
                primitives.append({
                    "name": prim.name,
                    "track": prim.track,
                    "version": prim.version,
                })

        handler._send_json({
            "skill_id": skill_id,
            "primitives": primitives,
            "count": len(primitives),
        })

    except Exception as e:
        logger.error(f"Skill primitives error for {skill_id}: {e}")
        handler._send_json({"error": str(e)}, 500)
