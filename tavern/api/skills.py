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


def _get_passives_summary():
    """Get passive eval summary for deriving skill levels."""
    try:
        passives_file = paths.get_base_dir() / "status" / "passives_ledger.json"
        if not passives_file.exists():
            return {}

        with open(passives_file) as f:
            data = json.load(f)

        results = data.get("results", [])
        by_passive = {}
        for r in results:
            name = r.get("passive_name", r.get("passive_id", "unknown"))
            acc = r.get("accuracy", 0)
            if name not in by_passive:
                by_passive[name] = {"count": 0, "best": 0, "recent": [], "history": []}
            by_passive[name]["count"] += 1
            by_passive[name]["best"] = max(by_passive[name]["best"], acc)
            by_passive[name]["recent"].append(acc)
            # Keep only last 10 for averaging
            if len(by_passive[name]["recent"]) > 10:
                by_passive[name]["recent"] = by_passive[name]["recent"][-10:]
            # Build accuracy_history compatible format for charts
            by_passive[name]["history"].append({
                "accuracy": acc,
                "timestamp": r.get("timestamp"),
                "training_level": 1,  # Passives don't have levels
            })
            # Keep last 20 for history
            if len(by_passive[name]["history"]) > 20:
                by_passive[name]["history"] = by_passive[name]["history"][-20:]

        return by_passive
    except Exception as e:
        logger.debug(f"Could not load passives summary: {e}")
        return {}


def _get_level_accuracy(skill_id: str, max_level: int = 20) -> dict:
    """Get per-level accuracy breakdown from eval ledger.

    Returns dict like: {1: 0.95, 2: 0.88, 3: 0.72, ...}
    Only includes levels that have been evaluated.
    """
    try:
        from core.evaluation_ledger import get_eval_ledger
        base_dir = paths.get_base_dir()
        ledger = get_eval_ledger(base_dir)

        level_acc = {}
        for level in range(1, max_level + 1):
            evals = ledger.get_by_skill(skill_id, level=level)
            if evals:
                # Use the latest eval for this level
                latest = evals[-1]
                level_acc[level] = round(latest.accuracy, 2)

        return level_acc
    except Exception as e:
        logger.debug(f"Could not load level accuracy for {skill_id}: {e}")
        return {}


def _get_strain_zone(accuracy: float) -> dict:
    """Get strain zone based on recent accuracy.

    Strain zones based on accuracy:
    - Recovery: Accuracy > 90% (too easy, under-challenged)
    - Productive: Accuracy 75-90% (optimal learning zone)
    - Stretch: Accuracy 50-75% (challenging but sustainable)
    - Overload: Accuracy < 50% (too hard)

    Returns zone info with color, icon, and hint.
    """
    if accuracy is None or accuracy == 0:
        return {
            "zone": "unknown",
            "zone_color": "#6b7280",
            "zone_icon": "?",
            "zone_hint": "No evaluation data"
        }

    # Accuracy is already a percentage (0-100)
    if accuracy > 90:
        return {
            "zone": "recovery",
            "zone_color": "#3b82f6",  # Blue
            "zone_icon": "↓",
            "zone_hint": "Under-challenged - ready to level up"
        }
    elif accuracy >= 75:
        return {
            "zone": "productive",
            "zone_color": "#22c55e",  # Green
            "zone_icon": "✓",
            "zone_hint": "Optimal learning zone"
        }
    elif accuracy >= 50:
        return {
            "zone": "stretch",
            "zone_color": "#f59e0b",  # Amber
            "zone_icon": "↑",
            "zone_hint": "Challenging - keep pushing"
        }
    else:
        return {
            "zone": "overload",
            "zone_color": "#ef4444",  # Red
            "zone_icon": "⚠",
            "zone_hint": "Struggling - may need remediation"
        }


def _get_skills_data():
    """Load skill data from CurriculumManager, falling back to passives when empty."""
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

        # Get passive eval summary for fallback
        passives_summary = _get_passives_summary()

        skills = []
        curriculum_skills = cm.state.get("skills", {})

        # If curriculum has skills, use those
        if curriculum_skills:
            for skill_id in curriculum_skills.keys():
                try:
                    config = load_skill_config(skill_id)
                    mastered = cm.get_mastered_level(skill_id)
                    training = cm.get_training_level(skill_id)
                    skill_state = cm.state["skills"][skill_id]
                    history = skill_state.get("accuracy_history", [])

                    recent_acc = 0
                    last_eval_time = None
                    last_single_acc = 0
                    if history:
                        recent = history[-3:]
                        recent_acc = (sum(r.get("accuracy", 0) for r in recent) / len(recent)) * 100
                        last_entry = history[-1]
                        last_eval_time = last_entry.get("timestamp")
                        last_single_acc = last_entry.get("accuracy", 0) * 100

                    ledger_count = eval_counts.get(skill_id, 0)

                    # Fallback to passive evals if ledger is empty
                    passive_count = 0
                    from_passives = False
                    passive_name_map = {
                        "bin": ["binary_arithmetic", "arithmetic", "bin"],
                        "sy": ["syllacrostic", "syllo", "sy", "word_puzzles"],
                    }
                    for name in passive_name_map.get(skill_id, [skill_id]):
                        if name in passives_summary:
                            passive_count += passives_summary[name].get("count", 0)
                            if not recent_acc and passives_summary[name].get("recent"):
                                recent_accs = passives_summary[name]["recent"]
                                recent_acc = (sum(recent_accs) / len(recent_accs)) * 100

                    # Use passive count as fallback if ledger is empty
                    final_eval_count = ledger_count if ledger_count > 0 else passive_count
                    if ledger_count == 0 and passive_count > 0:
                        from_passives = True

                    # Get per-level accuracy breakdown
                    level_accuracy = _get_level_accuracy(skill_id, max_level=config.max_level)

                    # Get strain zone based on recent accuracy
                    strain_zone = _get_strain_zone(recent_acc)

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
                        "eval_count": final_eval_count,
                        "from_passives": from_passives,
                        "category": config.category.value if hasattr(config.category, 'value') else str(config.category),
                        "description": config.description or "",
                        "level_accuracy": level_accuracy,
                        **strain_zone,  # zone, zone_color, zone_icon, zone_hint
                    })
                except Exception as e:
                    logger.warning(f"Failed to load skill {skill_id}: {e}")
                    continue
        else:
            # Fallback: Load all skill configs and use passive data
            skills_dir = base_dir / "configs" / "skills"
            if skills_dir.exists():
                for yaml_file in skills_dir.glob("*.yaml"):
                    skill_id = yaml_file.stem
                    try:
                        config = load_skill_config(skill_id)

                        # Get passive data for this skill (passives often match skill names)
                        passive_data = passives_summary.get(skill_id, {})
                        best_acc = passive_data.get("best", 0)
                        eval_count = passive_data.get("count", 0)
                        recent_accs = passive_data.get("recent", [])
                        recent_acc = (sum(recent_accs) / len(recent_accs) * 100) if recent_accs else 0

                        # Derive "effective level" from best accuracy:
                        # 80%+ = level 1, each additional 5% = +1 level (rough heuristic)
                        effective_level = 0
                        if best_acc >= 0.80:
                            effective_level = 1 + int((best_acc - 0.80) / 0.05)
                        effective_level = min(effective_level, config.max_level)

                        # Get per-level accuracy breakdown (may be empty for passives-only)
                        level_accuracy = _get_level_accuracy(skill_id, max_level=config.max_level)

                        # Get strain zone based on recent accuracy
                        strain_zone = _get_strain_zone(recent_acc)

                        skills.append({
                            "id": config.id,
                            "name": config.name,
                            "rpg_name": config.rpg_name or config.name,
                            "rpg_description": config.rpg_description or config.description or "",
                            "icon": config.display.icon if config.display else "⚔️",
                            "color": config.display.color if config.display else "#888",
                            "short_name": config.display.short_name if config.display else config.id.upper(),
                            "max_level": config.max_level,
                            "mastered_level": effective_level,
                            "training_level": effective_level + 1 if effective_level < config.max_level else effective_level,
                            "accuracy": round(recent_acc, 1),
                            "last_accuracy": round(best_acc * 100, 1),
                            "last_eval_time": None,
                            "eval_count": eval_count,
                            "category": config.category.value if hasattr(config.category, 'value') else str(config.category),
                            "description": config.description or "",
                            "from_passives": True,  # Indicate data source
                            "level_accuracy": level_accuracy,
                            **strain_zone,  # zone, zone_color, zone_icon, zone_hint
                        })
                    except Exception as e:
                        logger.debug(f"Skipping skill {skill_id}: {e}")
                        continue

        # Sort by name
        skills.sort(key=lambda s: s["name"])

        # Calculate total from passives if no curriculum
        total_passive_evals = sum(p.get("count", 0) for p in passives_summary.values())

        return {
            "skills": skills,
            "active_skill": cm.state.get("active_skill", ""),
            "total_mastered": sum(s["mastered_level"] for s in skills),
            "total_passive_evals": total_passive_evals,
            "passive_count": len(passives_summary),
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

        # Get passive eval counts for this skill (fallback when curriculum is empty)
        passives_summary = _get_passives_summary()
        # Map skill IDs to passive names (passives use different naming)
        passive_name_map = {
            "bin": ["binary_arithmetic", "arithmetic", "bin"],
            "sy": ["syllacrostic", "syllo", "sy", "word_puzzles"],
        }
        passive_count = 0
        passive_accuracy = 0
        passive_history = []  # Build accuracy_history from passives
        for name in passive_name_map.get(skill_id, [skill_id]):
            if name in passives_summary:
                passive_count += passives_summary[name].get("count", 0)
                if passives_summary[name].get("recent"):
                    passive_accuracy = sum(passives_summary[name]["recent"]) / len(passives_summary[name]["recent"]) * 100
                # Get full history from passives ledger for accuracy chart
                if passives_summary[name].get("history"):
                    passive_history.extend(passives_summary[name]["history"])

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

        # Recent accuracy - prefer curriculum, fallback to passives
        recent_acc = 0
        if history:
            recent = history[-3:]
            recent_acc = sum(r.get("accuracy", 0) for r in recent) / len(recent) * 100
        elif passive_accuracy > 0:
            recent_acc = passive_accuracy

        # Count evals at training level
        at_level = [r for r in history if r.get("training_level", r.get("level")) == training]

        # Use passive counts as fallback when curriculum is empty
        eval_count = len(at_level) if at_level else passive_count
        total_evals = len(history) if history else passive_count

        # Use passive_history if curriculum history is empty
        final_history = history[-20:] if history else passive_history[-20:]

        # Get per-level accuracy breakdown
        level_accuracy = _get_level_accuracy(skill_id, max_level=max_level)

        response = {
            "id": skill_id,
            "config": config,
            "state": {
                "mastered_level": mastered,
                "training_level": training,
                "accuracy": round(recent_acc, 1),
                "eval_count": eval_count,
                "total_evals": total_evals,
                "passive_evals": passive_count,  # Include passive count separately
                "accuracy_history": final_history,
                "level_accuracy": level_accuracy,
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
