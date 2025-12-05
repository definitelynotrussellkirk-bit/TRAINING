"""
Primitives API

Exposes primitive catalog and hero mastery data for visualization.
Handles:
- /api/primitives - Get all primitives by track
- /api/primitives/stats - Get hero's primitive mastery stats
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional

from guild.skills.primitives import (
    PRIMITIVE_CATALOG,
    list_primitives,
    list_tracks,
)

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent


def _get_hero_stats() -> Dict[str, Any]:
    """Get hero level and skill stats from curriculum state."""
    curriculum_file = BASE_DIR / "data_manager" / "curriculum_state.json"
    result = {"hero_level": 0, "skills": {}}

    if not curriculum_file.exists():
        return result

    try:
        with open(curriculum_file) as f:
            curriculum = json.load(f)

        # Get current step from latest accuracy entry in any skill
        latest_step = 0
        skills = curriculum.get("skills", {})
        for skill_id, skill_data in skills.items():
            acc_history = skill_data.get("accuracy_history", [])
            if acc_history:
                step = acc_history[-1].get("step", 0)
                if step > latest_step:
                    latest_step = step

            result["skills"][skill_id] = {
                "current_level": skill_data.get("current_level", 1),
            }

        # Hero level = steps / 1000
        result["hero_level"] = latest_step // 1000

        return result
    except Exception as e:
        logger.warning(f"Failed to get hero stats: {e}")
        return result


def serve_primitives(handler: "TavernHandler"):
    """
    GET /api/primitives - List all primitives organized by track.

    Returns:
        {
            "tracks": ["arithmetic", "binary", "logic", "string", "code"],
            "primitives": {
                "arithmetic": [
                    {"name": "add_single_digit_no_carry", "display_name": "...", ...},
                    ...
                ],
                ...
            },
            "total_count": 39
        }
    """
    try:
        tracks = list_tracks()
        primitives_by_track = {}

        for track in tracks:
            prims = PRIMITIVE_CATALOG.get(track, [])
            primitives_by_track[track] = [
                {
                    "id": str(p.id),
                    "name": p.id.name,
                    "track": p.id.track,
                    "display_name": p.display_name,
                    "description": p.description,
                    "difficulty": p.difficulty,
                    "prerequisites": p.prerequisites,
                    "tags": p.tags,
                }
                for p in prims
            ]

        total_count = sum(len(v) for v in primitives_by_track.values())

        handler._send_json({
            "tracks": tracks,
            "primitives": primitives_by_track,
            "total_count": total_count,
        })

    except Exception as e:
        logger.error(f"Primitives API error: {e}")
        handler._send_json({"error": str(e), "primitives": {}})


def serve_primitive_stats(handler: "TavernHandler"):
    """
    GET /api/primitives/stats - Get hero's mastery stats by track.

    Returns primitive mastery data for radar chart visualization.
    Currently returns difficulty-weighted stats since we don't track
    per-primitive accuracy yet.

    Returns:
        {
            "tracks": {
                "arithmetic": {
                    "name": "Arithmetic",
                    "count": 11,
                    "avg_difficulty": 1.9,
                    "mastery": 0.0,  # Future: computed from eval results
                    "icon": "🔢"
                },
                ...
            },
            "total_primitives": 39,
            "hero_level": 0
        }
    """
    try:
        tracks = list_tracks()
        track_stats = {}

        track_icons = {
            "arithmetic": "🔢",
            "binary": "💻",
            "logic": "🧠",
            "string": "📝",
            "code": "⚙️",
        }

        track_names = {
            "arithmetic": "Arithmetic",
            "binary": "Binary",
            "logic": "Logic",
            "string": "String",
            "code": "Code",
        }

        # Get hero stats for level and skill mastery proxy
        hero_stats = _get_hero_stats()

        # Map tracks to skills for mastery proxy
        # Tracks map to skills that exercise their primitives
        track_to_skills = {
            "arithmetic": ["bin"],
            "binary": ["bin"],
            "logic": ["bin", "sy"],
            "string": ["sy"],
            "code": [],
        }

        for track in tracks:
            prims = PRIMITIVE_CATALOG.get(track, [])
            if not prims:
                continue

            avg_difficulty = sum(p.difficulty for p in prims) / len(prims)

            # Compute mastery as proxy: average of skill_level / 50 for related skills
            # This is a rough approximation until per-primitive tracking is implemented
            mastery = 0.0
            related_skills = track_to_skills.get(track, [])
            if related_skills:
                skill_levels = []
                for skill_id in related_skills:
                    if skill_id in hero_stats.get("skills", {}):
                        level = hero_stats["skills"][skill_id].get("current_level", 1)
                        skill_levels.append(level / 50.0)  # Normalize to 0-1
                if skill_levels:
                    mastery = sum(skill_levels) / len(skill_levels)

            track_stats[track] = {
                "name": track_names.get(track, track.title()),
                "count": len(prims),
                "avg_difficulty": round(avg_difficulty, 1),
                "mastery": round(mastery, 2),
                "icon": track_icons.get(track, "📊"),
            }

        total_count = sum(s["count"] for s in track_stats.values())

        handler._send_json({
            "tracks": track_stats,
            "total_primitives": total_count,
            "hero_level": hero_stats["hero_level"],
        })

    except Exception as e:
        logger.error(f"Primitive stats API error: {e}")
        handler._send_json({"error": str(e), "tracks": {}})
