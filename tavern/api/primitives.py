"""
Primitives API

Exposes primitive catalog and hero mastery data for visualization.
Handles:
- /api/primitives - Get all primitives by track
- /api/primitives/stats - Get hero's primitive mastery stats
"""

import json
import logging
from typing import TYPE_CHECKING

from guild.skills.primitives import (
    PRIMITIVE_CATALOG,
    list_primitives,
    list_tracks,
)

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


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

        for track in tracks:
            prims = PRIMITIVE_CATALOG.get(track, [])
            if not prims:
                continue

            avg_difficulty = sum(p.difficulty for p in prims) / len(prims)

            # TODO: Compute actual mastery from evaluation results
            # For now, return 0 (no data) - will be populated when
            # per-primitive tracking is implemented
            mastery = 0.0

            track_stats[track] = {
                "name": track_names.get(track, track.title()),
                "count": len(prims),
                "avg_difficulty": round(avg_difficulty, 1),
                "mastery": mastery,
                "icon": track_icons.get(track, "📊"),
            }

        total_count = sum(s["count"] for s in track_stats.values())

        handler._send_json({
            "tracks": track_stats,
            "total_primitives": total_count,
            "hero_level": 0,  # TODO: Get from campaign state
        })

    except Exception as e:
        logger.error(f"Primitive stats API error: {e}")
        handler._send_json({"error": str(e), "tracks": {}})
