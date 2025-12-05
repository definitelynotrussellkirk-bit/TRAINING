"""
Strain API

Exposes strain/effort metrics for real-time training visualization.
Handles:
- /api/strain - Get current strain metrics and zone
- /api/strain/history - Get strain history for charts
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any

from core import paths
from guild.metrics.strain import StrainTracker, StrainZone, StrainMetrics

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)

# Global strain tracker (lazily initialized per skill)
_strain_trackers: Dict[str, StrainTracker] = {}


def _get_tracker(skill_id: str, floor: float = 0.01) -> StrainTracker:
    """Get or create a strain tracker for a skill."""
    if skill_id not in _strain_trackers:
        _strain_trackers[skill_id] = StrainTracker(floor=floor)
    return _strain_trackers[skill_id]


def _get_training_loss() -> Optional[float]:
    """Get current training loss from status file."""
    try:
        status_file = paths.get_base_dir() / "status" / "training_status.json"
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
            return data.get("loss")
    except Exception as e:
        logger.debug(f"Could not read training loss: {e}")
    return None


def _get_current_skill() -> Optional[str]:
    """Get current active skill from curriculum state."""
    try:
        curriculum_file = paths.get_base_dir() / "status" / "curriculum_state.json"
        if curriculum_file.exists():
            with open(curriculum_file) as f:
                data = json.load(f)
            return data.get("active_skill", "sy")
    except Exception:
        pass
    return "sy"


def _get_skill_floor(skill_id: str) -> float:
    """Get floor (target) loss for a skill from config or history."""
    # For now, use a reasonable default based on skill
    # In future, could derive from historical minimums
    floors = {
        "sy": 0.01,   # Syllacrostic - very low floor
        "bin": 0.02,  # Binary - slightly higher
    }
    return floors.get(skill_id, 0.02)


def serve_strain(handler: "TavernHandler"):
    """
    GET /api/strain - Current strain metrics.

    Returns:
        {
            "skill": "sy",
            "loss": 0.0233,
            "floor": 0.01,
            "strain": 0.0133,
            "zone": "productive",
            "zone_color": "#22c55e",
            "zone_icon": "✓",
            "hint": {
                "action": "continue",
                "reason": "Model is learning steadily"
            }
        }
    """
    try:
        # Get current state
        skill = _get_current_skill() or "sy"
        loss = _get_training_loss()
        floor = _get_skill_floor(skill)

        if loss is None:
            handler._send_json({
                "skill": skill,
                "loss": None,
                "floor": floor,
                "strain": None,
                "zone": "unknown",
                "zone_color": "#6b7280",
                "zone_icon": "?",
                "hint": {
                    "action": "wait",
                    "reason": "No training data available"
                },
                "training": False
            })
            return

        # Compute strain
        strain = max(0, loss - floor)

        # Determine zone
        if strain < 0.1:
            zone = StrainZone.RECOVERY
            zone_color = "#3b82f6"  # Blue
            zone_icon = "↓"
            hint_action = "level_up"
            hint_reason = "Model is coasting - consider increasing difficulty"
        elif strain < 0.3:
            zone = StrainZone.PRODUCTIVE
            zone_color = "#22c55e"  # Green
            zone_icon = "✓"
            hint_action = "continue"
            hint_reason = "Optimal learning zone - keep going"
        elif strain < 0.5:
            zone = StrainZone.STRETCH
            zone_color = "#f59e0b"  # Amber
            zone_icon = "↑"
            hint_action = "continue"
            hint_reason = "Challenging but sustainable"
        else:
            zone = StrainZone.OVERLOAD
            zone_color = "#ef4444"  # Red
            zone_icon = "⚠"
            hint_action = "back_off"
            hint_reason = "Too hard - consider reducing difficulty"

        handler._send_json({
            "skill": skill,
            "loss": round(loss, 4),
            "floor": floor,
            "strain": round(strain, 4),
            "zone": zone.value,
            "zone_color": zone_color,
            "zone_icon": zone_icon,
            "hint": {
                "action": hint_action,
                "reason": hint_reason
            },
            "training": True,
            # Zone thresholds for frontend visualization
            "thresholds": {
                "recovery": 0.1,
                "productive": 0.3,
                "stretch": 0.5,
            }
        })

    except Exception as e:
        logger.error(f"Strain API error: {e}")
        handler._send_json({
            "error": str(e),
            "zone": "unknown",
            "zone_color": "#6b7280",
            "zone_icon": "?",
        })


def serve_strain_zones(handler: "TavernHandler"):
    """
    GET /api/strain/zones - Zone definitions for UI.

    Returns zone metadata for building UI components.
    """
    zones = [
        {
            "id": "recovery",
            "name": "Recovery",
            "threshold_min": 0,
            "threshold_max": 0.1,
            "color": "#3b82f6",
            "icon": "↓",
            "description": "Under-challenged - model is coasting",
            "action": "Level up difficulty"
        },
        {
            "id": "productive",
            "name": "Productive",
            "threshold_min": 0.1,
            "threshold_max": 0.3,
            "color": "#22c55e",
            "icon": "✓",
            "description": "Optimal learning zone",
            "action": "Keep training"
        },
        {
            "id": "stretch",
            "name": "Stretch",
            "threshold_min": 0.3,
            "threshold_max": 0.5,
            "color": "#f59e0b",
            "icon": "↑",
            "description": "Challenging but sustainable",
            "action": "Monitor closely"
        },
        {
            "id": "overload",
            "name": "Overload",
            "threshold_min": 0.5,
            "threshold_max": 1.0,
            "color": "#ef4444",
            "icon": "⚠",
            "description": "Too hard - risk of destabilization",
            "action": "Reduce difficulty"
        },
    ]
    handler._send_json({"zones": zones})


def serve_efficiency(handler: "TavernHandler"):
    """
    GET /api/strain/efficiency - Skill efficiency comparison.

    Returns efficiency metrics per skill:
    - effort: Total cumulative strain spent
    - plastic_gain: Improvement (starting loss - current loss)
    - efficiency: plastic_gain / effort
    - accuracy: Current mastered level accuracy
    """
    try:
        # Load curriculum state
        curriculum_file = paths.get_base_dir() / "status" / "curriculum_state.json"
        curriculum = {}
        if curriculum_file.exists():
            with open(curriculum_file) as f:
                curriculum = json.load(f)

        # Load campaign data for peak metrics
        campaign_file = paths.get_base_dir() / "control" / "active_campaign.json"
        campaign = {}
        campaign_data = {}
        if campaign_file.exists():
            with open(campaign_file) as f:
                campaign = json.load(f)
            # Load the actual campaign.json
            campaign_path = paths.get_base_dir() / campaign.get("campaign_path", "") / "campaign.json"
            if campaign_path.exists():
                with open(campaign_path) as f:
                    campaign_data = json.load(f)

        # Load training status for current loss
        training_file = paths.get_base_dir() / "status" / "training_status.json"
        training = {}
        if training_file.exists():
            with open(training_file) as f:
                training = json.load(f)

        skills_data = curriculum.get("skills", {})
        skill_effort = campaign_data.get("skill_effort", {})
        peak_metrics = campaign_data.get("peak_metrics", {})

        # Build efficiency data per skill
        efficiency_data = {}
        all_skills = ["sy", "bin"]

        for skill_id in all_skills:
            skill_info = skills_data.get(skill_id, {})
            effort = skill_effort.get(skill_id, 0.0)

            # Get accuracy from latest eval (last entry in accuracy_history)
            accuracy_history = skill_info.get("accuracy_history", [])
            current_accuracy = 0.0
            if accuracy_history:
                # Get most recent accuracy for mastered level
                recent_evals = sorted(accuracy_history, key=lambda x: x.get("timestamp", ""), reverse=True)
                if recent_evals:
                    current_accuracy = recent_evals[0].get("accuracy", 0.0)

            # Get skill level
            current_level = skill_info.get("current_level", 0)

            # Estimate plastic gain from peak metrics
            # For now, use level as proxy for gain (each level = ~0.1 gain equivalent)
            plastic_gain = current_level * 0.1

            # Calculate efficiency
            efficiency = plastic_gain / effort if effort > 0 else 0.0

            efficiency_data[skill_id] = {
                "skill_id": skill_id,
                "skill_name": {"sy": "Syllacrostic", "bin": "Binary"}.get(skill_id, skill_id),
                "effort": round(effort, 2),
                "plastic_gain": round(plastic_gain, 3),
                "efficiency": round(efficiency, 4),
                "current_level": current_level,
                "current_accuracy": round(current_accuracy * 100, 1),
                "floor": _get_skill_floor(skill_id),
            }

        # Rank by efficiency
        ranked = sorted(efficiency_data.values(), key=lambda x: x["efficiency"], reverse=True)

        handler._send_json({
            "skills": efficiency_data,
            "ranking": [s["skill_id"] for s in ranked],
            "peak_metrics": peak_metrics,
            "current_loss": training.get("loss"),
            "current_step": training.get("step", 0),
        })

    except Exception as e:
        logger.error(f"Efficiency API error: {e}")
        handler._send_json({"error": str(e), "skills": {}})


def serve_level_transitions(handler: "TavernHandler"):
    """
    GET /api/strain/transitions - Level transition history.

    Returns when each skill level was mastered:
    - skill_id: Which skill
    - from_level: Previous mastered level
    - to_level: New mastered level
    - step: Training step when mastered
    - timestamp: When it happened
    """
    try:
        # Load curriculum state
        curriculum_file = paths.get_base_dir() / "status" / "curriculum_state.json"
        curriculum = {}
        if curriculum_file.exists():
            with open(curriculum_file) as f:
                curriculum = json.load(f)

        skills_data = curriculum.get("skills", {})
        all_transitions = []

        for skill_id, skill_info in skills_data.items():
            accuracy_history = skill_info.get("accuracy_history", [])

            # Sort by timestamp
            sorted_history = sorted(accuracy_history, key=lambda x: x.get("timestamp", ""))

            # Detect level transitions (when mastered_level changes)
            prev_mastered = 0
            for entry in sorted_history:
                current_mastered = entry.get("mastered_level", 0)
                if current_mastered > prev_mastered:
                    all_transitions.append({
                        "skill_id": skill_id,
                        "skill_name": {"sy": "Syllacrostic", "bin": "Binary"}.get(skill_id, skill_id),
                        "from_level": prev_mastered,
                        "to_level": current_mastered,
                        "step": entry.get("step", 0),
                        "timestamp": entry.get("timestamp", ""),
                        "accuracy": entry.get("accuracy", 0),
                    })
                    prev_mastered = current_mastered

        # Sort all transitions by timestamp (most recent first)
        all_transitions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Limit to recent transitions
        recent_transitions = all_transitions[:20]

        handler._send_json({
            "transitions": recent_transitions,
            "total_count": len(all_transitions),
        })

    except Exception as e:
        logger.error(f"Level transitions API error: {e}")
        handler._send_json({"error": str(e), "transitions": []})
