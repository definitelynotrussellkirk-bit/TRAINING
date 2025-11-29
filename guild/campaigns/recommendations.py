"""
Campaign Recommendations - "What should the user do next?"

Given the current campaign state, computes a concrete next action.
This is the "Next Action brain" - keeps things moving forward.

Usage:
    from guild.campaigns.recommendations import compute_recommendation

    campaign = load_active_campaign()
    rec = compute_recommendation(campaign)
    # rec = {
    #     "kind": "train_steps",
    #     "title": "Run a short training session",
    #     "description": "Push this hero further.",
    #     "suggested_steps": 2000,
    #     "reason": "Small sessions keep momentum.",
    # }
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from guild.campaigns.types import Campaign


def compute_recommendation(c: Campaign) -> Dict[str, Any]:
    """
    Compute the next recommended action for a campaign.

    Args:
        c: The active campaign

    Returns:
        Dict with:
        - kind: Action type ("train_steps", "run_eval", "create_quest", etc.)
        - title: Human-readable action title
        - description: What this will do
        - suggested_steps: For training, how many steps
        - reason: Why this is the recommendation
    """
    now = datetime.now(timezone.utc)

    # Default: "do something small"
    rec: Dict[str, Any] = {
        "kind": "train_steps",
        "title": "Run a short training session",
        "description": "Push this hero a bit further in the current campaign.",
        "suggested_steps": 2000,
        "reason": "Small, frequent sessions keep the hero improving.",
    }

    # First run ever
    if c.current_step == 0:
        rec.update({
            "title": "Start this hero's first training run",
            "description": "Begin the journey! This will train the hero on available quests.",
            "suggested_steps": 1000,
            "reason": "This campaign has never been trained; kick off the journey.",
        })
        return rec

    # Very early (< 5k steps) - keep sessions short
    if c.current_step < 5000:
        rec.update({
            "title": "Continue early training",
            "description": f"Hero is at step {c.current_step:,}. Keep building momentum.",
            "suggested_steps": 1000,
            "reason": "Early training benefits from short, focused sessions.",
        })
        return rec

    # Check if we have peak skills - if so, maybe suggest pushing further
    if c.peak_skill_levels:
        highest_skill = max(c.peak_skill_levels.items(), key=lambda x: x[1])
        skill_id, level = highest_skill
        rec.update({
            "description": f"Current best: {skill_id.upper()} at L{level}. Keep pushing!",
        })

    # Default session size scales with progress
    if c.current_step < 20000:
        rec["suggested_steps"] = 2000
    elif c.current_step < 50000:
        rec["suggested_steps"] = 3000
    else:
        rec["suggested_steps"] = 5000

    return rec


def compute_recommendation_with_context(
    c: Campaign,
    queue_files: int = 0,
    is_training: bool = False,
) -> Dict[str, Any]:
    """
    Compute recommendation with additional context.

    Args:
        c: Campaign
        queue_files: Number of files in training queue
        is_training: Whether training is currently running

    Returns:
        Recommendation dict
    """
    if is_training:
        return {
            "kind": "wait",
            "title": "Training in progress",
            "description": "The hero is currently training. Watch the battle unfold!",
            "suggested_steps": 0,
            "reason": "Training is already running.",
        }

    if queue_files == 0:
        return {
            "kind": "create_quest",
            "title": "Generate training data",
            "description": "The quest board is empty. Generate some training quests first.",
            "suggested_steps": 0,
            "reason": "No training data available in the queue.",
            "suggested_action": "open_quests",
        }

    return compute_recommendation(c)
