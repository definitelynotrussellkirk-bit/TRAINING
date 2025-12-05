"""
Curriculum Feedback - Connect Blessing Quality to Curriculum
=============================================================

Closes the feedback loop between evaluation (Temple/Blessing) and
curriculum (skill levels, difficulty adjustments).

The blessing quality_factor directly informs curriculum decisions:
- High quality (≥0.8): Consider level up, increase difficulty
- Medium quality (0.4-0.8): Continue current level
- Low quality (<0.4): Back off difficulty, focus on weak areas

Usage:
    from guild.curriculum_feedback import CurriculumFeedback

    feedback = CurriculumFeedback(campaign_id="campaign-001")

    # Get recommendation after blessing
    recommendation = feedback.get_recommendation(blessing)

    # Apply feedback to curriculum
    feedback.apply_to_curriculum(blessing, curriculum_manager)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# FEEDBACK ACTIONS
# =============================================================================

class FeedbackAction(Enum):
    """Actions the feedback system can recommend."""
    CONTINUE = "continue"           # Keep training at current level
    LEVEL_UP = "level_up"           # Progress to next skill level
    BACK_OFF = "back_off"           # Reduce difficulty (maybe level down)
    FOCUS_WEAK = "focus_weak"       # Focus on weak primitives/skills
    INCREASE_DATA = "increase_data" # Need more training data at this level
    CHANGE_SKILL = "change_skill"   # Switch to a different skill


@dataclass
class CurriculumRecommendation:
    """Recommendation for curriculum adjustment based on blessing."""
    action: FeedbackAction
    confidence: float  # 0-1: how confident in this recommendation
    reason: str
    details: Dict[str, Any]

    # Specific adjustments
    suggested_skill: Optional[str] = None
    suggested_level: Optional[int] = None
    data_multiplier: float = 1.0  # Increase/decrease data amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "details": self.details,
            "suggested_skill": self.suggested_skill,
            "suggested_level": self.suggested_level,
            "data_multiplier": self.data_multiplier,
        }


# =============================================================================
# FEEDBACK SYSTEM
# =============================================================================

class CurriculumFeedback:
    """
    Analyzes blessing quality and recommends curriculum adjustments.

    The core insight: blessing quality_factor encapsulates:
    - accuracy (60% weight)
    - skill coverage (20% weight)
    - improvement trend (20% weight)

    This is exactly what we need to decide curriculum actions.
    """

    def __init__(
        self,
        campaign_id: Optional[str] = None,
        high_quality_threshold: float = 0.8,
        low_quality_threshold: float = 0.4,
    ):
        self.campaign_id = campaign_id
        self.high_threshold = high_quality_threshold
        self.low_threshold = low_quality_threshold

    def get_recommendation(
        self,
        blessing: "Blessing",  # noqa: F821 - forward reference
        current_skill: Optional[str] = None,
        current_level: Optional[int] = None,
    ) -> CurriculumRecommendation:
        """
        Get curriculum recommendation based on blessing quality.

        Args:
            blessing: The blessing from Temple auto-blessing
            current_skill: Current skill being trained
            current_level: Current training level

        Returns:
            CurriculumRecommendation with suggested action
        """
        quality = blessing.quality_factor

        # High quality - consider advancement
        if quality >= self.high_threshold:
            return self._recommend_advancement(blessing, current_skill, current_level)

        # Low quality - need to back off
        if quality < self.low_threshold:
            return self._recommend_backoff(blessing, current_skill, current_level)

        # Medium quality - continue but maybe adjust
        return self._recommend_continue(blessing, current_skill, current_level)

    def _recommend_advancement(
        self,
        blessing: "Blessing",
        current_skill: Optional[str],
        current_level: Optional[int],
    ) -> CurriculumRecommendation:
        """Recommend advancement when quality is high."""
        # Check if we should level up or change skill
        reason_parts = blessing.reason.split(", ") if blessing.reason else []

        # Parse coverage from reason (e.g., "coverage=100%")
        coverage = 1.0
        for part in reason_parts:
            if "coverage=" in part:
                try:
                    coverage = float(part.split("=")[1].rstrip("%")) / 100
                except:
                    pass

        # If coverage is low, suggest switching skills
        if coverage < 0.5:
            return CurriculumRecommendation(
                action=FeedbackAction.CHANGE_SKILL,
                confidence=0.7,
                reason=f"High accuracy but low coverage ({coverage:.0%}). Try another skill.",
                details={
                    "quality": blessing.quality_factor,
                    "coverage": coverage,
                    "verdict": blessing.verdict,
                },
                data_multiplier=1.0,
            )

        # High quality + good coverage = level up
        return CurriculumRecommendation(
            action=FeedbackAction.LEVEL_UP,
            confidence=0.9,
            reason=f"Quality {blessing.quality_factor:.0%} >= {self.high_threshold:.0%}. Ready for next level.",
            details={
                "quality": blessing.quality_factor,
                "verdict": blessing.verdict,
            },
            suggested_skill=current_skill,
            suggested_level=(current_level + 1) if current_level else None,
            data_multiplier=1.0,
        )

    def _recommend_backoff(
        self,
        blessing: "Blessing",
        current_skill: Optional[str],
        current_level: Optional[int],
    ) -> CurriculumRecommendation:
        """Recommend backing off when quality is low."""
        # Parse trend from reason
        is_declining = "declining" in (blessing.reason or "")

        # If declining and low quality, definitely back off
        if is_declining:
            return CurriculumRecommendation(
                action=FeedbackAction.BACK_OFF,
                confidence=0.9,
                reason=f"Quality {blessing.quality_factor:.0%} and declining. Reduce difficulty.",
                details={
                    "quality": blessing.quality_factor,
                    "verdict": blessing.verdict,
                    "trend": "declining",
                },
                suggested_skill=current_skill,
                suggested_level=max(1, (current_level or 1) - 1),
                data_multiplier=0.5,  # Less data, more focused
            )

        # Low quality but stable - maybe need more data
        return CurriculumRecommendation(
            action=FeedbackAction.INCREASE_DATA,
            confidence=0.7,
            reason=f"Quality {blessing.quality_factor:.0%} is low. Need more training data.",
            details={
                "quality": blessing.quality_factor,
                "verdict": blessing.verdict,
            },
            suggested_skill=current_skill,
            data_multiplier=1.5,  # More data
        )

    def _recommend_continue(
        self,
        blessing: "Blessing",
        current_skill: Optional[str],
        current_level: Optional[int],
    ) -> CurriculumRecommendation:
        """Recommend continuing when quality is medium."""
        # Check trend
        is_improving = "improving" in (blessing.reason or "")

        if is_improving:
            return CurriculumRecommendation(
                action=FeedbackAction.CONTINUE,
                confidence=0.8,
                reason=f"Quality {blessing.quality_factor:.0%} and improving. Keep training.",
                details={
                    "quality": blessing.quality_factor,
                    "verdict": blessing.verdict,
                    "trend": "improving",
                },
                suggested_skill=current_skill,
                suggested_level=current_level,
                data_multiplier=1.0,
            )

        # Stable but not great - continue with attention
        return CurriculumRecommendation(
            action=FeedbackAction.CONTINUE,
            confidence=0.6,
            reason=f"Quality {blessing.quality_factor:.0%}. Continue training at current level.",
            details={
                "quality": blessing.quality_factor,
                "verdict": blessing.verdict,
            },
            suggested_skill=current_skill,
            suggested_level=current_level,
            data_multiplier=1.0,
        )

    def apply_to_curriculum(
        self,
        recommendation: CurriculumRecommendation,
        curriculum_manager: "CurriculumManager",  # noqa: F821
        skill: str,
        auto_apply: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply recommendation to curriculum manager.

        Args:
            recommendation: The recommendation to apply
            curriculum_manager: CurriculumManager instance
            skill: Skill to modify
            auto_apply: If True, actually make changes. If False, just return what would happen.

        Returns:
            Dict with action taken and new state
        """
        result = {
            "action": recommendation.action.value,
            "auto_applied": auto_apply,
            "changes": [],
        }

        if recommendation.action == FeedbackAction.LEVEL_UP:
            if auto_apply:
                new_level_config = curriculum_manager.progress_to_next_level(skill)
                result["changes"].append({
                    "type": "level_up",
                    "new_level": new_level_config.get("level"),
                    "new_name": new_level_config.get("name"),
                })
            else:
                result["would_do"] = f"Progress {skill} to next level"

        elif recommendation.action == FeedbackAction.BACK_OFF:
            if auto_apply and recommendation.suggested_level:
                # Set level manually (backing off)
                old_level = curriculum_manager.get_mastered_level(skill)
                curriculum_manager.state["skills"][skill]["current_level"] = recommendation.suggested_level
                curriculum_manager._save_state()
                result["changes"].append({
                    "type": "back_off",
                    "old_level": old_level,
                    "new_level": recommendation.suggested_level,
                })
            else:
                result["would_do"] = f"Back off {skill} to level {recommendation.suggested_level}"

        elif recommendation.action == FeedbackAction.CHANGE_SKILL:
            if auto_apply:
                # Get next skill from rotation
                next_skill = curriculum_manager.get_next_rotation_skill()
                result["changes"].append({
                    "type": "skill_change",
                    "new_skill": next_skill,
                })
            else:
                result["would_do"] = "Switch to next skill in rotation"

        elif recommendation.action == FeedbackAction.INCREASE_DATA:
            result["data_multiplier"] = recommendation.data_multiplier
            result["would_do"] = f"Increase data generation by {recommendation.data_multiplier}x"

        return result


# =============================================================================
# INTEGRATION HOOKS
# =============================================================================

def process_blessing_feedback(
    blessing: "Blessing",  # noqa: F821
    campaign_id: Optional[str] = None,
    auto_apply: bool = False,
) -> CurriculumRecommendation:
    """
    Process a blessing and get/apply curriculum feedback.

    This is the main entry point called after auto_bless_checkpoint().

    Args:
        blessing: The blessing from Temple
        campaign_id: Campaign ID for context
        auto_apply: Whether to automatically apply curriculum changes

    Returns:
        CurriculumRecommendation with action taken
    """
    from data_manager.curriculum_manager import CurriculumManager
    from core.paths import get_base_dir

    feedback = CurriculumFeedback(campaign_id=campaign_id)

    # Get curriculum manager
    cm = CurriculumManager(get_base_dir(), {})

    # Determine current skill/level
    rotation_status = cm.get_rotation_status()
    current_skill = rotation_status.get("next_skill") or cm.state.get("active_skill", "bin")
    current_level = cm.get_training_level(current_skill)

    # Get recommendation
    recommendation = feedback.get_recommendation(
        blessing,
        current_skill=current_skill,
        current_level=current_level,
    )

    # Apply if requested
    if auto_apply:
        feedback.apply_to_curriculum(recommendation, cm, current_skill, auto_apply=True)

    # Log the recommendation
    logger.info(
        f"[Curriculum Feedback] {recommendation.action.value}: {recommendation.reason} "
        f"(confidence={recommendation.confidence:.0%})"
    )

    return recommendation


def hook_auto_blessing(checkpoint_step: int, campaign_id: Optional[str] = None):
    """
    Hook to be called after auto_bless_checkpoint().

    Computes blessing and processes curriculum feedback.
    """
    from temple.auto_blessing import auto_bless_checkpoint

    # Get blessing
    blessing = auto_bless_checkpoint(checkpoint_step, campaign_id)

    if blessing is None:
        logger.debug(f"No blessing for checkpoint {checkpoint_step}, skipping feedback")
        return None

    # Process feedback (don't auto-apply by default)
    return process_blessing_feedback(blessing, campaign_id, auto_apply=False)
