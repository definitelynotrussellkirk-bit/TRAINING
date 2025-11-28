"""
Scheduler Bridge - Routes scheduling decisions through SkillEngine.

Integrates with task_master and quest scheduling to:
1. Pick "next quest" based on SkillState metrics
2. Balance training across skills
3. Prioritize weak primitives
4. Schedule evals when appropriate

Design: Scheduler sees SkillState, not raw metrics.
This module provides the decision logic.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class QuestType(Enum):
    """Type of quest to schedule."""
    TRAINING = "training"     # Generate + train on skill data
    EVAL = "eval"             # Run eval to measure accuracy
    SPARRING = "sparring"     # Targeted practice on weak primitives
    MIXED = "mixed"           # Training + inline eval


@dataclass
class QuestRecommendation:
    """A recommended quest from the scheduler."""
    skill_id: str
    level: int
    quest_type: QuestType
    priority: float           # 0.0-1.0, higher = more urgent
    reason: str               # Human-readable explanation
    count: int = 100          # Number of examples
    target_primitives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "level": self.level,
            "quest_type": self.quest_type.value,
            "priority": self.priority,
            "reason": self.reason,
            "count": self.count,
            "target_primitives": self.target_primitives,
            "metadata": self.metadata,
        }


@dataclass
class SkillMetrics:
    """Metrics for scheduling decisions."""
    skill_id: str
    level: int
    accuracy: float
    last_eval_time: Optional[datetime]
    total_evals: int
    weak_primitives: List[str]  # Primitives below threshold
    time_since_eval: Optional[timedelta]


class SchedulerBridge:
    """
    Bridges SkillEngine with quest scheduling.

    Decision factors:
    1. Accuracy - Low accuracy skills need more attention
    2. Time since eval - Skills not evaluated recently need evals
    3. Weak primitives - Specific concepts struggling need targeted practice
    4. Level balance - Don't neglect skills at lower levels
    5. Quest diversity - Rotate through skills

    Usage:
        bridge = SchedulerBridge()

        # Get next quest recommendation
        quest = bridge.recommend_next_quest(hero_id="dio")

        # Get multiple recommendations for queue
        quests = bridge.recommend_quests(hero_id="dio", count=5)

        # Check if skill needs eval
        needs_eval = bridge.needs_eval("bin", hero_id="dio")
    """

    # Thresholds
    ACCURACY_THRESHOLD = 0.80      # Below this = needs attention
    EVAL_INTERVAL_HOURS = 4        # Eval every N hours minimum
    WEAK_PRIMITIVE_THRESHOLD = 0.7 # Primitive below this = weak
    MIN_EVALS_FOR_PROGRESSION = 3  # Need N evals before level-up

    def __init__(self, base_dir: Optional[Path] = None):
        from core.paths import get_base_dir
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()

        self._engine = None
        self._hero_manager = None

    @property
    def engine(self):
        """Lazy-load SkillEngine."""
        if self._engine is None:
            from guild.skills import get_engine
            self._engine = get_engine()
        return self._engine

    @property
    def hero_manager(self):
        """Lazy-load HeroStateManager."""
        if self._hero_manager is None:
            from guild.integration.hero_state import get_hero_state_manager
            self._hero_manager = get_hero_state_manager()
        return self._hero_manager

    def _get_skill_metrics(self, skill_id: str, hero_id: str) -> SkillMetrics:
        """Gather metrics for scheduling decisions."""
        state = self.hero_manager.get(hero_id, skill_id)

        # Time since last eval
        last_eval_time = None
        time_since_eval = None
        if state.last_eval_timestamp:
            try:
                last_eval_time = datetime.fromisoformat(state.last_eval_timestamp)
                time_since_eval = datetime.now() - last_eval_time
            except Exception:
                pass

        # Find weak primitives
        weak = [
            prim for prim, acc in state.primitive_accuracy.items()
            if acc < self.WEAK_PRIMITIVE_THRESHOLD
        ]

        return SkillMetrics(
            skill_id=skill_id,
            level=state.level,
            accuracy=state.accuracy,
            last_eval_time=last_eval_time,
            total_evals=state.total_evals,
            weak_primitives=weak,
            time_since_eval=time_since_eval,
        )

    def _calculate_priority(self, metrics: SkillMetrics) -> float:
        """
        Calculate quest priority for a skill.

        Returns 0.0-1.0, higher = more urgent.
        """
        priority = 0.5  # Base priority

        # Low accuracy increases priority
        if metrics.accuracy < self.ACCURACY_THRESHOLD:
            acc_gap = self.ACCURACY_THRESHOLD - metrics.accuracy
            priority += acc_gap * 0.5  # Up to +0.4 for 0% accuracy

        # Long time since eval increases priority
        if metrics.time_since_eval:
            hours = metrics.time_since_eval.total_seconds() / 3600
            if hours > self.EVAL_INTERVAL_HOURS:
                overdue = (hours - self.EVAL_INTERVAL_HOURS) / 24
                priority += min(overdue * 0.1, 0.2)  # Up to +0.2

        # Weak primitives increase priority
        if metrics.weak_primitives:
            priority += min(len(metrics.weak_primitives) * 0.05, 0.2)

        # Low eval count increases priority (need more data)
        if metrics.total_evals < self.MIN_EVALS_FOR_PROGRESSION:
            priority += 0.1

        return min(priority, 1.0)

    def _decide_quest_type(self, metrics: SkillMetrics) -> QuestType:
        """Decide what type of quest to schedule."""
        # Need eval if not evaluated recently
        if metrics.time_since_eval is None:
            return QuestType.EVAL

        hours_since = metrics.time_since_eval.total_seconds() / 3600
        if hours_since > self.EVAL_INTERVAL_HOURS:
            return QuestType.EVAL

        # Need sparring if weak primitives
        if metrics.weak_primitives:
            return QuestType.SPARRING

        # Default to training
        return QuestType.TRAINING

    def _generate_reason(
        self,
        metrics: SkillMetrics,
        quest_type: QuestType,
    ) -> str:
        """Generate human-readable reason for quest."""
        reasons = []

        if quest_type == QuestType.EVAL:
            if metrics.time_since_eval is None:
                reasons.append("Never evaluated")
            else:
                hours = metrics.time_since_eval.total_seconds() / 3600
                reasons.append(f"Last eval {hours:.1f}h ago")

        if metrics.accuracy < self.ACCURACY_THRESHOLD:
            reasons.append(f"Accuracy {metrics.accuracy:.0%} below {self.ACCURACY_THRESHOLD:.0%}")

        if metrics.weak_primitives:
            weak_str = ", ".join(metrics.weak_primitives[:3])
            if len(metrics.weak_primitives) > 3:
                weak_str += f" (+{len(metrics.weak_primitives) - 3} more)"
            reasons.append(f"Weak: {weak_str}")

        if metrics.total_evals < self.MIN_EVALS_FOR_PROGRESSION:
            reasons.append(f"Only {metrics.total_evals}/{self.MIN_EVALS_FOR_PROGRESSION} evals")

        return "; ".join(reasons) if reasons else "Regular training"

    def recommend_next_quest(
        self,
        hero_id: str = "dio",
        exclude_skills: Optional[List[str]] = None,
    ) -> Optional[QuestRecommendation]:
        """
        Recommend the highest-priority quest.

        Args:
            hero_id: Hero to schedule for
            exclude_skills: Skills to skip (e.g., recently queued)

        Returns:
            QuestRecommendation or None if no skills available
        """
        exclude = set(exclude_skills or [])
        skills = [s for s in self.engine.list_skills() if s not in exclude]

        if not skills:
            return None

        # Score all skills
        candidates = []
        for skill_id in skills:
            metrics = self._get_skill_metrics(skill_id, hero_id)
            priority = self._calculate_priority(metrics)
            quest_type = self._decide_quest_type(metrics)
            reason = self._generate_reason(metrics, quest_type)

            candidates.append((priority, metrics, quest_type, reason))

        # Sort by priority descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        if not candidates:
            return None

        priority, metrics, quest_type, reason = candidates[0]

        return QuestRecommendation(
            skill_id=metrics.skill_id,
            level=metrics.level + 1,  # Train at next level
            quest_type=quest_type,
            priority=priority,
            reason=reason,
            count=100 if quest_type == QuestType.TRAINING else 10,
            target_primitives=metrics.weak_primitives[:5],
        )

    def recommend_quests(
        self,
        hero_id: str = "dio",
        count: int = 5,
    ) -> List[QuestRecommendation]:
        """
        Recommend multiple quests for queue filling.

        Returns list sorted by priority.
        """
        recommendations = []
        exclude: List[str] = []

        for _ in range(count):
            quest = self.recommend_next_quest(hero_id, exclude)
            if quest is None:
                break
            recommendations.append(quest)
            exclude.append(quest.skill_id)

        return recommendations

    def needs_eval(self, skill_id: str, hero_id: str = "dio") -> bool:
        """Check if a skill needs evaluation."""
        metrics = self._get_skill_metrics(skill_id, hero_id)

        if metrics.time_since_eval is None:
            return True

        hours = metrics.time_since_eval.total_seconds() / 3600
        return hours > self.EVAL_INTERVAL_HOURS

    def get_weak_primitives(
        self,
        skill_id: str,
        hero_id: str = "dio",
        threshold: float = 0.7,
    ) -> List[str]:
        """Get weak primitives for a skill."""
        state = self.hero_manager.get(hero_id, skill_id)
        return [
            prim for prim, acc in state.primitive_accuracy.items()
            if acc < threshold
        ]

    def get_schedule_summary(self, hero_id: str = "dio") -> Dict[str, Any]:
        """Get summary of scheduling state for all skills."""
        skills = self.engine.list_skills()
        summary = {
            "hero_id": hero_id,
            "timestamp": datetime.now().isoformat(),
            "skills": {},
        }

        for skill_id in skills:
            metrics = self._get_skill_metrics(skill_id, hero_id)
            priority = self._calculate_priority(metrics)
            quest_type = self._decide_quest_type(metrics)

            summary["skills"][skill_id] = {
                "level": metrics.level,
                "accuracy": metrics.accuracy,
                "priority": round(priority, 2),
                "quest_type": quest_type.value,
                "weak_primitives": metrics.weak_primitives,
                "total_evals": metrics.total_evals,
                "hours_since_eval": (
                    round(metrics.time_since_eval.total_seconds() / 3600, 1)
                    if metrics.time_since_eval else None
                ),
            }

        return summary


# Singleton instance
_bridge: Optional[SchedulerBridge] = None


def get_scheduler_bridge() -> SchedulerBridge:
    """Get singleton SchedulerBridge."""
    global _bridge
    if _bridge is None:
        _bridge = SchedulerBridge()
    return _bridge


def reset_scheduler_bridge():
    """Reset singleton (for testing)."""
    global _bridge
    _bridge = None
