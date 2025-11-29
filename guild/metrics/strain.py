"""
Strain & Effort Metrics - Materials Science Metaphor for Training

This module applies a materials science lens to ML training:

    Strain = instantaneous difficulty (loss - floor)
    Effort = cumulative strain over time (area under strain curve)
    Strain Rate = how fast things are changing (learning velocity)
    Plastic Gain = permanent improvement (before vs after)
    Efficiency = plastic_gain / effort (learning ROI)

The metaphor:
- Like stretching a material, training "strains" the model
- Some strain leads to permanent reshaping (plastic deformation = learning)
- Too much strain can cause damage (divergence, forgetting)
- The goal is productive strain: enough to learn, not so much to break

Usage:
    from guild.metrics.strain import StrainTracker, StrainZone

    tracker = StrainTracker(floor=0.5)  # comfort zone loss

    # During training
    metrics = tracker.update(current_loss)
    print(f"Strain: {metrics.strain:.3f}, Zone: {metrics.zone.name}")
    print(f"Cumulative effort: {metrics.cumulative_effort:.1f}")

    # Get curriculum hint
    hint = tracker.get_curriculum_hint()
    if hint.action == "back_off":
        print("Model is overloaded, reduce difficulty")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import math


class StrainZone(Enum):
    """
    Training zones based on strain level.

    Like heart rate zones for exercise, these indicate training intensity:
    - RECOVERY: Very easy, model is coasting
    - PRODUCTIVE: Optimal learning zone
    - STRETCH: Challenging but sustainable
    - OVERLOAD: Too hard, risk of destabilization
    """
    RECOVERY = "recovery"      # strain < 0.1: under-challenged
    PRODUCTIVE = "productive"  # 0.1 <= strain < 0.3: optimal
    STRETCH = "stretch"        # 0.3 <= strain < 0.5: challenging
    OVERLOAD = "overload"      # strain >= 0.5: too hard


class CurriculumAction(Enum):
    """Actions the curriculum system might take based on strain."""
    CONTINUE = "continue"       # Stay at current level
    LEVEL_UP = "level_up"       # Increase difficulty
    BACK_OFF = "back_off"       # Decrease difficulty
    WAIT = "wait"               # Need more data


@dataclass
class StrainMetrics:
    """
    Instantaneous strain metrics at a point in time.

    Attributes:
        strain: Current strain (loss - floor), always >= 0
        strain_rate: Rate of change (positive = getting harder)
        cumulative_effort: Total effort spent (sum of strain over time)
        zone: Current training zone
        floor: The baseline loss (comfort/target)
        raw_loss: Original loss value before floor subtraction
        step: Training step when measured
        timestamp: When this was recorded
    """
    strain: float
    strain_rate: float
    cumulative_effort: float
    zone: StrainZone
    floor: float
    raw_loss: float
    step: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_comfortable(self) -> bool:
        """Is the model in a comfortable state?"""
        return self.zone in (StrainZone.RECOVERY, StrainZone.PRODUCTIVE)

    @property
    def is_stressed(self) -> bool:
        """Is the model under significant stress?"""
        return self.zone in (StrainZone.STRETCH, StrainZone.OVERLOAD)

    @property
    def is_learning(self) -> bool:
        """Is the model actively learning (negative strain rate)?"""
        return self.strain_rate < -0.001  # Small threshold for noise

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strain": self.strain,
            "strain_rate": self.strain_rate,
            "cumulative_effort": self.cumulative_effort,
            "zone": self.zone.value,
            "floor": self.floor,
            "raw_loss": self.raw_loss,
            "step": self.step,
            "timestamp": self.timestamp,
        }


@dataclass
class CurriculumHint:
    """
    Hint for curriculum system based on strain analysis.

    Attributes:
        action: Suggested action (continue, level_up, back_off, wait)
        confidence: How confident are we (0-1)
        reason: Human-readable explanation
        zone: Current strain zone
        strain_trend: "improving", "stable", or "worsening"
    """
    action: CurriculumAction
    confidence: float
    reason: str
    zone: StrainZone
    strain_trend: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "zone": self.zone.value,
            "strain_trend": self.strain_trend,
        }


@dataclass
class EffortRecord:
    """Record of effort spent on a skill/level transition."""
    skill_id: str
    from_level: int
    to_level: int
    effort_spent: float  # Cumulative strain during this transition
    steps_taken: int
    start_loss: float
    end_loss: float
    plastic_gain: float  # start_loss - end_loss
    efficiency: float  # plastic_gain / effort_spent
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "from_level": self.from_level,
            "to_level": self.to_level,
            "effort_spent": self.effort_spent,
            "steps_taken": self.steps_taken,
            "start_loss": self.start_loss,
            "end_loss": self.end_loss,
            "plastic_gain": self.plastic_gain,
            "efficiency": self.efficiency,
            "timestamp": self.timestamp,
        }


class StrainTracker:
    """
    Tracks strain and effort over time for a training run.

    The tracker maintains:
    - Running strain history (for rate calculation)
    - Cumulative effort (area under strain curve)
    - Smoothed strain rate (to filter noise)

    Usage:
        tracker = StrainTracker(floor=0.5)  # Target/comfort loss

        for step, loss in training_loop():
            metrics = tracker.update(loss, step)

            # React to strain
            if metrics.zone == StrainZone.OVERLOAD:
                decrease_difficulty()
            elif metrics.zone == StrainZone.RECOVERY:
                increase_difficulty()
    """

    # Zone thresholds (can be customized)
    ZONE_THRESHOLDS = {
        StrainZone.RECOVERY: 0.1,
        StrainZone.PRODUCTIVE: 0.3,
        StrainZone.STRETCH: 0.5,
        # OVERLOAD is anything >= 0.5
    }

    def __init__(
        self,
        floor: float = 0.0,
        window_size: int = 50,
        ema_alpha: float = 0.1,
    ):
        """
        Initialize strain tracker.

        Args:
            floor: Baseline loss (comfort zone). Can be:
                   - 0.0 for theoretical minimum
                   - A target loss for the current skill level
                   - Best-seen-so-far (dynamic)
            window_size: How many samples to keep for rate calculation
            ema_alpha: Smoothing factor for strain rate (0-1, lower = smoother)
        """
        self.floor = floor
        self.window_size = window_size
        self.ema_alpha = ema_alpha

        # History
        self._history: List[float] = []  # strain values
        self._loss_history: List[float] = []  # raw loss values
        self._step_history: List[int] = []  # step numbers

        # Running totals
        self.cumulative_effort: float = 0.0
        self.steps_tracked: int = 0

        # Smoothed values
        self._strain_rate_ema: Optional[float] = None
        self._best_loss: Optional[float] = None

        # Level tracking (for effort-per-level)
        self._level_start_effort: float = 0.0
        self._level_start_loss: Optional[float] = None
        self._level_start_step: int = 0

    def update(self, loss: float, step: int = 0) -> StrainMetrics:
        """
        Update tracker with new loss value.

        Args:
            loss: Current training loss
            step: Current training step

        Returns:
            StrainMetrics with current state
        """
        # Calculate strain (always non-negative)
        strain = max(0.0, loss - self.floor)

        # Update cumulative effort
        self.cumulative_effort += strain
        self.steps_tracked += 1

        # Track best loss (for dynamic floor option)
        if self._best_loss is None or loss < self._best_loss:
            self._best_loss = loss

        # Calculate strain rate (smoothed)
        strain_rate = 0.0
        if self._history:
            # Raw rate: current - previous
            raw_rate = strain - self._history[-1]

            # Smooth with EMA
            if self._strain_rate_ema is None:
                self._strain_rate_ema = raw_rate
            else:
                self._strain_rate_ema = (
                    self.ema_alpha * raw_rate +
                    (1 - self.ema_alpha) * self._strain_rate_ema
                )
            strain_rate = self._strain_rate_ema

        # Update history
        self._history.append(strain)
        self._loss_history.append(loss)
        self._step_history.append(step)

        # Trim history
        if len(self._history) > self.window_size:
            self._history.pop(0)
            self._loss_history.pop(0)
            self._step_history.pop(0)

        # Determine zone
        zone = self._classify_zone(strain)

        # Initialize level tracking if needed
        if self._level_start_loss is None:
            self._level_start_loss = loss
            self._level_start_step = step

        return StrainMetrics(
            strain=strain,
            strain_rate=strain_rate,
            cumulative_effort=self.cumulative_effort,
            zone=zone,
            floor=self.floor,
            raw_loss=loss,
            step=step,
        )

    def _classify_zone(self, strain: float) -> StrainZone:
        """Classify strain into a training zone."""
        if strain < self.ZONE_THRESHOLDS[StrainZone.RECOVERY]:
            return StrainZone.RECOVERY
        elif strain < self.ZONE_THRESHOLDS[StrainZone.PRODUCTIVE]:
            return StrainZone.PRODUCTIVE
        elif strain < self.ZONE_THRESHOLDS[StrainZone.STRETCH]:
            return StrainZone.STRETCH
        else:
            return StrainZone.OVERLOAD

    def get_curriculum_hint(self) -> CurriculumHint:
        """
        Get curriculum guidance based on current strain state.

        Returns:
            CurriculumHint with suggested action
        """
        if len(self._history) < 10:
            return CurriculumHint(
                action=CurriculumAction.WAIT,
                confidence=0.3,
                reason="Need more data (< 10 samples)",
                zone=StrainZone.PRODUCTIVE,
                strain_trend="unknown",
            )

        # Get recent strain stats
        recent = self._history[-10:]
        avg_strain = sum(recent) / len(recent)
        zone = self._classify_zone(avg_strain)

        # Determine trend
        if self._strain_rate_ema is None:
            strain_trend = "stable"
        elif self._strain_rate_ema < -0.005:
            strain_trend = "improving"
        elif self._strain_rate_ema > 0.005:
            strain_trend = "worsening"
        else:
            strain_trend = "stable"

        # Decision logic
        # High strain + not improving -> back off
        if zone == StrainZone.OVERLOAD and strain_trend != "improving":
            return CurriculumHint(
                action=CurriculumAction.BACK_OFF,
                confidence=0.8,
                reason=f"Overloaded (strain={avg_strain:.2f}) and not improving",
                zone=zone,
                strain_trend=strain_trend,
            )

        # High strain but improving -> continue (it's working)
        if zone in (StrainZone.STRETCH, StrainZone.OVERLOAD) and strain_trend == "improving":
            return CurriculumHint(
                action=CurriculumAction.CONTINUE,
                confidence=0.7,
                reason=f"Challenging but improving (rate={self._strain_rate_ema:.4f})",
                zone=zone,
                strain_trend=strain_trend,
            )

        # Low strain + stable -> level up (too easy)
        if zone == StrainZone.RECOVERY and strain_trend == "stable":
            return CurriculumHint(
                action=CurriculumAction.LEVEL_UP,
                confidence=0.8,
                reason=f"Under-challenged (strain={avg_strain:.2f}) and stable",
                zone=zone,
                strain_trend=strain_trend,
            )

        # Productive zone + improving -> sweet spot
        if zone == StrainZone.PRODUCTIVE and strain_trend == "improving":
            return CurriculumHint(
                action=CurriculumAction.CONTINUE,
                confidence=0.9,
                reason="Optimal: productive zone with improving trend",
                zone=zone,
                strain_trend=strain_trend,
            )

        # Default: continue
        return CurriculumHint(
            action=CurriculumAction.CONTINUE,
            confidence=0.5,
            reason=f"Zone={zone.value}, trend={strain_trend}",
            zone=zone,
            strain_trend=strain_trend,
        )

    def mark_level_transition(
        self,
        skill_id: str,
        from_level: int,
        to_level: int,
        current_loss: float,
        current_step: int,
    ) -> EffortRecord:
        """
        Record effort spent on a level transition.

        Call this when a skill levels up to track effort-per-level.

        Args:
            skill_id: Skill that leveled up
            from_level: Previous level
            to_level: New level
            current_loss: Loss at transition
            current_step: Step at transition

        Returns:
            EffortRecord with stats for this transition
        """
        effort_spent = self.cumulative_effort - self._level_start_effort
        steps_taken = current_step - self._level_start_step
        start_loss = self._level_start_loss or current_loss
        plastic_gain = start_loss - current_loss

        # Avoid division by zero
        efficiency = plastic_gain / effort_spent if effort_spent > 0 else 0.0

        record = EffortRecord(
            skill_id=skill_id,
            from_level=from_level,
            to_level=to_level,
            effort_spent=effort_spent,
            steps_taken=steps_taken,
            start_loss=start_loss,
            end_loss=current_loss,
            plastic_gain=plastic_gain,
            efficiency=efficiency,
        )

        # Reset for next level
        self._level_start_effort = self.cumulative_effort
        self._level_start_loss = current_loss
        self._level_start_step = current_step

        return record

    def set_floor(self, floor: float):
        """Update the floor (comfort zone) value."""
        self.floor = floor

    def set_floor_to_best(self):
        """Set floor to best-seen-so-far (makes strain relative to personal best)."""
        if self._best_loss is not None:
            self.floor = self._best_loss

    def get_effort_per_step(self) -> float:
        """Average effort per step (efficiency metric)."""
        if self.steps_tracked == 0:
            return 0.0
        return self.cumulative_effort / self.steps_tracked

    def get_strain_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._history:
            return {
                "samples": 0,
                "cumulative_effort": 0.0,
                "avg_strain": 0.0,
                "min_strain": 0.0,
                "max_strain": 0.0,
                "current_zone": StrainZone.PRODUCTIVE.value,
                "strain_rate": 0.0,
                "effort_per_step": 0.0,
            }

        return {
            "samples": len(self._history),
            "cumulative_effort": self.cumulative_effort,
            "avg_strain": sum(self._history) / len(self._history),
            "min_strain": min(self._history),
            "max_strain": max(self._history),
            "current_zone": self._classify_zone(self._history[-1]).value,
            "strain_rate": self._strain_rate_ema or 0.0,
            "effort_per_step": self.get_effort_per_step(),
            "best_loss": self._best_loss,
            "floor": self.floor,
        }

    def reset(self, keep_floor: bool = True):
        """Reset tracker state."""
        old_floor = self.floor
        self._history.clear()
        self._loss_history.clear()
        self._step_history.clear()
        self.cumulative_effort = 0.0
        self.steps_tracked = 0
        self._strain_rate_ema = None
        self._best_loss = None
        self._level_start_effort = 0.0
        self._level_start_loss = None
        self._level_start_step = 0

        if keep_floor:
            self.floor = old_floor


class SkillStrainTracker:
    """
    Multi-skill strain tracker.

    Maintains separate strain tracking per skill, useful when
    training on multiple skills with different comfort zones.

    Usage:
        tracker = SkillStrainTracker()
        tracker.set_floor("sy", 0.8)  # SY comfort zone
        tracker.set_floor("bin", 0.5)  # BIN comfort zone

        # During training
        metrics = tracker.update("sy", loss, step)
    """

    def __init__(self):
        self._trackers: Dict[str, StrainTracker] = {}
        self._effort_records: List[EffortRecord] = []

    def get_or_create(self, skill_id: str, floor: float = 0.0) -> StrainTracker:
        """Get or create a tracker for a skill."""
        if skill_id not in self._trackers:
            self._trackers[skill_id] = StrainTracker(floor=floor)
        return self._trackers[skill_id]

    def update(
        self,
        skill_id: str,
        loss: float,
        step: int = 0,
    ) -> StrainMetrics:
        """Update strain for a skill."""
        tracker = self.get_or_create(skill_id)
        return tracker.update(loss, step)

    def set_floor(self, skill_id: str, floor: float):
        """Set floor for a skill."""
        tracker = self.get_or_create(skill_id)
        tracker.set_floor(floor)

    def get_curriculum_hint(self, skill_id: str) -> CurriculumHint:
        """Get curriculum hint for a skill."""
        if skill_id not in self._trackers:
            return CurriculumHint(
                action=CurriculumAction.WAIT,
                confidence=0.0,
                reason="No data for this skill",
                zone=StrainZone.PRODUCTIVE,
                strain_trend="unknown",
            )
        return self._trackers[skill_id].get_curriculum_hint()

    def mark_level_transition(
        self,
        skill_id: str,
        from_level: int,
        to_level: int,
        current_loss: float,
        current_step: int,
    ) -> EffortRecord:
        """Record level transition for a skill."""
        tracker = self.get_or_create(skill_id)
        record = tracker.mark_level_transition(
            skill_id, from_level, to_level, current_loss, current_step
        )
        self._effort_records.append(record)
        return record

    def get_effort_history(
        self,
        skill_id: Optional[str] = None,
    ) -> List[EffortRecord]:
        """Get effort records, optionally filtered by skill."""
        if skill_id is None:
            return self._effort_records.copy()
        return [r for r in self._effort_records if r.skill_id == skill_id]

    def get_total_effort(self, skill_id: Optional[str] = None) -> float:
        """Get total effort, optionally for a specific skill."""
        if skill_id is None:
            return sum(t.cumulative_effort for t in self._trackers.values())
        if skill_id in self._trackers:
            return self._trackers[skill_id].cumulative_effort
        return 0.0

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get strain summaries for all skills."""
        return {
            skill_id: tracker.get_strain_summary()
            for skill_id, tracker in self._trackers.items()
        }

    def compare_skills(self) -> Dict[str, Any]:
        """
        Compare effort across skills.

        Returns ranking of skills by effort spent and efficiency.
        """
        if not self._effort_records:
            return {"message": "No level transitions recorded yet"}

        # Aggregate by skill
        by_skill: Dict[str, Dict[str, Any]] = {}
        for record in self._effort_records:
            if record.skill_id not in by_skill:
                by_skill[record.skill_id] = {
                    "total_effort": 0.0,
                    "total_plastic_gain": 0.0,
                    "levels_gained": 0,
                    "transitions": [],
                }

            by_skill[record.skill_id]["total_effort"] += record.effort_spent
            by_skill[record.skill_id]["total_plastic_gain"] += record.plastic_gain
            by_skill[record.skill_id]["levels_gained"] += 1
            by_skill[record.skill_id]["transitions"].append(record.to_dict())

        # Calculate aggregate efficiency
        for skill_id, data in by_skill.items():
            if data["total_effort"] > 0:
                data["avg_efficiency"] = data["total_plastic_gain"] / data["total_effort"]
            else:
                data["avg_efficiency"] = 0.0
            data["effort_per_level"] = (
                data["total_effort"] / data["levels_gained"]
                if data["levels_gained"] > 0 else 0.0
            )

        # Rank by efficiency (higher is better)
        ranked = sorted(
            by_skill.items(),
            key=lambda x: x[1]["avg_efficiency"],
            reverse=True
        )

        return {
            "by_skill": by_skill,
            "efficiency_ranking": [skill_id for skill_id, _ in ranked],
            "most_expensive_skill": (
                max(by_skill.items(), key=lambda x: x[1]["effort_per_level"])[0]
                if by_skill else None
            ),
            "most_efficient_skill": (
                ranked[0][0] if ranked else None
            ),
        }
