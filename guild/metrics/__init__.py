"""
Guild Metrics - Materials Science Inspired Training Analytics

This module provides metrics that view training through a materials science lens:

    Strain = instantaneous difficulty (loss - floor)
    Effort = cumulative strain (area under curve)
    Strain Rate = learning velocity
    Plastic Gain = permanent improvement
    Efficiency = gain / effort

Usage:
    from guild.metrics import StrainTracker, StrainZone, SkillStrainTracker

    # Single skill tracking
    tracker = StrainTracker(floor=0.5)
    metrics = tracker.update(loss=0.8, step=100)
    hint = tracker.get_curriculum_hint()

    # Multi-skill tracking
    multi = SkillStrainTracker()
    multi.set_floor("sy", 0.8)
    multi.set_floor("bin", 0.5)
    metrics = multi.update("sy", loss=1.2, step=100)
"""

from guild.metrics.strain import (
    # Core types
    StrainZone,
    CurriculumAction,
    StrainMetrics,
    CurriculumHint,
    EffortRecord,
    # Trackers
    StrainTracker,
    SkillStrainTracker,
)

__all__ = [
    # Types
    "StrainZone",
    "CurriculumAction",
    "StrainMetrics",
    "CurriculumHint",
    "EffortRecord",
    # Trackers
    "StrainTracker",
    "SkillStrainTracker",
]
