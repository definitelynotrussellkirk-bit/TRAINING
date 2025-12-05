"""
Primitive Analytics - Per-Primitive Weakness Detection
=======================================================

Analyzes evaluation results to identify weak cognitive primitives.

The system stores per-problem results with optional primitive_id tags.
This module aggregates that data to answer:
- Which primitives are weakest?
- Which skills exercise which primitives?
- What training would strengthen a weak primitive?

Usage:
    from guild.primitives.analytics import PrimitiveAnalyzer

    analyzer = PrimitiveAnalyzer()

    # Get weakness report
    report = analyzer.get_weakness_report(campaign_id="campaign-001")

    # Get primitive profile for hero
    profile = analyzer.get_primitive_profile(hero_id="ojas-qwen3-8b")

    # Get training suggestions for weak primitives
    suggestions = analyzer.suggest_training(weak_primitives=["logic_chain"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# PRIMITIVE DEFINITIONS (from CLAUDE.md)
# =============================================================================

PRIMITIVE_CATEGORIES = {
    "seq": "Sequence Operations",
    "logic": "Logic Operations",
    "mem": "Memory Operations",
    "fmt": "Format Operations",
    "attn": "Attention Operations",
    "xfm": "Transform Operations",
}

# Known primitives and which skills exercise them
PRIMITIVE_SKILL_MAP = {
    # Sequence
    "seq_continue": ["sy"],
    "seq_reverse": ["sy"],
    "seq_transform": ["sy"],
    "seq_interleave": ["sy"],
    "seq_extract": ["sy"],
    # Logic
    "logic_deduce": ["bin"],
    "logic_contrapose": ["bin"],
    "logic_chain": ["bin"],
    "logic_disjunct": ["bin"],
    "logic_biconditional": ["bin"],
    # Memory
    "mem_recall": ["sy", "bin"],
    "mem_context": ["sy", "bin"],
    "mem_compose": ["bin"],
    "mem_update": ["bin"],
    # Format
    "fmt_json": ["sy", "bin"],
    "fmt_table": [],
    "fmt_code": [],
    "fmt_list": [],
    "fmt_structured": ["sy", "bin"],
    # Attention
    "attn_select": ["sy"],
    "attn_count": ["bin"],
    "attn_compare": ["bin"],
    "attn_filter": ["sy", "bin"],
    "attn_rank": [],
    # Transform
    "xfm_encode": ["bin"],
    "xfm_decode": ["bin"],
    "xfm_map": ["sy"],
    "xfm_reduce": ["bin"],
    "xfm_substitute": ["sy"],
}

# Reverse map: skill -> primitives
SKILL_PRIMITIVE_MAP: Dict[str, List[str]] = {}
for prim, skills in PRIMITIVE_SKILL_MAP.items():
    for skill in skills:
        if skill not in SKILL_PRIMITIVE_MAP:
            SKILL_PRIMITIVE_MAP[skill] = []
        SKILL_PRIMITIVE_MAP[skill].append(prim)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PrimitiveStats:
    """Statistics for a single primitive."""
    primitive_id: str
    total_samples: int = 0
    correct: int = 0
    accuracy: float = 0.0
    skills_exercised: List[str] = field(default_factory=list)
    trend: float = 0.0  # -1 to 1: declining to improving
    last_checkpoint: Optional[int] = None

    @property
    def category(self) -> str:
        """Get primitive category (seq, logic, etc.)."""
        prefix = self.primitive_id.split("_")[0]
        return PRIMITIVE_CATEGORIES.get(prefix, "unknown")

    @property
    def is_weak(self) -> bool:
        """Is this primitive below threshold?"""
        return self.accuracy < 0.7 and self.total_samples >= 5


@dataclass
class WeaknessReport:
    """Report of weak primitives for a campaign/hero."""
    weak_primitives: List[PrimitiveStats]
    strong_primitives: List[PrimitiveStats]
    untested_primitives: List[str]
    coverage: float  # 0-1: fraction of primitives tested
    campaign_id: Optional[str] = None
    hero_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weak_primitives": [
                {
                    "id": p.primitive_id,
                    "accuracy": p.accuracy,
                    "samples": p.total_samples,
                    "category": p.category,
                    "skills": p.skills_exercised,
                }
                for p in self.weak_primitives
            ],
            "strong_primitives": [
                {
                    "id": p.primitive_id,
                    "accuracy": p.accuracy,
                    "samples": p.total_samples,
                }
                for p in self.strong_primitives
            ],
            "untested_primitives": self.untested_primitives,
            "coverage": self.coverage,
            "campaign_id": self.campaign_id,
            "hero_id": self.hero_id,
        }


@dataclass
class TrainingSuggestion:
    """Suggestion for strengthening a weak primitive."""
    primitive_id: str
    current_accuracy: float
    suggested_skills: List[str]
    suggested_levels: Dict[str, int]  # skill -> recommended level
    rationale: str


# =============================================================================
# ANALYZER
# =============================================================================

class PrimitiveAnalyzer:
    """
    Analyzes evaluation results to identify weak primitives.

    Aggregates per-problem primitive_id data from EvaluationLedger
    to build a primitive profile for heroes/campaigns.
    """

    def __init__(self):
        self._primitive_cache: Dict[str, PrimitiveStats] = {}

    def get_primitive_stats(
        self,
        primitive_id: str,
        campaign_id: Optional[str] = None,
        hero_id: Optional[str] = None,
        limit: int = 100,
    ) -> PrimitiveStats:
        """
        Get statistics for a specific primitive.

        Scans recent evaluations looking for problems tagged with this primitive.
        """
        from core.evaluation_ledger import get_eval_ledger

        ledger = get_eval_ledger()
        all_evals = ledger.list_all(limit=500)

        # Filter by campaign/hero
        if campaign_id:
            all_evals = [e for e in all_evals if e.campaign_id == campaign_id]
        if hero_id:
            all_evals = [e for e in all_evals if e.hero_id == hero_id]

        # Aggregate problems with this primitive
        correct = 0
        total = 0
        skills_seen = set()
        last_checkpoint = None

        for eval_record in all_evals:
            for problem in eval_record.problems:
                # Check if problem has this primitive
                problem_primitives = problem.get("primitive_ids", [])
                if primitive_id in problem_primitives:
                    total += 1
                    if problem.get("correct", False):
                        correct += 1
                    skills_seen.add(eval_record.skill)

                    if last_checkpoint is None:
                        last_checkpoint = eval_record.checkpoint_step

        accuracy = correct / total if total > 0 else 0.0

        return PrimitiveStats(
            primitive_id=primitive_id,
            total_samples=total,
            correct=correct,
            accuracy=accuracy,
            skills_exercised=list(skills_seen),
            last_checkpoint=last_checkpoint,
        )

    def get_primitive_profile(
        self,
        campaign_id: Optional[str] = None,
        hero_id: Optional[str] = None,
    ) -> Dict[str, PrimitiveStats]:
        """
        Get complete primitive profile - stats for all known primitives.

        Returns dict mapping primitive_id to PrimitiveStats.
        """
        profile = {}
        for primitive_id in PRIMITIVE_SKILL_MAP.keys():
            stats = self.get_primitive_stats(
                primitive_id,
                campaign_id=campaign_id,
                hero_id=hero_id,
            )
            profile[primitive_id] = stats
        return profile

    def get_weakness_report(
        self,
        campaign_id: Optional[str] = None,
        hero_id: Optional[str] = None,
        threshold: float = 0.7,
    ) -> WeaknessReport:
        """
        Generate a weakness report identifying problem areas.

        Args:
            campaign_id: Filter to specific campaign
            hero_id: Filter to specific hero
            threshold: Accuracy below this is considered weak

        Returns:
            WeaknessReport with weak/strong/untested primitives
        """
        profile = self.get_primitive_profile(campaign_id, hero_id)

        weak = []
        strong = []
        untested = []

        for prim_id, stats in profile.items():
            if stats.total_samples == 0:
                untested.append(prim_id)
            elif stats.accuracy < threshold:
                weak.append(stats)
            else:
                strong.append(stats)

        # Sort weak by accuracy (worst first)
        weak.sort(key=lambda s: s.accuracy)

        # Coverage
        tested = len(weak) + len(strong)
        coverage = tested / max(len(PRIMITIVE_SKILL_MAP), 1)

        return WeaknessReport(
            weak_primitives=weak,
            strong_primitives=strong,
            untested_primitives=untested,
            coverage=coverage,
            campaign_id=campaign_id,
            hero_id=hero_id,
        )

    def suggest_training(
        self,
        weak_primitives: List[str],
        campaign_id: Optional[str] = None,
    ) -> List[TrainingSuggestion]:
        """
        Suggest training to strengthen weak primitives.

        Returns list of TrainingSuggestion with skills and levels to focus on.
        """
        suggestions = []

        for prim_id in weak_primitives:
            # Get skills that exercise this primitive
            skills = PRIMITIVE_SKILL_MAP.get(prim_id, [])

            if not skills:
                suggestions.append(TrainingSuggestion(
                    primitive_id=prim_id,
                    current_accuracy=0.0,
                    suggested_skills=[],
                    suggested_levels={},
                    rationale=f"No known skills exercise {prim_id}",
                ))
                continue

            # Get current accuracy
            stats = self.get_primitive_stats(prim_id, campaign_id=campaign_id)

            # Suggest levels based on current mastery
            suggested_levels = {}
            for skill in skills:
                # Start at level 1 if very weak, else current training level
                if stats.accuracy < 0.3:
                    suggested_levels[skill] = 1
                elif stats.accuracy < 0.5:
                    suggested_levels[skill] = 2
                else:
                    suggested_levels[skill] = 3

            suggestions.append(TrainingSuggestion(
                primitive_id=prim_id,
                current_accuracy=stats.accuracy,
                suggested_skills=skills,
                suggested_levels=suggested_levels,
                rationale=f"Train {', '.join(skills)} to strengthen {prim_id}",
            ))

        return suggestions

    def infer_primitives_from_skill_level(
        self,
        skill: str,
        level: int,
    ) -> List[str]:
        """
        Infer which primitives a skill/level exercises.

        Used when problems don't have explicit primitive_id tags.
        """
        base_primitives = SKILL_PRIMITIVE_MAP.get(skill, [])

        # Higher levels might exercise more complex primitives
        if level >= 5:
            # Add memory primitives at higher levels
            if "mem_context" not in base_primitives:
                base_primitives = base_primitives + ["mem_context"]

        return base_primitives

    def aggregate_from_eval_problems(
        self,
        problems: List[Dict[str, Any]],
        skill: str,
        level: int,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Aggregate primitive accuracy from problem list.

        Returns dict mapping primitive_id to (correct, total).
        """
        aggregated: Dict[str, List[int, int]] = defaultdict(lambda: [0, 0])

        # Infer primitives if not tagged
        inferred_primitives = self.infer_primitives_from_skill_level(skill, level)

        for problem in problems:
            # Use explicit primitives or inferred
            primitives = problem.get("primitive_ids", inferred_primitives)

            for prim in primitives:
                aggregated[prim][1] += 1  # total
                if problem.get("correct", False):
                    aggregated[prim][0] += 1  # correct

        return {k: tuple(v) for k, v in aggregated.items()}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_weakness_report(
    campaign_id: Optional[str] = None,
    hero_id: Optional[str] = None,
) -> WeaknessReport:
    """Get weakness report for campaign/hero."""
    analyzer = PrimitiveAnalyzer()
    return analyzer.get_weakness_report(campaign_id, hero_id)


def get_primitive_profile(
    campaign_id: Optional[str] = None,
    hero_id: Optional[str] = None,
) -> Dict[str, PrimitiveStats]:
    """Get primitive profile for campaign/hero."""
    analyzer = PrimitiveAnalyzer()
    return analyzer.get_primitive_profile(campaign_id, hero_id)


def suggest_training_for_weak(
    campaign_id: Optional[str] = None,
    threshold: float = 0.7,
) -> List[TrainingSuggestion]:
    """Get training suggestions for all weak primitives."""
    analyzer = PrimitiveAnalyzer()
    report = analyzer.get_weakness_report(campaign_id, threshold=threshold)
    weak_ids = [p.primitive_id for p in report.weak_primitives]
    return analyzer.suggest_training(weak_ids, campaign_id)
