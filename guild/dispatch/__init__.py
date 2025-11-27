"""
Guild Dispatch - Quest coordination from Trainers to the Quest Board.

The Dispatch module coordinates the flow of quests through the training pipeline:

    Skill Trainer -> Quality Gate -> Quest Board -> Hero (Training)

Components:
    - QuestDispatcher: Central coordinator, runs dispatch cycles
    - ProgressionAdvisor: Counsels on skill advancement and difficulty
    - QuestQualityGate: Inspects quest batches before posting

Usage:
    from guild.dispatch import QuestDispatcher

    dispatcher = QuestDispatcher(base_dir)

    # Check if hero needs quests
    needs_work, reason = dispatcher.hero_needs_work()

    # Run dispatch cycle
    result = dispatcher.run_dispatch()

    # Check status
    status = dispatcher.get_status()

Progression:
    from guild.dispatch import ProgressionAdvisor

    advisor = ProgressionAdvisor(base_dir)

    # What level should hero train at?
    params = advisor.recommend_quest_params("binary")

    # Record trial and check for advancement
    advanced, new_level = advisor.record_and_check("binary", accuracy=0.85, step=10000)

Quality Gate:
    from guild.dispatch import QuestQualityGate

    gate = QuestQualityGate()
    report = gate.inspect(quest_batch)

    if report.verdict == QuestVerdict.APPROVED:
        # Safe to post to quest board
        pass
"""

# Types
from guild.dispatch.types import (
    # Enums
    DispatchDecision,
    QuestVerdict,
    # Data classes
    DispatchStatus,
    DispatchResult,
    QualityReport,
    ProgressionStatus,
)

# Quality Gate
from guild.dispatch.quality_gate import QuestQualityGate

# Progression Advisor
from guild.dispatch.advisor import (
    ProgressionAdvisor,
    SKILL_CURRICULA,
)

# Quest Dispatcher
from guild.dispatch.dispatcher import QuestDispatcher


__all__ = [
    # Types - Enums
    "DispatchDecision",
    "QuestVerdict",
    # Types - Data classes
    "DispatchStatus",
    "DispatchResult",
    "QualityReport",
    "ProgressionStatus",
    # Quality Gate
    "QuestQualityGate",
    # Progression Advisor
    "ProgressionAdvisor",
    "SKILL_CURRICULA",
    # Quest Dispatcher
    "QuestDispatcher",
]
