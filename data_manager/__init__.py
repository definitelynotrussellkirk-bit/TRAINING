"""
Data Manager System (DEPRECATED - use guild.dispatch instead)

This module is maintained for backward compatibility.
New code should use guild.dispatch:

    # Old way (deprecated):
    from data_manager import DataManager, CurriculumManager, QualityChecker

    # New way (preferred):
    from guild.dispatch import QuestDispatcher, ProgressionAdvisor, QuestQualityGate

Mapping:
    DataManager      -> QuestDispatcher
    CurriculumManager -> ProgressionAdvisor
    QualityChecker   -> QuestQualityGate
"""

import warnings

# Legacy imports (still work but emit deprecation warning on direct use)
from .manager import DataManager
from .remote_client import RemoteGPUClient
from .quality_checker import QualityChecker
from .remote_evaluator import RemoteEvaluator
from .curriculum_manager import CurriculumManager

# New RPG-themed imports from guild.dispatch
try:
    from guild.dispatch import (
        QuestDispatcher,
        ProgressionAdvisor,
        QuestQualityGate,
        DispatchDecision,
        QuestVerdict,
        DispatchStatus,
        DispatchResult,
        QualityReport,
        ProgressionStatus,
        SKILL_CURRICULA,
    )
    _GUILD_DISPATCH_AVAILABLE = True
except ImportError:
    _GUILD_DISPATCH_AVAILABLE = False


__all__ = [
    # Legacy (deprecated)
    "DataManager",
    "RemoteGPUClient",
    "QualityChecker",
    "RemoteEvaluator",
    "CurriculumManager",
    # New RPG-themed (preferred)
    "QuestDispatcher",
    "ProgressionAdvisor",
    "QuestQualityGate",
    "DispatchDecision",
    "QuestVerdict",
    "DispatchStatus",
    "DispatchResult",
    "QualityReport",
    "ProgressionStatus",
    "SKILL_CURRICULA",
]
