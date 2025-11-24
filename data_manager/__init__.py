"""
Data Manager System

Coordinates data generation, quality testing, queue management, evaluation, and curriculum.
Acts as both a data pipeline and test suite to gauge training readiness.
"""

from .manager import DataManager
from .remote_client import RemoteGPUClient
from .quality_checker import QualityChecker
from .remote_evaluator import RemoteEvaluator
from .curriculum_manager import CurriculumManager

__all__ = ["DataManager", "RemoteGPUClient", "QualityChecker", "RemoteEvaluator", "CurriculumManager"]
