"""
Eval Suite System - A/B Testing Infrastructure
===============================================

Run the same evaluation suite across multiple checkpoints, configs,
or models for comparison.

Use cases:
1. Compare checkpoint A vs checkpoint B on same eval set
2. Compare model A vs model B (different heroes)
3. Track regression across training run
4. Benchmark new eval set against historical results

Usage:
    from core.eval_suite import EvalSuite, create_suite, run_suite

    # Create a suite
    suite = create_suite(
        name="Compare OJAS checkpoints",
        checkpoints=[1000, 2000, 3000],
        skills=["bin", "sy"],
        levels=[1, 2, 3],
    )

    # Run the suite
    results = run_suite(suite)

    # Compare results
    comparison = suite.compare_results()
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# SUITE TYPES
# =============================================================================

class SuiteStatus(Enum):
    """Status of an eval suite."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some evals completed


@dataclass
class SuiteConfig:
    """Configuration for an eval suite."""
    name: str
    description: str = ""
    checkpoints: List[int] = field(default_factory=list)
    hero_ids: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    levels: List[int] = field(default_factory=list)

    # Options
    use_same_seed: bool = True  # Use same random seed for all evals
    seed: int = 42
    problems_per_level: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuiteConfig":
        return cls(**data)


@dataclass
class SuiteResult:
    """Result of a single evaluation in a suite."""
    checkpoint_step: int
    hero_id: Optional[str]
    skill: str
    level: int
    accuracy: float
    correct: int
    total: int
    timestamp: str
    eval_id: str  # Reference to EvalRecord

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteComparison:
    """Comparison results between suite runs."""
    suite_id: str
    compared_at: str
    by_checkpoint: Dict[int, Dict[str, float]]  # checkpoint -> {skill: accuracy}
    by_skill: Dict[str, Dict[int, float]]  # skill -> {checkpoint: accuracy}
    best_checkpoint: Optional[int] = None
    best_accuracy: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSuite:
    """
    An evaluation suite that runs the same evals across multiple checkpoints.

    Enables A/B testing by ensuring identical eval conditions.
    """
    suite_id: str
    config: SuiteConfig
    status: SuiteStatus = SuiteStatus.PENDING
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: List[SuiteResult] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def total_evals_planned(self) -> int:
        """Total number of evaluations planned."""
        checkpoints = len(self.config.checkpoints) or 1
        heroes = len(self.config.hero_ids) or 1
        skills = len(self.config.skills) or 1
        levels = len(self.config.levels) or 1
        return checkpoints * heroes * skills * levels

    @property
    def total_evals_completed(self) -> int:
        """Number of evaluations completed."""
        return len(self.results)

    @property
    def progress(self) -> float:
        """Progress 0-1."""
        planned = self.total_evals_planned
        if planned == 0:
            return 1.0
        return self.total_evals_completed / planned

    def add_result(self, result: SuiteResult):
        """Add a result to the suite."""
        self.results.append(result)

    def add_error(self, checkpoint: int, skill: str, level: int, error: str):
        """Record an error."""
        self.errors.append({
            "checkpoint": checkpoint,
            "skill": skill,
            "level": level,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })

    def get_results_for_checkpoint(self, checkpoint_step: int) -> List[SuiteResult]:
        """Get all results for a specific checkpoint."""
        return [r for r in self.results if r.checkpoint_step == checkpoint_step]

    def get_results_for_skill(self, skill: str) -> List[SuiteResult]:
        """Get all results for a specific skill."""
        return [r for r in self.results if r.skill == skill]

    def compare_results(self) -> SuiteComparison:
        """Generate comparison of results across checkpoints."""
        by_checkpoint: Dict[int, Dict[str, float]] = {}
        by_skill: Dict[str, Dict[int, float]] = {}

        for result in self.results:
            # By checkpoint
            if result.checkpoint_step not in by_checkpoint:
                by_checkpoint[result.checkpoint_step] = {}
            key = f"{result.skill}:{result.level}"
            by_checkpoint[result.checkpoint_step][key] = result.accuracy

            # By skill
            if result.skill not in by_skill:
                by_skill[result.skill] = {}
            by_skill[result.skill][result.checkpoint_step] = result.accuracy

        # Find best checkpoint
        best_checkpoint = None
        best_accuracy = 0.0
        for cp, skill_accs in by_checkpoint.items():
            avg_acc = sum(skill_accs.values()) / max(len(skill_accs), 1)
            if avg_acc > best_accuracy:
                best_accuracy = avg_acc
                best_checkpoint = cp

        # Generate summary
        summary_parts = []
        summary_parts.append(f"Suite: {self.config.name}")
        summary_parts.append(f"Checkpoints: {len(self.config.checkpoints)}")
        summary_parts.append(f"Evals: {self.total_evals_completed}/{self.total_evals_planned}")
        if best_checkpoint:
            summary_parts.append(f"Best: checkpoint-{best_checkpoint} ({best_accuracy:.1%})")

        return SuiteComparison(
            suite_id=self.suite_id,
            compared_at=datetime.now().isoformat(),
            by_checkpoint=by_checkpoint,
            by_skill=by_skill,
            best_checkpoint=best_checkpoint,
            best_accuracy=best_accuracy,
            summary=", ".join(summary_parts),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
            "progress": self.progress,
            "total_planned": self.total_evals_planned,
            "total_completed": self.total_evals_completed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalSuite":
        return cls(
            suite_id=data["suite_id"],
            config=SuiteConfig.from_dict(data["config"]),
            status=SuiteStatus(data.get("status", "pending")),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            results=[SuiteResult(**r) for r in data.get("results", [])],
            errors=data.get("errors", []),
        )


# =============================================================================
# SUITE MANAGEMENT
# =============================================================================

class SuiteManager:
    """Manages eval suites - creation, storage, execution."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.suites_dir = self.base_dir / "status" / "eval_suites"
        self.suites_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, EvalSuite] = {}

    def _suite_file(self, suite_id: str) -> Path:
        """Get path to suite JSON file."""
        return self.suites_dir / f"{suite_id}.json"

    def create_suite(
        self,
        name: str,
        checkpoints: List[int],
        skills: Optional[List[str]] = None,
        levels: Optional[List[int]] = None,
        hero_ids: Optional[List[str]] = None,
        description: str = "",
    ) -> EvalSuite:
        """
        Create a new eval suite.

        Args:
            name: Human-readable name
            checkpoints: List of checkpoint steps to evaluate
            skills: Skills to evaluate (default: all)
            levels: Levels to evaluate (default: [1])
            hero_ids: Hero IDs (default: current hero)
            description: Optional description

        Returns:
            Created EvalSuite
        """
        # Generate suite ID from config hash
        config_str = f"{name}:{checkpoints}:{skills}:{levels}"
        suite_id = hashlib.md5(config_str.encode()).hexdigest()[:12]
        suite_id = f"suite-{suite_id}"

        # Default skills
        if skills is None:
            skills = ["bin", "sy"]

        # Default levels
        if levels is None:
            levels = [1]

        config = SuiteConfig(
            name=name,
            description=description,
            checkpoints=checkpoints,
            hero_ids=hero_ids or [],
            skills=skills,
            levels=levels,
        )

        suite = EvalSuite(
            suite_id=suite_id,
            config=config,
        )

        # Save immediately
        self.save_suite(suite)

        logger.info(f"Created eval suite: {suite_id} ({suite.total_evals_planned} evals planned)")
        return suite

    def save_suite(self, suite: EvalSuite):
        """Save suite to disk."""
        filepath = self._suite_file(suite.suite_id)
        with open(filepath, "w") as f:
            json.dump(suite.to_dict(), f, indent=2)
        self._cache[suite.suite_id] = suite

    def load_suite(self, suite_id: str) -> Optional[EvalSuite]:
        """Load suite from disk."""
        if suite_id in self._cache:
            return self._cache[suite_id]

        filepath = self._suite_file(suite_id)
        if not filepath.exists():
            return None

        try:
            with open(filepath) as f:
                data = json.load(f)
            suite = EvalSuite.from_dict(data)
            self._cache[suite_id] = suite
            return suite
        except Exception as e:
            logger.error(f"Failed to load suite {suite_id}: {e}")
            return None

    def list_suites(self, limit: int = 20) -> List[EvalSuite]:
        """List all suites."""
        suites = []
        for filepath in self.suites_dir.glob("suite-*.json"):
            suite = self.load_suite(filepath.stem)
            if suite:
                suites.append(suite)

        # Sort by created_at descending
        suites.sort(key=lambda s: s.created_at, reverse=True)
        return suites[:limit]

    def run_suite(self, suite: EvalSuite) -> EvalSuite:
        """
        Run an eval suite.

        Executes all planned evaluations and updates the suite.
        """
        from core.eval_runner import EvalRunner

        suite.status = SuiteStatus.RUNNING
        suite.started_at = datetime.now().isoformat()
        self.save_suite(suite)

        runner = EvalRunner(base_dir=self.base_dir)

        # Run each evaluation
        for checkpoint_step in suite.config.checkpoints:
            for skill in suite.config.skills:
                for level in suite.config.levels:
                    try:
                        result = runner.run_skill_evaluation(
                            checkpoint_step=checkpoint_step,
                            skill=skill,
                            level=level,
                            eval_type="suite",
                        )

                        if result:
                            suite.add_result(SuiteResult(
                                checkpoint_step=checkpoint_step,
                                hero_id=suite.config.hero_ids[0] if suite.config.hero_ids else None,
                                skill=skill,
                                level=level,
                                accuracy=result.accuracy,
                                correct=result.correct,
                                total=result.total,
                                timestamp=datetime.now().isoformat(),
                                eval_id=result.key,
                            ))
                        else:
                            suite.add_error(checkpoint_step, skill, level, "Eval returned None")

                    except Exception as e:
                        logger.error(f"Suite eval error: {e}")
                        suite.add_error(checkpoint_step, skill, level, str(e))

                    # Save progress
                    self.save_suite(suite)

        # Update status
        if len(suite.errors) == 0:
            suite.status = SuiteStatus.COMPLETED
        elif len(suite.results) > 0:
            suite.status = SuiteStatus.PARTIAL
        else:
            suite.status = SuiteStatus.FAILED

        suite.completed_at = datetime.now().isoformat()
        self.save_suite(suite)

        logger.info(
            f"Suite {suite.suite_id} {suite.status.value}: "
            f"{suite.total_evals_completed}/{suite.total_evals_planned} evals, "
            f"{len(suite.errors)} errors"
        )

        return suite

    def delete_suite(self, suite_id: str) -> bool:
        """Delete a suite."""
        filepath = self._suite_file(suite_id)
        if filepath.exists():
            filepath.unlink()
            if suite_id in self._cache:
                del self._cache[suite_id]
            return True
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_manager: Optional[SuiteManager] = None


def get_suite_manager() -> SuiteManager:
    """Get singleton suite manager."""
    global _manager
    if _manager is None:
        _manager = SuiteManager()
    return _manager


def create_suite(
    name: str,
    checkpoints: List[int],
    skills: Optional[List[str]] = None,
    levels: Optional[List[int]] = None,
    **kwargs,
) -> EvalSuite:
    """Create a new eval suite."""
    return get_suite_manager().create_suite(
        name=name,
        checkpoints=checkpoints,
        skills=skills,
        levels=levels,
        **kwargs,
    )


def run_suite(suite_or_id: EvalSuite | str) -> EvalSuite:
    """Run an eval suite."""
    manager = get_suite_manager()

    if isinstance(suite_or_id, str):
        suite = manager.load_suite(suite_or_id)
        if suite is None:
            raise ValueError(f"Suite not found: {suite_or_id}")
    else:
        suite = suite_or_id

    return manager.run_suite(suite)


def compare_checkpoints(
    checkpoints: List[int],
    skills: Optional[List[str]] = None,
    levels: Optional[List[int]] = None,
) -> SuiteComparison:
    """
    Quick comparison of multiple checkpoints.

    Creates and runs a suite, returns comparison.
    """
    suite = create_suite(
        name=f"Quick compare {len(checkpoints)} checkpoints",
        checkpoints=checkpoints,
        skills=skills,
        levels=levels,
    )

    suite = run_suite(suite)
    return suite.compare_results()
