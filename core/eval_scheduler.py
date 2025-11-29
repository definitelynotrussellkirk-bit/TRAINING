"""
Eval Scheduler - Policy-driven evaluation suite scheduling.

This module decides WHEN to run WHICH eval suites based on triggers:
- P0 (Gatekeeping): Run on every checkpoint
- P1 (Coverage): Run periodically (every N steps / N minutes)
- P2 (Exploratory): Run when GPU is idle

Usage:
    from core.eval_scheduler import get_scheduler

    scheduler = get_scheduler()

    # Call on checkpoint save (triggers P0 + maybe P1)
    scheduler.on_checkpoint_saved(run_ctx, checkpoint_step)

    # Call when GPU becomes idle (triggers P2)
    scheduler.on_idle_gpu(run_ctx)

    # Manual trigger
    scheduler.trigger_suite("skill_coverage", run_ctx, checkpoint_step)
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Callable

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# TYPES
# =============================================================================

class PriorityClass(str, Enum):
    """Suite priority classes."""
    P0 = "P0"  # Gatekeeping - run on every checkpoint
    P1 = "P1"  # Coverage - run periodically
    P2 = "P2"  # Exploratory - run when idle


@dataclass
class SkillEvalSpec:
    """Specification for a single skill eval within a suite."""
    skill_id: str
    level: int = 1
    batch_size: int = 100

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillEvalSpec":
        return cls(
            skill_id=data["skill_id"],
            level=data.get("level", 1),
            batch_size=data.get("batch_size", 100),
        )


@dataclass
class EvalSuite:
    """Definition of an evaluation suite."""
    id: str
    name: str
    description: str
    priority_class: PriorityClass
    skills: List[SkillEvalSpec]

    # Triggers
    on_new_checkpoint: bool = True
    background_only: bool = False
    on_idle_gpu: bool = False
    manual: bool = False
    min_steps_between_runs: int = 0
    min_minutes_between_runs: int = 0

    # Limits
    max_examples: int = 1000
    max_duration_minutes: int = 30

    # Behavior
    interruptible: bool = False
    resumable: bool = False

    # Actions (for P0 suites)
    actions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, suite_id: str, data: Dict[str, Any]) -> "EvalSuite":
        triggers = data.get("triggers", {})
        limits = data.get("limits", {})

        return cls(
            id=suite_id,
            name=data.get("name", suite_id),
            description=data.get("description", ""),
            priority_class=PriorityClass(data.get("priority_class", "P1")),
            skills=[SkillEvalSpec.from_dict(s) for s in data.get("skills", [])],
            # Triggers
            on_new_checkpoint=triggers.get("on_new_checkpoint", True),
            background_only=triggers.get("background_only", False),
            on_idle_gpu=triggers.get("on_idle_gpu", False),
            manual=triggers.get("manual", False),
            min_steps_between_runs=triggers.get("min_steps_between_runs", 0),
            min_minutes_between_runs=triggers.get("min_minutes_between_runs", 0),
            # Limits
            max_examples=limits.get("max_examples", 1000),
            max_duration_minutes=limits.get("max_duration_minutes", 30),
            # Behavior
            interruptible=data.get("interruptible", False),
            resumable=data.get("resumable", False),
            # Actions
            actions=data.get("actions", {}),
        )

    @property
    def total_examples(self) -> int:
        """Total examples across all skills in this suite."""
        return sum(s.batch_size for s in self.skills)


@dataclass
class SuiteRun:
    """Record of a suite execution."""
    suite_id: str
    run_id: str
    hero_id: str
    campaign_id: str
    checkpoint_step: int
    started_at: str
    completed_at: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, interrupted
    jobs_submitted: int = 0
    jobs_completed: int = 0
    results: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuiteRun":
        return cls(
            suite_id=data["suite_id"],
            run_id=data["run_id"],
            hero_id=data["hero_id"],
            campaign_id=data["campaign_id"],
            checkpoint_step=data["checkpoint_step"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            status=data.get("status", "pending"),
            jobs_submitted=data.get("jobs_submitted", 0),
            jobs_completed=data.get("jobs_completed", 0),
            results=data.get("results", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# CONFIG LOADER
# =============================================================================

class SuiteConfig:
    """Loader for eval_suites.yaml config."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "eval_suites.yaml"
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None
        self._suites: Dict[str, EvalSuite] = {}
        self._loaded = False

    def _load(self):
        """Load config from disk."""
        if self._loaded:
            return

        if not self.config_path.exists():
            logger.warning(f"Eval suites config not found: {self.config_path}")
            self._config = {}
            self._loaded = True
            return

        try:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f) or {}

            # Parse suites
            for suite_id, suite_data in self._config.get("suites", {}).items():
                self._suites[suite_id] = EvalSuite.from_dict(suite_id, suite_data)

            logger.info(f"Loaded {len(self._suites)} eval suites from config")
            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load eval suites config: {e}")
            self._config = {}
            self._loaded = True

    @property
    def suites(self) -> Dict[str, EvalSuite]:
        """Get all suites."""
        self._load()
        return self._suites

    def get_suite(self, suite_id: str) -> Optional[EvalSuite]:
        """Get a suite by ID."""
        self._load()
        return self._suites.get(suite_id)

    def get_suites_by_priority(self, priority_class: PriorityClass) -> List[EvalSuite]:
        """Get all suites of a given priority class."""
        self._load()
        return [s for s in self._suites.values() if s.priority_class == priority_class]

    def get_priority_mapping(self) -> Dict[str, str]:
        """Get priority class to job priority mapping."""
        self._load()
        return self._config.get("priority_mapping", {
            "P0": "critical",
            "P1": "high",
            "P2": "low",
        })

    def get_scheduler_settings(self) -> Dict[str, Any]:
        """Get scheduler settings."""
        self._load()
        return self._config.get("scheduler", {})


# =============================================================================
# SUITE RUN HISTORY
# =============================================================================

class SuiteRunHistory:
    """
    Track when suites were last run.

    Stored in status/eval_suite_runs.json
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = base_dir
        self.history_file = base_dir / "status" / "eval_suite_runs.json"
        self._runs: Dict[str, List[SuiteRun]] = {}  # suite_id -> list of runs
        self._lock = Lock()
        self._loaded = False

    def _ensure_loaded(self):
        """Load history from disk if not already loaded."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            if self.history_file.exists():
                try:
                    with open(self.history_file) as f:
                        data = json.load(f)
                    for suite_id, runs in data.get("runs", {}).items():
                        self._runs[suite_id] = [SuiteRun.from_dict(r) for r in runs]
                    logger.info(f"Loaded suite run history: {sum(len(r) for r in self._runs.values())} runs")
                except Exception as e:
                    logger.error(f"Failed to load suite run history: {e}")

            self._loaded = True

    def _save(self):
        """Save history to disk."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "runs": {
                suite_id: [r.to_dict() for r in runs[-100:]]  # Keep last 100 per suite
                for suite_id, runs in self._runs.items()
            }
        }

        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_last_run(
        self,
        suite_id: str,
        hero_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> Optional[SuiteRun]:
        """Get the most recent run of a suite."""
        self._ensure_loaded()

        runs = self._runs.get(suite_id, [])
        if not runs:
            return None

        # Filter by hero/campaign if specified
        if hero_id:
            runs = [r for r in runs if r.hero_id == hero_id]
        if campaign_id:
            runs = [r for r in runs if r.campaign_id == campaign_id]

        if not runs:
            return None

        # Return most recent
        return max(runs, key=lambda r: r.started_at)

    def record_run(self, run: SuiteRun):
        """Record a new suite run."""
        self._ensure_loaded()

        with self._lock:
            if run.suite_id not in self._runs:
                self._runs[run.suite_id] = []
            self._runs[run.suite_id].append(run)
            self._save()

    def update_run(self, run: SuiteRun):
        """Update an existing run."""
        self._ensure_loaded()

        with self._lock:
            runs = self._runs.get(run.suite_id, [])
            for i, r in enumerate(runs):
                if r.run_id == run.run_id:
                    runs[i] = run
                    self._save()
                    return

            # If not found, add it
            if run.suite_id not in self._runs:
                self._runs[run.suite_id] = []
            self._runs[run.suite_id].append(run)
            self._save()


# =============================================================================
# SCHEDULER
# =============================================================================

class EvalScheduler:
    """
    Main scheduler that decides when to run which suites.

    Integrates with:
    - RunContext for model identity
    - Job system for submitting eval jobs
    - Suite run history for cooldown checks
    """

    def __init__(
        self,
        config: Optional[SuiteConfig] = None,
        history: Optional[SuiteRunHistory] = None,
        job_submitter: Optional[Callable] = None,
    ):
        self.config = config or SuiteConfig()
        self.history = history or SuiteRunHistory()
        self._job_submitter = job_submitter  # Injected for testing

    def _get_job_submitter(self):
        """Get job submitter, lazy-loading if not injected."""
        if self._job_submitter:
            return self._job_submitter

        # Try to import job dispatcher
        try:
            from vault.server import submit_job
            self._job_submitter = submit_job
            return self._job_submitter
        except ImportError:
            logger.warning("Job dispatcher not available")
            return None

    def _should_run_suite(
        self,
        suite: EvalSuite,
        hero_id: str,
        campaign_id: str,
        checkpoint_step: int,
        trigger: str,  # "checkpoint", "idle", "manual"
    ) -> tuple[bool, str]:
        """
        Check if a suite should run based on triggers and cooldowns.

        Returns (should_run, reason).
        """
        # Check RealmMode first
        try:
            from core.realm_state import can_run_evals
            priority_class = suite.priority_class.value  # "P0", "P1", "P2"
            if not can_run_evals(priority_class):
                return False, f"realm mode doesn't allow {priority_class} evals"
        except ImportError:
            pass  # If realm_state not available, allow all

        # Check trigger type matches suite config
        if trigger == "checkpoint":
            if not suite.on_new_checkpoint:
                return False, "not triggered on checkpoint"
            if suite.background_only:
                return False, "background_only suite"

        elif trigger == "idle":
            if not suite.on_idle_gpu:
                return False, "not triggered on idle GPU"

        elif trigger == "manual":
            pass  # Manual always allowed

        else:
            return False, f"unknown trigger: {trigger}"

        # Check cooldowns
        last_run = self.history.get_last_run(suite.id, hero_id, campaign_id)

        if last_run:
            # Check step cooldown
            if suite.min_steps_between_runs > 0:
                steps_since = checkpoint_step - last_run.checkpoint_step
                if steps_since < suite.min_steps_between_runs:
                    return False, f"step cooldown ({steps_since} < {suite.min_steps_between_runs})"

            # Check time cooldown
            if suite.min_minutes_between_runs > 0:
                try:
                    last_time = datetime.fromisoformat(last_run.started_at)
                    minutes_since = (datetime.now() - last_time).total_seconds() / 60
                    if minutes_since < suite.min_minutes_between_runs:
                        return False, f"time cooldown ({minutes_since:.1f} < {suite.min_minutes_between_runs} min)"
                except Exception:
                    pass

        return True, "ok"

    def _enqueue_suite_jobs(
        self,
        suite: EvalSuite,
        hero_id: str,
        campaign_id: str,
        checkpoint_step: int,
        checkpoint_path: str,
        context_hash: str,
    ) -> SuiteRun:
        """
        Submit jobs for all skills in a suite.

        Returns a SuiteRun record.
        """
        import uuid

        # Create run record
        run = SuiteRun(
            suite_id=suite.id,
            run_id=str(uuid.uuid4())[:8],
            hero_id=hero_id,
            campaign_id=campaign_id,
            checkpoint_step=checkpoint_step,
            started_at=datetime.now().isoformat(),
            status="running",
        )

        # Get priority mapping
        priority_mapping = self.config.get_priority_mapping()
        job_priority = priority_mapping.get(suite.priority_class.value, "normal")

        # Import eval_job constructor
        from guild.job_types import eval_job, JobPriority

        # Map string priority to enum
        priority_enum = JobPriority(job_priority)

        # Get job submitter
        submitter = self._get_job_submitter()
        if not submitter:
            logger.error("No job submitter available, cannot enqueue suite jobs")
            run.status = "failed"
            run.results["error"] = "no job submitter"
            return run

        # Submit jobs for each skill
        jobs_submitted = 0
        for skill_spec in suite.skills:
            try:
                spec = eval_job(
                    skill_id=skill_spec.skill_id,
                    level=skill_spec.level,
                    batch_size=skill_spec.batch_size,
                    priority=priority_enum,
                    hero_id=hero_id,
                    campaign_id=campaign_id,
                    checkpoint_id=f"checkpoint-{checkpoint_step}",
                    checkpoint_path=checkpoint_path,
                    context_hash=context_hash,
                )

                # Add suite_id to payload
                spec.payload["suite_id"] = suite.id
                spec.payload["suite_run_id"] = run.run_id

                # Submit the job
                submitter(spec)
                jobs_submitted += 1

                logger.info(
                    f"[{suite.id}] Submitted eval job: {skill_spec.skill_id} L{skill_spec.level} "
                    f"(batch={skill_spec.batch_size}, priority={job_priority})"
                )

            except Exception as e:
                logger.error(f"Failed to submit eval job for {skill_spec.skill_id}: {e}")

        run.jobs_submitted = jobs_submitted

        if jobs_submitted == 0:
            run.status = "failed"
            run.results["error"] = "no jobs submitted"
        else:
            logger.info(f"[{suite.id}] Suite queued: {jobs_submitted} jobs for checkpoint-{checkpoint_step}")

        # Record the run
        self.history.record_run(run)

        return run

    def on_checkpoint_saved(
        self,
        run_ctx: Any,  # RunContext
        checkpoint_step: int,
        checkpoint_path: Optional[str] = None,
    ) -> List[SuiteRun]:
        """
        Called when a new checkpoint is saved.

        Triggers P0 and eligible P1 suites.

        Args:
            run_ctx: RunContext with hero/campaign identity
            checkpoint_step: The checkpoint step number
            checkpoint_path: Path to checkpoint (optional, derived from run_ctx if not provided)

        Returns:
            List of SuiteRun records for suites that were triggered
        """
        hero_id = run_ctx.hero_id or "unknown"
        campaign_id = run_ctx.campaign_id or "unknown"
        context_hash = run_ctx.context_hash()

        if checkpoint_path is None:
            checkpoint_path = run_ctx.current_model_dir or ""

        triggered_runs = []

        # Check P0 suites first (gatekeeping)
        for suite in self.config.get_suites_by_priority(PriorityClass.P0):
            should_run, reason = self._should_run_suite(
                suite, hero_id, campaign_id, checkpoint_step, "checkpoint"
            )
            if should_run:
                run = self._enqueue_suite_jobs(
                    suite, hero_id, campaign_id, checkpoint_step, checkpoint_path, context_hash
                )
                triggered_runs.append(run)
            else:
                logger.debug(f"[{suite.id}] Skipped: {reason}")

        # Check P1 suites (coverage)
        for suite in self.config.get_suites_by_priority(PriorityClass.P1):
            should_run, reason = self._should_run_suite(
                suite, hero_id, campaign_id, checkpoint_step, "checkpoint"
            )
            if should_run:
                run = self._enqueue_suite_jobs(
                    suite, hero_id, campaign_id, checkpoint_step, checkpoint_path, context_hash
                )
                triggered_runs.append(run)
            else:
                logger.debug(f"[{suite.id}] Skipped: {reason}")

        return triggered_runs

    def on_idle_gpu(
        self,
        run_ctx: Any,  # RunContext
        checkpoint_step: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ) -> List[SuiteRun]:
        """
        Called when GPU becomes idle.

        Triggers eligible P2 suites.

        Args:
            run_ctx: RunContext with hero/campaign identity
            checkpoint_step: The checkpoint step (latest if not provided)
            checkpoint_path: Path to checkpoint

        Returns:
            List of SuiteRun records for suites that were triggered
        """
        hero_id = run_ctx.hero_id or "unknown"
        campaign_id = run_ctx.campaign_id or "unknown"
        context_hash = run_ctx.context_hash()

        if checkpoint_path is None:
            checkpoint_path = run_ctx.current_model_dir or ""

        if checkpoint_step is None:
            # Try to get latest checkpoint step from ledger
            try:
                from core.checkpoint_ledger import get_ledger
                ledger = get_ledger()
                latest = ledger.get_latest()
                checkpoint_step = latest.step if latest else 0
            except Exception:
                checkpoint_step = 0

        triggered_runs = []

        # Check P2 suites (exploratory)
        for suite in self.config.get_suites_by_priority(PriorityClass.P2):
            should_run, reason = self._should_run_suite(
                suite, hero_id, campaign_id, checkpoint_step, "idle"
            )
            if should_run:
                run = self._enqueue_suite_jobs(
                    suite, hero_id, campaign_id, checkpoint_step, checkpoint_path, context_hash
                )
                triggered_runs.append(run)
            else:
                logger.debug(f"[{suite.id}] Skipped: {reason}")

        return triggered_runs

    def trigger_suite(
        self,
        suite_id: str,
        run_ctx: Any,  # RunContext
        checkpoint_step: int,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[SuiteRun]:
        """
        Manually trigger a specific suite.

        Bypasses cooldown checks.

        Args:
            suite_id: ID of the suite to run
            run_ctx: RunContext with hero/campaign identity
            checkpoint_step: The checkpoint step
            checkpoint_path: Path to checkpoint

        Returns:
            SuiteRun record if triggered, None if suite not found
        """
        suite = self.config.get_suite(suite_id)
        if not suite:
            logger.warning(f"Suite not found: {suite_id}")
            return None

        hero_id = run_ctx.hero_id or "unknown"
        campaign_id = run_ctx.campaign_id or "unknown"
        context_hash = run_ctx.context_hash()

        if checkpoint_path is None:
            checkpoint_path = run_ctx.current_model_dir or ""

        return self._enqueue_suite_jobs(
            suite, hero_id, campaign_id, checkpoint_step, checkpoint_path, context_hash
        )

    def get_suite_status(
        self,
        hero_id: str,
        campaign_id: str,
        checkpoint_step: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all suites for a checkpoint.

        Returns dict mapping suite_id to status info:
        - last_run: when suite last ran
        - last_score: average accuracy from last run
        - is_stale: whether suite is overdue
        - can_run: whether suite can be triggered
        """
        status = {}

        for suite_id, suite in self.config.suites.items():
            last_run = self.history.get_last_run(suite_id, hero_id, campaign_id)

            info: Dict[str, Any] = {
                "suite_id": suite_id,
                "name": suite.name,
                "priority_class": suite.priority_class.value,
                "last_run": None,
                "last_score": None,
                "is_stale": False,
                "can_run": True,
                "reason": "ok",
            }

            if last_run:
                info["last_run"] = last_run.started_at
                info["last_score"] = last_run.results.get("avg_accuracy")
                info["last_checkpoint"] = last_run.checkpoint_step

                # Check staleness
                steps_since = checkpoint_step - last_run.checkpoint_step
                if suite.min_steps_between_runs > 0 and steps_since > suite.min_steps_between_runs * 2:
                    info["is_stale"] = True

            # Check if can run
            can_run, reason = self._should_run_suite(
                suite, hero_id, campaign_id, checkpoint_step, "checkpoint"
            )
            info["can_run"] = can_run
            info["reason"] = reason

            status[suite_id] = info

        return status


# =============================================================================
# SINGLETON
# =============================================================================

_scheduler: Optional[EvalScheduler] = None
_scheduler_lock = Lock()


def get_scheduler() -> EvalScheduler:
    """Get the singleton scheduler instance."""
    global _scheduler

    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = EvalScheduler()
        return _scheduler


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval Suite Scheduler")
    parser.add_argument("--list", action="store_true", help="List all suites")
    parser.add_argument("--status", action="store_true", help="Show suite status")
    parser.add_argument("--trigger", type=str, help="Manually trigger a suite")
    parser.add_argument("--checkpoint", type=int, default=0, help="Checkpoint step")

    args = parser.parse_args()

    scheduler = get_scheduler()

    if args.list:
        print("Eval Suites:")
        print("=" * 60)
        for suite_id, suite in scheduler.config.suites.items():
            print(f"\n{suite.priority_class.value} | {suite_id}: {suite.name}")
            print(f"   {suite.description}")
            print(f"   Skills: {', '.join(f'{s.skill_id}:L{s.level}' for s in suite.skills)}")
            print(f"   Triggers: checkpoint={suite.on_new_checkpoint}, idle={suite.on_idle_gpu}")
            if suite.min_steps_between_runs:
                print(f"   Cooldown: {suite.min_steps_between_runs} steps")

    elif args.status:
        from core.run_context import get_run_context
        ctx = get_run_context()

        if not ctx.hero_id or not ctx.campaign_id:
            print("No active campaign")
            exit(1)

        print(f"Suite Status for {ctx.hero_id}/{ctx.campaign_id}")
        print("=" * 60)

        status = scheduler.get_suite_status(
            ctx.hero_id, ctx.campaign_id, args.checkpoint
        )

        for suite_id, info in status.items():
            stale = " [STALE]" if info["is_stale"] else ""
            can_run = "yes" if info["can_run"] else f"no ({info['reason']})"
            last = info.get("last_run", "never")[:10] if info.get("last_run") else "never"
            print(f"\n{info['priority_class']} | {suite_id}: {info['name']}{stale}")
            print(f"   Last run: {last}")
            print(f"   Can run: {can_run}")

    elif args.trigger:
        from core.run_context import get_run_context
        ctx = get_run_context()

        if not ctx.hero_id:
            print("No active campaign")
            exit(1)

        print(f"Triggering suite: {args.trigger}")
        run = scheduler.trigger_suite(args.trigger, ctx, args.checkpoint)
        if run:
            print(f"Suite run started: {run.run_id}")
            print(f"Jobs submitted: {run.jobs_submitted}")
        else:
            print("Failed to trigger suite")

    else:
        parser.print_help()
