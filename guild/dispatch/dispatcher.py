"""
Quest Dispatcher - Coordinates quest flow from trainers to the Quest Board.

The Quest Dispatcher is the central coordinator that:
1. Checks if the hero needs work (queue is empty, cooldown passed)
2. Consults the Progression Advisor for quest difficulty
3. Requests quests from Skill Trainers
4. Passes quests through the Quality Gate
5. Posts approved quests to the Quest Board (training queue)

RPG Flavor:
    The Quest Dispatcher is the busy clerk at the Guild Hall reception desk.
    When the Quest Board is empty, the Dispatcher:
    - Asks the Advisor: "What level is the hero ready for?"
    - Visits the Skill Trainer: "Give me 100 Level 5 challenges"
    - Takes the scrolls to the Quality Gate: "Are these quests valid?"
    - Posts approved quests to the Quest Board for the hero

Usage:
    from guild.dispatch import QuestDispatcher

    dispatcher = QuestDispatcher(base_dir)

    # Run a single dispatch cycle
    result = dispatcher.run_dispatch()

    # Check if hero needs quests
    needs_work, reason = dispatcher.hero_needs_work()
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from guild.skills.loader import get_trainer, list_trainers
from guild.skills.contract import Batch, Sample
from guild.dispatch.types import (
    DispatchDecision,
    DispatchResult,
    DispatchStatus,
    QuestVerdict,
)
from guild.dispatch.quality_gate import QuestQualityGate
from guild.dispatch.advisor import ProgressionAdvisor


logger = logging.getLogger(__name__)


class QuestDispatcher:
    """
    Coordinates quest flow from Skill Trainers to the Quest Board.

    The Dispatcher is the central coordinator for the quest pipeline:
    Trainer -> Quality Gate -> Quest Board -> Hero

    Usage:
        dispatcher = QuestDispatcher(base_dir, config)

        # Single dispatch cycle
        result = dispatcher.run_dispatch()

        # Check status
        status = dispatcher.get_status()
    """

    def __init__(
        self,
        base_dir: Path | str,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Quest Dispatcher.

        Args:
            base_dir: Base training directory
            config: Optional config dict (loads config.json if None)
        """
        self.base_dir = Path(base_dir)

        # Load config if not provided
        if config is None:
            config_file = self.base_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
            else:
                config = {}
        self.config = config

        # Initialize components
        self.quality_gate = QuestQualityGate(
            min_tokens=config.get("min_length", 10),
            max_tokens=config.get("max_length", 4096),
        )
        self.advisor = ProgressionAdvisor(base_dir, config)

        # Directories
        self.inbox_dir = self.base_dir / "inbox"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.base_dir / "guild" / "dispatch" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir = self.base_dir / "guild" / "dispatch" / "rejected"
        self.rejected_dir.mkdir(parents=True, exist_ok=True)

        # Dispatch state
        self._status = DispatchStatus(
            active_skill=self.advisor.get_active_skill(),
        )
        self._last_dispatch_time: float = 0

        # Auto-dispatch config
        self._auto_config = config.get("auto_generate", {})

    def hero_needs_work(self) -> tuple[bool, str]:
        """
        Check if the hero needs quests.

        Checks:
        - Auto-dispatch enabled
        - Queue depth below threshold
        - Not currently processing
        - Cooldown passed
        - Trainer online

        Returns:
            (needs_work, reason) - True if should dispatch, with explanation
        """
        if not self._auto_config.get("enabled", False):
            return False, "Auto-dispatch disabled in config"

        # Check queue depth
        queue_status = self._get_queue_status()
        threshold = self._auto_config.get("threshold", 0)

        if queue_status["total_queued"] > threshold:
            return False, f"Queue depth ({queue_status['total_queued']}) > threshold ({threshold})"

        # Check if currently processing
        if queue_status["processing"] > 0:
            return False, "Already processing a quest file"

        # Check cooldown
        cooldown = self._auto_config.get("cooldown_sec", 180)
        time_since_last = time.time() - self._last_dispatch_time

        if time_since_last < cooldown:
            remaining = int(cooldown - time_since_last)
            return False, f"Cooldown: {remaining}s remaining"

        # Check trainer availability
        active_skill = self.advisor.get_active_skill()
        try:
            trainer = get_trainer(active_skill)
            if not trainer.health():
                return False, f"{active_skill} trainer offline"
            self._status.trainer_online = True
        except Exception as e:
            return False, f"Trainer error: {e}"

        return True, "Hero is ready for quests"

    def request_quests(
        self,
        skill_id: str | None = None,
        count: int | None = None,
        level: int | None = None,
        seed: int | None = None,
    ) -> tuple[bool, Batch | None, dict[str, Any]]:
        """
        Request quests from a Skill Trainer.

        If no skill/level specified, uses Progression Advisor recommendation.

        Args:
            skill_id: Skill to request (default: active skill)
            count: Number of quests (default: from config)
            level: Difficulty level (default: from advisor)
            seed: Random seed for reproducibility

        Returns:
            (success, batch, metadata) - Batch of quests and metadata
        """
        # Get defaults from config/advisor
        if skill_id is None:
            skill_id = self.advisor.get_active_skill()

        if count is None:
            count = self._auto_config.get("count", 100)

        if level is None:
            params = self.advisor.recommend_quest_params(skill_id, count)
            level = params["level"]

        level_info = self.advisor.get_current_level(skill_id)

        logger.info(
            f"Requesting {count:,} {skill_id} quests at Level {level} "
            f"({level_info['name']})"
        )

        start_time = time.time()

        try:
            trainer = get_trainer(skill_id)
            batch = trainer.sample(level=level, count=count, seed=seed)

            generation_time = time.time() - start_time

            logger.info(
                f"Received {len(batch.samples):,} quests in {generation_time:.1f}s "
                f"({len(batch.samples) / generation_time:.1f} quests/sec)"
            )

            metadata = {
                "skill": skill_id,
                "level": level,
                "level_name": level_info["name"],
                "count": len(batch.samples),
                "generation_time": generation_time,
                "rate": len(batch.samples) / generation_time if generation_time > 0 else 0,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
            }

            self._status.total_quests_requested += len(batch.samples)

            return True, batch, metadata

        except Exception as e:
            logger.error(f"Quest request failed: {e}")
            return False, None, {"error": str(e)}

    def inspect_quests(self, batch: Batch) -> tuple[bool, list[dict], dict]:
        """
        Pass quests through the Quality Gate.

        Converts batch to message format and runs quality checks.

        Args:
            batch: Batch of quests from trainer

        Returns:
            (approved, quest_dicts, report_dict) - Approved quests and report
        """
        logger.info(f"Inspecting {len(batch.samples):,} quests...")

        # Convert samples to dict format for inspection
        quest_dicts = [
            {
                "messages": sample.messages,
                "metadata": sample.metadata,
            }
            for sample in batch.samples
        ]

        report = self.quality_gate.inspect(quest_dicts)

        if report.verdict == QuestVerdict.APPROVED:
            logger.info(f"Quality Gate: APPROVED ({report.passed_checks}/{report.total_checks} checks)")
            self._status.total_quests_approved += len(quest_dicts)
        elif report.verdict == QuestVerdict.CONDITIONAL:
            logger.warning(f"Quality Gate: CONDITIONAL - {report.recommendation}")
            self._status.total_quests_approved += len(quest_dicts)
        else:
            logger.warning(f"Quality Gate: REJECTED - {report.recommendation}")
            self._status.total_quests_rejected += len(quest_dicts)

        # Track quality scores
        self._status.quality_history.append(report.pass_rate)
        if len(self._status.quality_history) > 100:
            self._status.quality_history = self._status.quality_history[-100:]
        self._status.avg_quality_score = (
            sum(self._status.quality_history) / len(self._status.quality_history)
        )

        approved = report.verdict in (QuestVerdict.APPROVED, QuestVerdict.CONDITIONAL)

        return approved, quest_dicts, report.to_dict()

    def post_to_board(
        self,
        quests: Sequence[dict],
        metadata: dict[str, Any],
        priority: str = "normal",
    ) -> str | None:
        """
        Post quests to the Quest Board (training queue).

        Writes quests to inbox as JSONL for the training daemon to pick up.

        Args:
            quests: List of quest dicts with messages
            metadata: Quest batch metadata
            priority: Queue priority (high/normal/low)

        Returns:
            Path to the created quest file, or None if failed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        skill = metadata.get("skill", "unknown")
        level = metadata.get("level", 0)
        filename = f"quest_{skill}_L{level}_{len(quests)}_{timestamp}.jsonl"
        filepath = self.inbox_dir / filename

        logger.info(f"Posting {len(quests):,} quests to Quest Board: {filename}")

        try:
            # Write JSONL
            with open(filepath, 'w') as f:
                for quest in quests:
                    f.write(json.dumps(quest) + '\n')

            # Save metadata sidecar
            meta_file = self.reports_dir / f"{filename}.meta.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Posted: {filename} (priority: {priority})")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to post quests: {e}")
            return None

    def run_dispatch(self, force: bool = False) -> DispatchResult:
        """
        Run a single dispatch cycle.

        The main workflow:
        1. Check if hero needs work
        2. Request quests from trainer
        3. Pass through quality gate
        4. Post approved quests to board

        Args:
            force: Skip hero_needs_work check

        Returns:
            DispatchResult with outcome details
        """
        start_time = time.time()

        # Check if should dispatch
        if not force:
            needs_work, reason = self.hero_needs_work()
            if not needs_work:
                return DispatchResult(
                    success=False,
                    decision=DispatchDecision.WAIT,
                    reason=reason,
                    duration_seconds=time.time() - start_time,
                )

        # Request quests
        success, batch, metadata = self.request_quests()
        if not success or batch is None:
            return DispatchResult(
                success=False,
                decision=DispatchDecision.UNAVAILABLE,
                reason=metadata.get("error", "Request failed"),
                duration_seconds=time.time() - start_time,
            )

        # Inspect quests
        approved, quest_dicts, report_dict = self.inspect_quests(batch)

        # Save quality report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"quality_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)

        # Add report to metadata
        metadata["quality_report"] = report_dict
        metadata["approved"] = approved

        # Post if approved
        if approved:
            priority = self._auto_config.get("priority", "normal")
            output_file = self.post_to_board(quest_dicts, metadata, priority)

            if output_file:
                self._last_dispatch_time = time.time()
                self._status.dispatch_count += 1
                self._status.last_dispatch_time = datetime.now()

                logger.info(
                    f"Dispatch #{self._status.dispatch_count}: "
                    f"{len(quest_dicts):,} quests posted"
                )

                return DispatchResult(
                    success=True,
                    decision=DispatchDecision.DISPATCH,
                    reason="Quests posted to Quest Board",
                    skill_id=batch.skill_id,
                    level=batch.level,
                    quests_requested=len(batch.samples),
                    quests_approved=len(quest_dicts),
                    quality_report=self.quality_gate.last_report,
                    output_file=output_file,
                    duration_seconds=time.time() - start_time,
                )
        else:
            # Save rejected quests for analysis
            reject_file = self.rejected_dir / f"rejected_{timestamp}.jsonl"
            with open(reject_file, 'w') as f:
                for quest in quest_dicts:
                    f.write(json.dumps(quest) + '\n')
            logger.info(f"Rejected quests saved to {reject_file}")

        return DispatchResult(
            success=False,
            decision=DispatchDecision.WAIT,
            reason="Quests failed quality gate",
            skill_id=batch.skill_id,
            level=batch.level,
            quests_requested=len(batch.samples),
            quests_approved=0,
            quality_report=self.quality_gate.last_report,
            duration_seconds=time.time() - start_time,
        )

    def get_status(self) -> dict[str, Any]:
        """Get current dispatcher status."""
        self._update_status()
        return {
            **self._status.to_dict(),
            "queue_status": self._get_queue_status(),
            "progression": self.advisor.get_status(),
        }

    def _update_status(self):
        """Update status with current state."""
        self._status.active_skill = self.advisor.get_active_skill()

        # Update approval rate
        total = self._status.total_quests_requested
        if total > 0:
            self._status.approval_rate = self._status.total_quests_approved / total

    def _get_queue_status(self) -> dict[str, Any]:
        """Get training queue status."""
        queue_dir = self.base_dir / "queue"
        inbox_count = len(list(self.inbox_dir.glob("*.jsonl")))

        return {
            "inbox": inbox_count,
            "high": len(list((queue_dir / "high").glob("*.jsonl"))) if (queue_dir / "high").exists() else 0,
            "normal": len(list((queue_dir / "normal").glob("*.jsonl"))) if (queue_dir / "normal").exists() else 0,
            "low": len(list((queue_dir / "low").glob("*.jsonl"))) if (queue_dir / "low").exists() else 0,
            "processing": len(list((queue_dir / "processing").glob("*.jsonl"))) if (queue_dir / "processing").exists() else 0,
            "total_queued": inbox_count + sum([
                len(list((queue_dir / p).glob("*.jsonl")))
                for p in ["high", "normal", "low"]
                if (queue_dir / p).exists()
            ]),
        }

    def check_trainer_status(self, skill_id: str | None = None) -> bool:
        """
        Check if a skill trainer is online.

        Args:
            skill_id: Skill to check (default: active skill)

        Returns:
            True if trainer is online
        """
        if skill_id is None:
            skill_id = self.advisor.get_active_skill()

        try:
            trainer = get_trainer(skill_id)
            online = trainer.health()

            if online:
                info = trainer.info()
                logger.info(f"{skill_id} trainer online: {info.name} v{info.version}")
                level_status = self.advisor.get_status(skill_id)
                logger.info(
                    f"Hero at Level {level_status.current_level}/{level_status.total_levels} "
                    f"({level_status.level_name})"
                )
            else:
                logger.warning(f"{skill_id} trainer offline")

            return online

        except Exception as e:
            logger.error(f"Trainer check failed: {e}")
            return False


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for Quest Dispatcher."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quest Dispatcher - Coordinate quest flow to the Quest Board"
    )
    parser.add_argument('--base-dir', default='/path/to/training')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Status
    subparsers.add_parser('status', help='Show dispatcher status')

    # Check trainer
    check_p = subparsers.add_parser('check', help='Check trainer status')
    check_p.add_argument('--skill', help='Specific skill to check')

    # Dispatch
    dispatch_p = subparsers.add_parser('dispatch', help='Run dispatch cycle')
    dispatch_p.add_argument('--force', action='store_true', help='Force dispatch')
    dispatch_p.add_argument('--count', type=int, help='Override quest count')
    dispatch_p.add_argument('--level', type=int, help='Override level')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    dispatcher = QuestDispatcher(Path(args.base_dir))

    if args.command == 'status':
        print("\n" + "=" * 80)
        print("QUEST DISPATCHER STATUS")
        print("=" * 80 + "\n")

        status = dispatcher.get_status()
        print(json.dumps(status, indent=2, default=str))

        print("\n" + "=" * 80 + "\n")

    elif args.command == 'check':
        online = dispatcher.check_trainer_status(args.skill)
        if online:
            print(f"\nTrainer is ONLINE and ready\n")
        else:
            print(f"\nTrainer is OFFLINE\n")

    elif args.command == 'dispatch':
        result = dispatcher.run_dispatch(force=args.force)

        print("\n" + "=" * 80)
        print("DISPATCH RESULT")
        print("=" * 80 + "\n")
        print(json.dumps(result.to_dict(), indent=2, default=str))

        if result.success:
            print(f"\nDispatched {result.quests_approved:,} quests to Quest Board\n")
        else:
            print(f"\nDispatch failed: {result.reason}\n")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
