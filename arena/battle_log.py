"""
Battle Log - Records the hero's combat progress.

The Battle Log tracks training metrics and writes status for the War Room
(monitoring dashboard) to display.

RPG Flavor:
    The Battle Log is a magical tome that records every swing, hit, and
    miss during combat. The War Room scrying pool reads from this tome
    to show the hero's progress in real-time.

Metrics Mapping:
    loss → damage_taken (lower is better, hero is hurt less)
    validation_loss → validation_damage
    accuracy → hit_rate
    step → round
    epoch → campaign
    steps/sec → rounds_per_second

This module wraps core/training_status.py with RPG-themed naming.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from arena.types import BattleState, BattleStatus


class BattleLog:
    """
    Records and writes battle (training) status.

    Writes to status/battle_status.json for the War Room to read.

    Usage:
        log = BattleLog(base_dir)

        # Start battle
        log.begin_battle("quest_binary_L5.jsonl", total_rounds=1000)

        # Record combat round
        log.record_round(
            round_num=100,
            damage_taken=0.45,
            rounds_per_second=2.5,
        )

        # End battle
        log.end_battle(victory=True)
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.status_dir = self.base_dir / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)

        # Status files
        self.battle_status_file = self.status_dir / "battle_status.json"
        self.training_status_file = self.status_dir / "training_status.json"  # Legacy

        # Current status
        self._status = BattleStatus()

    def begin_battle(
        self,
        quest_file: str,
        total_rounds: int,
        total_campaigns: int = 1,
    ):
        """
        Begin a new battle.

        Args:
            quest_file: Name of the quest being fought
            total_rounds: Total training steps
            total_campaigns: Total epochs
        """
        self._status = BattleStatus(
            state=BattleState.FIGHTING,
            quest_file=quest_file,
            total_rounds=total_rounds,
            total_campaigns=total_campaigns,
            battle_started=datetime.now(),
            last_update=datetime.now(),
        )
        self._write_status()

    def record_round(
        self,
        round_num: int,
        damage_taken: float,
        campaign: int = 0,
        validation_damage: Optional[float] = None,
        hit_rate: Optional[float] = None,
        rounds_per_second: float = 0.0,
        time_remaining: Optional[str] = None,
        eta: Optional[str] = None,
        hero_checkpoint: Optional[str] = None,
        hero_vram_mb: float = 0.0,
    ):
        """
        Record a combat round (training step).

        Args:
            round_num: Current step number
            damage_taken: Training loss
            campaign: Current epoch
            validation_damage: Validation loss (if evaluated)
            hit_rate: Accuracy (if evaluated)
            rounds_per_second: Training speed
            time_remaining: Estimated time remaining
            eta: Estimated completion time
            hero_checkpoint: Current checkpoint path
            hero_vram_mb: GPU memory usage
        """
        self._status.current_round = round_num
        self._status.damage_taken = damage_taken
        self._status.campaign = campaign

        if validation_damage is not None:
            self._status.validation_damage = validation_damage
        if hit_rate is not None:
            self._status.hit_rate = hit_rate

        self._status.rounds_per_second = rounds_per_second
        self._status.time_remaining = time_remaining
        self._status.eta = eta
        self._status.hero_checkpoint = hero_checkpoint
        self._status.hero_vram_mb = hero_vram_mb
        self._status.last_update = datetime.now()

        self._write_status()

    def pause_battle(self):
        """Pause the current battle."""
        self._status.state = BattleState.PAUSED
        self._status.last_update = datetime.now()
        self._write_status()

    def resume_battle(self):
        """Resume a paused battle."""
        self._status.state = BattleState.FIGHTING
        self._status.last_update = datetime.now()
        self._write_status()

    def end_battle(self, victory: bool = True, reason: Optional[str] = None):
        """
        End the current battle.

        Args:
            victory: True if completed successfully
            reason: Optional reason for ending
        """
        if victory:
            self._status.state = BattleState.VICTORY
        else:
            self._status.state = BattleState.RETREAT

        self._status.last_update = datetime.now()
        self._write_status()

    def report_defeat(self, error: str):
        """
        Report that the hero was defeated (training crashed).

        Args:
            error: Error message describing the defeat
        """
        self._status.state = BattleState.DEFEATED
        self._status.last_update = datetime.now()
        self._write_status()

        # Also write to legacy format for compatibility
        self._write_legacy_status(error=error)

    def get_status(self) -> BattleStatus:
        """Get current battle status."""
        return self._status

    def _write_status(self):
        """Write status to JSON files."""
        # Write new format
        status_dict = self._status.to_dict()
        with open(self.battle_status_file, 'w') as f:
            json.dump(status_dict, f, indent=2)

        # Also write legacy format for backward compat
        self._write_legacy_status()

    def _write_legacy_status(self, error: Optional[str] = None):
        """Write status in legacy training_status.json format."""
        legacy = {
            "status": self._status.state.value,
            "current_file": self._status.quest_file,
            "current_step": self._status.current_round,
            "total_steps": self._status.total_rounds,
            "current_epoch": self._status.campaign,
            "total_epochs": self._status.total_campaigns,
            "loss": self._status.damage_taken,
            "validation_loss": self._status.validation_damage,
            "accuracy": self._status.hit_rate,
            "steps_per_second": self._status.rounds_per_second,
            "time_remaining": self._status.time_remaining,
            "eta": self._status.eta,
            "checkpoint": self._status.hero_checkpoint,
            "vram_mb": self._status.hero_vram_mb,
            "started_at": self._status.battle_started.isoformat() if self._status.battle_started else None,
            "last_update": self._status.last_update.isoformat() if self._status.last_update else None,
        }

        if error:
            legacy["error"] = error

        with open(self.training_status_file, 'w') as f:
            json.dump(legacy, f, indent=2)

    @classmethod
    def load_status(cls, base_dir: str | Path) -> Optional[BattleStatus]:
        """
        Load battle status from file.

        Args:
            base_dir: Base training directory

        Returns:
            BattleStatus or None if no status file
        """
        status_file = Path(base_dir) / "status" / "battle_status.json"

        if not status_file.exists():
            # Try legacy file
            legacy_file = Path(base_dir) / "status" / "training_status.json"
            if legacy_file.exists():
                return cls._load_legacy_status(legacy_file)
            return None

        with open(status_file) as f:
            data = json.load(f)

        return BattleStatus(
            state=BattleState(data.get("state", "idle")),
            quest_file=data.get("quest_file"),
            current_round=data.get("current_round", 0),
            total_rounds=data.get("total_rounds", 0),
            campaign=data.get("campaign", 0),
            total_campaigns=data.get("total_campaigns", 1),
            damage_taken=data.get("damage_taken", 0.0),
            validation_damage=data.get("validation_damage", 0.0),
            hit_rate=data.get("hit_rate"),
            rounds_per_second=data.get("rounds_per_second", 0.0),
            time_remaining=data.get("time_remaining"),
            eta=data.get("eta"),
            hero_checkpoint=data.get("hero_checkpoint"),
            hero_vram_mb=data.get("hero_vram_mb", 0.0),
            battle_started=datetime.fromisoformat(data["battle_started"]) if data.get("battle_started") else None,
            last_update=datetime.fromisoformat(data["last_update"]) if data.get("last_update") else None,
        )

    @classmethod
    def _load_legacy_status(cls, legacy_file: Path) -> BattleStatus:
        """Load from legacy training_status.json format."""
        with open(legacy_file) as f:
            data = json.load(f)

        # Map legacy status to BattleState
        status_map = {
            "training": BattleState.FIGHTING,
            "paused": BattleState.PAUSED,
            "completed": BattleState.VICTORY,
            "stopped": BattleState.RETREAT,
            "error": BattleState.DEFEATED,
            "idle": BattleState.IDLE,
        }

        return BattleStatus(
            state=status_map.get(data.get("status", "idle"), BattleState.IDLE),
            quest_file=data.get("current_file"),
            current_round=data.get("current_step", 0),
            total_rounds=data.get("total_steps", 0),
            campaign=data.get("current_epoch", 0),
            total_campaigns=data.get("total_epochs", 1),
            damage_taken=data.get("loss", 0.0),
            validation_damage=data.get("validation_loss"),
            hit_rate=data.get("accuracy"),
            rounds_per_second=data.get("steps_per_second", 0.0),
            time_remaining=data.get("time_remaining"),
            eta=data.get("eta"),
            hero_checkpoint=data.get("checkpoint"),
            hero_vram_mb=data.get("vram_mb", 0.0),
        )


# Convenience function
def get_battle_log(base_dir: str | Path) -> BattleLog:
    """Get a BattleLog instance for the given base directory."""
    return BattleLog(base_dir)
