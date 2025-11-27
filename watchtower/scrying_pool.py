"""
Scrying Pool - Real-time observation of training battles.

The Scrying Pool is a magical basin in the Watchtower that shows
real-time visions of what's happening in the Arena below.

RPG Flavor:
    Watchtower sentries gather around the Scrying Pool to observe
    the hero's progress in battle. The pool's surface ripples with
    each combat round, showing damage taken, hit rates, and progress
    toward victory.

Observation Sources:
    status/training_status.json → Arena battle status
    status/battle_status.json   → New format status
    control/state.json          → Command state

This module provides a unified interface for observing training state.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from watchtower.types import ScryingVision, AlertLevel, HeraldMessage


class ScryingPool:
    """
    The Scrying Pool - observe real-time training state.

    Reads status files and provides a unified view of training progress.

    Usage:
        pool = ScryingPool(base_dir)

        # Get current vision
        vision = pool.gaze()
        print(f"Battle: {vision.battle_state}")
        print(f"Round: {vision.current_round}/{vision.total_rounds}")
        print(f"Damage: {vision.damage_taken:.4f}")

        # Check for alerts
        alerts = pool.detect_disturbances()
        for alert in alerts:
            print(f"[{alert.level.name}] {alert.title}")
    """

    def __init__(self, base_dir: str | Path):
        """
        Initialize the Scrying Pool.

        Args:
            base_dir: Base training directory
        """
        self.base_dir = Path(base_dir)
        self.status_dir = self.base_dir / "status"
        self.control_dir = self.base_dir / "control"

        # Status file locations
        self.battle_status_file = self.status_dir / "battle_status.json"
        self.training_status_file = self.status_dir / "training_status.json"
        self.control_state_file = self.control_dir / "state.json"

        # Alert thresholds
        self._damage_warning_threshold = 2.0   # Loss > 2.0
        self._damage_alarm_threshold = 5.0     # Loss > 5.0
        self._stale_vision_seconds = 120       # No update in 2 min

    # =========================================================================
    # OBSERVATION METHODS
    # =========================================================================

    def gaze(self) -> ScryingVision:
        """
        Gaze into the Scrying Pool to see current training state.

        Returns:
            ScryingVision with current battle state
        """
        # Try new format first
        if self.battle_status_file.exists():
            return self._read_battle_status()

        # Fall back to legacy format
        if self.training_status_file.exists():
            return self._read_training_status()

        # No status - idle
        return ScryingVision(
            battle_state="idle",
            observed_at=datetime.now(),
        )

    def _read_battle_status(self) -> ScryingVision:
        """Read from new battle_status.json format."""
        try:
            with open(self.battle_status_file) as f:
                data = json.load(f)

            return ScryingVision(
                battle_state=data.get("state", "idle"),
                quest_file=data.get("quest_file"),
                current_round=data.get("current_round", 0),
                total_rounds=data.get("total_rounds", 0),
                campaign=data.get("campaign", 0),
                total_campaigns=data.get("total_campaigns", 1),
                damage_taken=data.get("damage_taken", 0.0),
                hit_rate=data.get("hit_rate"),
                rounds_per_second=data.get("rounds_per_second", 0.0),
                time_remaining=data.get("time_remaining"),
                eta=data.get("eta"),
                hero_vram_mb=data.get("hero_vram_mb", 0.0),
                hero_checkpoint=data.get("hero_checkpoint"),
                observed_at=datetime.now(),
            )
        except Exception:
            return ScryingVision(battle_state="unknown", observed_at=datetime.now())

    def _read_training_status(self) -> ScryingVision:
        """Read from legacy training_status.json format."""
        try:
            with open(self.training_status_file) as f:
                data = json.load(f)

            # Map legacy status to battle state
            status_map = {
                "training": "fighting",
                "paused": "paused",
                "completed": "victory",
                "stopped": "retreat",
                "error": "defeated",
                "idle": "idle",
            }

            return ScryingVision(
                battle_state=status_map.get(data.get("status", "idle"), "unknown"),
                quest_file=data.get("current_file"),
                current_round=data.get("current_step", 0),
                total_rounds=data.get("total_steps", 0),
                campaign=data.get("current_epoch", 0),
                total_campaigns=data.get("total_epochs", 1),
                damage_taken=data.get("loss", 0.0),
                hit_rate=data.get("accuracy"),
                rounds_per_second=data.get("steps_per_second", 0.0),
                time_remaining=data.get("time_remaining"),
                eta=data.get("eta"),
                hero_vram_mb=data.get("vram_mb", 0.0),
                hero_checkpoint=data.get("checkpoint"),
                observed_at=datetime.now(),
            )
        except Exception:
            return ScryingVision(battle_state="unknown", observed_at=datetime.now())

    def is_battle_active(self) -> bool:
        """Check if a battle is currently active."""
        vision = self.gaze()
        return vision.battle_state in ("fighting", "paused")

    def get_battle_progress(self) -> Dict[str, Any]:
        """
        Get simplified battle progress for display.

        Returns:
            Dict with progress info
        """
        vision = self.gaze()
        return {
            "state": vision.battle_state,
            "progress_percent": vision.progress_percent,
            "current_round": vision.current_round,
            "total_rounds": vision.total_rounds,
            "damage_taken": vision.damage_taken,
            "eta": vision.eta,
        }

    # =========================================================================
    # ALERT DETECTION
    # =========================================================================

    def detect_disturbances(self) -> list[HeraldMessage]:
        """
        Detect any disturbances that should be announced by heralds.

        Returns:
            List of HeraldMessage alerts
        """
        alerts = []
        vision = self.gaze()

        # Check for high damage (loss)
        if vision.damage_taken > self._damage_alarm_threshold:
            alerts.append(HeraldMessage(
                level=AlertLevel.ALARM,
                title="Critical Damage!",
                message=f"Hero taking severe damage: loss={vision.damage_taken:.4f}",
                source="scrying_pool",
                details={"loss": vision.damage_taken, "threshold": self._damage_alarm_threshold},
                announced_at=datetime.now(),
            ))
        elif vision.damage_taken > self._damage_warning_threshold:
            alerts.append(HeraldMessage(
                level=AlertLevel.WARNING,
                title="High Damage",
                message=f"Hero taking significant damage: loss={vision.damage_taken:.4f}",
                source="scrying_pool",
                details={"loss": vision.damage_taken, "threshold": self._damage_warning_threshold},
                announced_at=datetime.now(),
            ))

        # Check for stale vision (training might be stuck)
        if vision.battle_state == "fighting" and vision.observed_at:
            file_mtime = self._get_status_mtime()
            if file_mtime:
                age_seconds = (datetime.now() - file_mtime).total_seconds()
                if age_seconds > self._stale_vision_seconds:
                    alerts.append(HeraldMessage(
                        level=AlertLevel.WARNING,
                        title="Vision Stale",
                        message=f"No updates for {int(age_seconds)}s - hero may be stuck",
                        source="scrying_pool",
                        details={"age_seconds": age_seconds},
                        announced_at=datetime.now(),
                    ))

        # Check for defeat
        if vision.battle_state == "defeated":
            alerts.append(HeraldMessage(
                level=AlertLevel.ALARM,
                title="Hero Defeated!",
                message="Training has crashed",
                source="scrying_pool",
                announced_at=datetime.now(),
            ))

        return alerts

    def _get_status_mtime(self) -> Optional[datetime]:
        """Get modification time of status file."""
        for f in [self.battle_status_file, self.training_status_file]:
            if f.exists():
                return datetime.fromtimestamp(f.stat().st_mtime)
        return None

    # =========================================================================
    # QUEUE OBSERVATION
    # =========================================================================

    def observe_quest_board(self) -> Dict[str, Any]:
        """
        Observe the Quest Board (training queue) from the Scrying Pool.

        Returns:
            Dict with queue status
        """
        queue_dir = self.base_dir / "queue"

        counts = {
            "urgent": 0,
            "standard": 0,
            "reserves": 0,
            "active_duty": 0,
            "fallen": 0,
        }

        priority_map = {
            "high": "urgent",
            "normal": "standard",
            "low": "reserves",
            "processing": "active_duty",
            "failed": "fallen",
        }

        for tech_name, rpg_name in priority_map.items():
            subdir = queue_dir / tech_name
            if subdir.exists():
                counts[rpg_name] = len(list(subdir.glob("*.jsonl")))

        counts["total_pending"] = counts["urgent"] + counts["standard"] + counts["reserves"]

        return counts


# Convenience function
def get_scrying_pool(base_dir: str | Path) -> ScryingPool:
    """Get a ScryingPool instance for the given base directory."""
    return ScryingPool(base_dir)
