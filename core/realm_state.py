"""
Realm State - Global mode and state for the training system.

The Realm has exactly ONE mode at any time:
- OFFLINE: Nothing should be running (startup/reset)
- IDLE: System ready, no auto-training
- TRAINING: Normal operation, auto-run + evals allowed
- EVAL_ONLY: No training, but eval suites may run
- MAINTENANCE: Only admin jobs allowed

Usage:
    from core.realm_state import get_realm_mode, set_realm_mode, RealmMode

    # Check current mode
    mode = get_realm_mode()
    if mode == RealmMode.TRAINING:
        # OK to auto-run training

    # Change mode (e.g., from UI "Stop" button)
    set_realm_mode(RealmMode.IDLE, changed_by="tavern", reason="user clicked stop")

    # Check if action is allowed
    from core.realm_state import can_start_training, can_run_evals
    if can_start_training():
        submit_training_job()
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class RealmMode(str, Enum):
    """
    Global operating modes for the Realm.

    Each mode defines what jobs are allowed to run.
    """
    OFFLINE = "offline"          # Nothing running (startup/reset)
    IDLE = "idle"                # Ready but no auto-training
    TRAINING = "training"        # Normal operation
    EVAL_ONLY = "eval_only"      # No training, evals OK
    MAINTENANCE = "maintenance"  # Admin jobs only

    @property
    def allows_training(self) -> bool:
        """Whether this mode allows training jobs."""
        return self == RealmMode.TRAINING

    @property
    def allows_auto_training(self) -> bool:
        """Whether this mode allows auto-queued training."""
        return self == RealmMode.TRAINING

    @property
    def allows_manual_training(self) -> bool:
        """Whether this mode allows manually-submitted training."""
        return self in {RealmMode.TRAINING, RealmMode.IDLE}

    @property
    def allows_evals(self) -> bool:
        """Whether this mode allows eval jobs."""
        return self in {RealmMode.TRAINING, RealmMode.EVAL_ONLY, RealmMode.IDLE}

    @property
    def allows_p0_evals(self) -> bool:
        """Whether P0 (gatekeeping) evals auto-trigger."""
        return self == RealmMode.TRAINING

    @property
    def allows_p1_evals(self) -> bool:
        """Whether P1 (coverage) evals auto-trigger."""
        return self == RealmMode.TRAINING

    @property
    def allows_p2_evals(self) -> bool:
        """Whether P2 (exploratory) evals can run."""
        return self in {RealmMode.TRAINING, RealmMode.EVAL_ONLY}

    @property
    def description(self) -> str:
        """Human-readable description."""
        descriptions = {
            RealmMode.OFFLINE: "System offline - nothing running",
            RealmMode.IDLE: "Ready - no auto-training, manual OK",
            RealmMode.TRAINING: "Training - auto-run and evals enabled",
            RealmMode.EVAL_ONLY: "Eval only - no training allowed",
            RealmMode.MAINTENANCE: "Maintenance - admin jobs only",
        }
        return descriptions.get(self, "Unknown mode")


@dataclass
class RealmState:
    """
    Current state of the Realm.

    Stored in control/realm_state.json
    """
    mode: RealmMode
    changed_at: str  # ISO timestamp
    changed_by: str  # "cli", "tavern", "system", "daemon"
    reason: Optional[str] = None

    # Optional: track what triggered the mode change
    previous_mode: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "changed_at": self.changed_at,
            "changed_by": self.changed_by,
            "reason": self.reason,
            "previous_mode": self.previous_mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RealmState":
        return cls(
            mode=RealmMode(data.get("mode", "idle")),
            changed_at=data.get("changed_at", datetime.now().isoformat()),
            changed_by=data.get("changed_by", "unknown"),
            reason=data.get("reason"),
            previous_mode=data.get("previous_mode"),
        )


# =============================================================================
# STATE FILE MANAGEMENT
# =============================================================================

_state_lock = Lock()


def _get_state_file() -> Path:
    """Get path to realm state file."""
    try:
        from core.paths import get_base_dir
        base_dir = get_base_dir()
    except ImportError:
        base_dir = Path(__file__).parent.parent
    return base_dir / "control" / "realm_state.json"


def _load_state() -> RealmState:
    """Load state from disk."""
    state_file = _get_state_file()

    if not state_file.exists():
        # Default to IDLE if no state file
        return RealmState(
            mode=RealmMode.IDLE,
            changed_at=datetime.now().isoformat(),
            changed_by="system",
            reason="initial state",
        )

    try:
        with open(state_file) as f:
            data = json.load(f)
        return RealmState.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to load realm state: {e}")
        return RealmState(
            mode=RealmMode.IDLE,
            changed_at=datetime.now().isoformat(),
            changed_by="system",
            reason=f"error loading state: {e}",
        )


def _save_state(state: RealmState):
    """Save state to disk."""
    state_file = _get_state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)

    with open(state_file, "w") as f:
        json.dump(state.to_dict(), f, indent=2)


# =============================================================================
# PUBLIC API
# =============================================================================

def get_realm_state() -> RealmState:
    """Get the current realm state."""
    with _state_lock:
        return _load_state()


def get_realm_mode() -> RealmMode:
    """Get the current realm mode."""
    return get_realm_state().mode


def set_realm_mode(
    mode: RealmMode,
    changed_by: str = "system",
    reason: Optional[str] = None,
) -> RealmState:
    """
    Set the realm mode.

    Args:
        mode: The new mode
        changed_by: Who/what triggered this change ("cli", "tavern", "daemon", "system")
        reason: Human-readable reason for the change

    Returns:
        The new RealmState
    """
    with _state_lock:
        current = _load_state()
        previous_mode = current.mode.value

        new_state = RealmState(
            mode=mode,
            changed_at=datetime.now().isoformat(),
            changed_by=changed_by,
            reason=reason,
            previous_mode=previous_mode,
        )

        _save_state(new_state)

        logger.info(f"Realm mode changed: {previous_mode} -> {mode.value} (by {changed_by})")

        # Emit event
        try:
            from core.events import emit_event
            emit_event(
                "mode_changed",
                from_mode=previous_mode,
                to_mode=mode.value,
                changed_by=changed_by,
                reason=reason,
            )
        except ImportError:
            pass

        return new_state


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def can_start_training(manual: bool = False) -> bool:
    """
    Check if training can be started.

    Args:
        manual: True if this is a manual/explicit request, False if auto-queued
    """
    mode = get_realm_mode()
    if manual:
        return mode.allows_manual_training
    return mode.allows_auto_training


def can_run_evals(priority_class: Optional[str] = None) -> bool:
    """
    Check if evals can run.

    Args:
        priority_class: Optional "P0", "P1", or "P2" to check specific class
    """
    mode = get_realm_mode()

    if priority_class == "P0":
        return mode.allows_p0_evals
    elif priority_class == "P1":
        return mode.allows_p1_evals
    elif priority_class == "P2":
        return mode.allows_p2_evals

    return mode.allows_evals


def start_training_mode(changed_by: str = "system") -> RealmState:
    """Convenience: switch to TRAINING mode."""
    return set_realm_mode(
        RealmMode.TRAINING,
        changed_by=changed_by,
        reason="start training",
    )


def stop_training_mode(changed_by: str = "system") -> RealmState:
    """Convenience: switch to IDLE mode (stop auto-training)."""
    return set_realm_mode(
        RealmMode.IDLE,
        changed_by=changed_by,
        reason="stop training",
    )


def enter_maintenance_mode(changed_by: str = "system", reason: str = "maintenance") -> RealmState:
    """Convenience: switch to MAINTENANCE mode."""
    return set_realm_mode(
        RealmMode.MAINTENANCE,
        changed_by=changed_by,
        reason=reason,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Realm State Manager")
    parser.add_argument("--get", action="store_true", help="Get current state")
    parser.add_argument("--set", type=str, choices=[m.value for m in RealmMode], help="Set mode")
    parser.add_argument("--reason", type=str, default="cli command", help="Reason for change")

    args = parser.parse_args()

    if args.set:
        new_mode = RealmMode(args.set)
        state = set_realm_mode(new_mode, changed_by="cli", reason=args.reason)
        print(f"Mode set to: {state.mode.value}")
        print(f"  Changed at: {state.changed_at}")
        print(f"  Changed by: {state.changed_by}")
        print(f"  Reason: {state.reason}")
        print(f"  Previous: {state.previous_mode}")
    else:
        state = get_realm_state()
        print(f"Current Realm State:")
        print(f"  Mode: {state.mode.value}")
        print(f"  Description: {state.mode.description}")
        print(f"  Changed at: {state.changed_at}")
        print(f"  Changed by: {state.changed_by}")
        print(f"  Reason: {state.reason}")
        print()
        print(f"Permissions:")
        print(f"  Auto-training: {state.mode.allows_auto_training}")
        print(f"  Manual training: {state.mode.allows_manual_training}")
        print(f"  P0 evals: {state.mode.allows_p0_evals}")
        print(f"  P1 evals: {state.mode.allows_p1_evals}")
        print(f"  P2 evals: {state.mode.allows_p2_evals}")
