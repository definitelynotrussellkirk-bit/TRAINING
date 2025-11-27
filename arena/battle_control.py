"""
Battle Control - Command the hero during combat.

Battle Control provides commands to manage the training battle:

    Rally      (pause)  - Pause after current round to regroup
    Charge     (resume) - Continue the battle
    Retreat    (stop)   - End the battle gracefully
    Abandon    (skip)   - Abandon current quest, move to next
    Status              - Check battle state

RPG Flavor:
    The Battle Commander uses signal flags to communicate with the hero
    during combat. When the Rally flag is raised, the hero completes the
    current exchange and waits for orders. The Charge flag sends them
    back into the fray.

Signal Files (control/):
    .rally    (.pause)  - Hero pauses after current round
    .charge   (.resume) - Hero continues from rally
    .retreat  (.stop)   - Hero finishes round and retreats
    .abandon  (.skip)   - Hero abandons current quest

This module wraps core/training_controller.py with RPG-themed naming.
"""

from pathlib import Path
from typing import Any, Optional

from core.training_controller import TrainingController as _TrainingController


class BattleControl(_TrainingController):
    """
    Command and control for the training battle.

    RPG wrapper around TrainingController with themed method names.

    Usage:
        control = BattleControl(base_dir)

        # Command the hero
        control.signal_rally("Need to check the hero's wounds")
        control.signal_charge()  # Resume battle
        control.signal_retreat("Battle won, hero needs rest")
        control.signal_abandon("This quest is cursed")

        # Check status (from training daemon)
        if control.should_rally():
            control.wait_for_charge()

        # Query state
        status = control.get_battle_state()
    """

    # =========================================================================
    # SIGNAL COMMANDS (for CLI/external use)
    # =========================================================================

    def signal_rally(self, reason: Optional[str] = None):
        """
        Signal the hero to rally (pause).

        Hero will complete current round and wait for orders.

        Args:
            reason: Why we're calling a rally
        """
        self.signal_pause(reason)

    def signal_charge(self):
        """
        Signal the hero to charge (resume).

        Hero continues battle from the rally point.
        """
        self.signal_resume()

    def signal_retreat(self, reason: Optional[str] = None):
        """
        Signal the hero to retreat (stop).

        Hero finishes current round, saves checkpoint, and exits battle.

        Args:
            reason: Why we're retreating
        """
        self.signal_stop(reason)

    def signal_abandon(self, reason: Optional[str] = None):
        """
        Signal hero to abandon current quest (skip).

        Current quest is marked as failed, hero moves to next quest.

        Args:
            reason: Why we're abandoning this quest
        """
        self.signal_skip(reason)

    def clear_all_signals(self):
        """Clear all signal flags. Emergency reset."""
        self.clear_all()

    # =========================================================================
    # SIGNAL CHECKS (for training daemon use)
    # =========================================================================

    def should_rally(self) -> bool:
        """
        Check if hero should rally after current round.

        Returns:
            True if rally signal is active
        """
        return self.should_pause_after_batch()

    def should_retreat(self) -> bool:
        """
        Check if hero should retreat after current round.

        Returns:
            True if retreat signal is active
        """
        return self.should_stop_after_batch()

    def should_abandon(self) -> bool:
        """
        Check if hero should abandon current quest.

        Returns:
            True if abandon signal is active
        """
        return self.should_skip_current_file()

    def wait_for_charge(self):
        """
        Wait for charge signal to continue battle.

        Blocks until charge (resume) signal is received.
        Called after rally to wait for orders.
        """
        self.wait_for_resume()

    def clear_rally(self):
        """Clear the rally signal after handling."""
        self.clear_pause()

    def clear_retreat(self):
        """Clear the retreat signal after handling."""
        self.clear_stop()

    def clear_abandon(self):
        """Clear the abandon signal after handling."""
        self.clear_skip()

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def set_battle_state(
        self,
        state: str,
        quest_file: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """
        Update the battle state.

        Args:
            state: "idle", "fighting", "rallied", "retreating", "abandoning"
            quest_file: Current quest being fought
            reason: Reason for current state
        """
        # Map RPG states to technical states
        state_map = {
            "idle": "idle",
            "fighting": "training",
            "rallied": "paused",
            "retreating": "stopping",
            "abandoning": "skipping",
        }
        tech_state = state_map.get(state, state)
        self.set_status(tech_state, current_file=quest_file, reason=reason)

    def get_battle_state(self) -> dict[str, Any]:
        """
        Get current battle state.

        Returns:
            Dict with state, signals, quest info
        """
        status = self.get_status()

        # Map technical states to RPG states
        state_map = {
            "idle": "idle",
            "training": "fighting",
            "paused": "rallied",
            "stopping": "retreating",
            "skipping": "abandoning",
        }

        # Map technical signals to RPG signals
        signal_map = {
            "pause": "rally",
            "resume": "charge",
            "stop": "retreat",
            "skip": "abandon",
        }

        rpg_signals = [
            signal_map.get(s, s)
            for s in status.get("signals", [])
        ]

        return {
            "state": state_map.get(status.get("status", "idle"), status.get("status")),
            "last_update": status.get("last_update"),
            "quest_file": status.get("current_file"),
            "rallied_at": status.get("paused_at"),
            "reason": status.get("reason"),
            "signals": rpg_signals,
            # Include raw status for debugging
            "_raw": status,
        }


# Convenience function
def get_battle_control(base_dir: str | Path) -> BattleControl:
    """Get a BattleControl instance for the given base directory."""
    return BattleControl(str(base_dir))


# Re-export original class for backward compatibility
TrainingController = _TrainingController
