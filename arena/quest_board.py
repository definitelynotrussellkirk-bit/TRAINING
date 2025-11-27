"""
Quest Board - Where quests await the hero.

The Quest Board manages the queue of training data files waiting to be
processed. Quests are organized by priority:

    Urgent (high)    - Critical quests, processed first
    Standard (normal) - Regular quests
    Reserves (low)    - Lower priority, processed when queue is empty

RPG Flavor:
    The Quest Board is a large wooden board at the Guild Hall entrance.
    Urgent quests are pinned at the top in red. The hero takes quests
    from the board, completes them in the Arena, and gains experience.

This module wraps core/training_queue.py with RPG-themed naming.
"""

from pathlib import Path
from typing import Any, Optional

# Re-export from core with RPG aliases
from core.training_queue import TrainingQueue as _TrainingQueue


class QuestBoard(_TrainingQueue):
    """
    The Quest Board - manages training data queue.

    RPG wrapper around TrainingQueue with themed method names.

    Usage:
        board = QuestBoard(base_dir)

        # Post a quest
        board.post_quest("train_binary_L5_100.jsonl", priority="urgent")

        # Check what's available
        status = board.get_board_status()

        # Take next quest for battle
        quest = board.take_next_quest()
    """

    def post_quest(
        self,
        quest_file: str | Path,
        priority: str = "standard",
    ) -> bool:
        """
        Post a quest to the board.

        Args:
            quest_file: Path to quest scroll (JSONL file)
            priority: "urgent", "standard", or "reserves"

        Returns:
            True if successfully posted
        """
        # Map RPG names to technical names
        priority_map = {
            "urgent": "high",
            "standard": "normal",
            "reserves": "low",
        }
        tech_priority = priority_map.get(priority, priority)
        return self.add_to_queue(str(quest_file), tech_priority)

    def take_next_quest(self) -> Optional[Path]:
        """
        Take the next quest from the board.

        Returns highest priority quest available.

        Returns:
            Path to quest file, or None if board is empty
        """
        result = self.get_next_file()
        return Path(result) if result else None

    def return_quest(self, quest_file: str | Path, failed: bool = False):
        """
        Return a quest to the board (incomplete or failed).

        Args:
            quest_file: Path to quest file
            failed: If True, quest goes to fallen pile
        """
        if failed:
            self.mark_failed(str(quest_file))
        else:
            self.mark_complete(str(quest_file))

    def get_board_status(self) -> dict[str, Any]:
        """
        Get current Quest Board status.

        Returns:
            Dict with quest counts per priority level
        """
        status = self.get_queue_status()

        # Map technical names to RPG names
        return {
            "urgent": status.get("high", 0),
            "standard": status.get("normal", 0),
            "reserves": status.get("low", 0),
            "active_duty": status.get("processing", 0),
            "fallen": status.get("failed", 0),
            "total_pending": status.get("total_queued", 0),
            "inbox": status.get("inbox", 0),
        }

    def list_quests(self, priority: Optional[str] = None) -> list[Path]:
        """
        List quests on the board.

        Args:
            priority: Filter by priority ("urgent", "standard", "reserves")
                     or None for all

        Returns:
            List of quest file paths
        """
        priority_map = {
            "urgent": "high",
            "standard": "normal",
            "reserves": "low",
        }

        if priority:
            tech_priority = priority_map.get(priority, priority)
            files = self.list_queue(tech_priority)
        else:
            files = []
            for p in ["high", "normal", "low"]:
                files.extend(self.list_queue(p))

        return [Path(f) for f in files]

    def clear_fallen(self) -> int:
        """
        Clear the fallen quest pile.

        Returns:
            Number of quests cleared
        """
        return self.clear_failed()

    def resurrect_fallen(self, to_priority: str = "standard") -> int:
        """
        Move fallen quests back to the board for retry.

        Args:
            to_priority: Priority level for resurrected quests

        Returns:
            Number of quests resurrected
        """
        priority_map = {
            "urgent": "high",
            "standard": "normal",
            "reserves": "low",
        }
        tech_priority = priority_map.get(to_priority, to_priority)
        return self.retry_failed(tech_priority)


# Convenience function
def get_quest_board(base_dir: str | Path) -> QuestBoard:
    """Get a QuestBoard instance for the given base directory."""
    return QuestBoard(str(base_dir))


# Re-export original class for backward compatibility
TrainingQueue = _TrainingQueue
