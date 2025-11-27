"""
Chronicle - The historical record of model evolution.

The Chronicle is a great tome in the Vault that records the lineage
and evolution of every model version. Each entry tells the story of
how the hero grew stronger through training.

RPG Flavor:
    The Chronicle is kept by the Lorekeeper, who records every
    significant moment in the hero's journey. Each version entry
    includes what training was done, what metrics were achieved,
    and how the hero evolved from previous versions.

Version Format:
    v001, v002, v003... - Sequential version IDs
    Each version can restore a specific point in training

This module wraps management/model_versioner.py with RPG-themed naming.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vault.types import ChronicleEntry

# Import the underlying versioner
from management.model_versioner import ModelVersioner as _ModelVersioner


class Chronicle(_ModelVersioner):
    """
    The Chronicle - records model version history.

    RPG wrapper around ModelVersioner with themed method names.

    Usage:
        chronicle = Chronicle(base_dir)

        # Record a new chapter (version)
        entry = chronicle.record_chapter(
            checkpoint_path="/path/to/checkpoint",
            description="Completed SYLLO L5 training",
            training_scrolls=["train_syllo_L5.jsonl"]
        )

        # Read the chronicle
        entries = chronicle.read_history()

        # Restore to a previous chapter
        chronicle.restore_chapter("v003")
    """

    def __init__(self, base_dir: str = "/path/to/training"):
        """
        Initialize the Chronicle.

        Args:
            base_dir: Base training directory
        """
        super().__init__(base_dir)

    # =========================================================================
    # RECORDING HISTORY
    # =========================================================================

    def record_chapter(
        self,
        checkpoint_path: str | Path,
        description: str,
        training_scrolls: Optional[List[str]] = None,
        evolution_notes: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[ChronicleEntry]:
        """
        Record a new chapter in the Chronicle.

        Creates a versioned snapshot of the current model state.

        Args:
            checkpoint_path: Path to checkpoint to version
            description: Description of this version
            training_scrolls: List of training data files used
            evolution_notes: Notes about changes from previous version
            metrics: Validation metrics at this point

        Returns:
            ChronicleEntry if successful
        """
        training_scrolls = training_scrolls or []
        metrics = metrics or {}

        # Create version using underlying method
        result = self.create_version(
            adapter_path=str(checkpoint_path),
            description=description,
            training_data=training_scrolls,
            metrics=metrics,
        )

        if not result:
            return None

        # Create ChronicleEntry from result
        entry = ChronicleEntry(
            version_id=result.get("version_id", ""),
            version_number=result.get("version_number", 0),
            checkpoint_step=result.get("step", 0),
            source_checkpoint=str(checkpoint_path),
            description=description,
            training_data=training_scrolls,
            validation_loss=metrics.get("validation_loss"),
            validation_accuracy=metrics.get("validation_accuracy"),
            created_at=datetime.now(),
            evolution_notes=evolution_notes,
        )

        return entry

    def record_milestone(
        self,
        checkpoint_path: str | Path,
        milestone_name: str,
        metrics: Dict[str, float],
    ) -> Optional[ChronicleEntry]:
        """
        Record a training milestone (significant achievement).

        Args:
            checkpoint_path: Path to checkpoint
            milestone_name: Name of milestone (e.g., "SYLLO L5 Mastery")
            metrics: Achieved metrics

        Returns:
            ChronicleEntry if successful
        """
        return self.record_chapter(
            checkpoint_path=checkpoint_path,
            description=f"Milestone: {milestone_name}",
            metrics=metrics,
            evolution_notes=f"Achieved {milestone_name}",
        )

    # =========================================================================
    # READING HISTORY
    # =========================================================================

    def read_history(
        self,
        limit: Optional[int] = None,
    ) -> List[ChronicleEntry]:
        """
        Read the Chronicle's history.

        Args:
            limit: Maximum entries to return (newest first)

        Returns:
            List of ChronicleEntry
        """
        versions = self.list_versions()

        entries = []
        for v in versions:
            entry = ChronicleEntry(
                version_id=v.get("version_id", ""),
                version_number=int(v.get("version_id", "v0")[1:]) if v.get("version_id") else 0,
                checkpoint_step=v.get("step", 0),
                source_checkpoint=v.get("adapter_path", ""),
                description=v.get("description", ""),
                training_data=v.get("training_data", []),
                validation_loss=v.get("metrics", {}).get("validation_loss"),
                validation_accuracy=v.get("metrics", {}).get("validation_accuracy"),
                created_at=datetime.fromisoformat(v["created_at"]) if v.get("created_at") else None,
            )
            entries.append(entry)

        # Sort by version number descending (newest first)
        entries.sort(key=lambda e: e.version_number, reverse=True)

        if limit:
            entries = entries[:limit]

        return entries

    def read_chapter(self, version_id: str) -> Optional[ChronicleEntry]:
        """
        Read a specific chapter from the Chronicle.

        Args:
            version_id: Version ID (e.g., "v003")

        Returns:
            ChronicleEntry or None if not found
        """
        version = self.get_version(version_id)

        if not version:
            return None

        return ChronicleEntry(
            version_id=version.get("version_id", ""),
            version_number=int(version.get("version_id", "v0")[1:]),
            checkpoint_step=version.get("step", 0),
            source_checkpoint=version.get("adapter_path", ""),
            description=version.get("description", ""),
            training_data=version.get("training_data", []),
            validation_loss=version.get("metrics", {}).get("validation_loss"),
            validation_accuracy=version.get("metrics", {}).get("validation_accuracy"),
            created_at=datetime.fromisoformat(version["created_at"]) if version.get("created_at") else None,
        )

    def get_latest_chapter(self) -> Optional[ChronicleEntry]:
        """
        Get the most recent chapter in the Chronicle.

        Returns:
            Latest ChronicleEntry or None
        """
        history = self.read_history(limit=1)
        return history[0] if history else None

    # =========================================================================
    # RESTORATION
    # =========================================================================

    def restore_chapter(
        self,
        version_id: str,
        restore_path: Optional[str | Path] = None,
    ) -> bool:
        """
        Restore model to a specific chapter in history.

        Args:
            version_id: Version to restore (e.g., "v003")
            restore_path: Where to restore (default: current_model)

        Returns:
            True if successful
        """
        if restore_path:
            return self.restore_version(version_id, str(restore_path))
        return self.restore_version(version_id)

    def compare_chapters(
        self,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """
        Compare two chapters in the Chronicle.

        Args:
            version_a: First version ID
            version_b: Second version ID

        Returns:
            Dict with comparison results
        """
        chapter_a = self.read_chapter(version_a)
        chapter_b = self.read_chapter(version_b)

        if not chapter_a or not chapter_b:
            return {"error": "One or both chapters not found"}

        return {
            "version_a": chapter_a.to_dict(),
            "version_b": chapter_b.to_dict(),
            "step_difference": (chapter_b.checkpoint_step - chapter_a.checkpoint_step),
            "loss_change": (
                (chapter_b.validation_loss or 0) - (chapter_a.validation_loss or 0)
                if chapter_a.validation_loss and chapter_b.validation_loss
                else None
            ),
            "accuracy_change": (
                (chapter_b.validation_accuracy or 0) - (chapter_a.validation_accuracy or 0)
                if chapter_a.validation_accuracy and chapter_b.validation_accuracy
                else None
            ),
        }

    # =========================================================================
    # LINEAGE
    # =========================================================================

    def get_lineage(self) -> List[Dict[str, Any]]:
        """
        Get the full lineage (evolution tree) of the model.

        Returns:
            List of version summaries showing evolution
        """
        history = self.read_history()

        lineage = []
        for i, entry in enumerate(reversed(history)):
            node = {
                "version": entry.version_id,
                "step": entry.checkpoint_step,
                "description": entry.description,
                "metrics": {
                    "loss": entry.validation_loss,
                    "accuracy": entry.validation_accuracy,
                },
                "parent": history[len(history) - i].version_id if i > 0 else None,
            }
            lineage.append(node)

        return lineage

    def get_chronicle_status(self) -> Dict[str, Any]:
        """
        Get Chronicle status summary.

        Returns:
            Dict with chronicle statistics
        """
        history = self.read_history()
        latest = history[0] if history else None

        return {
            "total_chapters": len(history),
            "latest_version": latest.version_id if latest else None,
            "latest_step": latest.checkpoint_step if latest else None,
            "oldest_version": history[-1].version_id if history else None,
        }


# Convenience function
def get_chronicle(base_dir: str = "/path/to/training") -> Chronicle:
    """Get a Chronicle instance for the given base directory."""
    return Chronicle(base_dir)


# Re-export original for backward compatibility
ModelVersioner = _ModelVersioner
