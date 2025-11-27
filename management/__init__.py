"""
Management modules for model and backup operations.

Contains:
- backup_manager.py: Verified backup system
- model_versioner.py: Version history
- checkpoint_retention.py: Retention policies
- consolidate_model.py: Model consolidation
- daily_snapshot.py: Daily snapshots

NEW: RPG-themed wrappers available in vault/:
- Archivist: Backup management (seals, verifies, restores)
- Chronicle: Version history (chapters, lineage, milestones)
- Treasury: Resource management (inventory, retention, cleanup)

Usage:
    # Traditional
    from management.backup_manager import BackupManager
    from management.model_versioner import ModelVersioner

    # RPG-themed (new)
    from vault import Archivist, Chronicle, Treasury
"""

# Note: vault/ imports from management/, so we can't re-export here
# to avoid circular imports. Use 'from vault import ...' directly.
