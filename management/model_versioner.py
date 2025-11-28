#!/usr/bin/env python3
"""
Model Version Management System

Prevents catastrophic data loss by:
1. Creating versioned snapshots of trained models
2. Tracking metadata (what was trained, when, metrics)
3. Enabling restoration of any previous version
4. Maintaining evolution data with each version

Never lose a trained model again!
"""

import json
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use centralized path resolution for default
try:
    from core.paths import get_base_dir
    _DEFAULT_BASE_DIR = get_base_dir()
except ImportError:
    _DEFAULT_BASE_DIR = Path(__file__).parent.parent  # Fallback: parent of management/


class ModelVersioner:
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else _DEFAULT_BASE_DIR
        self.versions_dir = self.base_dir / "models" / "versions"
        self.backups_dir = self.base_dir / "models" / "backups"

        # Create directories
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def get_next_version_number(self) -> int:
        """Get next available version number"""
        versions = self.list_versions()
        if not versions:
            return 1

        # Extract version numbers (v001 -> 1)
        version_nums = [int(v['version_id'][1:]) for v in versions]
        return max(version_nums) + 1

    def create_version(
        self,
        adapter_path: str,
        description: str,
        training_data: List[str],
        metadata: Dict
    ) -> str:
        """
        Create a new model version

        Args:
            adapter_path: Path to the adapter to version
            description: Human-readable description
            training_data: List of training files used
            metadata: Additional metadata (steps, loss, metrics, etc.)

        Returns:
            version_id (e.g., "v001")
        """
        version_num = self.get_next_version_number()
        version_id = f"v{version_num:03d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"{version_id}_{timestamp}_{description.replace(' ', '_')}"

        version_path = self.versions_dir / version_name

        logger.info(f"Creating version {version_id}: {description}")

        try:
            # Create version directory
            version_path.mkdir(parents=True, exist_ok=True)

            # Copy adapter
            adapter_src = Path(adapter_path)
            if adapter_src.exists():
                adapter_dst = version_path / "adapter"
                logger.info(f"Copying adapter from {adapter_src} to {adapter_dst}")
                shutil.copytree(adapter_src, adapter_dst, dirs_exist_ok=True)
            else:
                logger.warning(f"Adapter path {adapter_src} does not exist")

            # Copy evolution snapshots if they exist
            evolution_src = self.base_dir / "data" / "evolution_snapshots"
            if evolution_src.exists():
                evolution_dst = version_path / "evolution_snapshots"
                logger.info(f"Copying evolution snapshots to {evolution_dst}")
                shutil.copytree(evolution_src, evolution_dst, dirs_exist_ok=True)
                snapshot_count = len(list(evolution_dst.glob("*.json")))
            else:
                snapshot_count = 0

            # Create metadata
            version_metadata = {
                "version_id": version_id,
                "version_name": version_name,
                "created_at": datetime.now().isoformat(),
                "description": description,
                "training_data": training_data,
                "evolution_snapshots": snapshot_count,
                **metadata  # Merge in additional metadata
            }

            # Save metadata
            metadata_path = version_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(version_metadata, f, indent=2)

            # Update "latest" symlink
            latest_link = self.versions_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(version_name, target_is_directory=True)

            logger.info(f"✅ Created version {version_id} successfully")
            logger.info(f"   Path: {version_path}")
            logger.info(f"   Evolution snapshots: {snapshot_count}")

            return version_id

        except Exception as e:
            logger.error(f"❌ Failed to create version {version_id}: {e}")
            # Cleanup on failure
            if version_path.exists():
                shutil.rmtree(version_path)
            raise

    def list_versions(self) -> List[Dict]:
        """List all versions with their metadata"""
        versions = []

        for version_dir in sorted(self.versions_dir.iterdir()):
            if version_dir.is_dir() and not version_dir.name == "latest":
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                    # Add size info
                    metadata['size_mb'] = sum(
                        f.stat().st_size for f in version_dir.rglob('*') if f.is_file()
                    ) / (1024 * 1024)

                    versions.append(metadata)

        return sorted(versions, key=lambda x: x['version_id'])

    def get_version_metadata(self, version_id: str) -> Optional[Dict]:
        """Get metadata for a specific version"""
        for version in self.list_versions():
            if version['version_id'] == version_id:
                return version
        return None

    def get_version_path(self, version_id: str) -> Optional[Path]:
        """Get the filesystem path for a version"""
        for version_dir in self.versions_dir.iterdir():
            if version_dir.is_dir() and version_dir.name.startswith(version_id):
                return version_dir
        return None

    def restore_version(self, version_id: str, target_path: str = None) -> bool:
        """
        Restore a version to current_model/ or specified path

        Args:
            version_id: Version to restore (e.g., "v001")
            target_path: Where to restore (default: current_model/)

        Returns:
            True if successful
        """
        version_path = self.get_version_path(version_id)
        if not version_path:
            logger.error(f"Version {version_id} not found")
            return False

        if target_path is None:
            target_path = self.base_dir / "current_model"
        else:
            target_path = Path(target_path)

        try:
            # Remove existing target
            if target_path.exists():
                logger.info(f"Removing existing {target_path}")
                shutil.rmtree(target_path)

            # Copy adapter
            adapter_src = version_path / "adapter"
            if adapter_src.exists():
                logger.info(f"Restoring {version_id} to {target_path}")
                shutil.copytree(adapter_src, target_path)
                logger.info(f"✅ Restored version {version_id} successfully")
                return True
            else:
                logger.error(f"Adapter not found in version {version_id}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to restore version {version_id}: {e}")
            return False

    def delete_version(self, version_id: str, confirm: bool = False) -> bool:
        """
        Delete a version (requires confirmation)

        Args:
            version_id: Version to delete
            confirm: Must be True to actually delete

        Returns:
            True if deleted
        """
        if not confirm:
            logger.error("❌ Delete requires confirm=True")
            return False

        version_path = self.get_version_path(version_id)
        if not version_path:
            logger.error(f"Version {version_id} not found")
            return False

        try:
            # Create backup before deletion
            backup_path = self.backups_dir / "deleted_versions" / f"{version_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Backing up {version_id} before deletion to {backup_path}")
            shutil.copytree(version_path, backup_path)

            # Now delete
            logger.info(f"Deleting version {version_id} from {version_path}")
            shutil.rmtree(version_path)

            logger.info(f"✅ Deleted version {version_id} (backup saved)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete version {version_id}: {e}")
            return False

    def get_latest_version(self) -> Optional[Dict]:
        """Get the latest version metadata"""
        versions = self.list_versions()
        return versions[-1] if versions else None

    def print_version_summary(self):
        """Print a summary of all versions"""
        versions = self.list_versions()

        if not versions:
            print("No versions found")
            return

        print(f"\n{'='*80}")
        print(f"MODEL VERSIONS ({len(versions)} total)")
        print(f"{'='*80}\n")

        for v in versions:
            print(f"📦 {v['version_id']} - {v['description']}")
            print(f"   Created: {v['created_at']}")
            print(f"   Training data: {', '.join(v['training_data']) if v['training_data'] else 'N/A'}")
            print(f"   Evolution snapshots: {v.get('evolution_snapshots', 0)}")
            print(f"   Size: {v['size_mb']:.1f} MB")

            if 'total_steps' in v:
                print(f"   Steps: {v['total_steps']}")
            if 'final_loss' in v:
                print(f"   Final loss: {v['final_loss']:.4f}")

            print()

        print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model Version Management")
    parser.add_argument('--base-dir', default=None, help='Base directory')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List versions
    subparsers.add_parser('list', help='List all versions')

    # Create version
    create_parser = subparsers.add_parser('create', help='Create new version')
    create_parser.add_argument('--adapter', required=True, help='Path to adapter')
    create_parser.add_argument('--description', required=True, help='Version description')
    create_parser.add_argument('--data', nargs='+', help='Training data files used')
    create_parser.add_argument('--steps', type=int, help='Total steps')
    create_parser.add_argument('--loss', type=float, help='Final loss')

    # Restore version
    restore_parser = subparsers.add_parser('restore', help='Restore a version')
    restore_parser.add_argument('version_id', help='Version to restore (e.g., v001)')
    restore_parser.add_argument('--target', help='Target path (default: current_model/)')

    # Delete version
    delete_parser = subparsers.add_parser('delete', help='Delete a version')
    delete_parser.add_argument('version_id', help='Version to delete')
    delete_parser.add_argument('--confirm', action='store_true', help='Confirm deletion')

    args = parser.parse_args()

    versioner = ModelVersioner(args.base_dir)

    if args.command == 'list':
        versioner.print_version_summary()

    elif args.command == 'create':
        metadata = {}
        if args.steps:
            metadata['total_steps'] = args.steps
        if args.loss:
            metadata['final_loss'] = args.loss

        version_id = versioner.create_version(
            adapter_path=args.adapter,
            description=args.description,
            training_data=args.data or [],
            metadata=metadata
        )
        print(f"✅ Created version {version_id}")

    elif args.command == 'restore':
        success = versioner.restore_version(args.version_id, args.target)
        if success:
            print(f"✅ Restored {args.version_id}")
        else:
            print(f"❌ Failed to restore {args.version_id}")

    elif args.command == 'delete':
        if not args.confirm:
            print("❌ Must use --confirm to delete")
            print(f"   Run: python3 model_versioner.py delete {args.version_id} --confirm")
        else:
            success = versioner.delete_version(args.version_id, confirm=True)
            if success:
                print(f"✅ Deleted {args.version_id}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
