#!/usr/bin/env python3
"""
Forge Queue Integration - Validation hooks for the training queue.

This module provides validated queue operations that run Forge validation
before adding files to the training queue.

Usage:
    from forge.queue_integration import validated_process_inbox

    # Process inbox with validation
    results = validated_process_inbox(queue)
    print(f"Added: {results['added']}, Rejected: {results['rejected']}")

Integration with training_queue.py:
    The training daemon can use these functions instead of direct queue.process_inbox()
    to ensure all data is validated before training.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def validated_add_to_queue(
    queue,
    file_path: Path,
    priority: str = "normal",
    validate: bool = True,
    skill_id: Optional[str] = None,
    reject_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Add a file to the queue with optional Forge validation.

    Args:
        queue: TrainingQueue instance
        file_path: Path to .jsonl file
        priority: "high", "normal", or "low"
        validate: If True, run Forge validation first
        skill_id: Optional skill ID for leakage check (auto-detected if None)
        reject_dir: Directory to move rejected files (defaults to queue/rejected/)

    Returns:
        {
            "success": bool,
            "file": str,
            "priority": str,
            "validated": bool,
            "validation_result": {...} or None,
            "error": str or None
        }
    """
    file_path = Path(file_path)
    result = {
        "success": False,
        "file": file_path.name,
        "priority": priority,
        "validated": False,
        "validation_result": None,
        "error": None,
    }

    if not file_path.exists():
        result["error"] = f"File not found: {file_path}"
        return result

    # Set up reject directory
    if reject_dir is None:
        reject_dir = queue.queue_dir / "rejected"
    reject_dir.mkdir(parents=True, exist_ok=True)

    # Run validation if enabled
    if validate:
        try:
            from forge.validator import validate_for_queue

            validation = validate_for_queue(file_path, skill_id=skill_id)
            result["validated"] = True
            result["validation_result"] = validation.to_dict()

            if not validation.passed:
                # Move to rejected
                reject_path = reject_dir / file_path.name
                shutil.move(str(file_path), str(reject_path))

                # Save rejection report
                report_path = reject_dir / f"{file_path.stem}.rejection.json"
                with open(report_path, "w") as f:
                    json.dump({
                        "file": file_path.name,
                        "rejected_at": datetime.utcnow().isoformat() + "Z",
                        "validation": validation.to_dict(),
                    }, f, indent=2)

                result["error"] = f"Validation failed: {validation.summary}"
                logger.warning(f"Rejected {file_path.name}: {validation.summary}")

                # Log to Battle Log
                try:
                    from core.battle_log import log_jobs
                    log_jobs(
                        f"🚫 Rejected {file_path.name}: {validation.summary}",
                        source="forge.queue_integration",
                        severity="warning",
                        details={
                            "file": file_path.name,
                            "errors": validation.errors[:5],
                            "leakage_count": validation.leakage_count,
                        }
                    )
                except Exception:
                    pass

                return result

        except Exception as e:
            logger.error(f"Validation error for {file_path}: {e}")
            result["error"] = f"Validation error: {e}"
            # On validation error, still add to queue (fail-open for availability)
            result["validation_result"] = {"error": str(e)}

    # Add to queue
    if queue.add_to_queue(file_path, priority):
        result["success"] = True

        # Log success to Battle Log
        try:
            from core.battle_log import log_jobs
            status = "validated" if result["validated"] else "unvalidated"
            log_jobs(
                f"✅ Added {file_path.name} to {priority} queue ({status})",
                source="forge.queue_integration",
                severity="success" if result["validated"] else "info",
                details={
                    "file": file_path.name,
                    "priority": priority,
                    "validated": result["validated"],
                }
            )
        except Exception:
            pass
    else:
        result["error"] = "Failed to add to queue"

    return result


def validated_process_inbox(
    queue,
    default_priority: str = "normal",
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Process all files from inbox with Forge validation.

    Args:
        queue: TrainingQueue instance
        default_priority: Default priority for inbox files
        validate: If True, run Forge validation on each file

    Returns:
        {
            "added": int,
            "rejected": int,
            "files": [
                {"file": str, "success": bool, "priority": str, ...},
                ...
            ]
        }
    """
    new_files = queue.scan_inbox()
    results = {
        "added": 0,
        "rejected": 0,
        "files": [],
    }

    for file_path in new_files:
        # Detect priority from filename (sparring = high)
        priority = default_priority
        if "sparring" in file_path.name.lower():
            priority = "high"

        file_result = validated_add_to_queue(
            queue=queue,
            file_path=file_path,
            priority=priority,
            validate=validate,
        )

        results["files"].append(file_result)

        if file_result["success"]:
            results["added"] += 1
        else:
            results["rejected"] += 1

    if results["added"] > 0 or results["rejected"] > 0:
        logger.info(
            f"📥 Processed inbox: {results['added']} added, {results['rejected']} rejected"
        )

    return results


def get_forge_queue_status(queue) -> Dict[str, Any]:
    """
    Get queue status with Forge-specific info.

    Returns queue status augmented with:
    - Rejected files count
    - Recent validation failures
    """
    # Get base queue status
    status = queue.get_queue_status()

    # Add rejected files
    rejected_dir = queue.queue_dir / "rejected"
    if rejected_dir.exists():
        rejected_files = list(rejected_dir.glob("*.jsonl"))
        status["rejected"] = {
            "count": len(rejected_files),
            "files": [f.name for f in rejected_files[:10]],
        }

        # Get recent rejection reasons
        recent_rejections = []
        for report_file in sorted(rejected_dir.glob("*.rejection.json"))[-5:]:
            try:
                with open(report_file) as f:
                    report = json.load(f)
                    recent_rejections.append({
                        "file": report.get("file"),
                        "rejected_at": report.get("rejected_at"),
                        "summary": report.get("validation", {}).get("summary", "Unknown"),
                    })
            except Exception:
                pass

        status["recent_rejections"] = recent_rejections
    else:
        status["rejected"] = {"count": 0, "files": []}
        status["recent_rejections"] = []

    return status


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Quick test
    from core.training_queue import TrainingQueue

    try:
        from core.paths import get_base_dir
        base_dir = get_base_dir()
    except ImportError:
        base_dir = Path.cwd()

    queue = TrainingQueue(str(base_dir))

    print("Forge Queue Status:")
    status = get_forge_queue_status(queue)

    print(f"  Queued: {status.get('total_queued', 0)}")
    print(f"  Rejected: {status.get('rejected', {}).get('count', 0)}")

    if status.get("recent_rejections"):
        print("\n  Recent Rejections:")
        for rej in status["recent_rejections"]:
            print(f"    - {rej['file']}: {rej['summary']}")
