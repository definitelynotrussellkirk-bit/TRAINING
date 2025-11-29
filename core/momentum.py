"""
Momentum Engine - Forward progress tracking and blocker management.

The Momentum Engine's job is simple:
"Given the current state, what's the next thing we want to do?
If we can't do it, why not, and how can the user fix it?"

Instead of just failing with logs, the system:
1. Tries to do X (train, analyze, autogenerate, etc.)
2. If it can't, records a blocker with:
   - what_i_was_trying: "start a 2000-step training session"
   - why_i_failed: machine-readable + human message
   - how_to_fix: concrete instructions + UI target
3. Exposes that via API, and the UI surfaces it loudly

Usage:
    from core.momentum import report_blocker, clear_blocker, get_momentum_state

    # Report a blocker
    report_blocker(
        code="NO_ACTIVE_CAMPAIGN",
        what="Start a 2000-step training session",
        why="There is no active campaign selected.",
        how_to_fix="Go to the Guild, create or select a campaign, and mark it active.",
        suggested_action="open_guild",
    )

    # Clear when resolved
    clear_blocker("NO_ACTIVE_CAMPAIGN")

    # Check state
    state = get_momentum_state()
    print(state.status)  # "go" | "blocked" | "idle"
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.paths import get_control_dir

logger = logging.getLogger(__name__)


def _get_momentum_file() -> Path:
    """Get path to momentum state file."""
    return get_control_dir() / "momentum.json"


@dataclass
class Blocker:
    """A specific thing blocking forward progress."""

    code: str  # e.g. "NO_ACTIVE_CAMPAIGN", "MISSING_HERO_CONFIG"
    severity: str  # "error" | "warning"
    what_i_was_trying: str
    why_i_failed: str
    how_to_fix: str
    suggested_action: Optional[str] = None  # e.g. "open_guild", "open_settings"
    context: Dict[str, Any] = field(default_factory=dict)
    first_seen_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_seen_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class MomentumState:
    """Overall forward momentum state."""

    # High-level: are we "ready to go"?
    status: str  # "go" | "blocked" | "idle"

    # One primary blocker (the most actionable one)
    primary_blocker: Optional[Blocker] = None

    # All known blockers
    blockers: Dict[str, Blocker] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "primary_blocker": asdict(self.primary_blocker)
            if self.primary_blocker
            else None,
            "blockers": {code: asdict(b) for code, b in self.blockers.items()},
            "blocker_count": len(self.blockers),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MomentumState":
        blockers = {}
        for code, b in (data.get("blockers") or {}).items():
            blockers[code] = Blocker(**b)
        pb = data.get("primary_blocker")
        primary_blocker = Blocker(**pb) if pb else None
        return cls(
            status=data.get("status", "idle"),
            primary_blocker=primary_blocker,
            blockers=blockers,
        )


def _load_state() -> MomentumState:
    """Load momentum state from disk."""
    momentum_file = _get_momentum_file()
    if not momentum_file.exists():
        return MomentumState(status="idle")
    try:
        with open(momentum_file) as f:
            data = json.load(f)
        return MomentumState.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load momentum state: {e}")
        return MomentumState(status="idle")


def _save_state(state: MomentumState) -> None:
    """Save momentum state to disk."""
    momentum_file = _get_momentum_file()
    try:
        momentum_file.parent.mkdir(parents=True, exist_ok=True)
        with open(momentum_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save momentum state: {e}")


def report_blocker(
    code: str,
    what: str,
    why: str,
    how_to_fix: str,
    severity: str = "error",
    suggested_action: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Report a blocker that's preventing forward progress.

    Args:
        code: Machine-readable blocker code (e.g. "NO_ACTIVE_CAMPAIGN")
        what: What the system was trying to do
        why: Why it failed (human-readable)
        how_to_fix: Concrete instructions to resolve
        severity: "error" (blocks progress) or "warning" (degraded but can continue)
        suggested_action: UI action hint (e.g. "open_guild", "open_settings")
        context: Additional context data
    """
    state = _load_state()
    now = datetime.now(timezone.utc).isoformat()
    context = context or {}

    existing = state.blockers.get(code)
    if existing:
        # Update existing blocker
        existing.last_seen_at = now
        existing.what_i_was_trying = what
        existing.why_i_failed = why
        existing.how_to_fix = how_to_fix
        existing.suggested_action = suggested_action
        existing.context.update(context)
        blocker = existing
    else:
        # Create new blocker
        blocker = Blocker(
            code=code,
            severity=severity,
            what_i_was_trying=what,
            why_i_failed=why,
            how_to_fix=how_to_fix,
            suggested_action=suggested_action,
            context=context,
        )
        state.blockers[code] = blocker

    # Simple policy: any error-level blocker makes status "blocked"
    if severity == "error":
        state.status = "blocked"
        state.primary_blocker = blocker

    _save_state(state)
    logger.info(f"[Momentum] Blocker reported: {code} - {why}")


def clear_blocker(code: str) -> None:
    """
    Clear a blocker (condition has been resolved).

    Args:
        code: The blocker code to clear
    """
    state = _load_state()
    if code not in state.blockers:
        return  # Nothing to clear

    del state.blockers[code]

    # Recompute primary/status
    error_blockers = {
        c: b for c, b in state.blockers.items() if b.severity == "error"
    }

    if not error_blockers:
        state.status = "go"
        state.primary_blocker = None
    else:
        # Pick first remaining error as primary
        _, first = next(iter(error_blockers.items()))
        state.status = "blocked"
        state.primary_blocker = first

    _save_state(state)
    logger.info(f"[Momentum] Blocker cleared: {code}")


def clear_all_blockers() -> None:
    """Clear all blockers and reset to 'go' status."""
    state = MomentumState(status="go")
    _save_state(state)
    logger.info("[Momentum] All blockers cleared")


def get_momentum_state() -> MomentumState:
    """Get current momentum state."""
    return _load_state()


def set_status(status: str) -> None:
    """
    Manually set momentum status.

    Args:
        status: "go" | "blocked" | "idle"
    """
    state = _load_state()
    state.status = status
    _save_state(state)


# =============================================================================
# COMMON BLOCKER CODES (for reference)
# =============================================================================

# Campaign/Hero
BLOCKER_NO_ACTIVE_CAMPAIGN = "NO_ACTIVE_CAMPAIGN"
BLOCKER_MISSING_HERO_CONFIG = "MISSING_HERO_CONFIG"
BLOCKER_NO_HERO_SELECTED = "NO_HERO_SELECTED"

# Training
BLOCKER_NO_TRAINING_DATA = "NO_TRAINING_DATA"
BLOCKER_QUEUE_EMPTY = "QUEUE_EMPTY"
BLOCKER_TRAINING_PAUSED = "TRAINING_PAUSED"

# Infrastructure
BLOCKER_INFERENCE_OFFLINE = "INFERENCE_OFFLINE"
BLOCKER_GPU_UNAVAILABLE = "GPU_UNAVAILABLE"
BLOCKER_DISK_FULL = "DISK_FULL"

# Analysis
BLOCKER_NO_ANALYSIS_DATA = "NO_ANALYSIS_DATA"
BLOCKER_NO_CHECKPOINTS = "NO_CHECKPOINTS"


# =============================================================================
# COMPREHENSIVE MOMENTUM CHECK
# =============================================================================


def run_momentum_checks() -> MomentumState:
    """
    Run all momentum checks and update the state.

    This function checks various conditions and reports/clears blockers:
    - Active campaign exists
    - Training queue has files
    - Training paused state
    - Checkpoints exist

    Call this periodically or on-demand to keep momentum state up-to-date.

    Returns:
        Current MomentumState after all checks
    """
    # Check 1: Active campaign (critical - blocks everything)
    _check_active_campaign()

    # Only run other checks if we have a campaign
    state = _load_state()
    if BLOCKER_NO_ACTIVE_CAMPAIGN not in state.blockers:
        # Check 2: Training queue
        _check_training_queue()

        # Check 3: Training paused
        _check_training_paused()

        # Check 4: Checkpoints exist
        _check_checkpoints()

    return get_momentum_state()


def _check_active_campaign() -> None:
    """Check if there's an active campaign."""
    try:
        from guild.campaigns.loader import load_active_campaign

        campaign = load_active_campaign()
        if campaign is None:
            report_blocker(
                code=BLOCKER_NO_ACTIVE_CAMPAIGN,
                what="Start training",
                why="There is no active campaign selected.",
                how_to_fix="Go to the Campaign page, create a new campaign or select an existing one, and mark it as active.",
                suggested_action="open_campaign",
            )
        else:
            clear_blocker(BLOCKER_NO_ACTIVE_CAMPAIGN)
    except Exception as e:
        logger.warning(f"Failed to check active campaign: {e}")


def _check_training_queue() -> None:
    """Check if there are files in the training queue."""
    try:
        from core.paths import get_queue_dir, get_inbox_dir

        queue_dir = get_queue_dir()
        inbox_dir = get_inbox_dir()

        # Count files in queue priorities
        queue_files = 0
        for priority in ["high", "normal", "low"]:
            priority_dir = queue_dir / priority
            if priority_dir.exists():
                queue_files += len(list(priority_dir.glob("*.jsonl")))

        # Count files in inbox
        inbox_files = 0
        if inbox_dir.exists():
            inbox_files = len(list(inbox_dir.glob("*.jsonl")))
            inbox_files += len(list(inbox_dir.glob("**/*.jsonl")))

        total_files = queue_files + inbox_files

        if total_files == 0:
            report_blocker(
                code=BLOCKER_QUEUE_EMPTY,
                what="Continue training",
                why="No training data in the queue or inbox.",
                how_to_fix="Drop JSONL training files into the inbox folder, or enable auto-generation in Settings.",
                suggested_action="open_quests",
                severity="warning",  # Warning, not error - system can idle
                context={"queue_files": queue_files, "inbox_files": inbox_files},
            )
        else:
            clear_blocker(BLOCKER_QUEUE_EMPTY)
    except Exception as e:
        logger.warning(f"Failed to check training queue: {e}")


def _check_training_paused() -> None:
    """Check if training is paused."""
    try:
        from core.paths import get_control_dir

        pause_file = get_control_dir() / ".pause"
        if pause_file.exists():
            report_blocker(
                code=BLOCKER_TRAINING_PAUSED,
                what="Continue training",
                why="Training is currently paused.",
                how_to_fix="Click the Resume button in the Tavern, or delete the .pause file in the control directory.",
                suggested_action="open_tavern",
                severity="warning",
            )
        else:
            clear_blocker(BLOCKER_TRAINING_PAUSED)
    except Exception as e:
        logger.warning(f"Failed to check training paused state: {e}")


def _check_checkpoints() -> None:
    """Check if there are any checkpoints."""
    try:
        from guild.campaigns.loader import load_active_campaign

        campaign = load_active_campaign()
        if campaign is None:
            return  # Campaign check will handle this

        # Check campaign checkpoints directory
        checkpoints_dir = campaign.path / "checkpoints"
        if not checkpoints_dir.exists():
            checkpoint_count = 0
        else:
            checkpoint_count = len(list(checkpoints_dir.glob("checkpoint-*")))

        if checkpoint_count == 0:
            report_blocker(
                code=BLOCKER_NO_CHECKPOINTS,
                what="Resume from checkpoint or analyze progress",
                why="No checkpoints saved yet for this campaign.",
                how_to_fix="Start training to create checkpoints. The first checkpoint will be saved after the configured save_steps.",
                suggested_action="open_quests",
                severity="warning",
                context={"campaign_id": campaign.id},
            )
        else:
            clear_blocker(BLOCKER_NO_CHECKPOINTS)
    except Exception as e:
        logger.warning(f"Failed to check checkpoints: {e}")
