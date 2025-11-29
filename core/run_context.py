"""
Run Context - Single source of truth for "what are we training?"

This module provides a unified RunContext that combines:
- Active campaign info (hero_id, campaign_id, campaign_path)
- Hero metadata (model name, architecture, context, vocab)
- Campaign config (model_path, current_model_dir, hyperparams)
- Runtime state (daemon status, auto_run, etc.)

All UI and system components should use RunContext instead of mixing
/config, /api/hero-model-info, and other sources.

Usage:
    from core.run_context import get_run_context

    ctx = get_run_context()
    print(ctx.model_path)        # From active campaign's config
    print(ctx.locked.base_model) # From hero's model definition
    print(ctx.context_hash())    # For job payload verification
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class LockedConfig:
    """Immutable model identity - cannot change mid-campaign."""
    base_model: str          # e.g., "Qwen/Qwen3-0.6B" or "models/Qwen3-0.6B"
    architecture: str        # e.g., "Qwen3ForCausalLM"
    context_length: int      # e.g., 4096
    vocab_size: int          # e.g., 151936


@dataclass
class DaemonStatus:
    """Training daemon runtime state."""
    running: bool = False
    pid: Optional[int] = None
    last_heartbeat: Optional[str] = None


@dataclass
class RunContext:
    """
    Single source of truth for the current training run.

    All fields are derived from the active campaign, not the root config.json.
    If there's no active campaign, we fall back to root config.json (legacy mode).
    """
    # Identity
    hero_id: Optional[str] = None
    hero_name: Optional[str] = None
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    campaign_path: Optional[str] = None

    # Model paths (from campaign config, NOT root config.json)
    model_path: Optional[str] = None
    current_model_dir: Optional[str] = None
    base_model: Optional[str] = None  # Convenience field = model_path

    # Locked model identity (from hero config)
    locked: Optional[LockedConfig] = None

    # Display info
    hero_icon: str = "🦸"
    hero_rpg_name: Optional[str] = None

    # Runtime state
    daemon: DaemonStatus = field(default_factory=DaemonStatus)
    auto_run: bool = False
    auto_generate: bool = False

    # Flags
    is_legacy_mode: bool = False  # True if using root config.json (no campaign)
    is_first_run: bool = False    # True if no campaigns exist

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result = asdict(self)
        # Handle nested dataclasses
        if self.locked:
            result["locked"] = asdict(self.locked)
        result["daemon"] = asdict(self.daemon)
        return result

    def context_hash(self) -> str:
        """
        Compute a hash of the stable identity fields.

        This hash can be included in job payloads for verification.
        Workers can check that their view of RunContext matches the
        hash in the job payload to detect context drift.

        Only includes immutable identity fields, not runtime state:
        - hero_id, campaign_id
        - model_path, base_model
        - locked config (architecture, context_length, vocab_size)
        """
        identity = {
            "hero_id": self.hero_id,
            "campaign_id": self.campaign_id,
            "model_path": self.model_path,
            "base_model": self.base_model,
        }
        if self.locked:
            identity["locked"] = {
                "base_model": self.locked.base_model,
                "architecture": self.locked.architecture,
                "context_length": self.locked.context_length,
                "vocab_size": self.locked.vocab_size,
            }
        # Stable JSON representation
        canonical = json.dumps(identity, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def identity_summary(self) -> Dict[str, Any]:
        """
        Return the identity fields used for hashing.

        Useful for debugging context mismatch errors.
        """
        return {
            "hero_id": self.hero_id,
            "campaign_id": self.campaign_id,
            "model_path": self.model_path,
            "context_hash": self.context_hash(),
        }


def _get_base_dir() -> Path:
    """Get the base directory."""
    try:
        from core.paths import get_base_dir
        return get_base_dir()
    except ImportError:
        return Path(__file__).parent.parent


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return {}


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return {}


def _get_daemon_status(base_dir: Path) -> DaemonStatus:
    """Check if training daemon is running."""
    import os

    pid_file = base_dir / ".pids" / "training_daemon.pid"
    status = DaemonStatus()

    if not pid_file.exists():
        return status

    try:
        pid = int(pid_file.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        status.running = True
        status.pid = pid

        # Try to get last heartbeat from status file
        status_file = base_dir / "status" / "training_status.json"
        if status_file.exists():
            data = _load_json(status_file)
            status.last_heartbeat = data.get("last_update")

    except (ValueError, ProcessLookupError, PermissionError):
        # PID file exists but process is gone
        pass

    return status


def get_run_context() -> RunContext:
    """
    Build RunContext from current system state.

    Resolution order:
    1. Check control/active_campaign.json for active campaign
    2. If active campaign exists:
       - Load campaign config from campaigns/{hero}/{campaign}/config.json
       - Load hero config from configs/heroes/{hero}.yaml
       - Merge into RunContext
    3. If no active campaign:
       - Fall back to root config.json (legacy mode)
    """
    base_dir = _get_base_dir()
    ctx = RunContext()

    # Check for active campaign
    active_campaign_file = base_dir / "control" / "active_campaign.json"
    active_campaign = _load_json(active_campaign_file) if active_campaign_file.exists() else {}

    hero_id = active_campaign.get("hero_id")
    campaign_id = active_campaign.get("campaign_id")
    campaign_path = active_campaign.get("campaign_path")

    # Check if any campaigns exist at all
    campaigns_dir = base_dir / "campaigns"
    has_any_campaigns = campaigns_dir.exists() and any(campaigns_dir.iterdir())
    ctx.is_first_run = not has_any_campaigns

    if hero_id and campaign_id and campaign_path:
        # Active campaign mode
        ctx.hero_id = hero_id
        ctx.campaign_id = campaign_id
        ctx.campaign_path = campaign_path

        # Load campaign config - try campaign-specific first, fall back to root
        campaign_config_file = base_dir / campaign_path / "config.json"
        root_config_file = base_dir / "config.json"

        if campaign_config_file.exists():
            # Campaign has its own config (e.g., titan-qwen3-4b)
            campaign_config = _load_json(campaign_config_file)
            using_campaign_config = True
        elif root_config_file.exists():
            # Fall back to root config (e.g., dio-qwen3-0.6b)
            campaign_config = _load_json(root_config_file)
            using_campaign_config = False
        else:
            campaign_config = {}
            using_campaign_config = False

        # Load hero config for metadata
        hero_config_file = base_dir / "configs" / "heroes" / f"{hero_id}.yaml"
        hero_config = _load_yaml(hero_config_file) if hero_config_file.exists() else {}

        # Extract hero display info
        ctx.hero_name = hero_config.get("name", hero_id)
        ctx.hero_rpg_name = hero_config.get("rpg_name")
        display = hero_config.get("display", {})
        ctx.hero_icon = display.get("emoji", "🦸")

        # Extract model paths - from campaign config (or root if no campaign config)
        ctx.model_path = campaign_config.get("model_path")
        ctx.current_model_dir = campaign_config.get("current_model_dir")
        ctx.base_model = campaign_config.get("base_model") or ctx.model_path

        # Extract locked model identity FROM HERO CONFIG (authoritative source)
        model_info = hero_config.get("model", {})
        if model_info:
            # Use hero's model.hf_name as the authoritative base_model
            hf_name = model_info.get("hf_name", ctx.model_path or "")
            ctx.locked = LockedConfig(
                base_model=hf_name,
                architecture=model_info.get("architecture", "Unknown"),
                context_length=model_info.get("context_length", 4096),
                vocab_size=model_info.get("vocab_size", 151936),
            )
            # If model_path wasn't in config, derive from hero's hf_name
            if not ctx.model_path:
                ctx.model_path = hf_name
                ctx.base_model = hf_name

        # If current_model_dir not specified, derive from campaign path
        if not ctx.current_model_dir and campaign_path:
            ctx.current_model_dir = f"{campaign_path}/checkpoints"

        # Load campaign.json for metadata if it exists
        campaign_meta_file = base_dir / campaign_path / "campaign.json"
        campaign_meta = _load_json(campaign_meta_file) if campaign_meta_file.exists() else {}

        # Extract campaign name from metadata or config
        ctx.campaign_name = campaign_meta.get("name") or \
                           campaign_config.get("model_display_name") or \
                           campaign_config.get("hero_name")

        # Extract auto flags from config
        auto_run = campaign_config.get("auto_run", {})
        ctx.auto_run = auto_run.get("enabled", False) if isinstance(auto_run, dict) else bool(auto_run)

        auto_gen = campaign_config.get("auto_generate", {})
        ctx.auto_generate = auto_gen.get("enabled", False) if isinstance(auto_gen, dict) else bool(auto_gen)

    else:
        # Legacy mode - use root config.json
        ctx.is_legacy_mode = True

        root_config_file = base_dir / "config.json"
        root_config = _load_json(root_config_file) if root_config_file.exists() else {}

        # Extract from root config
        ctx.model_path = root_config.get("model_path")
        ctx.current_model_dir = root_config.get("current_model_dir")
        ctx.base_model = root_config.get("base_model") or ctx.model_path

        # Extract locked block
        locked = root_config.get("locked", {})
        if locked:
            ctx.locked = LockedConfig(
                base_model=locked.get("base_model", ctx.model_path or ""),
                architecture=locked.get("model_architecture", "Unknown"),
                context_length=locked.get("max_context_length", 4096),
                vocab_size=locked.get("vocab_size", 151936),
            )

        # Extract auto flags
        auto_run = root_config.get("auto_run", {})
        ctx.auto_run = auto_run.get("enabled", False) if isinstance(auto_run, dict) else bool(auto_run)

        auto_gen = root_config.get("auto_generate", {})
        ctx.auto_generate = auto_gen.get("enabled", False) if isinstance(auto_gen, dict) else bool(auto_gen)

    # Get daemon status
    ctx.daemon = _get_daemon_status(base_dir)

    return ctx


def validate_run_context(ctx: RunContext) -> list[str]:
    """
    Validate that RunContext is internally consistent.

    Returns list of error messages (empty if valid).
    """
    errors = []

    # Check model paths match locked config
    if ctx.locked and ctx.model_path:
        # The locked.base_model should correspond to model_path
        # Allow either local path or HF name
        locked_name = ctx.locked.base_model
        model_path = ctx.model_path

        # Extract model name from path for comparison
        locked_short = locked_name.split("/")[-1] if "/" in locked_name else locked_name
        path_short = model_path.split("/")[-1] if "/" in model_path else model_path

        # Check if they refer to the same model (fuzzy match)
        if locked_short.lower().replace("-", "").replace("_", "") not in \
           path_short.lower().replace("-", "").replace("_", "") and \
           path_short.lower().replace("-", "").replace("_", "") not in \
           locked_short.lower().replace("-", "").replace("_", ""):
            errors.append(
                f"Model mismatch: locked.base_model='{locked_name}' but model_path='{model_path}'"
            )

    # Check current_model_dir exists if specified
    if ctx.current_model_dir:
        base_dir = _get_base_dir()
        current_model_path = base_dir / ctx.current_model_dir
        if not current_model_path.exists():
            # This is a warning, not an error - the dir might be created during training
            pass

    return errors


if __name__ == "__main__":
    import pprint
    ctx = get_run_context()
    print("Run Context:")
    pprint.pprint(ctx.to_dict())

    errors = validate_run_context(ctx)
    if errors:
        print("\nValidation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nContext is valid.")
