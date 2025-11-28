"""
Realm - The unified entry point to the RPG Training System.

The Realm is the entire training kingdom, providing access to all
locations and systems through a single import:

    from realm import Realm
    r = Realm()

    # Access any location
    r.arena.board.take_next_quest()
    r.watchtower.pool.gaze()
    r.vault.treasury.get_status()
    r.sentinels.inspector.full_patrol()

Or use the global convenience functions:
    from realm import gaze, patrol, invoke_scroll

Locations:
    Guild       - Skills, quests, dispatch, progression
    Arena       - Training execution (battles)
    Watchtower  - Monitoring and observation
    Vault       - Storage and versioning
    Sentinels   - System protection
    Armory      - Equipment and configuration
    Scrolls     - Utility scripts
    Tavern      - Game UI (hero view, battle status)

This is the recommended entry point for the RPG-themed training system.
"""

from pathlib import Path
from typing import Any, Dict, Optional

# Default base directory
DEFAULT_BASE_DIR = "/path/to/training"


class Realm:
    """
    The Training Realm - unified access to all RPG systems.

    Usage:
        realm = Realm()

        # Quick status
        print(realm.status())

        # Access locations
        realm.arena.board.take_next_quest()
        realm.watchtower.pool.gaze()
        realm.vault.treasury.get_status()
    """

    def __init__(self, base_dir: str = DEFAULT_BASE_DIR):
        """
        Initialize the Realm.

        Args:
            base_dir: Base training directory
        """
        self.base_dir = Path(base_dir)
        self._arena = None
        self._watchtower = None
        self._vault = None
        self._sentinels = None
        self._armory = None
        self._scrolls = None
        self._guild = None
        self._tavern = None

    # =========================================================================
    # LAZY-LOADED LOCATIONS
    # =========================================================================

    @property
    def arena(self) -> "ArenaAccess":
        """Access the Arena (training execution)."""
        if self._arena is None:
            self._arena = ArenaAccess(self.base_dir)
        return self._arena

    @property
    def watchtower(self) -> "WatchtowerAccess":
        """Access the Watchtower (monitoring)."""
        if self._watchtower is None:
            self._watchtower = WatchtowerAccess(self.base_dir)
        return self._watchtower

    @property
    def vault(self) -> "VaultAccess":
        """Access the Vault (storage)."""
        if self._vault is None:
            self._vault = VaultAccess(self.base_dir)
        return self._vault

    @property
    def sentinels(self) -> "SentinelsAccess":
        """Access the Sentinels (protection)."""
        if self._sentinels is None:
            self._sentinels = SentinelsAccess(self.base_dir)
        return self._sentinels

    @property
    def armory(self):
        """Access the Armory (equipment)."""
        if self._armory is None:
            import armory
            self._armory = armory
        return self._armory

    @property
    def scrolls(self):
        """Access the Scrolls (utilities)."""
        if self._scrolls is None:
            import scrolls
            self._scrolls = scrolls
        return self._scrolls

    @property
    def guild(self):
        """Access the Guild (skills, quests, dispatch)."""
        if self._guild is None:
            import guild
            self._guild = guild
        return self._guild

    @property
    def tavern(self):
        """Access the Tavern (game UI)."""
        if self._tavern is None:
            import tavern
            self._tavern = tavern
        return self._tavern

    # =========================================================================
    # QUICK STATUS
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """
        Get overall realm status.

        Returns:
            Dict with status from all locations
        """
        status = {
            "realm": "active",
            "base_dir": str(self.base_dir),
        }

        # Arena status
        try:
            vision = self.watchtower.pool.gaze()
            status["arena"] = {
                "battle_state": vision.battle_state,
                "progress": f"{vision.current_round}/{vision.total_rounds}",
            }
        except Exception as e:
            status["arena"] = {"error": str(e)}

        # Vault status
        try:
            treasury = self.vault.treasury.get_status()
            status["vault"] = {
                "free_gb": round(treasury.free_disk_gb, 1),
                "health": treasury.health,
            }
        except Exception as e:
            status["vault"] = {"error": str(e)}

        # Sentinels status
        try:
            patrol = self.sentinels.inspector.quick_check()
            status["sentinels"] = {
                "healthy": patrol["healthy"],
                "status": patrol["status"],
            }
        except Exception as e:
            status["sentinels"] = {"error": str(e)}

        return status

    def quick_check(self) -> bool:
        """Quick health check - True if all systems healthy."""
        try:
            patrol = self.sentinels.inspector.full_patrol()
            return patrol.is_all_clear
        except Exception:
            return False


# =============================================================================
# LOCATION ACCESS CLASSES
# =============================================================================

class ArenaAccess:
    """Lazy-loaded access to Arena components."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._board = None
        self._control = None
        self._log = None

    @property
    def board(self):
        """Quest Board (training queue)."""
        if self._board is None:
            from arena import QuestBoard
            self._board = QuestBoard(str(self.base_dir))
        return self._board

    @property
    def control(self):
        """Battle Control (training controller)."""
        if self._control is None:
            from arena import BattleControl
            self._control = BattleControl(str(self.base_dir))
        return self._control

    @property
    def log(self):
        """Battle Log (training status)."""
        if self._log is None:
            from arena import BattleLog
            self._log = BattleLog(self.base_dir)
        return self._log


class WatchtowerAccess:
    """Lazy-loaded access to Watchtower components."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._pool = None
        self._oracle = None

    @property
    def pool(self):
        """Scrying Pool (real-time observation)."""
        if self._pool is None:
            from watchtower import ScryingPool
            self._pool = ScryingPool(self.base_dir)
        return self._pool

    @property
    def oracle(self):
        """Oracle Client (inference)."""
        if self._oracle is None:
            from watchtower import OracleClient
            self._oracle = OracleClient()
        return self._oracle


class VaultAccess:
    """Lazy-loaded access to Vault components."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._archivist = None
        self._chronicle = None
        self._treasury = None

    @property
    def archivist(self):
        """Archivist (backup management)."""
        if self._archivist is None:
            from vault import Archivist
            self._archivist = Archivist(str(self.base_dir))
        return self._archivist

    @property
    def chronicle(self):
        """Chronicle (version history)."""
        if self._chronicle is None:
            from vault import Chronicle
            self._chronicle = Chronicle(str(self.base_dir))
        return self._chronicle

    @property
    def treasury(self):
        """Treasury (resource management)."""
        if self._treasury is None:
            from vault import Treasury
            self._treasury = Treasury(self.base_dir)
        return self._treasury


class SentinelsAccess:
    """Lazy-loaded access to Sentinels components."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._guardian = None
        self._inspector = None

    @property
    def guardian(self):
        """Guardian (daemon watchdog)."""
        if self._guardian is None:
            from sentinels import Guardian
            self._guardian = Guardian(self.base_dir)
        return self._guardian

    @property
    def inspector(self):
        """Health Inspector."""
        if self._inspector is None:
            from sentinels import HealthInspector
            self._inspector = HealthInspector(self.base_dir)
        return self._inspector


# =============================================================================
# GLOBAL CONVENIENCE FUNCTIONS
# =============================================================================

_realm: Optional[Realm] = None


def get_realm(base_dir: str = DEFAULT_BASE_DIR) -> Realm:
    """Get or create the global Realm instance."""
    global _realm
    if _realm is None:
        _realm = Realm(base_dir)
    return _realm


def gaze() -> Dict[str, Any]:
    """
    Quick gaze into the Scrying Pool.

    Returns current training state.
    """
    return get_realm().watchtower.pool.gaze().to_dict()


def patrol() -> Dict[str, Any]:
    """
    Run a sentinel patrol.

    Returns health check results.
    """
    return get_realm().sentinels.inspector.full_patrol().to_dict()


def get_champion() -> Optional[Dict[str, Any]]:
    """
    Get current champion (best checkpoint by train_loss).

    Returns champion info or None.
    """
    try:
        from core.checkpoint_ledger import get_ledger
        ledger = get_ledger()
        best = ledger.get_best(metric="train_loss")
        if best:
            return {
                "step": best.step,
                "train_loss": best.train_loss,
                "canonical_name": best.canonical_name,
                "saved_at": best.saved_at,
            }
    except Exception:
        pass
    return None


def invoke_scroll(scroll_name: str, args: list = None) -> Dict[str, Any]:
    """
    Invoke a utility scroll.

    Args:
        scroll_name: Name of scroll to invoke
        args: Command line arguments

    Returns:
        Result dict
    """
    from scrolls import invoke_scroll as _invoke
    return _invoke(scroll_name, str(get_realm().base_dir), args)


def realm_status() -> Dict[str, Any]:
    """Get full realm status."""
    return get_realm().status()


def open_tavern(port: int = 8888):
    """
    Open the Tavern (start game UI server).

    Args:
        port: Port to serve on (default 8888)
    """
    from tavern import run_tavern
    run_tavern(port=port)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "Realm",
    "get_realm",
    # Access classes
    "ArenaAccess",
    "WatchtowerAccess",
    "VaultAccess",
    "SentinelsAccess",
    # Convenience functions
    "gaze",
    "patrol",
    "get_champion",
    "invoke_scroll",
    "realm_status",
    "open_tavern",
]
