"""
Scrolls - Utility scripts and tools for the training realm.

The Scrolls are magical scripts that perform various utility tasks:

    DataScrolls     - Data processing utilities
    AnalysisScrolls - Analysis and comparison tools
    ConfigScrolls   - Configuration editing

RPG Mapping:
    tools/data/       → DataScrolls (data preparation)
    tools/analysis/   → AnalysisScrolls (insights)
    tools/config/     → ConfigScrolls (configuration)

Quick Start:
    from scrolls import invoke_scroll

    # Generate training data
    invoke_scroll("generate_training_data", count=100)

    # Analyze state
    invoke_scroll("state_tracker", action="check")
"""

__version__ = "0.1.0"

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


# Scroll registry - maps scroll names to scripts
SCROLL_REGISTRY = {
    # Data scrolls
    "generate_training_data": "tools/generate_training_data.py",
    "validate_data": "tools/data/validate_data.py",

    # Analysis scrolls
    "state_tracker": "tools/analysis/state_tracker.py",
    "compare_models": "tools/analysis/compare_models.py",
    "calculate_data_value": "tools/analysis/calculate_data_value.py",

    # Config scrolls
    "edit_config": "tools/config/edit_config.py",
}


def list_scrolls() -> List[str]:
    """List all available scrolls."""
    return list(SCROLL_REGISTRY.keys())


def get_scroll_path(scroll_name: str, base_dir: str = None) -> Optional[Path]:
    """
    Get the path to a scroll script.

    Args:
        scroll_name: Name of the scroll
        base_dir: Base training directory (None = auto-detect)

    Returns:
        Path to script or None if not found
    """
    if scroll_name not in SCROLL_REGISTRY:
        return None

    if base_dir is None:
        try:
            from core.paths import get_base_dir
            base_dir = str(get_base_dir())
        except ImportError:
            base_dir = str(Path(__file__).parent.parent)

    return Path(base_dir) / SCROLL_REGISTRY[scroll_name]


def invoke_scroll(
    scroll_name: str,
    base_dir: str = None,
    args: List[str] = None,
    capture_output: bool = True,
) -> Dict[str, Any]:
    """
    Invoke a scroll (run a utility script).

    Args:
        scroll_name: Name of the scroll to invoke
        base_dir: Base training directory
        args: Additional command line arguments
        capture_output: Whether to capture output

    Returns:
        Dict with result, stdout, stderr, return_code
    """
    script_path = get_scroll_path(scroll_name, base_dir)

    if not script_path:
        return {
            "success": False,
            "error": f"Unknown scroll: {scroll_name}",
            "available": list_scrolls(),
        }

    if not script_path.exists():
        return {
            "success": False,
            "error": f"Scroll not found: {script_path}",
        }

    cmd = ["python3", str(script_path)]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            cwd=base_dir,
        )

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout if capture_output else None,
            "stderr": result.stderr if capture_output else None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


class ScrollInvoker:
    """
    Convenience class for invoking scrolls.

    Usage:
        scrolls = ScrollInvoker(base_dir)

        # Invoke with arguments
        result = scrolls.invoke("edit_config", ["batch_size", "32"])

        # Check available
        print(scrolls.available)
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            try:
                from core.paths import get_base_dir
                self.base_dir = str(get_base_dir())
            except ImportError:
                self.base_dir = str(Path(__file__).parent.parent)
        else:
            self.base_dir = base_dir

    @property
    def available(self) -> List[str]:
        """List available scrolls."""
        return list_scrolls()

    def invoke(self, scroll_name: str, args: List[str] = None) -> Dict[str, Any]:
        """Invoke a scroll."""
        return invoke_scroll(scroll_name, self.base_dir, args)

    def exists(self, scroll_name: str) -> bool:
        """Check if a scroll exists."""
        path = get_scroll_path(scroll_name, self.base_dir)
        return path is not None and path.exists()


def get_scroll_invoker(base_dir: str = None) -> ScrollInvoker:
    """Get a ScrollInvoker instance."""
    return ScrollInvoker(base_dir)


__all__ = [
    "SCROLL_REGISTRY",
    "list_scrolls",
    "get_scroll_path",
    "invoke_scroll",
    "ScrollInvoker",
    "get_scroll_invoker",
]


# =============================================================================
# RPG TERMINOLOGY GUIDE
# =============================================================================

"""
SCROLLS GLOSSARY
================

The Scrolls use RPG terminology for utility operations:

SCROLL TYPES
------------
DataScrolls     = Data processing scripts
AnalysisScrolls = Analysis and insight tools
ConfigScrolls   = Configuration editing

SCROLL ACTIONS
--------------
Invoke          = Run a scroll (execute script)
Read            = Get scroll contents
List            = Show available scrolls

SPECIFIC SCROLLS
----------------
generate_training_data  = Create new training examples
validate_data          = Check data format
state_tracker          = Track system state
compare_models         = Compare checkpoints
edit_config            = Modify configuration
"""
