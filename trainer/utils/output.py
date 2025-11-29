"""CLI output helpers for consistent formatting."""

from typing import Optional


def banner(title: str, width: int = 80) -> None:
    """Print a section banner.

    Args:
        title: Banner text (will be uppercased)
        width: Total banner width
    """
    print("\n" + "=" * width)
    print(title.upper())
    print("=" * width + "\n")


def step(n: int, msg: str) -> None:
    """Print a numbered step.

    Args:
        n: Step number
        msg: Step description
    """
    print(f"Step {n}: {msg}")


def section(title: str, width: int = 60) -> None:
    """Print a subsection header.

    Args:
        title: Section title
        width: Total section width
    """
    print(f"\n{'-' * width}")
    print(title)
    print(f"{'-' * width}")


def success(msg: str) -> None:
    """Print a success message."""
    print(f"[OK] {msg}")


def error(msg: str) -> None:
    """Print an error message."""
    print(f"[ERROR] {msg}")


def warning(msg: str) -> None:
    """Print a warning message."""
    print(f"[WARN] {msg}")


def info(msg: str) -> None:
    """Print an info message."""
    print(f"[INFO] {msg}")


def bullet(msg: str, indent: int = 2) -> None:
    """Print a bullet point.

    Args:
        msg: Bullet text
        indent: Number of leading spaces
    """
    print(f"{' ' * indent}- {msg}")


def kv(key: str, value: str, indent: int = 2) -> None:
    """Print a key-value pair.

    Args:
        key: Label
        value: Value
        indent: Number of leading spaces
    """
    print(f"{' ' * indent}{key}: {value}")


def progress(current: int, total: int, label: Optional[str] = None) -> None:
    """Print progress indicator.

    Args:
        current: Current count
        total: Total count
        label: Optional label (default: "Progress")
    """
    pct = 100 * current / total if total > 0 else 0
    label = label or "Progress"
    print(f"{label}: {current}/{total} ({pct:.0f}%)")
