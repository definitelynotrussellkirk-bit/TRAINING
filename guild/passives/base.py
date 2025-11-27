"""
Base class for Passive modules.

To create a new passive:
1. Create a new file in guild/passives/ (e.g., my_passive.py)
2. Create a class that inherits from PassiveModule
3. Implement the required methods
4. The passive is auto-discovered at runtime

Example:
    # guild/passives/logic_gates.py
    from guild.passives.base import PassiveModule

    class LogicGatesPassive(PassiveModule):
        id = "logic_gates"
        name = "Logic Gates"
        category = "logic"
        description = "Boolean AND, OR, XOR operations"

        def generate_problems(self, count: int) -> List[Dict]:
            # Return list of {"prompt": ..., "expected": ...}
            ...

        def check_answer(self, expected: str, got: str) -> bool:
            # Return True if correct
            ...

That's it! The passive will be auto-discovered and used.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class PassiveConfig:
    """
    Configuration for a passive.

    IMPORTANT: version tracks the passive definition version.
    When you change problem generation or answer checking, bump the version!
    Results are only comparable within the same version.
    """
    id: str
    name: str
    category: str
    description: str
    version: str  # Semantic version: "1.0.0"
    lite_count: int = 5
    full_count: int = 30
    enabled: bool = True

    def config_hash(self) -> str:
        """Short hash identifying this config version."""
        import hashlib
        content = f"{self.id}:{self.version}:{self.lite_count}:{self.full_count}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


class PassiveModule(ABC):
    """
    Base class for passive evaluation modules.

    Each passive module defines:
    - What problems to generate
    - How to check if answers are correct

    Subclasses MUST define these class attributes:
    - id: str - Unique identifier (e.g., "logic_gates")
    - name: str - Display name (e.g., "Logic Gates")
    - category: str - Category from GUILD_VOCABULARY.md
    - description: str - Brief description

    Optional attributes:
    - lite_count: int = 5 - Problems for LITE mode
    - full_count: int = 30 - Problems for FULL mode
    - enabled: bool = True - Whether this passive is active
    """

    # Required class attributes (subclasses must override)
    id: str = None
    name: str = None
    category: str = None
    description: str = None
    version: str = "1.0.0"  # BUMP THIS when changing problem gen or answer checking!

    # Optional class attributes
    lite_count: int = 5
    full_count: int = 30
    enabled: bool = True

    def __init__(self):
        """Initialize the passive module."""
        if self.id is None:
            raise ValueError(f"{self.__class__.__name__} must define 'id' attribute")
        if self.name is None:
            raise ValueError(f"{self.__class__.__name__} must define 'name' attribute")
        if self.category is None:
            raise ValueError(f"{self.__class__.__name__} must define 'category' attribute")

    @abstractmethod
    def generate_problems(self, count: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate problems for evaluation.

        Args:
            count: Number of problems to generate
            seed: Optional random seed for reproducibility

        Returns:
            List of dicts with at least:
            - prompt: str - The question/task for the model
            - expected: str - The correct answer
            - (optional) metadata: dict - Additional problem info
        """
        pass

    @abstractmethod
    def check_answer(self, expected: str, got: str) -> bool:
        """
        Check if the model's answer is correct.

        Args:
            expected: The correct answer
            got: The model's response

        Returns:
            True if correct, False otherwise
        """
        pass

    def get_config(self) -> PassiveConfig:
        """Get this passive's configuration."""
        return PassiveConfig(
            id=self.id,
            name=self.name,
            category=self.category,
            description=self.description or "",
            version=self.version,
            lite_count=self.lite_count,
            full_count=self.full_count,
            enabled=self.enabled,
        )

    def __repr__(self):
        return f"<Passive:{self.id} ({self.category})>"
