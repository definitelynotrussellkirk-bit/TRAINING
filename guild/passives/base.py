"""
Base class for Passive modules.

===========================================================================
                    PASSIVE MODULE CONTRACT (v2.0)
===========================================================================

Every passive MUST adhere to this contract to work with the evaluation system.

REQUIRED CLASS ATTRIBUTES:
    id: str           - Unique identifier (snake_case, e.g., "logic_gates")
    name: str         - Display name (e.g., "Logic Gates")
    category: str     - Category from GUILD_VOCABULARY.md
    description: str  - Brief description of what this tests
    version: str      - Semantic version "X.Y.Z" (BUMP when changing logic!)

OPTIONAL CLASS ATTRIBUTES:
    tier: str         - "core" (run always) or "extended" (on-demand)
    priority: int     - Lower runs first within tier (0-100, default 50)
    lite_count: int   - Problems for LITE mode (default 5)
    full_count: int   - Problems for FULL mode (default 30)
    enabled: bool     - Whether this passive is active (default True)

REQUIRED METHODS:
    generate_problems(count, seed=None, level=1) -> List[Dict]
        MUST accept all three parameters (even if level is unused!)
        MUST return list of dicts with at least:
            - prompt: str       - The question for the model
            - expected: str     - The correct answer
            - primitive_id: str - Identifier for per-primitive tracking

    check_answer(expected, got) -> bool
        MUST return True if answer is correct, False otherwise

EXAMPLE:
    class MyPassive(PassiveModule):
        id = "my_passive"
        name = "My Passive"
        category = "logic"
        description = "Tests XYZ reasoning"
        version = "1.0.0"

        def generate_problems(self, count: int, seed: Optional[int] = None, level: int = 1):
            # level can be used for difficulty scaling (1=easy, 30=hard)
            # Must include primitive_id in each problem!
            return [{"prompt": "...", "expected": "...", "primitive_id": "my_prim"}]

        def check_answer(self, expected: str, got: str) -> bool:
            return expected.lower() in got.lower()

===========================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


class PassiveTier:
    """Tier constants for passive evaluation scheduling."""
    CORE = "core"          # Run on every checkpoint (sentinel passives)
    EXTENDED = "extended"  # Run on-demand (full library)


@dataclass
class PassiveConfig:
    """
    Configuration for a passive.

    IMPORTANT: version tracks the passive definition version.
    When you change problem generation or answer checking, bump the version!
    Results are only comparable within the same version.

    Tier System:
    - CORE: Small set of passives that run on every checkpoint save.
            These are "sentinel" passives that catch catastrophic forgetting.
    - EXTENDED: Large library of passives run on-demand for comprehensive
                capability assessment.
    """
    id: str
    name: str
    category: str
    description: str
    version: str  # Semantic version: "1.0.0"
    tier: str = PassiveTier.EXTENDED  # "core" or "extended"
    priority: int = 50  # Lower = runs first (0-100)
    lite_count: int = 5
    full_count: int = 30
    enabled: bool = True

    def config_hash(self) -> str:
        """Short hash identifying this config version."""
        import hashlib
        content = f"{self.id}:{self.version}:{self.lite_count}:{self.full_count}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    @property
    def is_core(self) -> bool:
        """Check if this is a core (sentinel) passive."""
        return self.tier == PassiveTier.CORE


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
    - tier: str = "extended" - "core" (run always) or "extended" (on-demand)
    - priority: int = 50 - Lower runs first within tier (0-100)
    - lite_count: int = 5 - Problems for LITE mode
    - full_count: int = 30 - Problems for FULL mode
    - enabled: bool = True - Whether this passive is active

    Tier System:
    - CORE passives run on every checkpoint save (sentinel passives)
    - EXTENDED passives run on-demand for comprehensive assessment
    - Use tier = PassiveTier.CORE for critical capability checks
    """

    # Required class attributes (subclasses must override)
    id: str = None
    name: str = None
    category: str = None
    description: str = None
    version: str = "1.0.0"  # BUMP THIS when changing problem gen or answer checking!

    # Tier system attributes
    tier: str = PassiveTier.EXTENDED  # "core" or "extended"
    priority: int = 50  # Lower = runs first (0-100)

    # Optional class attributes
    lite_count: int = 5
    full_count: int = 30
    enabled: bool = True

    def __init__(self):
        """Initialize the passive module."""
        cls_name = self.__class__.__name__
        if self.id is None:
            raise ValueError(f"{cls_name} must define 'id' attribute")
        if self.name is None:
            raise ValueError(f"{cls_name} must define 'name' attribute")
        if self.category is None:
            raise ValueError(f"{cls_name} must define 'category' attribute")
        if self.description is None:
            raise ValueError(f"{cls_name} must define 'description' attribute")
        if self.version is None or self.version == "1.0.0" and cls_name != "PassiveModule":
            # version="1.0.0" is the base class default - subclasses should explicitly set it
            pass  # Allow default for now to not break existing passives during transition

    @abstractmethod
    def generate_problems(self, count: int, seed: Optional[int] = None, level: int = 1) -> List[Dict[str, Any]]:
        """
        Generate problems for evaluation.

        Args:
            count: Number of problems to generate
            seed: Optional random seed for reproducibility
            level: Skill level (1-30) for difficulty scaling. Passives that
                   support level-based difficulty should use this to adjust
                   problem complexity. Default is 1 (easiest).

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
            tier=self.tier,
            priority=self.priority,
            lite_count=self.lite_count,
            full_count=self.full_count,
            enabled=self.enabled,
        )

    @property
    def is_core(self) -> bool:
        """Check if this is a core (sentinel) passive."""
        return self.tier == PassiveTier.CORE

    def __repr__(self):
        return f"<Passive:{self.id} ({self.category})>"
