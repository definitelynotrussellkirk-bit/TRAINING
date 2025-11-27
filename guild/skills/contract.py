"""
Skill API Contract
==================

Defines what a skill API MUST provide. Skills are external APIs (in singleSKILL).
This contract standardizes the interface so any compliant API works with the Guild.

Usage:
    from guild.skills.contract import SkillClient

    client = SkillClient("binary", "http://localhost:8090")

    # Get skill metadata
    info = client.info()

    # Get level configs
    levels = client.levels()

    # Generate samples
    batch = client.sample(level=5, count=100)

    # Get eval requirements
    req = client.eval_requirements(level=5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence
import requests
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class SkillDefinition:
    """
    Static metadata about a skill.
    Populated from API's /info endpoint.
    """
    id: str                     # "binary", "syllo"
    name: str                   # "Binary Arithmetic"
    description: str
    category: str               # "math", "reasoning"
    tags: Sequence[str]         # ["arithmetic", "binary", "notation"]
    min_level: int              # 1
    max_level: int              # 30 or 50
    version: str = "1.0.0"      # For lineage tracking


@dataclass(frozen=True)
class LevelConfig:
    """
    Configuration for a single level.
    Level = RANGE of difficulty (not exact).
    Populated from API's /levels endpoint.
    """
    level: int
    name: str                   # "2-bit (tiny)" or "Beginner"
    description: str

    # Eval requirements
    eval_samples: int           # How many problems to test (e.g., 5)
    pass_threshold: float       # Accuracy to advance (e.g., 0.80)

    # Level-specific parameters (varies by skill)
    params: dict = field(default_factory=dict)
    # e.g., {"bits": 2, "symbol_language": False} for binary
    # e.g., {"words": 4, "distractors": 0, "hint_channel": "plain"} for syllo


@dataclass(frozen=True)
class EvalRequirements:
    """What it takes to pass a level and advance."""
    level: int
    samples_needed: int         # How many to test
    pass_threshold: float       # 0.80 = 80%
    consecutive_passes: int = 3 # How many evals in a row (optional)
    passing_score: str = ""     # Human readable: "4/5"


@dataclass
class Sample:
    """
    Single training/eval sample.
    Format matches TRAINING inbox contract.
    """
    messages: list[dict]        # [{"role": "user", ...}, {"role": "assistant", ...}]
    metadata: dict              # skill_id, level, tags, etc.
    golden_answers: list[str] = field(default_factory=list)  # For eval grading


@dataclass
class Batch:
    """Collection of samples from a single request."""
    skill_id: str
    level: int
    samples: Sequence[Sample]
    level_info: dict = field(default_factory=dict)  # From API response


# =============================================================================
# API CONTRACT - What endpoints must exist
# =============================================================================

class SkillAPIContract(Protocol):
    """
    Protocol defining what a skill API MUST provide.

    Required Endpoints:
        GET  /health  -> {"status": "ok"}
        GET  /info    -> SkillDefinition fields
        GET  /levels  -> List of LevelConfig
        POST /generate -> Batch of samples

    Optional Endpoints:
        GET  /eval/{level} -> Pre-generated eval set
    """

    def health(self) -> bool:
        """Returns True if API is up."""
        ...

    def info(self) -> SkillDefinition:
        """Get skill metadata."""
        ...

    def levels(self) -> list[LevelConfig]:
        """Get all level configurations."""
        ...

    def sample(self, level: int, count: int) -> Batch:
        """Generate samples at given level."""
        ...

    def eval_requirements(self, level: int) -> EvalRequirements:
        """Get eval requirements for a level."""
        ...


# =============================================================================
# SKILL CLIENT - Calls any compliant API
# =============================================================================

class SkillClient:
    """
    Client for any skill API that follows the contract.

    Usage:
        client = SkillClient("binary", "http://localhost:8090")
        batch = client.sample(level=5, count=100)
    """

    def __init__(self, skill_id: str, api_url: str, timeout: int = 30):
        self.skill_id = skill_id
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self._info_cache: SkillDefinition | None = None
        self._levels_cache: list[LevelConfig] | None = None

    def health(self) -> bool:
        """Check if API is up."""
        try:
            r = requests.get(f"{self.api_url}/health", timeout=5)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def info(self) -> SkillDefinition:
        """Get skill metadata from /info."""
        if self._info_cache:
            return self._info_cache

        r = requests.get(f"{self.api_url}/info", timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        # Map API response to SkillDefinition
        self._info_cache = SkillDefinition(
            id=data.get("skill", self.skill_id),
            name=data.get("name", data.get("skill", self.skill_id).title()),
            description=data.get("description", ""),
            category=data.get("category", "unknown"),
            tags=data.get("tags", data.get("features", [])),
            min_level=data.get("min_level", 1),
            max_level=data.get("max_level", data.get("levels", 30)),
            version=data.get("version", "1.0.0"),
        )
        return self._info_cache

    def levels(self) -> list[LevelConfig]:
        """Get all level configs from /levels."""
        if self._levels_cache:
            return self._levels_cache

        r = requests.get(f"{self.api_url}/levels", timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        levels_data = data.get("levels", data) if isinstance(data, dict) else data

        configs = []
        for lvl in levels_data:
            # Extract known fields, put rest in params
            known = {"level", "name", "description", "eval_samples", "pass_threshold"}
            params = {k: v for k, v in lvl.items() if k not in known}

            configs.append(LevelConfig(
                level=lvl.get("level", 1),
                name=lvl.get("name", lvl.get("description", f"Level {lvl.get('level')}")),
                description=lvl.get("description", ""),
                eval_samples=lvl.get("eval_samples", lvl.get("num_problems", 5)),
                pass_threshold=lvl.get("pass_threshold", 0.80),
                params=params,
            ))

        self._levels_cache = configs
        return self._levels_cache

    def sample(self, level: int, count: int, seed: int | None = None) -> Batch:
        """Generate samples at given level via /generate."""
        payload = {"level": level, "count": count}
        if seed is not None:
            payload["seed"] = seed

        r = requests.post(
            f"{self.api_url}/generate",
            json=payload,
            timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()

        # Convert API response to Sample objects
        samples = []
        raw_samples = data.get("samples", data.get("puzzles", []))

        for s in raw_samples:
            # Handle different response formats
            if "messages" in s:
                messages = s["messages"]
            elif "user_prompt" in s and "assistant_response" in s:
                messages = [
                    {"role": "user", "content": s["user_prompt"]},
                    {"role": "assistant", "content": s["assistant_response"]}
                ]
            elif "prompt" in s:
                messages = [
                    {"role": "user", "content": s["prompt"]},
                    {"role": "assistant", "content": s.get("response", s.get("solution", ""))}
                ]
            else:
                logger.warning(f"Unknown sample format: {list(s.keys())}")
                continue

            samples.append(Sample(
                messages=messages,
                metadata={
                    "skill_id": self.skill_id,
                    "level": level,
                    **{k: v for k, v in s.items()
                       if k not in ("messages", "user_prompt", "assistant_response", "prompt", "response")}
                },
                golden_answers=s.get("golden_answers", []),
            ))

        return Batch(
            skill_id=self.skill_id,
            level=level,
            samples=samples,
            level_info=data.get("level_info", {}),
        )

    def eval_requirements(self, level: int) -> EvalRequirements:
        """Get eval requirements for a level."""
        levels = self.levels()
        for lvl in levels:
            if lvl.level == level:
                return EvalRequirements(
                    level=level,
                    samples_needed=lvl.eval_samples,
                    pass_threshold=lvl.pass_threshold,
                    consecutive_passes=3,  # Default
                    passing_score=f"{int(lvl.eval_samples * lvl.pass_threshold)}/{lvl.eval_samples}",
                )

        # Default if level not found
        return EvalRequirements(
            level=level,
            samples_needed=5,
            pass_threshold=0.80,
            consecutive_passes=3,
            passing_score="4/5",
        )

    def clear_cache(self):
        """Clear cached info/levels."""
        self._info_cache = None
        self._levels_cache = None


# =============================================================================
# REQUIRED API METADATA - What skills should provide
# =============================================================================

"""
REQUIRED METADATA FOR SKILL APIS
================================

Every skill API should provide these fields in /info:

Required:
    skill: str              # Unique ID: "binary", "syllo"
    description: str        # What the skill teaches
    min_level: int          # First level (usually 1)
    max_level: int          # Last level (e.g., 30, 50)

Recommended:
    name: str               # Display name: "Binary Arithmetic"
    version: str            # For lineage: "1.0.0"
    category: str           # "math", "reasoning", "language"
    tags: list[str]         # ["arithmetic", "binary"]
    features: list[str]     # What it teaches
    operations: dict        # Available operations

Every level in /levels should provide:

Required:
    level: int              # Level number
    description: str        # What this level covers

Recommended:
    name: str               # Human-friendly name
    eval_samples: int       # Problems to test (default: 5)
    pass_threshold: float   # Accuracy to pass (default: 0.80)

Skill-specific (in params):
    Binary: bits, symbol_language, symbol_probability, difficulty_weights
    Syllo: words, distractors, hint_channel, zipf_threshold
"""
