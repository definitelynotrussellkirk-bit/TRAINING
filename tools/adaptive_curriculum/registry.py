#!/usr/bin/env python3
"""Generator registry for managing multiple data generators."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from urllib import request, error


@dataclass
class GeneratorToggles:
    """Difficulty toggles for a generator.

    Each toggle affects difficulty in some way. The difficulty_of_toggles
    function maps toggle values -> numeric difficulty level.
    """
    toggles: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.toggles.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GeneratorToggles:
        return cls(toggles=data)


@dataclass
class GeneratorConfig:
    """Configuration for a single generator."""
    generator_id: str
    api_url: str
    api_port: int = 8765

    # Difficulty mapping: level -> toggles
    difficulty_levels: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Optional custom difficulty calculator
    difficulty_calculator: Optional[Callable[[Dict[str, Any]], int]] = None

    # Generation parameters
    default_count: int = 1000
    default_priority: str = "normal"

    def to_dict(self) -> dict:
        """Serialize config (excludes callable)."""
        return {
            "generator_id": self.generator_id,
            "api_url": self.api_url,
            "api_port": self.api_port,
            "difficulty_levels": self.difficulty_levels,
            "default_count": self.default_count,
            "default_priority": self.default_priority
        }


class DataGenerator:
    """Wrapper for a single data generator (e.g., SYLLO API)."""

    def __init__(self, config: GeneratorConfig):
        """Initialize generator.

        Args:
            config: Generator configuration
        """
        self.config = config

    def generate(self, difficulty: int, count: int,
                 custom_toggles: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Generate training examples at specified difficulty.

        Args:
            difficulty: Difficulty level (0-N)
            count: Number of examples to generate
            custom_toggles: Optional override toggles (if None, uses config)

        Returns:
            List of generated examples with metadata
        """
        # Get toggles for this difficulty
        if custom_toggles:
            toggles = custom_toggles
        else:
            toggles = self.config.difficulty_levels.get(difficulty, {})

        # Build API payload
        payload = {
            "count": count,
            **toggles  # Merge in difficulty toggles
        }

        # Call generator API
        raw_examples = self._call_api(payload)

        # Add metadata to each example
        examples_with_meta = []
        for example in raw_examples:
            # Ensure metadata exists
            if "metadata" not in example:
                example["metadata"] = {}

            # Add generator info
            example["metadata"]["generator_id"] = self.config.generator_id
            example["metadata"]["difficulty_level"] = difficulty
            example["metadata"]["toggles"] = toggles

            examples_with_meta.append(example)

        return examples_with_meta

    def _call_api(self, payload: Dict[str, Any]) -> List[Dict]:
        """Call generator API.

        Args:
            payload: Request payload

        Returns:
            List of generated examples
        """
        url = f"{self.config.api_url}:{self.config.api_port}/generate"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        req = request.Request(url, data=data, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=300) as resp:
                body = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"Generator API call failed: {exc}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"API returned invalid JSON: {exc}") from exc

        # Handle different response formats
        if isinstance(parsed, dict):
            if "examples" in parsed:
                return parsed["examples"]
            elif "puzzles" in parsed:
                return parsed["puzzles"]
            else:
                return parsed.get("data", [])
        elif isinstance(parsed, list):
            return parsed
        else:
            raise RuntimeError(f"Unexpected API response format: {type(parsed)}")

    def difficulty_of_toggles(self, toggles: Dict[str, Any]) -> int:
        """Calculate difficulty level from toggle values.

        Uses custom calculator if provided, otherwise returns 0.
        """
        if self.config.difficulty_calculator:
            return self.config.difficulty_calculator(toggles)

        # Default: try to find matching level
        for level, level_toggles in self.config.difficulty_levels.items():
            if level_toggles == toggles:
                return level

        return 0  # Unknown -> easy


class GeneratorRegistry:
    """Registry of all available data generators."""

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize registry.

        Args:
            config_file: Optional path to JSON config file
        """
        self._generators: Dict[str, DataGenerator] = {}
        self._config_file = config_file

        if config_file and config_file.exists():
            self._load_from_file(config_file)

    def register(self, config: GeneratorConfig) -> None:
        """Register a new generator.

        Args:
            config: Generator configuration
        """
        generator = DataGenerator(config)
        self._generators[config.generator_id] = generator

    def get(self, generator_id: str) -> Optional[DataGenerator]:
        """Get generator by ID."""
        return self._generators.get(generator_id)

    def list_generators(self) -> List[str]:
        """List all registered generator IDs."""
        return list(self._generators.keys())

    def generate(self, generator_id: str, difficulty: int, count: int) -> List[Dict]:
        """Generate examples from a specific generator.

        Args:
            generator_id: Generator to use
            difficulty: Difficulty level
            count: Number of examples

        Returns:
            Generated examples with metadata

        Raises:
            ValueError: If generator not found
        """
        generator = self.get(generator_id)
        if not generator:
            raise ValueError(f"Generator '{generator_id}' not found")

        return generator.generate(difficulty, count)

    def _load_from_file(self, config_file: Path) -> None:
        """Load generator configs from JSON file."""
        with config_file.open("r") as f:
            data = json.load(f)

        for gen_config in data.get("generators", []):
            config = GeneratorConfig(
                generator_id=gen_config["generator_id"],
                api_url=gen_config["api_url"],
                api_port=gen_config.get("api_port", 8765),
                difficulty_levels=gen_config.get("difficulty_levels", {}),
                default_count=gen_config.get("default_count", 1000),
                default_priority=gen_config.get("default_priority", "normal")
            )
            self.register(config)

    def save_to_file(self, config_file: Path) -> None:
        """Save registry to JSON file."""
        data = {
            "generators": [
                gen.config.to_dict()
                for gen in self._generators.values()
            ]
        }

        config_file.parent.mkdir(parents=True, exist_ok=True)
        with config_file.open("w") as f:
            json.dump(data, f, indent=2)


# Example SYLLO generator config
def create_syllo_generator() -> GeneratorConfig:
    """Create config for SYLLO puzzle generator.

    Difficulty levels:
    0 (EASY): 4-5 words, no red herrings, simple definitions
    1 (MEDIUM): 5-6 words, 1-2 red herrings, moderate definitions
    2 (HARD): 6-8 words, 2-4 red herrings, tricky definitions
    """

    def syllo_difficulty_calc(toggles: Dict[str, Any]) -> int:
        """Calculate SYLLO difficulty from toggles."""
        score = 0

        # Word count affects difficulty
        word_count = toggles.get("word_count", 4)
        if word_count >= 6:
            score += 1
        if word_count >= 8:
            score += 1

        # Red herrings add difficulty
        red_herrings = toggles.get("red_herring_count", 0)
        if red_herrings >= 2:
            score += 1

        # Difficulty label
        difficulty = toggles.get("difficulty", "EASY")
        if difficulty == "MEDIUM":
            score = max(score, 1)
        elif difficulty == "HARD":
            score = max(score, 2)

        return min(score, 2)  # Cap at level 2

    return GeneratorConfig(
        generator_id="syllo",
        api_url="http://127.0.0.1",
        api_port=8765,
        difficulty_levels={
            0: {  # EASY
                "difficulty": "EASY",
                "min_words": 4,
                "max_words": 5,
                "red_herring_count": 0
            },
            1: {  # MEDIUM
                "difficulty": "MEDIUM",
                "min_words": 5,
                "max_words": 6,
                "red_herring_count": 1
            },
            2: {  # HARD
                "difficulty": "HARD",
                "min_words": 6,
                "max_words": 8,
                "red_herring_count": 3
            }
        },
        difficulty_calculator=syllo_difficulty_calc,
        default_count=1000,
        default_priority="normal"
    )
