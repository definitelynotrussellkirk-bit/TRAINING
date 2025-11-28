"""
GeneratorAdapter - wraps skill API (singleSKILL) for training data generation.

This adapter connects existing SkillClient API calls to the Skill interface,
allowing skills to generate training data via their external APIs.
"""

from typing import Optional
import logging

from guild.skills.contract import SkillClient, Batch

logger = logging.getLogger(__name__)


class GeneratorAdapter:
    """
    Adapts skill API (singleSKILL) to training generation interface.

    Uses SkillClient internally to call /generate endpoint on the
    skill's API server.

    Example:
        adapter = GeneratorAdapter("binary", "http://localhost:8090")
        batch = adapter.generate_training_batch(level=5, count=100)
    """

    def __init__(
        self,
        skill_id: str,
        api_url: str,
        timeout: int = 30
    ):
        """
        Initialize generator adapter.

        Args:
            skill_id: Skill identifier (e.g., "binary", "sy")
            api_url: Base URL for skill API (e.g., "http://localhost:8090")
            timeout: Request timeout in seconds
        """
        self.skill_id = skill_id
        self.api_url = api_url
        self.client = SkillClient(skill_id, api_url, timeout=timeout)

    def generate_training_batch(
        self,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> list[dict]:
        """
        Generate training samples via API.

        Calls the skill API's /generate endpoint and converts the
        response to inbox format.

        Args:
            level: Skill level (1 to max_level)
            count: Number of samples to generate
            seed: Optional random seed for reproducibility

        Returns:
            List of training examples in inbox format:
            [{"messages": [...], "metadata": {...}}, ...]

        Raises:
            requests.HTTPError: If API request fails
            requests.Timeout: If request times out
        """
        logger.debug(f"Generating {count} samples for {self.skill_id} level {level}")

        batch = self.client.sample(level=level, count=count, seed=seed)

        # Convert to inbox format
        training_samples = []
        for sample in batch.samples:
            training_samples.append({
                "messages": sample.messages,
                "metadata": {
                    "skill_id": self.skill_id,
                    "level": level,
                    **sample.metadata,
                },
            })

        logger.debug(f"Generated {len(training_samples)} samples")
        return training_samples

    def health(self) -> bool:
        """
        Check if API is available.

        Returns:
            True if API responds with healthy status
        """
        return self.client.health()

    def info(self):
        """Get skill info from API."""
        return self.client.info()

    def levels(self):
        """Get level configs from API."""
        return self.client.levels()

    def clear_cache(self):
        """Clear cached API responses."""
        self.client.clear_cache()

    def __repr__(self) -> str:
        return f"GeneratorAdapter({self.skill_id!r}, {self.api_url!r})"
