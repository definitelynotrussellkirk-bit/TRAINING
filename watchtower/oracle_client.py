"""
Oracle Client - Communicate with the Oracle for prophecies (inference).

The Oracle is a mystical entity residing in the Crystal Tower (3090 server)
that can see the future - given a question, it speaks prophecies (generates
text responses).

RPG Flavor:
    The Oracle sits in the Crystal Tower, connected to the training realm
    via magical crystal links. Seekers approach with questions, and the
    Oracle speaks prophecies. The Oracle can be reborn (model reload) to
    gain new wisdom from the latest champion.

API Mapping:
    chat completion → seek_prophecy
    model reload → rebirth_oracle
    model info → oracle_wisdom
    health check → oracle_pulse

This module wraps monitoring/prediction_client.py with RPG-themed naming.
"""

import time
from typing import Any, Dict, List, Optional

from watchtower.types import OracleResponse

# Import the underlying client
from monitoring.prediction_client import PredictionClient as _PredictionClient


class OracleClient(_PredictionClient):
    """
    Client for communicating with the Oracle (inference server).

    RPG wrapper around PredictionClient with themed method names.

    Usage:
        oracle = OracleClient()

        # Seek a prophecy (inference)
        response = oracle.seek_prophecy(
            question="What is 2 + 2?",
            context="You are a helpful assistant."
        )
        print(response.prophecy)

        # Check Oracle status
        if oracle.oracle_pulse():
            wisdom = oracle.oracle_wisdom()
            print(f"Oracle: {wisdom['model_name']}")

        # Rebirth with new checkpoint
        oracle.rebirth_oracle(champion_path="/path/to/checkpoint")
    """

    def __init__(
        self,
        crystal_tower: Optional[str] = None,
        timeout: int = 120,
        max_attempts: int = 3,
        attempt_delay: float = 2.0,
        seeker_key: Optional[str] = None,
        high_priest_key: Optional[str] = None,
    ):
        """
        Initialize Oracle client.

        Args:
            crystal_tower: URL of the Crystal Tower (inference server) - auto-detects from host registry if None
            timeout: Request timeout in seconds
            max_attempts: Maximum prophecy attempts
            attempt_delay: Delay between attempts
            seeker_key: Read-level API key (for seeking prophecies)
            high_priest_key: Admin-level API key (for rebirth)
        """
        # Auto-detect inference server URL from host registry
        if crystal_tower is None:
            try:
                from core.hosts import get_service_url
                crystal_tower = get_service_url("inference")
            except Exception:
                # Fallback if host registry unavailable
                crystal_tower = "http://inference.local:8765"

        super().__init__(
            base_url=crystal_tower,
            timeout=timeout,
            max_retries=max_attempts,
            retry_backoff=attempt_delay,
            api_key=seeker_key,
            admin_key=high_priest_key,
        )

    # =========================================================================
    # PROPHECY METHODS (Inference)
    # =========================================================================

    def seek_prophecy(
        self,
        question: str,
        context: Optional[str] = None,
        conversation: Optional[List[Dict[str, str]]] = None,
        max_words: int = 512,
        creativity: float = 0.7,
    ) -> OracleResponse:
        """
        Seek a prophecy from the Oracle.

        Args:
            question: The question to ask
            context: System context for the Oracle
            conversation: Previous conversation history
            max_words: Maximum words in prophecy (maps to max_tokens)
            creativity: How creative the prophecy should be (temperature)

        Returns:
            OracleResponse with the prophecy
        """
        # Build messages
        messages = []

        if context:
            messages.append({"role": "system", "content": context})

        if conversation:
            messages.extend(conversation)

        messages.append({"role": "user", "content": question})

        # Estimate max tokens from words (~1.3 tokens per word)
        max_tokens = int(max_words * 1.3)

        start_time = time.time()

        try:
            # Call underlying chat method
            result = self.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=creativity,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            prophecy_text = result.get("content", "")
            tokens = result.get("tokens_generated", len(prophecy_text.split()))

            return OracleResponse(
                prophecy=prophecy_text,
                tokens_generated=tokens,
                time_taken_ms=elapsed_ms,
                tokens_per_second=tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
                oracle_name=result.get("model", ""),
                success=True,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return OracleResponse(
                prophecy="",
                time_taken_ms=elapsed_ms,
                success=False,
                error=str(e),
            )

    def seek_batch_prophecies(
        self,
        questions: List[str],
        context: Optional[str] = None,
        max_words: int = 512,
        creativity: float = 0.7,
    ) -> List[OracleResponse]:
        """
        Seek multiple prophecies from the Oracle.

        Args:
            questions: List of questions to ask
            context: System context for all questions
            max_words: Maximum words per prophecy
            creativity: Temperature

        Returns:
            List of OracleResponse
        """
        responses = []
        for question in questions:
            response = self.seek_prophecy(
                question=question,
                context=context,
                max_words=max_words,
                creativity=creativity,
            )
            responses.append(response)
        return responses

    # =========================================================================
    # ORACLE STATE METHODS
    # =========================================================================

    def oracle_pulse(self) -> bool:
        """
        Check if the Oracle is alive and responsive.

        Returns:
            True if Oracle responds
        """
        try:
            result = self.health()
            return result.get("status") == "healthy"
        except Exception:
            return False

    def oracle_wisdom(self) -> Dict[str, Any]:
        """
        Get the Oracle's current wisdom (model info).

        Returns:
            Dict with model name, path, VRAM usage
        """
        try:
            info = self.get_model_info()
            return {
                "oracle_name": info.get("model_name", "unknown"),
                "oracle_path": info.get("model_path", ""),
                "crystal_power_mb": info.get("vram_mb", 0),  # VRAM as "crystal power"
                "loaded_at": info.get("loaded_at"),
            }
        except Exception as e:
            return {
                "oracle_name": "unknown",
                "error": str(e),
            }

    def rebirth_oracle(self, champion_path: str) -> bool:
        """
        Rebirth the Oracle with a new champion checkpoint.

        The Oracle dies and is reborn with wisdom from the new champion.

        Args:
            champion_path: Path to checkpoint to load

        Returns:
            True if rebirth successful
        """
        try:
            result = self.reload_model(model_path=champion_path)
            return result.get("success", False)
        except Exception:
            return False

    def get_crystal_metrics(self) -> Dict[str, Any]:
        """
        Get Crystal Tower metrics (server stats).

        Returns:
            Dict with request counts, latencies, etc.
        """
        try:
            return self.get_metrics()
        except Exception as e:
            return {"error": str(e)}


# Convenience function
def get_oracle_client(
    crystal_tower: Optional[str] = None,
) -> OracleClient:
    """Get an OracleClient for the Crystal Tower (auto-detects URL if not provided)."""
    return OracleClient(crystal_tower=crystal_tower)


# Re-export original for backward compatibility
PredictionClient = _PredictionClient
