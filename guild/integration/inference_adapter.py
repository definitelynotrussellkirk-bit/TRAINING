"""
Inference adapter for model interaction.

Wraps the RTX 3090 inference server API for guild combat evaluation.

Features:
- Execute quests against the model
- Parse and validate responses
- Batch inference support
- Retry and error handling
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

from guild.integration.adapters import (
    BaseAdapter,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
)

logger = logging.getLogger(__name__)


# Try to import requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None  # type: ignore


@dataclass
class InferenceRequest:
    """A request for model inference."""
    messages: List[Dict[str, str]]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Metadata for tracking
    quest_id: str = ""
    skill: str = ""
    difficulty: int = 1

    def to_api_request(self, model: str = "Qwen3-0.6B") -> Dict[str, Any]:
        """Convert to API request format."""
        return {
            "model": model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass
class InferenceResponse:
    """Response from model inference."""
    content: str = ""
    finish_reason: str = ""
    model: str = ""

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Timing
    latency_ms: float = 0.0

    # Original response
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_response(cls, response: Dict[str, Any], latency_ms: float = 0.0) -> "InferenceResponse":
        """Parse API response into InferenceResponse."""
        choices = response.get("choices", [])
        if not choices:
            return cls(raw_response=response, latency_ms=latency_ms)

        choice = choices[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})

        return cls(
            content=message.get("content", ""),
            finish_reason=choice.get("finish_reason", ""),
            model=response.get("model", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            raw_response=response,
        )


@dataclass
class QuestExecution:
    """
    Result of executing a quest against the model.

    Contains the inference response plus evaluation context.
    """
    quest_id: str
    request: InferenceRequest
    response: InferenceResponse
    executed_at: datetime = field(default_factory=datetime.now)

    @property
    def model_answer(self) -> str:
        """Get the model's answer content."""
        return self.response.content


class InferenceAdapter(BaseAdapter):
    """
    Adapter for model inference via RTX 3090 server.

    Features:
    - Chat completions for quest execution
    - Retry logic with backoff
    - Response parsing and validation
    - Health monitoring
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self._session: Optional[requests.Session] = None

    @property
    def name(self) -> str:
        return "inference"

    @property
    def api_url(self) -> str:
        """Base URL for inference API."""
        return self.config.inference_url

    def _get_session(self) -> "requests.Session":
        """Get or create HTTP session."""
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Install: pip install requests")
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def health_check(self) -> bool:
        """Check if inference server is available."""
        if not HAS_REQUESTS:
            return False

        try:
            session = self._get_session()
            response = session.get(
                f"{self.api_url}/health",
                timeout=self.config.timeout_seconds
            )
            data = response.json()
            return data.get("status") == "ok"
        except Exception as e:
            logger.debug(f"Inference health check failed: {e}")
            return False

    def get_model_info(self) -> AdapterResult[Dict[str, Any]]:
        """Get information about the currently loaded model."""
        try:
            session = self._get_session()
            response = session.get(
                f"{self.api_url}/models/info",
                timeout=self.config.timeout_seconds
            )
            return AdapterResult.ok(response.json())
        except Exception as e:
            return AdapterResult.fail(str(e))

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: str = "Qwen3-0.6B",
    ) -> AdapterResult[InferenceResponse]:
        """
        Execute a chat completion.

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            model: Model name

        Returns:
            AdapterResult with InferenceResponse
        """
        if not HAS_REQUESTS:
            return AdapterResult.fail("requests library not available")

        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        start_time = time.time()

        for attempt in range(self.config.max_retries):
            try:
                session = self._get_session()
                response = session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=request_data,
                    timeout=self.config.timeout_seconds,
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    inference_response = InferenceResponse.from_api_response(data, latency_ms)
                    return AdapterResult.ok(
                        inference_response,
                        attempt=attempt + 1,
                        latency_ms=latency_ms,
                    )
                else:
                    logger.warning(f"Inference request failed: {response.status_code}")

            except requests.exceptions.Timeout:
                logger.warning(f"Inference timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Inference error on attempt {attempt + 1}: {e}")

            # Wait before retry
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay_seconds * (attempt + 1))

        return AdapterResult.timeout(f"Failed after {self.config.max_retries} attempts")

    def execute_quest(
        self,
        quest,  # QuestInstance
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AdapterResult[QuestExecution]:
        """
        Execute a quest against the model.

        Args:
            quest: QuestInstance to execute
            system_prompt: Optional system prompt override
            max_tokens: Max generation tokens
            temperature: Sampling temperature

        Returns:
            AdapterResult with QuestExecution
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": quest.prompt})

        # Create request
        request = InferenceRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            quest_id=quest.id,
            skill=quest.skills[0] if quest.skills else "",
            difficulty=quest.difficulty_level,
        )

        # Execute
        result = self.chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        if result.success:
            execution = QuestExecution(
                quest_id=quest.id,
                request=request,
                response=result.data,
            )
            return AdapterResult.ok(
                execution,
                latency_ms=result.metadata.get("latency_ms", 0),
            )
        else:
            return AdapterResult.fail(result.error)

    def execute_batch(
        self,
        quests: List,  # List[QuestInstance]
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        on_complete: Optional[Callable[[QuestExecution], None]] = None,
    ) -> AdapterResult[List[QuestExecution]]:
        """
        Execute multiple quests in sequence.

        Args:
            quests: List of QuestInstance objects
            system_prompt: System prompt for all quests
            max_tokens: Max generation tokens
            temperature: Sampling temperature
            on_complete: Callback for each completed quest

        Returns:
            AdapterResult with list of QuestExecutions
        """
        executions: List[QuestExecution] = []
        errors: List[str] = []

        for quest in quests:
            result = self.execute_quest(
                quest,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if result.success:
                executions.append(result.data)
                if on_complete:
                    on_complete(result.data)
            else:
                errors.append(f"{quest.id}: {result.error}")

        if errors:
            if executions:
                return AdapterResult(
                    status=AdapterStatus.PARTIAL,
                    data=executions,
                    error=f"{len(errors)} quests failed",
                    metadata={"errors": errors},
                )
            else:
                return AdapterResult.fail(f"All {len(errors)} quests failed")

        return AdapterResult.ok(
            executions,
            total=len(quests),
            successful=len(executions),
        )

    def generate_raw(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AdapterResult[str]:
        """
        Generate raw text (non-chat format).

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            AdapterResult with generated text
        """
        if not HAS_REQUESTS:
            return AdapterResult.fail("requests library not available")

        try:
            session = self._get_session()
            response = session.post(
                f"{self.api_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=self.config.timeout_seconds,
            )

            if response.status_code == 200:
                data = response.json()
                return AdapterResult.ok(data.get("text", ""))
            else:
                return AdapterResult.fail(f"Status {response.status_code}")

        except Exception as e:
            return AdapterResult.fail(str(e))


# Global adapter instance
_inference_adapter: Optional[InferenceAdapter] = None


def init_inference_adapter(config: Optional[AdapterConfig] = None) -> InferenceAdapter:
    """Initialize the global inference adapter."""
    global _inference_adapter
    _inference_adapter = InferenceAdapter(config)
    return _inference_adapter


def get_inference_adapter() -> InferenceAdapter:
    """Get the global inference adapter."""
    global _inference_adapter
    if _inference_adapter is None:
        _inference_adapter = InferenceAdapter()
    return _inference_adapter


def reset_inference_adapter() -> None:
    """Reset the global inference adapter (for testing)."""
    global _inference_adapter
    _inference_adapter = None
