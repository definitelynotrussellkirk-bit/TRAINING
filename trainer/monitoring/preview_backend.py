#!/usr/bin/env python3
"""
Preview Backend Abstraction

Separates preview generation (expensive) from training loop.
Supports local (on training GPU) and remote (3090) backends.

Architecture:
- PreviewBackend: Protocol defining interface
- LocalPreviewBackend: Current behavior (runs on training GPU)
- Remote3090Backend: Send requests to 3090 API (future-ready)

Usage:
    from trainer.monitoring.preview_backend import create_preview_backend

    backend = create_preview_backend(config.monitoring)
    result = backend.preview(prompt, golden_answer, step, max_new_tokens=256)

    if result.success:
        print(f"Generated: {result.text}")
        print(f"Tokens/sec: {result.tokens_per_sec}")
"""

from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any
from datetime import datetime
import time
import requests


@dataclass
class PreviewResult:
    """Result from preview generation"""
    success: bool
    text: str
    prompt: str
    golden_answer: Optional[str]
    step: int

    # Performance metrics
    tokens_generated: int
    generation_time_sec: float
    tokens_per_sec: float

    # Metadata
    backend_type: str  # "local" or "remote_3090"
    timestamp: str
    error_message: Optional[str] = None

    @classmethod
    def from_error(cls, prompt: str, step: int, backend_type: str, error: str) -> 'PreviewResult':
        """Create error result"""
        return cls(
            success=False,
            text="",
            prompt=prompt,
            golden_answer=None,
            step=step,
            tokens_generated=0,
            generation_time_sec=0.0,
            tokens_per_sec=0.0,
            backend_type=backend_type,
            timestamp=datetime.now().isoformat(),
            error_message=error
        )


class PreviewBackend(Protocol):
    """
    Protocol for preview generation backends.

    Implementations must provide:
    - preview() method
    - backend_type property
    """

    @property
    def backend_type(self) -> str:
        """Backend identifier (e.g., 'local', 'remote_3090')"""
        ...

    def preview(
        self,
        prompt: str,
        golden_answer: Optional[str],
        step: int,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> PreviewResult:
        """
        Generate preview text.

        Args:
            prompt: Input prompt
            golden_answer: Expected answer (for comparison)
            step: Training step number
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            PreviewResult with generated text and metrics
        """
        ...


class LocalPreviewBackend:
    """
    Local preview backend - runs generation on training GPU.

    This is the current default behavior. Useful for:
    - Quick tests
    - Single-GPU setups
    - When 3090 is unavailable

    Warning: Runs on training GPU, competes with training for VRAM/compute.
    """

    def __init__(self, model, tokenizer, logits_processor=None):
        """
        Initialize local backend.

        Args:
            model: Loaded model (on training GPU)
            tokenizer: Tokenizer
            logits_processor: Optional logits processors for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.logits_processor = logits_processor

    @property
    def backend_type(self) -> str:
        return "local"

    def preview(
        self,
        prompt: str,
        golden_answer: Optional[str],
        step: int,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> PreviewResult:
        """Generate preview locally on training GPU"""
        try:
            import torch

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate
            start_time = time.time()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    logits_processor=self.logits_processor,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            generation_time = time.time() - start_time

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]

            # Calculate metrics
            tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

            return PreviewResult(
                success=True,
                text=generated_text,
                prompt=prompt,
                golden_answer=golden_answer,
                step=step,
                tokens_generated=tokens_generated,
                generation_time_sec=generation_time,
                tokens_per_sec=tokens_per_sec,
                backend_type=self.backend_type,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return PreviewResult.from_error(
                prompt=prompt,
                step=step,
                backend_type=self.backend_type,
                error=str(e)
            )


class Remote3090Backend:
    """
    Remote 3090 backend - sends generation requests to RTX 3090 API.

    Benefits:
    - Offloads compute from training GPU
    - No VRAM competition with training
    - Can use different model/settings for preview

    Configuration:
    - host: 3090 server IP (default: 192.168.x.x)
    - port: API port (default: 8765)
    - timeout: Request timeout in seconds (default: 30)

    API Contract:
    - POST /generate
    - Body: {prompt, max_tokens, temperature, model_id}
    - Response: {job_id, status} (async) or {text, ...} (sync when worker ready)
    """

    def __init__(
        self,
        host: str = "192.168.x.x",
        port: int = 8765,
        timeout: int = 30,
        model_id: Optional[str] = None
    ):
        """
        Initialize remote backend.

        Args:
            host: 3090 server IP
            port: API port
            timeout: Request timeout (seconds)
            model_id: Model to use on 3090 (None = use active model)
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.model_id = model_id
        self.base_url = f"http://{host}:{port}"

    @property
    def backend_type(self) -> str:
        return "remote_3090"

    def preview(
        self,
        prompt: str,
        golden_answer: Optional[str],
        step: int,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> PreviewResult:
        """
        Generate preview via 3090 API.

        Note: Currently the 3090 API queues jobs asynchronously.
        This implementation polls for results. When the 3090 worker
        is fully implemented, this will work seamlessly.
        """
        try:
            start_time = time.time()

            # Send generation request
            payload = {
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "model_id": self.model_id
            }

            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            # Check if response is synchronous (worker ready) or async (job queued)
            if "text" in result:
                # Synchronous response (future state when worker is ready)
                generation_time = time.time() - start_time
                tokens_generated = result.get("tokens_generated", len(result["text"].split()))

                return PreviewResult(
                    success=True,
                    text=result["text"],
                    prompt=prompt,
                    golden_answer=golden_answer,
                    step=step,
                    tokens_generated=tokens_generated,
                    generation_time_sec=generation_time,
                    tokens_per_sec=tokens_generated / generation_time if generation_time > 0 else 0,
                    backend_type=self.backend_type,
                    timestamp=datetime.now().isoformat()
                )
            else:
                # Asynchronous response (current state - job queued)
                # For now, return error explaining worker not ready
                return PreviewResult.from_error(
                    prompt=prompt,
                    step=step,
                    backend_type=self.backend_type,
                    error=f"3090 API queued job but worker not running. Job ID: {result.get('job_id')}"
                )

        except requests.Timeout:
            return PreviewResult.from_error(
                prompt=prompt,
                step=step,
                backend_type=self.backend_type,
                error=f"Request timeout after {self.timeout}s"
            )
        except requests.RequestException as e:
            return PreviewResult.from_error(
                prompt=prompt,
                step=step,
                backend_type=self.backend_type,
                error=f"Network error: {str(e)}"
            )
        except Exception as e:
            return PreviewResult.from_error(
                prompt=prompt,
                step=step,
                backend_type=self.backend_type,
                error=f"Unexpected error: {str(e)}"
            )


def create_preview_backend(
    backend_type: str,
    model=None,
    tokenizer=None,
    logits_processor=None,
    **backend_config
) -> PreviewBackend:
    """
    Factory function to create preview backend.

    Args:
        backend_type: "local" or "remote_3090"
        model: Model (required for local backend)
        tokenizer: Tokenizer (required for local backend)
        logits_processor: Logits processor (optional, for local backend)
        **backend_config: Additional config (host, port for remote)

    Returns:
        PreviewBackend instance

    Examples:
        # Local backend
        backend = create_preview_backend("local", model=model, tokenizer=tokenizer)

        # Remote 3090 backend
        backend = create_preview_backend(
            "remote_3090",
            host="192.168.x.x",
            port=8765
        )
    """
    if backend_type == "local":
        if model is None or tokenizer is None:
            raise ValueError("Local backend requires model and tokenizer")
        return LocalPreviewBackend(model, tokenizer, logits_processor)

    elif backend_type == "remote_3090":
        return Remote3090Backend(**backend_config)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Must be 'local' or 'remote_3090'")


__all__ = [
    "PreviewBackend",
    "PreviewResult",
    "LocalPreviewBackend",
    "Remote3090Backend",
    "create_preview_backend"
]
