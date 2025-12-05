"""
Output Validator - Detects error messages masquerading as model output.

The inference server may return errors as valid HTTP 200 responses with
the error message as the assistant's content. This module detects such
cases and flags them appropriately.

Usage:
    from core.output_validator import is_error_response, validate_model_output

    output = get_model_response(...)
    if is_error_response(output):
        logger.error(f"Got error instead of model output: {output}")
        return None

    # Or use the wrapper that logs and returns None on error
    output = validate_model_output(output, skill="binary", level=5)
"""

import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Error patterns that indicate the response is an error, not real output
ERROR_PATTERNS = [
    # CUDA/PyTorch errors
    r"Expected all tensors to be on the same device",
    r"CUDA out of memory",
    r"RuntimeError:",
    r"CUDA error:",
    r"device-side assert",
    r"index_select",
    r"different from other tensors on",
    # Python exceptions
    r"Traceback \(most recent call last\)",
    r"AttributeError:",
    r"TypeError:",
    r"ValueError:",
    r"KeyError:",
    r"ImportError:",
    r"ModuleNotFoundError:",
    r"FileNotFoundError:",
    r"IndexError:",
    r"ZeroDivisionError:",
    r"AssertionError:",
    # Explicit error prefixes from inference server
    r"^Error:",
    r"^ERROR:",
    r"Generation failed:",
    r"Inference failed:",
    r"Model not loaded:",
    # OOM and resource errors
    r"out of memory",
    r"OutOfMemoryError",
    r"resource exhausted",
    # Network/connection errors
    r"ConnectionError:",
    r"TimeoutError:",
    r"Connection refused",
    r"Connection reset",
]

# Compiled regex for efficiency
_ERROR_REGEX = re.compile(
    "|".join(f"({pattern})" for pattern in ERROR_PATTERNS),
    re.IGNORECASE | re.MULTILINE
)


def is_error_response(output: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a model output is actually an error message.

    Args:
        output: The model's output string

    Returns:
        Tuple of (is_error: bool, matched_pattern: Optional[str])
        matched_pattern is the error pattern that was detected, or None if not an error
    """
    if not output:
        return False, None

    match = _ERROR_REGEX.search(output)
    if match:
        # Return the matched group
        return True, match.group(0)

    return False, None


def validate_model_output(
    output: str,
    skill: Optional[str] = None,
    level: Optional[int] = None,
    problem_idx: Optional[int] = None,
    log_error: bool = True,
) -> Optional[str]:
    """
    Validate model output and return None if it's an error message.

    This is the recommended wrapper for inference responses. It handles
    error detection, logging, and returns None to signal the eval should
    mark this as a failed attempt.

    Args:
        output: The model's output string
        skill: Skill ID for logging context
        level: Level for logging context
        problem_idx: Problem index for logging context
        log_error: Whether to log errors (default True)

    Returns:
        The output if valid, None if it was an error message

    Example:
        response = get_model_response(prompt)
        validated = validate_model_output(response, skill="bin", level=5)
        if validated is None:
            # Handle as inference error
            continue
    """
    if not output:
        return None

    is_error, matched_pattern = is_error_response(output)

    if is_error:
        if log_error:
            context_parts = []
            if skill:
                context_parts.append(f"skill={skill}")
            if level is not None:
                context_parts.append(f"level={level}")
            if problem_idx is not None:
                context_parts.append(f"problem={problem_idx}")
            context = f" [{', '.join(context_parts)}]" if context_parts else ""

            # Truncate long error messages for logging
            truncated = output[:200] + "..." if len(output) > 200 else output
            logger.error(
                f"🚨 INFERENCE ERROR{context}: Got error instead of model output. "
                f"Pattern: '{matched_pattern}' | Output: {truncated}"
            )
        return None

    return output


def classify_error(output: str) -> Optional[str]:
    """
    Classify the type of error in a response.

    Returns:
        Error category string: "cuda_device", "oom", "python_exception",
        "inference_error", "network", or None if not an error
    """
    if not output:
        return None

    output_lower = output.lower()

    # CUDA device mismatch
    if "tensors to be on the same device" in output_lower or "index_select" in output_lower:
        return "cuda_device"

    # Out of memory
    if "out of memory" in output_lower or "oom" in output_lower:
        return "oom"

    # Python exceptions
    if "traceback" in output_lower or any(
        err in output for err in [
            "AttributeError:", "TypeError:", "ValueError:",
            "KeyError:", "ImportError:", "IndexError:"
        ]
    ):
        return "python_exception"

    # Network errors
    if any(err in output_lower for err in ["connection", "timeout"]):
        return "network"

    # Generic inference errors
    is_error, _ = is_error_response(output)
    if is_error:
        return "inference_error"

    return None


def get_error_guidance(error_type: str) -> str:
    """
    Get debugging guidance for an error type.

    Args:
        error_type: From classify_error()

    Returns:
        Human-readable debugging guidance
    """
    guidance = {
        "cuda_device": (
            "CUDA device mismatch: The model has tensors on different devices (CPU/CUDA). "
            "This usually happens with PEFT adapters or models loaded with device_map='auto'. "
            "Try: 1) Reload the model with explicit device placement, "
            "2) Check that both base model and adapter are on the same device, "
            "3) Restart the inference server."
        ),
        "oom": (
            "Out of memory: GPU VRAM is exhausted. "
            "Try: 1) Reduce max_tokens, 2) Unload unused models from pool, "
            "3) Restart inference server to clear fragmented memory."
        ),
        "python_exception": (
            "Python exception during inference. Check inference server logs for full traceback. "
            "Common causes: corrupted checkpoint, incompatible model version, missing dependencies."
        ),
        "network": (
            "Network error connecting to inference server. "
            "Check: 1) Inference server is running, 2) Network connectivity, "
            "3) Firewall rules allow the connection."
        ),
        "inference_error": (
            "Generic inference error. Check inference server logs for details."
        ),
    }
    return guidance.get(error_type, "Unknown error type")
