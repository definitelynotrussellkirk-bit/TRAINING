#!/usr/bin/env python3
"""
Logit Penalty System - Control token generation during inference.

Provides LogitsProcessor implementations that modify generation probabilities
to discourage unwanted tokens (<think> tags, premature EOS) and enforce
stop signal behavior.

Key Components:
    - TokenLogitPenalty: Penalize specific tokens with optional schedules
    - PostStopPenalty: Penalize all tokens after stop emoji sequences
    - build_think_penalty_processor(): Factory for <think> tag penalties
    - build_eos_penalty_processor(): Factory for EOS token penalties
    - build_post_stop_penalty_processor(): Factory for post-stop penalties

Penalty Schedule Format:
    Schedules control how penalties change over generation steps.

    Format: List of dicts with "steps" and "multiplier" keys:
        [
            {"steps": 4, "multiplier": 8.0},   # First 4 steps: penalty * 8.0
            {"steps": 4, "multiplier": 2.0},   # Next 4 steps: penalty * 2.0
            {"steps": 4, "multiplier": 1.5},   # Next 4 steps: penalty * 1.5
            # After step 12: penalty * 1.0 (no multiplier)
        ]

    - "steps": int, number of generation steps for this window
    - "multiplier": float >= 1.0, penalty multiplier for this window
    - Windows apply sequentially (cumulative step counting)
    - After all windows complete, multiplier defaults to 1.0

Usage:
    from core.logit_penalty import build_think_penalty_processor

    # Basic penalty
    processors = build_think_penalty_processor(tokenizer, penalty=80.0)

    # With schedule (strong early, taper off)
    schedule = [
        {"steps": 4, "multiplier": 8.0},
        {"steps": 4, "multiplier": 2.0},
    ]
    processors = build_think_penalty_processor(
        tokenizer,
        penalty=80.0,
        schedule=schedule
    )

    # Use with generation
    outputs = model.generate(
        input_ids,
        logits_processor=processors,
        max_new_tokens=100
    )
"""

from collections.abc import Iterable
from typing import List, Sequence, Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList


# Default penalty schedule: Strong early penalties that taper off
# Steps 0-3:   penalty * 8.0 (very strong)
# Steps 4-7:   penalty * 2.0 (moderate)
# Steps 8-11:  penalty * 1.5 (mild)
# Steps 12+:   penalty * 1.0 (baseline)
DEFAULT_PENALTY_SCHEDULE = [
    {"steps": 4, "multiplier": 8.0},
    {"steps": 4, "multiplier": 2.0},
    {"steps": 4, "multiplier": 1.5},
]


class PostStopPenalty(LogitsProcessor):
    """
    Penalize all tokens after stop emoji sequences with exponentially escalating penalties.

    Detects stop emoji sequences (e.g., "🛑🛑", "✓✓✓") and applies growing penalties
    to all subsequent tokens except EOT (end-of-turn). Forces model to end generation
    cleanly after stop signals rather than continuing to produce text.

    Responsibilities:
        - Detect stop emoji sequences in generated tokens
        - Apply exponentially increasing penalties after stop detected
        - Exempt EOT tokens from penalties (+ optional reward boost)
        - Track statistics (hits, steps, detected patterns)

    Penalty Formula:
        penalty = base_penalty * (escalation_rate ^ tokens_after_stop)

        Example (base=5.0, rate=2.0):
            Token 1 after stop: 5.0 * (2.0^1) = 10.0
            Token 2 after stop: 5.0 * (2.0^2) = 20.0
            Token 3 after stop: 5.0 * (2.0^3) = 40.0
            ...exponentially growing

    Attributes:
        base_penalty: Starting penalty value
        escalation_rate: Exponential growth rate (e.g., 2.0 = double each step)
        eot_reward: Extra reward for EOT tokens (on top of penalty exemption)
        stop_count_min: Minimum emoji repetitions to detect (default: 2)
        stop_count_max: Maximum emoji repetitions to detect (default: 4)
        stop_emoji_pool: List of stop emojis (default: ["🛑"])
        stop_sequences: Dict mapping (emoji, count) → token IDs
        eot_ids: Set of EOT token IDs to exempt
        stop_seen: bool, whether stop sequence detected yet
        tokens_after_stop: int, count of tokens generated after stop
        detected_emoji: str, which emoji was detected (or None)
        detected_count: int, how many repetitions detected (or None)

    Example:
        >>> penalty = PostStopPenalty(
        ...     tokenizer,
        ...     stop_emoji_pool=["🛑", "✓"],
        ...     stop_count_min=2,
        ...     stop_count_max=3,
        ...     base_penalty=5.0,
        ...     escalation_rate=2.0,
        ...     eot_reward=5.0
        ... )
        >>> processors = LogitsProcessorList([penalty])
        >>> outputs = model.generate(inputs, logits_processor=processors)
    """

    def __init__(
        self,
        tokenizer,
        stop_emoji_pool: Optional[List[str]] = None,
        stop_count_min: int = 2,
        stop_count_max: int = 4,
        base_penalty: float = 5.0,
        escalation_rate: float = 2.0,
        eot_reward: float = 0.0,
        eot_sequence: Optional[str] = None,
        label: Optional[str] = None,
    ):
        """
        Initialize post-stop penalty processor.

        Args:
            tokenizer: HuggingFace tokenizer for encoding stop sequences
            stop_emoji_pool: List of stop emojis to detect (default: ["🛑"])
            stop_count_min: Minimum repetitions to detect (default: 2)
            stop_count_max: Maximum repetitions to detect (default: 4)
            base_penalty: Starting penalty value (default: 5.0)
            escalation_rate: Exponential growth rate (default: 2.0)
            eot_reward: Extra reward for EOT tokens (default: 0.0)
            eot_sequence: Custom EOT sequence (e.g., "<|end|>"). If None, uses tokenizer.eos_token_id
            label: Optional label for stats tracking (default: "post_stop")

        Side Effects:
            - Tokenizes all (emoji, count) combinations to build stop_sequences dict
            - Validates EOT token IDs are within vocab bounds
        """
        self.base_penalty = float(base_penalty)
        self.escalation_rate = float(escalation_rate)
        self.eot_reward = float(eot_reward)
        self.stop_count_min = max(2, int(stop_count_min))
        self.stop_count_max = max(self.stop_count_min, int(stop_count_max))
        self.label = label or "post_stop"

        # Use pool or default to single emoji
        if stop_emoji_pool is None:
            stop_emoji_pool = ["🛑"]
        self.stop_emoji_pool = stop_emoji_pool

        # Encode all possible stop sequences (emoji x count combinations)
        self.stop_sequences = {}  # Maps (emoji, count) -> list of token IDs
        for emoji in self.stop_emoji_pool:
            for count in range(self.stop_count_min, self.stop_count_max + 1):
                sequence = emoji * count
                token_ids = tuple(tokenizer.encode(sequence, add_special_tokens=False))
                self.stop_sequences[(emoji, count)] = token_ids

        # Get EOT token IDs (forward compatible with custom sequences)
        self.eot_ids = set()
        if eot_sequence is not None:
            # Custom EOT sequence provided (e.g., "<|end|>" or "<|end|><|end|>")
            eot_tokens = tokenizer.encode(eot_sequence, add_special_tokens=False)
            vocab_size = len(tokenizer)
            for token_id in eot_tokens:
                if 0 <= token_id < vocab_size:
                    self.eot_ids.add(token_id)
        else:
            # Default: use tokenizer's EOS token
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                # Validate token ID is within vocab bounds
                vocab_size = len(tokenizer)
                if 0 <= eos_token_id < vocab_size:
                    self.eot_ids.add(eos_token_id)

        # Track state
        self.stop_seen = False
        self.tokens_after_stop = 0
        self.generated_steps = 0
        self.hit_count = 0
        self.detected_emoji = None
        self.detected_count = None

    def reset_state(self):
        self.stop_seen = False
        self.tokens_after_stop = 0
        self.generated_steps = 0
        self.detected_emoji = None
        self.detected_count = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.generated_steps += 1

        # Check if we've seen any stop sequence in the generated tokens
        if not self.stop_seen:
            # Check for any stop sequence match
            # IMPORTANT: Check longer sequences FIRST to avoid matching prefixes
            sorted_sequences = sorted(
                self.stop_sequences.items(),
                key=lambda x: len(x[1]),  # Sort by token sequence length
                reverse=True  # Longest first
            )

            for (emoji, count), token_ids in sorted_sequences:
                seq_len = len(token_ids)
                if input_ids.shape[1] >= seq_len:
                    # Check if the last N tokens match this stop sequence
                    recent_tokens = tuple(input_ids[0, -seq_len:].tolist())
                    if recent_tokens == token_ids:
                        self.stop_seen = True
                        self.tokens_after_stop = 0
                        self.detected_emoji = emoji
                        self.detected_count = count
                        break  # Found a match, stop checking

        # If we've seen the stop, penalize everything except EOT
        if self.stop_seen:
            self.tokens_after_stop += 1

            # Calculate escalating penalty: base * (escalation_rate ^ tokens_after_stop)
            current_penalty = self.base_penalty * (self.escalation_rate ** self.tokens_after_stop)

            # Apply penalty to all tokens
            adjusted = scores.clone()
            adjusted -= current_penalty

            # Remove penalty from EOT tokens AND add extra reward
            for eot_id in self.eot_ids:
                adjusted[:, eot_id] += current_penalty + self.eot_reward

            self.hit_count += 1
            return adjusted

        return scores

    def snapshot_stats(self) -> dict:
        return {
            "label": self.label,
            "hits": self.hit_count,
            "stop_seen": self.stop_seen,
            "tokens_after_stop": self.tokens_after_stop,
            "generated_steps": self.generated_steps,
            "eot_reward": self.eot_reward,
            "detected_emoji": self.detected_emoji,
            "detected_count": self.detected_count,
        }


class TokenLogitPenalty(LogitsProcessor):
    """
    Penalize specific token IDs with optional time-varying schedules.

    Subtracts a fixed penalty from specified token logits during generation.
    Supports two modes for time-varying penalties:
    1. Prefix window: Higher penalty for first N steps, then baseline
    2. Schedule: Multiple windows with different multipliers (more flexible)

    Use this for discouraging specific tokens like <think>, </s>, or any
    unwanted tokens throughout generation.

    Responsibilities:
        - Apply constant or time-varying penalties to specified tokens
        - Track generation steps and adjust multiplier based on schedule
        - Collect statistics (hit count, steps, penalties applied)

    Attributes:
        token_ids: Tuple of token IDs to penalize (sorted, deduplicated)
        penalty: Base penalty value to subtract from logits
        prefix_steps: Number of initial steps for prefix mode
        prefix_multiplier: Multiplier for prefix window (≥1.0)
        schedule: Optional list of {"steps": int, "multiplier": float} dicts
        generated_steps: Counter for generation steps
        hit_count: Counter for how many times penalty was applied
        label: Optional label for stats tracking

    Example (prefix mode):
        >>> # Strong penalty for first 10 steps, then baseline
        >>> penalty = TokenLogitPenalty(
        ...     token_ids=[think_token_id],
        ...     penalty=80.0,
        ...     prefix_steps=10,
        ...     prefix_multiplier=4.0
        ... )
        >>> # Steps 0-9:  penalty * 4.0 = 320.0
        >>> # Steps 10+:  penalty * 1.0 = 80.0

    Example (schedule mode):
        >>> # Tapering penalty: strong → moderate → mild
        >>> schedule = [
        ...     {"steps": 4, "multiplier": 8.0},
        ...     {"steps": 4, "multiplier": 2.0},
        ...     {"steps": 4, "multiplier": 1.5},
        ... ]
        >>> penalty = TokenLogitPenalty(
        ...     token_ids=[eos_id],
        ...     penalty=80.0,
        ...     schedule=schedule
        ... )
        >>> # Steps 0-3:   penalty * 8.0 = 640.0
        >>> # Steps 4-7:   penalty * 2.0 = 160.0
        >>> # Steps 8-11:  penalty * 1.5 = 120.0
        >>> # Steps 12+:   penalty * 1.0 = 80.0
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        penalty: float = 5.0,
        prefix_steps: int = 0,
        prefix_multiplier: float = 1.0,
        schedule: Sequence[dict] | None = None,
        label: Optional[str] = None,
    ):
        """
        Initialize token penalty processor.

        Args:
            token_ids: Sequence of token IDs to penalize
            penalty: Base penalty value (default: 5.0). Must be non-negative.
            prefix_steps: Number of steps for prefix mode (default: 0, disabled)
            prefix_multiplier: Multiplier for prefix window (default: 1.0). Must be ≥1.0.
            schedule: Optional penalty schedule (overrides prefix mode if provided).
                Format: [{"steps": int, "multiplier": float >= 1.0}, ...]
            label: Optional label for stats tracking (default: "penalty")

        Raises:
            ValueError: If penalty < 0

        Side Effects:
            - Normalizes schedule (validates steps > 0, multiplier ≥ 1.0)
            - Deduplicates and sorts token_ids
        """
        if penalty < 0:
            raise ValueError("penalty must be non-negative")

        self.token_ids = tuple(sorted(set(int(t) for t in token_ids)))
        self.penalty = float(penalty)
        self.prefix_steps = max(0, int(prefix_steps))
        self.prefix_multiplier = max(float(prefix_multiplier), 1.0)
        self.schedule = self._normalize_schedule(schedule)
        self.generated_steps = 0
        self.label = label or "penalty"
        self.hit_count = 0

    def reset_state(self):
        self.generated_steps = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.token_ids or self.penalty == 0.0:
            return scores

        factor = self._current_multiplier()
        self.generated_steps += 1

        adjusted = scores.clone()
        token_indices = torch.tensor(self.token_ids, device=adjusted.device, dtype=torch.long)
        adjusted[:, token_indices] -= self.penalty * factor
        self.hit_count += 1
        return adjusted

    def _current_multiplier(self) -> float:
        if self.schedule:
            step = self.generated_steps
            cumulative = 0
            for window in self.schedule:
                cumulative += window["steps"]
                if step < cumulative:
                    return window["multiplier"]
            return 1.0

        if self.generated_steps < self.prefix_steps:
            return self.prefix_multiplier
        return 1.0

    @staticmethod
    def _normalize_schedule(schedule: Sequence[dict] | None) -> list[dict]:
        normalized = []
        if not schedule:
            return normalized

        for window in schedule:
            if not window:
                continue
            steps = max(int(window.get("steps", 0)), 0)
            multiplier = max(float(window.get("multiplier", 1.0)), 1.0)
            if steps <= 0:
                continue
            normalized.append({"steps": steps, "multiplier": multiplier})
        return normalized

    def snapshot_stats(self) -> dict:
        return {
            "label": self.label,
            "hits": self.hit_count,
            "generated_steps": self.generated_steps,
            "penalty": self.penalty,
            "schedule": self.schedule or None,
        }


def build_think_penalty_processor(
    tokenizer,
    penalty: float = 80.0,
    prefix_steps: int = 0,
    prefix_multiplier: float = 1.0,
    schedule: Sequence[dict] | None = None,
) -> LogitsProcessorList:
    """
    Build a logits processor list that penalizes <think>/<think> style tags.

    Args:
        tokenizer: HF tokenizer with `encode` and `convert_ids_to_tokens`.
        penalty: Value to subtract from logits of the penalized tokens.

    Returns:
        LogitsProcessorList configured for use with `generate()`.
    """
    think_tokens: List[str] = ["<think>", "</think>"]
    return _build_penalty_processor(
        tokenizer,
        think_tokens,
        penalty=penalty,
        prefix_steps=prefix_steps,
        prefix_multiplier=prefix_multiplier,
        schedule=schedule,
        label="think",
    )


def build_eos_penalty_processor(
    tokenizer,
    penalty: float = 80.0,
    schedule: Sequence[dict] | None = None,
) -> LogitsProcessorList:
    """Penalize EOS tokens (e.g., </s>) to discourage premature endings."""
    eos_ids: List[int] = []

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)

    eos_ids.extend(_coerce_token_ids(eos_token_id))
    eos_ids.extend(_coerce_token_ids(eos_token_ids))

    eos_tokens = getattr(tokenizer, "eos_token", None)
    if eos_tokens:
        if isinstance(eos_tokens, str):
            eos_ids.extend(tokenizer.encode(eos_tokens, add_special_tokens=False))
        else:
            for token in _coerce_token_sequence(eos_tokens):
                if isinstance(token, str):
                    eos_ids.extend(tokenizer.encode(token, add_special_tokens=False))
                else:
                    eos_ids.append(int(token))

    unique_ids = sorted(set(eos_ids))
    if not unique_ids:
        return LogitsProcessorList()

    return LogitsProcessorList(
        [
            TokenLogitPenalty(
                unique_ids,
                penalty=penalty,
                schedule=schedule,
                label="eos",
            )
        ]
    )


def build_post_stop_penalty_processor(
    tokenizer,
    stop_emoji_pool: Optional[List[str]] = None,
    stop_count_min: int = 2,
    stop_count_max: int = 4,
    base_penalty: float = 5.0,
    escalation_rate: float = 2.0,
    eot_reward: float = 0.0,
    eot_sequence: Optional[str] = None,
) -> LogitsProcessorList:
    """
    Build processor that penalizes tokens after stop emoji sequences.

    Supports variable stop emojis (pool) and variable repetition counts (min-max).
    Penalties escalate exponentially: base * (escalation_rate ^ tokens_after_stop)
    EOT tokens are exempt from the penalty and optionally receive extra reward.

    Args:
        tokenizer: HF tokenizer
        stop_emoji_pool: List of stop emojis to detect (default: ["🛑"])
        stop_count_min: Minimum repetitions to detect (default: 2)
        stop_count_max: Maximum repetitions to detect (default: 4)
        base_penalty: Starting penalty value
        escalation_rate: Exponential growth rate for penalties
        eot_reward: Extra reward boost for EOT tokens (default: 0.0)
        eot_sequence: Optional custom EOT sequence string (e.g., "<|end|>" or "<|end|><|end|>")
                     If None, uses tokenizer.eos_token_id

    Returns:
        LogitsProcessorList with the post-stop penalty processor
    """
    return LogitsProcessorList(
        [
            PostStopPenalty(
                tokenizer=tokenizer,
                stop_emoji_pool=stop_emoji_pool,
                stop_count_min=stop_count_min,
                stop_count_max=stop_count_max,
                base_penalty=base_penalty,
                escalation_rate=escalation_rate,
                eot_reward=eot_reward,
                eot_sequence=eot_sequence,
                label="post_stop",
            )
        ]
    )


def reset_processor_states(processors: LogitsProcessorList):
    if processors is None:
        return
    for proc in processors:
        if hasattr(proc, "reset_state"):
            proc.reset_state()


def collect_penalty_stats(processors: LogitsProcessorList) -> List[dict]:
    stats: List[dict] = []
    if not processors:
        return stats
    for proc in processors:
        if hasattr(proc, "snapshot_stats"):
            try:
                stats.append(proc.snapshot_stats())
            except Exception:
                continue
    return stats


def _build_penalty_processor(
    tokenizer,
    token_strs: Sequence[str],
    penalty: float,
    prefix_steps: int = 0,
    prefix_multiplier: float = 1.0,
    schedule: Sequence[dict] | None = None,
    label: Optional[str] = None,
) -> LogitsProcessorList:
    token_ids: List[int] = []
    for token_str in token_strs:
        token_ids.extend(tokenizer.encode(token_str, add_special_tokens=False))

    unique_ids = sorted(set(token_ids))
    if not unique_ids:
        return LogitsProcessorList()

    return LogitsProcessorList(
        [
            TokenLogitPenalty(
                unique_ids,
                penalty=penalty,
                prefix_steps=prefix_steps,
                prefix_multiplier=prefix_multiplier,
                schedule=schedule,
                label=label,
            )
        ]
    )


def _coerce_token_ids(value) -> List[int]:
    """Return a flat list of int token ids from diverse inputs."""
    if value is None:
        return []

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [int(value.item())]
        return [int(v) for v in value.reshape(-1).tolist()]

    if isinstance(value, int):
        return [int(value)]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        coerced: List[int] = []
        for item in value:
            if item is None:
                continue
            coerced.extend(_coerce_token_ids(item))
        return coerced

    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _coerce_token_sequence(value) -> List:
    """Ensure eos token specs are iterable for encoding."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        result = []
        for item in value:
            if item is None:
                continue
            result.append(item)
        return result
    return [value]
