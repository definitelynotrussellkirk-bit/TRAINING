#!/usr/bin/env python3
"""
Data Profiler - Compute distributional statistics for training data.

Profiles a shard to understand:
- Token length distributions
- Field value distributions
- Content characteristics

Usage:
    from forge.profiler import profile_shard

    profile = profile_shard(Path("data.jsonl"))
    print(profile.summary())
"""

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)

PROFILER_VERSION = "1.0.0"


@dataclass
class ShardProfile:
    """Distributional statistics for a shard."""
    file_path: str
    rows_total: int = 0

    # Token lengths (if tokenizer provided)
    token_lengths: Dict[str, Any] = field(default_factory=dict)

    # Character lengths
    char_lengths: Dict[str, Any] = field(default_factory=dict)

    # Field presence
    field_counts: Dict[str, int] = field(default_factory=dict)

    # Value distributions
    role_distribution: Dict[str, int] = field(default_factory=dict)
    message_count_distribution: Dict[int, int] = field(default_factory=dict)

    # Content characteristics
    avg_user_length: float = 0
    avg_assistant_length: float = 0
    user_assistant_ratio: float = 0

    # Metadata
    profiled_at: str = ""
    profiler_version: str = PROFILER_VERSION
    sample_size: int = 0

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [
            f"Rows: {self.rows_total}",
            f"Avg tokens: {self.token_lengths.get('mean', 'N/A')}",
            f"User/Assistant ratio: {self.user_assistant_ratio:.2f}" if self.user_assistant_ratio else "",
        ]
        return " | ".join(p for p in parts if p)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "rows_total": self.rows_total,
            "token_lengths": self.token_lengths,
            "char_lengths": self.char_lengths,
            "field_counts": self.field_counts,
            "role_distribution": self.role_distribution,
            "message_count_distribution": {str(k): v for k, v in self.message_count_distribution.items()},
            "avg_user_length": self.avg_user_length,
            "avg_assistant_length": self.avg_assistant_length,
            "user_assistant_ratio": self.user_assistant_ratio,
            "profiled_at": self.profiled_at,
            "profiler_version": self.profiler_version,
            "sample_size": self.sample_size,
        }


def profile_shard(
    file_path: Path,
    tokenizer=None,
    max_samples: int = 10000,
) -> ShardProfile:
    """
    Profile a shard to understand its characteristics.

    Args:
        file_path: Path to JSONL file
        tokenizer: Optional tokenizer for token length analysis
        max_samples: Maximum samples to analyze

    Returns:
        ShardProfile with statistics
    """
    file_path = Path(file_path)
    profile = ShardProfile(
        file_path=str(file_path),
        profiled_at=datetime.utcnow().isoformat() + "Z",
    )

    # Collectors
    char_lengths = []
    token_lengths = []
    user_lengths = []
    assistant_lengths = []
    field_counter = Counter()
    role_counter = Counter()
    message_counts = Counter()

    rows_read = 0

    try:
        with open(file_path) as f:
            for line in f:
                if rows_read >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rows_read += 1

                # Track field presence
                for field_name in example.keys():
                    field_counter[field_name] += 1

                # Analyze messages
                messages = example.get("messages", [])
                message_counts[len(messages)] += 1

                full_text = ""
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    role_counter[role] += 1
                    full_text += content

                    if role == "user":
                        user_lengths.append(len(content))
                    elif role == "assistant":
                        assistant_lengths.append(len(content))

                # Character length
                char_lengths.append(len(full_text))

                # Token length (if tokenizer provided)
                if tokenizer:
                    try:
                        tokens = tokenizer.encode(full_text)
                        token_lengths.append(len(tokens))
                    except Exception:
                        pass

    except Exception as e:
        logger.error(f"Error profiling {file_path}: {e}")

    # Compute statistics
    profile.rows_total = rows_read
    profile.sample_size = rows_read
    profile.field_counts = dict(field_counter)
    profile.role_distribution = dict(role_counter)
    profile.message_count_distribution = dict(message_counts)

    # Character length stats
    if char_lengths:
        profile.char_lengths = _compute_stats(char_lengths)

    # Token length stats
    if token_lengths:
        profile.token_lengths = _compute_stats(token_lengths)

    # Content ratio
    if user_lengths:
        profile.avg_user_length = statistics.mean(user_lengths)
    if assistant_lengths:
        profile.avg_assistant_length = statistics.mean(assistant_lengths)
    if profile.avg_assistant_length > 0:
        profile.user_assistant_ratio = profile.avg_user_length / profile.avg_assistant_length

    return profile


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute standard statistics for a list of values."""
    if not values:
        return {}

    sorted_values = sorted(values)
    n = len(sorted_values)

    return {
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if n > 1 else 0,
        "p10": sorted_values[int(n * 0.10)],
        "p25": sorted_values[int(n * 0.25)],
        "p75": sorted_values[int(n * 0.75)],
        "p90": sorted_values[int(n * 0.90)],
        "p95": sorted_values[int(n * 0.95)],
        "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        "count": n,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m forge.profiler <file.jsonl>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    print(f"Profiling: {file_path}")

    profile = profile_shard(file_path)

    print(f"\nProfile Summary:")
    print(f"  Rows: {profile.rows_total}")

    if profile.char_lengths:
        print(f"\n  Character Lengths:")
        for k, v in profile.char_lengths.items():
            print(f"    {k}: {v:.1f}" if isinstance(v, float) else f"    {k}: {v}")

    if profile.field_counts:
        print(f"\n  Field Presence:")
        for field_name, count in sorted(profile.field_counts.items(), key=lambda x: -x[1])[:10]:
            pct = count / profile.rows_total * 100
            print(f"    {field_name}: {count} ({pct:.1f}%)")

    if profile.role_distribution:
        print(f"\n  Role Distribution:")
        for role, count in profile.role_distribution.items():
            print(f"    {role}: {count}")

    print(f"\n  User/Assistant Ratio: {profile.user_assistant_ratio:.2f}")
