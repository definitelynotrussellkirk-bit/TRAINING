"""
Probe dataset management for activation analysis.

Probes are fixed input sequences used to measure activation statistics.
Using the same probes across all checkpoints ensures comparability.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("analysis.probes")


# =============================================================================
# DEFAULT PROBES
# =============================================================================

# A fixed set of diverse prompts for consistent activation measurement.
# These cover different types of reasoning the model should handle.

DEFAULT_PROBES = [
    # === MATH REASONING ===
    "What is 15 + 27? Think step by step.",
    "If x + 5 = 12, what is x?",
    "Calculate 144 / 12.",
    "What is 7 times 8?",
    "If I have 3 apples and get 5 more, how many do I have?",

    # === LOGIC ===
    "If all cats are mammals and all mammals breathe air, do cats breathe air?",
    "True or False: If A implies B, and B is false, then A must be false.",
    "If it is raining, the ground is wet. The ground is wet. Is it raining?",
    "All squares are rectangles. Is this statement true?",

    # === LANGUAGE ===
    "What is the opposite of 'happy'?",
    "Complete this sentence: The quick brown fox jumps over the lazy ___.",
    "What rhymes with 'cat'?",
    "What is a synonym for 'big'?",
    "Is 'run' a noun or a verb?",

    # === BINARY (BIN skill) ===
    "Convert 5 to binary.",
    "What is 1011 in decimal?",
    "What is the binary AND of 1010 and 1100?",

    # === SYLLACROSTIC (SY skill) ===
    "What are the first letters of: Apple, Banana, Cherry?",
    "Spell out the word formed by the last letters of: caT, doG, piG.",
    "Count the syllables in 'elephant'.",

    # === GENERAL KNOWLEDGE ===
    "What color is the sky?",
    "How many days are in a week?",
    "What is the capital of France?",

    # === INSTRUCTION FOLLOWING ===
    "List three fruits.",
    "Explain what a computer is in one sentence.",
    "Say hello in Spanish.",
]


def get_default_probes() -> List[str]:
    """
    Get the default probe sequences.

    Returns:
        List of probe strings (copy to avoid mutation).
    """
    return DEFAULT_PROBES.copy()


def load_probe_dataset(
    dataset_id: str,
    probes_dir: Optional[Path] = None,
    max_probes: int = 256,
) -> List[str]:
    """
    Load a probe dataset by ID.

    Args:
        dataset_id: One of:
            - 'default': Use DEFAULT_PROBES
            - Path to a .jsonl file with prompts
            - Path to a .txt file with one prompt per line
        probes_dir: Base directory for probe files
        max_probes: Maximum number of probes to load

    Returns:
        List of probe strings.

    Raises:
        FileNotFoundError: If probe file doesn't exist.
        ValueError: If file format is unsupported.
    """
    if dataset_id == "default":
        return get_default_probes()[:max_probes]

    # Resolve file path
    if probes_dir:
        probe_file = probes_dir / dataset_id
    else:
        probe_file = Path(dataset_id)

    if not probe_file.exists():
        raise FileNotFoundError(f"Probe dataset not found: {probe_file}")

    suffix = probe_file.suffix.lower()
    probes = []

    if suffix == ".jsonl":
        # JSONL format: {"prompt": "...", ...} or {"text": "...", ...}
        with open(probe_file) as f:
            for line in f:
                if len(probes) >= max_probes:
                    break

                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Support multiple field names
                for field in ["prompt", "text", "instruction", "input", "question"]:
                    if field in data:
                        probes.append(data[field])
                        break

    elif suffix == ".txt":
        # Plain text: one prompt per line
        with open(probe_file) as f:
            for line in f:
                if len(probes) >= max_probes:
                    break

                line = line.strip()
                if line and not line.startswith("#"):
                    probes.append(line)

    elif suffix == ".json":
        # JSON array of prompts
        with open(probe_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data[:max_probes]:
                if isinstance(item, str):
                    probes.append(item)
                elif isinstance(item, dict):
                    for field in ["prompt", "text", "instruction"]:
                        if field in item:
                            probes.append(item[field])
                            break
        elif isinstance(data, dict) and "probes" in data:
            probes = data["probes"][:max_probes]

    else:
        raise ValueError(f"Unsupported probe file format: {suffix}")

    logger.info(f"Loaded {len(probes)} probes from {probe_file}")
    return probes


def create_skill_probes(skill_id: str, level: int = 1, count: int = 50) -> List[str]:
    """
    Generate skill-specific probes for targeted analysis.

    Args:
        skill_id: Skill identifier ('bin', 'sy', etc.)
        level: Skill level for difficulty scaling
        count: Number of probes to generate

    Returns:
        List of skill-specific probe strings.
    """
    probes = []

    if skill_id in ("bin", "binary"):
        # Binary arithmetic probes
        import random
        for _ in range(count):
            a = random.randint(0, 2**level - 1)
            b = random.randint(0, 2**level - 1)
            op = random.choice(["+", "-", "&", "|"])
            probes.append(f"Calculate {a} {op} {b} in binary.")

    elif skill_id in ("sy", "syllo"):
        # Syllacrostic probes
        words = ["apple", "banana", "cherry", "dog", "elephant", "frog", "grape"]
        import random
        for _ in range(count):
            sample = random.sample(words, min(3 + level // 10, len(words)))
            probes.append(f"What word do the first letters of {', '.join(sample)} spell?")

    else:
        # Generic probes
        probes = get_default_probes()[:count]

    return probes


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("Default Probes:")
    print("=" * 60)
    for i, probe in enumerate(get_default_probes(), 1):
        print(f"{i:2}. {probe[:60]}{'...' if len(probe) > 60 else ''}")

    print(f"\nTotal: {len(get_default_probes())} probes")
