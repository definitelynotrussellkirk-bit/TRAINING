#!/usr/bin/env python3
"""
Eval Bank Manager - Prevents training on evaluation data.

This module manages "eval banks" - sets of IDs and prompt hashes from
evaluation datasets. Training data is checked against these banks to
prevent contamination (training on test data).

Usage:
    from forge.leakage import EvalBankManager

    manager = EvalBankManager()
    bank = manager.get_bank("bin")  # Get eval bank for binary skill

    if bank and example_id in bank.ids:
        print("LEAKAGE: This example is in the eval set!")

Eval banks are stored in:
    data/eval_banks/{skill_id}.jsonl

Each line is a JSON object with at minimum:
    {"id": "...", "prompt": "..."}

The bank stores:
    - IDs (exact match)
    - Prompt hashes (normalized, lowercase, stripped)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalBank:
    """
    Eval bank for a single skill.

    Contains IDs and prompt hashes for leakage detection.
    """
    skill_id: str
    ids: Set[str] = field(default_factory=set)
    prompt_hashes: Set[int] = field(default_factory=set)
    source_path: Optional[str] = None
    loaded_at: Optional[str] = None

    @property
    def size(self) -> int:
        """Total number of items in bank."""
        return len(self.ids) + len(self.prompt_hashes)

    def contains_id(self, item_id: str) -> bool:
        """Check if ID is in bank."""
        return item_id in self.ids

    def contains_prompt(self, prompt: str) -> bool:
        """Check if prompt (normalized) is in bank."""
        prompt_hash = hash(prompt.strip().lower())
        return prompt_hash in self.prompt_hashes

    def contains(self, item_id: Optional[str] = None, prompt: Optional[str] = None) -> bool:
        """Check if either ID or prompt is in bank."""
        if item_id and self.contains_id(item_id):
            return True
        if prompt and self.contains_prompt(prompt):
            return True
        return False


class EvalBankManager:
    """
    Manages eval banks for all skills.

    Eval banks are loaded lazily from data/eval_banks/{skill_id}.jsonl.
    Each JSONL file contains evaluation examples that should NOT appear
    in training data.

    Attributes:
        eval_banks_dir: Directory containing eval bank files
    """

    def __init__(self, eval_banks_dir: Optional[Path] = None):
        """
        Initialize eval bank manager.

        Args:
            eval_banks_dir: Directory containing eval bank files.
                           Defaults to data/eval_banks/
        """
        if eval_banks_dir is None:
            try:
                from core.paths import get_base_dir
                eval_banks_dir = get_base_dir() / "data" / "eval_banks"
            except ImportError:
                eval_banks_dir = Path("data/eval_banks")

        self.eval_banks_dir = Path(eval_banks_dir)
        self._cache: Dict[str, EvalBank] = {}

    def get_bank(self, skill_id: str) -> Optional[EvalBank]:
        """
        Get eval bank for a skill.

        Loads from disk if not cached. Returns None if no bank exists.

        Args:
            skill_id: Skill identifier (e.g., "bin", "sy")

        Returns:
            EvalBank or None if not found
        """
        # Check cache
        if skill_id in self._cache:
            return self._cache[skill_id]

        # Try to load from disk
        bank = self._load_bank(skill_id)
        if bank:
            self._cache[skill_id] = bank

        return bank

    def _load_bank(self, skill_id: str) -> Optional[EvalBank]:
        """Load eval bank from JSONL file."""
        bank_path = self.eval_banks_dir / f"{skill_id}.jsonl"

        if not bank_path.exists():
            # Try alternate names
            alt_paths = [
                self.eval_banks_dir / f"{skill_id}_eval.jsonl",
                self.eval_banks_dir / f"eval_{skill_id}.jsonl",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    bank_path = alt_path
                    break
            else:
                logger.debug(f"No eval bank found for skill: {skill_id}")
                return None

        bank = EvalBank(
            skill_id=skill_id,
            source_path=str(bank_path),
        )

        try:
            with open(bank_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Add ID
                    item_id = item.get("id") or item.get("puzzle_id") or item.get("example_id")
                    if item_id:
                        bank.ids.add(str(item_id))

                    # Add prompt hash
                    prompt = (
                        item.get("prompt") or
                        item.get("user_prompt") or
                        item.get("question") or
                        item.get("input")
                    )
                    if prompt and isinstance(prompt, str):
                        bank.prompt_hashes.add(hash(prompt.strip().lower()))

            from datetime import datetime
            bank.loaded_at = datetime.utcnow().isoformat() + "Z"

            logger.info(
                f"Loaded eval bank for '{skill_id}': "
                f"{len(bank.ids)} IDs, {len(bank.prompt_hashes)} prompt hashes"
            )

        except Exception as e:
            logger.error(f"Failed to load eval bank for {skill_id}: {e}")
            return None

        return bank

    def list_banks(self) -> Dict[str, Dict[str, int]]:
        """
        List available eval banks.

        Returns:
            Dict of skill_id → {"ids": count, "prompt_hashes": count}
        """
        result = {}

        if not self.eval_banks_dir.exists():
            return result

        for jsonl_file in self.eval_banks_dir.glob("*.jsonl"):
            skill_id = jsonl_file.stem.replace("_eval", "").replace("eval_", "")
            bank = self.get_bank(skill_id)
            if bank:
                result[skill_id] = {
                    "ids": len(bank.ids),
                    "prompt_hashes": len(bank.prompt_hashes),
                }

        return result

    def create_bank_from_validation_dir(
        self,
        skill_id: str,
        validation_dir: Path,
    ) -> Optional[EvalBank]:
        """
        Create an eval bank from a validation data directory.

        This is useful for populating eval banks from existing eval sets.

        Args:
            skill_id: Skill identifier
            validation_dir: Directory containing eval JSONL files

        Returns:
            Created EvalBank or None if failed
        """
        validation_dir = Path(validation_dir)
        if not validation_dir.exists():
            logger.error(f"Validation directory not found: {validation_dir}")
            return None

        bank = EvalBank(skill_id=skill_id)

        # Process all JSONL files in directory
        for jsonl_file in validation_dir.glob("**/*.jsonl"):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Add ID
                        item_id = item.get("id") or item.get("puzzle_id")
                        if item_id:
                            bank.ids.add(str(item_id))

                        # Add prompt hash
                        prompt = item.get("prompt") or item.get("user_prompt")
                        if prompt:
                            bank.prompt_hashes.add(hash(prompt.strip().lower()))

            except Exception as e:
                logger.warning(f"Error processing {jsonl_file}: {e}")

        if bank.size == 0:
            logger.warning(f"No items found in {validation_dir}")
            return None

        # Save to disk
        output_path = self.eval_banks_dir / f"{skill_id}.jsonl"
        self.eval_banks_dir.mkdir(parents=True, exist_ok=True)

        # We just store a summary since we only need IDs and hashes
        # The actual eval data stays in validation_dir
        with open(output_path, "w") as f:
            for item_id in bank.ids:
                f.write(json.dumps({"id": item_id}) + "\n")

        bank.source_path = str(output_path)
        self._cache[skill_id] = bank

        logger.info(
            f"Created eval bank for '{skill_id}': "
            f"{len(bank.ids)} IDs from {validation_dir}"
        )

        return bank


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    manager = EvalBankManager()

    print("Available eval banks:")
    banks = manager.list_banks()

    if not banks:
        print("  (none)")
        print("\nTo create an eval bank:")
        print("  1. Place eval JSONL files in data/eval_banks/{skill_id}.jsonl")
        print("  2. Or use: manager.create_bank_from_validation_dir('bin', Path('data/validation/binary'))")
    else:
        for skill_id, counts in banks.items():
            print(f"  {skill_id}: {counts['ids']} IDs, {counts['prompt_hashes']} prompt hashes")
