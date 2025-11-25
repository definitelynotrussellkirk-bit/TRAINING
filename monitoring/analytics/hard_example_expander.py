#!/usr/bin/env python3
"""
Hard Example Expander - Discover new hard examples from training failures.

Analyzes actual model failures during training to find patterns that could
become new canonical hard examples for the rogues gallery.

The process:
1. Scan flagged_examples/ for model failures
2. Analyze failure patterns to identify common errors
3. Generate candidate hard examples
4. Test candidates against current model
5. Add persistent failures to hard_examples.json

This grows the test suite organically based on real weaknesses.

Usage:
    # Analyze recent failures and suggest candidates
    python3 hard_example_expander.py --analyze

    # Test candidates and add good ones
    python3 hard_example_expander.py --expand

    # Show current hard examples
    python3 hard_example_expander.py --list

Output:
    config/hard_examples.json - Updated hard example set
    status/hard_example_candidates.json - Pending candidates
"""

import argparse
import json
import logging
import random
import re
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HardExampleCandidate:
    """A candidate hard example from failure analysis."""
    prompt: str
    expected: str
    category: str
    source: str  # Where this was found
    failure_count: int  # How many times model failed this
    discovered_at: str
    tested: bool = False
    confirmed_hard: bool = False
    notes: Optional[str] = None


# Pattern categories for classification
LOGIC_PATTERNS = {
    'negation': [
        r'\bno\b.*\bare\b', r'\bnot\b.*\ball\b', r'\bnone\b',
        r'\bnever\b', r'not true that'
    ],
    'quantifier': [
        r'\ball\b.*\bare\b', r'\bsome\b.*\bare\b', r'\bevery\b',
        r'\beach\b', r'\bany\b'
    ],
    'conditional': [
        r'\bif\b.*\bthen\b', r'\bwhenever\b', r'\bwhen\b.*\bthen\b',
        r'\bunless\b'
    ],
    'transitivity': [
        r'all\s+\w+\s+are\s+\w+.*all\s+\w+\s+are\s+\w+',
        r'\w+\s*->\s*\w+.*->'
    ],
    'contradiction': [
        r'consistent', r'contradict', r'possible that',
        r'both.*and'
    ],
    'existential': [
        r'\bsome\b(?!.*\ball\b)', r'\bthere exist',
        r'\bat least one\b'
    ],
}


class HardExampleExpander:
    """Discover and manage hard examples from failures."""

    def __init__(
        self,
        base_dir: str = "/path/to/training",
        api_url: str = "http://192.168.x.x:8765"
    ):
        self.base_dir = Path(base_dir)
        self.api_url = api_url

        # Paths
        self.flagged_dir = self.base_dir / "data" / "flagged_examples"
        self.examples_file = self.base_dir / "config" / "hard_examples.json"
        self.candidates_file = self.base_dir / "status" / "hard_example_candidates.json"

        self.flagged_dir.mkdir(parents=True, exist_ok=True)
        self.examples_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing
        self.hard_examples = self._load_examples()
        self.candidates: List[HardExampleCandidate] = self._load_candidates()

    def _load_examples(self) -> List[Dict]:
        """Load existing hard examples."""
        if self.examples_file.exists():
            with open(self.examples_file) as f:
                return json.load(f)
        return []

    def _save_examples(self):
        """Save hard examples."""
        with open(self.examples_file, 'w') as f:
            json.dump(self.hard_examples, f, indent=2)

    def _load_candidates(self) -> List[HardExampleCandidate]:
        """Load candidate hard examples."""
        if self.candidates_file.exists():
            with open(self.candidates_file) as f:
                data = json.load(f)
                return [HardExampleCandidate(**c) for c in data.get("candidates", [])]
        return []

    def _save_candidates(self):
        """Save candidates."""
        with open(self.candidates_file, 'w') as f:
            json.dump({
                "candidates": [asdict(c) for c in self.candidates],
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)

    def classify_prompt(self, prompt: str) -> str:
        """Classify a prompt into a logic category."""
        prompt_lower = prompt.lower()

        for category, patterns in LOGIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    return category

        return "unknown"

    def extract_expected_answer(self, prompt: str, expected: str) -> str:
        """Normalize expected answer."""
        expected_lower = expected.lower().strip()

        if "cannot" in expected_lower or "not determined" in expected_lower:
            return "Cannot determine"
        elif expected_lower.startswith("yes"):
            return "Yes"
        elif expected_lower.startswith("no"):
            return "No"

        return expected

    def scan_flagged_examples(self) -> List[Dict]:
        """Scan flagged examples directory for failures."""
        failures = []

        for jsonl_file in self.flagged_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file) as f:
                    for line_num, line in enumerate(f):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            # Extract prompt and expected from messages format
                            if "messages" in record:
                                for msg in record["messages"]:
                                    if msg.get("role") == "user":
                                        prompt = msg.get("content", "")
                                    elif msg.get("role") == "assistant":
                                        expected = msg.get("content", "")
                                failures.append({
                                    "prompt": prompt,
                                    "expected": expected,
                                    "source": jsonl_file.name,
                                    "line": line_num
                                })
                            elif "prompt" in record:
                                failures.append({
                                    "prompt": record["prompt"],
                                    "expected": record.get("expected", record.get("answer", "")),
                                    "source": jsonl_file.name,
                                    "line": line_num
                                })
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Error scanning {jsonl_file}: {e}")

        return failures

    def scan_training_errors(self) -> List[Dict]:
        """Scan training data for model prediction errors."""
        errors = []

        # Check self-correction status
        sc_file = self.base_dir / "status" / "self_correction.json"
        if sc_file.exists():
            with open(sc_file) as f:
                data = json.load(f)
                for error in data.get("recent_errors", []):
                    errors.append({
                        "prompt": error.get("prompt", ""),
                        "expected": error.get("expected", ""),
                        "got": error.get("got", ""),
                        "source": "self_correction"
                    })

        return errors

    def analyze_failures(self) -> Dict[str, List[Dict]]:
        """Analyze failures by category."""
        flagged = self.scan_flagged_examples()
        training_errors = self.scan_training_errors()

        all_failures = flagged + training_errors

        # Group by category
        by_category: Dict[str, List[Dict]] = {}
        for failure in all_failures:
            category = self.classify_prompt(failure.get("prompt", ""))
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(failure)

        return by_category

    def generate_candidates(self, max_per_category: int = 3) -> List[HardExampleCandidate]:
        """Generate candidate hard examples from failures."""
        failures_by_category = self.analyze_failures()

        new_candidates = []
        existing_prompts = {e["prompt"].lower() for e in self.hard_examples}
        existing_prompts |= {c.prompt.lower() for c in self.candidates}

        for category, failures in failures_by_category.items():
            if category == "unknown":
                continue

            # Take unique, interesting failures
            seen_prompts = set()
            for failure in failures[:max_per_category * 3]:
                prompt = failure.get("prompt", "").strip()
                expected = failure.get("expected", "").strip()

                if not prompt or not expected:
                    continue

                # Skip if too similar to existing
                prompt_lower = prompt.lower()
                if prompt_lower in existing_prompts or prompt_lower in seen_prompts:
                    continue

                # Skip very short or very long prompts
                if len(prompt) < 30 or len(prompt) > 500:
                    continue

                seen_prompts.add(prompt_lower)

                candidate = HardExampleCandidate(
                    prompt=prompt,
                    expected=self.extract_expected_answer(prompt, expected),
                    category=category,
                    source=failure.get("source", "unknown"),
                    failure_count=1,
                    discovered_at=datetime.now().isoformat()
                )
                new_candidates.append(candidate)

                if len([c for c in new_candidates if c.category == category]) >= max_per_category:
                    break

        return new_candidates

    def test_candidate(self, candidate: HardExampleCandidate) -> Tuple[bool, str]:
        """Test a candidate against the current model."""
        try:
            full_prompt = f"Solve this logic problem. Answer with just Yes, No, or Cannot determine.\n\nProblem: {candidate.prompt}\n\nAnswer:"

            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": full_prompt}],
                    "max_tokens": 50,
                    "temperature": 0.0
                },
                timeout=30
            )

            if not response.ok:
                return False, "API error"

            data = response.json()
            model_answer = data["choices"][0]["message"]["content"].strip().lower()

            # Check if model got it wrong
            expected_lower = candidate.expected.lower()
            if "cannot" in expected_lower:
                correct = "cannot" in model_answer
            elif expected_lower == "yes":
                correct = model_answer.startswith("yes")
            elif expected_lower == "no":
                correct = model_answer.startswith("no") and "cannot" not in model_answer
            else:
                correct = expected_lower in model_answer

            return not correct, model_answer  # Return True if model failed (it's hard)

        except Exception as e:
            return False, str(e)

    def expand_hard_examples(self, max_new: int = 5) -> List[Dict]:
        """Test candidates and add confirmed hard ones."""
        # Generate new candidates if needed
        if len(self.candidates) < max_new:
            new = self.generate_candidates()
            self.candidates.extend(new)
            logger.info(f"Generated {len(new)} new candidates")

        added = []
        tested = 0

        for candidate in self.candidates:
            if candidate.tested:
                continue

            is_hard, model_answer = self.test_candidate(candidate)
            candidate.tested = True
            candidate.confirmed_hard = is_hard
            candidate.notes = f"Model said: {model_answer}"
            tested += 1

            if is_hard:
                # Add to hard examples
                new_example = {
                    "id": f"{candidate.category}_{len(self.hard_examples)}",
                    "prompt": candidate.prompt,
                    "expected": candidate.expected,
                    "category": candidate.category,
                    "difficulty": "hard",
                    "notes": f"Auto-discovered from {candidate.source}"
                }
                self.hard_examples.append(new_example)
                added.append(new_example)
                logger.info(f"Added hard example: {new_example['id']}")

            if len(added) >= max_new:
                break

        # Save results
        self._save_examples()
        self._save_candidates()

        # Clean up tested candidates
        self.candidates = [c for c in self.candidates if not c.tested or c.confirmed_hard]
        self._save_candidates()

        logger.info(f"Tested {tested} candidates, added {len(added)} new hard examples")
        return added

    def get_summary(self) -> Dict:
        """Get summary of hard example expansion."""
        failures_by_cat = self.analyze_failures()

        return {
            "current_hard_examples": len(self.hard_examples),
            "pending_candidates": len([c for c in self.candidates if not c.tested]),
            "confirmed_candidates": len([c for c in self.candidates if c.confirmed_hard]),
            "failures_by_category": {k: len(v) for k, v in failures_by_cat.items()},
            "categories_covered": list({e["category"] for e in self.hard_examples}),
            "last_expansion": self.candidates_file.stat().st_mtime if self.candidates_file.exists() else None
        }


def main():
    parser = argparse.ArgumentParser(description="Hard Example Expander")
    parser.add_argument('--base-dir', default='/path/to/training',
                       help='Base directory')
    parser.add_argument('--api-url', default='http://192.168.x.x:8765',
                       help='Inference API URL')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze failures and generate candidates')
    parser.add_argument('--expand', action='store_true',
                       help='Test candidates and expand hard examples')
    parser.add_argument('--list', action='store_true',
                       help='List current hard examples')
    parser.add_argument('--max-new', type=int, default=5,
                       help='Maximum new examples to add')

    args = parser.parse_args()

    expander = HardExampleExpander(args.base_dir, args.api_url)

    if args.analyze:
        candidates = expander.generate_candidates()
        expander.candidates.extend(candidates)
        expander._save_candidates()
        print(f"Generated {len(candidates)} candidates:")
        for c in candidates:
            print(f"  [{c.category}] {c.prompt[:60]}...")

    elif args.expand:
        added = expander.expand_hard_examples(args.max_new)
        print(f"Added {len(added)} new hard examples:")
        for ex in added:
            print(f"  [{ex['category']}] {ex['id']}")

    elif args.list:
        examples = expander.hard_examples
        print(f"\nHard Examples ({len(examples)}):")
        for ex in examples:
            print(f"  [{ex.get('category', '?')}] {ex['id']}: {ex['expected']}")

    else:
        summary = expander.get_summary()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
