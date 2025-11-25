#!/usr/bin/env python3
"""
Pattern Generator - Generate targeted training data for specific error types.

Analyzes hard example failures and generates training data to correct them.
Each error type has a template-based generator that creates variations.

Error Types Addressed:
- over_confident: Says Yes/No when should be "Cannot determine"
- under_confident: Says "Cannot determine" when answer is clear
- false_positive: Says Yes when answer is No
- false_negative: Says No when answer is Yes

Usage:
    # Generate corrections for all error types found in last eval
    python3 pattern_generator.py --auto

    # Generate specific type
    python3 pattern_generator.py --type over_confident --count 50

    # Queue generated data for training
    python3 pattern_generator.py --auto --queue

Output:
    inbox/corrections_<type>_<timestamp>.jsonl
"""

import argparse
import json
import logging
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example."""
    messages: List[Dict[str, str]]
    category: str
    difficulty: str = "correction"
    source: str = "pattern_generator"


# =============================================================================
# PATTERN TEMPLATES
# =============================================================================

# Categories of entities for variation
ENTITIES = {
    'animals': ['cats', 'dogs', 'birds', 'fish', 'horses', 'cows', 'pigs', 'sheep',
                'lions', 'tigers', 'bears', 'wolves', 'foxes', 'rabbits', 'deer'],
    'people': ['students', 'teachers', 'doctors', 'lawyers', 'artists', 'musicians',
               'engineers', 'scientists', 'athletes', 'writers', 'chefs', 'pilots'],
    'objects': ['books', 'cars', 'houses', 'phones', 'computers', 'tables', 'chairs',
                'tools', 'machines', 'devices', 'instruments', 'vehicles'],
    'abstract': ['ideas', 'theories', 'concepts', 'methods', 'processes', 'systems',
                 'patterns', 'structures', 'forms', 'types', 'kinds', 'classes'],
}

PROPERTIES = {
    'animals': ['fast', 'large', 'small', 'wild', 'domestic', 'nocturnal', 'herbivore',
                'dangerous', 'friendly', 'intelligent', 'social', 'solitary'],
    'people': ['tall', 'skilled', 'experienced', 'certified', 'trained', 'licensed',
               'professional', 'amateur', 'retired', 'active', 'senior', 'junior'],
    'objects': ['expensive', 'heavy', 'portable', 'durable', 'fragile', 'electronic',
                'mechanical', 'digital', 'automatic', 'manual', 'modern', 'antique'],
}

VERBS = {
    'ability': ['can fly', 'can swim', 'can run fast', 'can climb', 'can jump high',
                'can speak', 'can read', 'can write', 'can sing', 'can dance'],
    'state': ['are happy', 'are tired', 'are busy', 'are free', 'are available',
              'are present', 'are absent', 'are active', 'are idle', 'are ready'],
    'action': ['work hard', 'travel often', 'study daily', 'exercise regularly',
               'eat meat', 'sleep late', 'wake early', 'read books', 'watch TV'],
}


def random_entity(category: str = None) -> str:
    """Get a random entity."""
    if category and category in ENTITIES:
        return random.choice(ENTITIES[category])
    return random.choice(random.choice(list(ENTITIES.values())))


def random_property(category: str = 'animals') -> str:
    """Get a random property."""
    if category in PROPERTIES:
        return random.choice(PROPERTIES[category])
    return random.choice(PROPERTIES['animals'])


def random_verb(category: str = 'ability') -> str:
    """Get a random verb phrase."""
    if category in VERBS:
        return random.choice(VERBS[category])
    return random.choice(random.choice(list(VERBS.values())))


def random_name() -> str:
    """Generate a random name."""
    names = ['Alex', 'Sam', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Quinn',
             'Avery', 'Parker', 'Skyler', 'Dakota', 'Reese', 'Finley', 'Rowan']
    return random.choice(names)


# =============================================================================
# OVER-CONFIDENCE GENERATORS
# =============================================================================

def gen_existential_trap() -> TrainingExample:
    """
    Generate "Some X are Y. Z is X. Is Z Y?" problems.
    Correct answer: Cannot determine (some != all)
    """
    group = random_entity()
    subgroup = random_entity()
    property_ = random_verb()
    individual = random_name()

    # Ensure variety in phrasing
    templates = [
        f"Some {group} {property_}. {individual} is a {group[:-1] if group.endswith('s') else group}. Does {individual} {property_.replace('can ', '').replace('are ', 'be ')}?",
        f"Some {group} {property_}. All {subgroup} are {group}. Do all {subgroup} {property_.replace('can ', '').replace('are ', '')}?",
        f"A few {group} {property_}. {individual} is among the {group}. Can we conclude that {individual} {property_.replace('can ', '').replace('are ', '')}?",
    ]

    prompt = random.choice(templates)

    # Build reasoning
    reasoning = f"""Let me analyze this step by step.

The premise states that SOME {group} {property_} - this means at least one, but not necessarily all.

We know {individual if 'individual' in prompt.lower() or individual in prompt else subgroup} is/are {group}.

However, "some" does not mean "all". Without knowing whether this specific case is among those that {property_}, we cannot determine the answer.

Answer: Cannot determine"""

    return TrainingExample(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning}
        ],
        category="existential_trap"
    )


def gen_negation_chain() -> TrainingExample:
    """
    Generate "No X are Y. No Y are Z. Are any X Z?" problems.
    Correct answer: Cannot determine (negations don't chain)
    """
    A = random_entity()
    B = random_entity()
    C = random_entity()

    # Ensure distinct
    while B == A:
        B = random_entity()
    while C == A or C == B:
        C = random_entity()

    templates = [
        f"No {A} are {B}. No {B} are {C}. Are any {A} also {C}?",
        f"None of the {A} are {B}. None of the {B} are {C}. Can any {A} be {C}?",
        f"{A.capitalize()} are never {B}. {B.capitalize()} are never {C}. Does this mean {A} are never {C}?",
    ]

    prompt = random.choice(templates)

    reasoning = f"""Let me think through this carefully.

Premise 1: No {A} are {B}
Premise 2: No {B} are {C}

A common mistake is to assume negation chains transitively, but it doesn't.

Consider: No cats are dogs. No dogs are fish. This does NOT tell us whether cats can be fish - they could be or couldn't be, independent of dogs.

The relationship between {A} and {C} is not determined by their separate relationships to {B}.

Answer: Cannot determine"""

    return TrainingExample(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning}
        ],
        category="negation_chain"
    )


def gen_vacuous_truth() -> TrainingExample:
    """
    Generate "All X are Y. Are there any X?" problems.
    Correct answer: Cannot determine (statement about empty set)
    """
    mythical = random.choice(['unicorns', 'dragons', 'phoenixes', 'griffins', 'mermaids',
                               'centaurs', 'pegasi', 'basilisks', 'hydras', 'chimeras'])
    property_ = random_property()

    templates = [
        f"All {mythical} are {property_}. Are there any {mythical}?",
        f"Every {mythical[:-1] if mythical.endswith('s') else mythical} is {property_}. Does this mean {mythical} exist?",
        f"If something is a {mythical[:-1] if mythical.endswith('s') else mythical}, it must be {property_}. Can we conclude {mythical} are real?",
    ]

    prompt = random.choice(templates)

    reasoning = f"""This is a question about vacuous truth.

The statement "All {mythical} are {property_}" can be true even if no {mythical} exist. In logic, a universal statement about an empty set is considered vacuously true.

For example, "All unicorns are purple" is technically true if no unicorns exist, because there are no counterexamples.

The statement tells us a property that {mythical} would have IF they existed, but it says nothing about whether they actually exist.

Answer: Cannot determine"""

    return TrainingExample(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning}
        ],
        category="vacuous_truth"
    )


def gen_conditional_existence() -> TrainingExample:
    """
    Generate "If X exists, then Y. Does Y exist?" problems.
    Correct answer: Cannot determine
    """
    A = random_entity()
    B = random_entity()

    templates = [
        f"If there are any {A} in this room, then there must be {B} too. Are there {B} in this room?",
        f"Whenever {A} are present, {B} are also present. Are {B} present?",
        f"The presence of {A} implies the presence of {B}. Can we conclude {B} are here?",
    ]

    prompt = random.choice(templates)

    reasoning = f"""Let me analyze the logical structure.

We have a conditional: IF {A} exist THEN {B} exist.

However, we're not told whether the antecedent ({A} existing) is true.

If {A} don't exist, the conditional tells us nothing about {B}.
If {A} do exist, then yes, {B} would exist too.

Without knowing whether {A} are present, we cannot determine whether {B} are present.

Answer: Cannot determine"""

    return TrainingExample(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning}
        ],
        category="conditional_existence"
    )


# =============================================================================
# FALSE NEGATIVE/POSITIVE GENERATORS
# =============================================================================

def gen_modus_tollens() -> TrainingExample:
    """
    Generate modus tollens problems.
    If P then Q. Not Q. Therefore not P.
    """
    conditions = [
        ("it rains", "the ground is wet"),
        ("you study", "you pass"),
        ("you exercise", "you get stronger"),
        ("you practice", "you improve"),
        ("the alarm rings", "people leave"),
        ("the light is on", "the room is bright"),
    ]

    P, Q = random.choice(conditions)

    templates = [
        f"If {P}, then {Q}. {Q.replace('is', 'is not').replace('are', 'are not').capitalize() if 'is' in Q or 'are' in Q else 'Not: ' + Q}. Is it true that {P}?",
        f"Whenever {P}, {Q}. But {Q.replace('is', 'is not').replace('are', 'are not') if 'is' in Q or 'are' in Q else Q + ' is not happening'}. Can we conclude {P}?",
    ]

    prompt = templates[0]  # Use simpler form
    prompt = f"If {P}, then {Q}. The {Q.split()[-1] if len(Q.split()) > 2 else Q.split()[0]} is not {Q.split()[-1] if 'is' in Q else 'happening'}. Is {P.split()[0]} {P.split()[1] if len(P.split()) > 1 else ''}?"

    # Simplified version
    prompt = f"If {P}, {Q}. {Q.split()[0].capitalize()} {Q.split()[1]} not {' '.join(Q.split()[2:])}. Is it the case that {P}?"

    reasoning = f"""This is modus tollens - a valid logical inference.

Structure:
- If P then Q
- Not Q
- Therefore: Not P

Given: If {P}, then {Q}
Given: {Q} is not the case

By modus tollens: {P} is not the case.

Answer: No"""

    return TrainingExample(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning}
        ],
        category="modus_tollens"
    )


def gen_transitivity() -> TrainingExample:
    """
    Generate transitivity chain problems.
    All A are B. All B are C. Therefore all A are C.
    """
    letters = random.sample(['A', 'B', 'C', 'D', 'E', 'F'], 4)
    chain_length = random.choice([3, 4])

    premises = []
    for i in range(chain_length - 1):
        premises.append(f"All {letters[i]} are {letters[i+1]}")

    prompt = ". ".join(premises) + f". Are all {letters[0]} also {letters[chain_length-1]}?"

    chain_explanation = " -> ".join(letters[:chain_length])

    reasoning = f"""This is a transitivity chain.

Given premises:
{chr(10).join('- ' + p for p in premises)}

Transitivity of "All X are Y":
{chain_explanation}

Following the chain: Since all {letters[0]} are {letters[1]}, and all {letters[1]} are {letters[2]}, all {letters[0]} must be {letters[2]}.
{f"Continuing: all {letters[0]} are also {letters[3]}." if chain_length > 3 else ""}

Answer: Yes"""

    return TrainingExample(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning}
        ],
        category="transitivity"
    )


def gen_contradiction() -> TrainingExample:
    """
    Generate contradiction detection problems.
    """
    group = random_entity()
    property_ = random_property()
    individual = random_name()

    templates = [
        (f"All {group} are {property_}. This {group[:-1] if group.endswith('s') else group} is not {property_}. Is this consistent?",
         f"We have a universal claim that ALL {group} are {property_}, but a specific instance that is not {property_}. This is a direct contradiction."),
        (f"No {group} are {property_}. {individual}'s {group[:-1] if group.endswith('s') else group} is {property_}. Is this logically consistent?",
         f"The universal claim says NO {group} are {property_}, but we have an example of one that is. This is a contradiction."),
    ]

    prompt, explanation = random.choice(templates)

    reasoning = f"""Let me check for logical consistency.

{explanation}

A statement and its negation cannot both be true. This is the law of non-contradiction.

Answer: No"""

    return TrainingExample(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning}
        ],
        category="contradiction"
    )


def gen_double_negation() -> TrainingExample:
    """
    Generate double negation problems.
    It is not true that no X are Y = Some X are Y.
    """
    group = random_entity()
    property_ = random_verb()

    templates = [
        f"It is not true that no {group} {property_}. Do some {group} {property_.replace('can ', '').replace('are ', '')}?",
        f"It is false that none of the {group} {property_}. {property_.replace('can ', 'Can any ').replace('are ', 'Are any ')}{group}?",
        f"The claim 'no {group} {property_}' is false. Does this mean some {group} {property_.replace('can ', '').replace('are ', '')}?",
    ]

    prompt = random.choice(templates)

    reasoning = f"""This involves double negation.

"It is not true that no {group} {property_}"

Breaking it down:
- "No {group} {property_}" means zero {group} have this property
- "It is not true that..." negates this
- So: It's false that zero {group} {property_}
- Therefore: At least one {group[:-1] if group.endswith('s') else group} does {property_.replace('can ', '').replace('are ', '')}

Double negation: NOT(NO X) = SOME X

Answer: Yes"""

    return TrainingExample(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning}
        ],
        category="double_negation"
    )


# =============================================================================
# GENERATOR REGISTRY
# =============================================================================

GENERATORS = {
    # Over-confidence corrections (answer should be "Cannot determine")
    'over_confident': [
        gen_existential_trap,
        gen_negation_chain,
        gen_vacuous_truth,
        gen_conditional_existence,
    ],
    # False negative corrections (answer should be "Yes")
    'false_negative': [
        gen_transitivity,
        gen_double_negation,
    ],
    # False positive corrections (answer should be "No")
    'false_positive': [
        gen_modus_tollens,
        gen_contradiction,
    ],
}

# Map error types from hard_example_tracker to generator categories
# This handles the wrong_* errors from classify_error()
ERROR_TYPE_MAP = {
    # Direct mappings
    'over_confident': 'over_confident',
    'under_confident': 'over_confident',  # Train on "Cannot determine" cases
    'false_negative': 'false_negative',
    'false_positive': 'false_positive',
    # Category-specific mappings (from hard_example_tracker.classify_error)
    'wrong_negation': 'over_confident',           # Negation errors → need "Cannot determine"
    'wrong_double_negation': 'false_negative',    # Should be "Yes"
    'wrong_quantifier': 'over_confident',         # Quantifier scope → "Cannot determine"
    'wrong_modus_tollens': 'false_positive',      # Should be "No"
    'wrong_transitivity': 'false_negative',       # Should be "Yes"
    'wrong_contradiction': 'false_positive',      # Should be "No"
    'wrong_existential': 'over_confident',        # Should be "Cannot determine"
    'wrong_negation_chain': 'over_confident',     # Should be "Cannot determine"
    'wrong_implicit_quantifier': 'false_negative', # Should be "Yes"
    'wrong_vacuous': 'over_confident',            # Should be "Cannot determine"
    'api_error': None,                            # Skip these
    'unknown': None,                              # Skip these
}


class PatternGenerator:
    """Generate targeted training data for error types."""

    def __init__(self, base_dir: str = "/path/to/training"):
        self.base_dir = Path(base_dir)
        self.inbox_dir = self.base_dir / "inbox"
        self.status_dir = self.base_dir / "status"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    def get_error_distribution(self) -> Dict[str, int]:
        """Get error distribution from hard example board."""
        board_file = self.status_dir / "hard_example_board.json"

        if not board_file.exists():
            return {}

        with open(board_file) as f:
            data = json.load(f)

        entries = data.get("entries", [])
        if not entries:
            return {}

        # Get most recent entry
        latest = entries[-1]
        return latest.get("error_types", {})

    def generate_for_type(
        self,
        error_type: str,
        count: int = 50
    ) -> List[TrainingExample]:
        """Generate training examples for a specific error type."""
        if error_type not in GENERATORS:
            logger.warning(f"Unknown error type: {error_type}")
            return []

        generators = GENERATORS[error_type]
        examples = []

        for _ in range(count):
            generator = random.choice(generators)
            try:
                example = generator()
                examples.append(example)
            except Exception as e:
                logger.warning(f"Generator failed: {e}")

        return examples

    def generate_auto(self, per_error: int = 30) -> Dict[str, List[TrainingExample]]:
        """
        Auto-generate based on recent error distribution.
        Generates more examples for more frequent errors.
        Maps error types from hard_example_tracker to generator categories.
        """
        error_dist = self.get_error_distribution()

        if not error_dist:
            logger.info("No error distribution found, generating balanced set")
            error_dist = {k: 1 for k in GENERATORS.keys()}

        # Aggregate by generator category
        category_counts: Dict[str, int] = {}
        unmapped_errors: List[str] = []

        for error_type, count in error_dist.items():
            # Use ERROR_TYPE_MAP for proper mapping
            if error_type in ERROR_TYPE_MAP:
                mapped = ERROR_TYPE_MAP[error_type]
                if mapped is None:
                    continue  # Skip api_error, unknown, etc.
            elif error_type in GENERATORS:
                mapped = error_type
            else:
                # Try fallback heuristics for unknown error types
                if 'over' in error_type.lower() or 'confident' in error_type.lower():
                    mapped = 'over_confident'
                elif 'wrong' in error_type.lower():
                    # Most "wrong_*" errors that aren't mapped likely need more training
                    mapped = 'over_confident'  # Default to "Cannot determine" training
                else:
                    unmapped_errors.append(error_type)
                    continue

            category_counts[mapped] = category_counts.get(mapped, 0) + count

        if unmapped_errors:
            logger.warning(f"Unmapped error types (skipped): {unmapped_errors}")

        # Generate examples for each category
        all_examples = {}
        total_errors = sum(category_counts.values())

        for category, count in category_counts.items():
            # Scale count by frequency - more errors = more training data
            scaled_count = max(per_error, int(per_error * count / max(total_errors, 1) * 3))

            logger.info(f"Generating {scaled_count} examples for {category} (from {count} errors)")
            examples = self.generate_for_type(category, scaled_count)
            if examples:
                all_examples[category] = examples

        return all_examples

    def save_to_inbox(
        self,
        examples: List[TrainingExample],
        error_type: str = "mixed"
    ) -> Path:
        """Save examples to inbox as JSONL."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"corrections_{error_type}_{timestamp}.jsonl"
        filepath = self.inbox_dir / filename

        with open(filepath, 'w') as f:
            for ex in examples:
                record = {
                    "messages": ex.messages,
                    "metadata": {
                        "category": ex.category,
                        "difficulty": ex.difficulty,
                        "source": ex.source,
                    }
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Saved {len(examples)} examples to {filepath}")
        return filepath

    def generate_and_queue(self, per_error: int = 30) -> Dict[str, Path]:
        """Generate examples and queue them for training."""
        all_examples = self.generate_auto(per_error)

        output_files = {}
        for error_type, examples in all_examples.items():
            if examples:
                filepath = self.save_to_inbox(examples, error_type)
                output_files[error_type] = filepath

        return output_files


def main():
    parser = argparse.ArgumentParser(description="Pattern Generator")
    parser.add_argument('--base-dir', default='/path/to/training',
                       help='Base directory')
    parser.add_argument('--type', type=str, choices=list(GENERATORS.keys()),
                       help='Error type to generate for')
    parser.add_argument('--count', type=int, default=50,
                       help='Number of examples to generate')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-generate based on error distribution')
    parser.add_argument('--queue', action='store_true',
                       help='Queue generated data for training')
    parser.add_argument('--preview', action='store_true',
                       help='Preview examples without saving')

    args = parser.parse_args()

    generator = PatternGenerator(args.base_dir)

    if args.auto:
        if args.queue:
            files = generator.generate_and_queue(args.count)
            print(f"Generated and queued {len(files)} files:")
            for error_type, path in files.items():
                print(f"  {error_type}: {path}")
        else:
            all_examples = generator.generate_auto(args.count)
            total = sum(len(ex) for ex in all_examples.values())
            print(f"Generated {total} examples:")
            for error_type, examples in all_examples.items():
                print(f"  {error_type}: {len(examples)}")
                if args.preview and examples:
                    print(f"    Sample: {examples[0].messages[0]['content'][:80]}...")

    elif args.type:
        examples = generator.generate_for_type(args.type, args.count)

        if args.preview:
            for ex in examples[:3]:
                print(f"\n--- {ex.category} ---")
                print(f"Q: {ex.messages[0]['content']}")
                print(f"A: {ex.messages[1]['content'][:200]}...")
        elif args.queue:
            filepath = generator.save_to_inbox(examples, args.type)
            print(f"Queued {len(examples)} examples: {filepath}")
        else:
            print(f"Generated {len(examples)} examples for {args.type}")
            print("Use --queue to save, --preview to see samples")

    else:
        print("Available error types:", list(GENERATORS.keys()))
        print("\nUsage:")
        print("  --auto            Generate based on recent errors")
        print("  --type TYPE       Generate for specific error type")
        print("  --count N         Number of examples (default: 50)")
        print("  --queue           Save to inbox for training")
        print("  --preview         Show sample examples")


if __name__ == "__main__":
    main()
