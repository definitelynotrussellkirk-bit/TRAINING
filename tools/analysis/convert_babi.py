#!/usr/bin/env python3
"""
Convert bAbI dataset from Facebook's raw format to our validation format.

bAbI format:
  1 Mary moved to the bathroom.
  2 John went to the hallway.
  3 Where is Mary?	bathroom	1

Each line is either:
- A numbered fact (no tab)
- A question with answer and supporting fact IDs (has tab)

Usage:
    python3 convert_babi.py /tmp/tasks_1-20_v1-2/en/ --output data/validation/benchmarks/
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

TASK_INFO = {
    "qa1": {"name": "single-supporting-fact", "difficulty": "easy", "description": "Single supporting fact"},
    "qa2": {"name": "two-supporting-facts", "difficulty": "medium", "description": "Two supporting facts"},
    "qa3": {"name": "three-supporting-facts", "difficulty": "hard", "description": "Three supporting facts"},
    "qa4": {"name": "two-arg-relations", "difficulty": "medium", "description": "Two argument relations"},
    "qa5": {"name": "three-arg-relations", "difficulty": "hard", "description": "Three argument relations"},
    "qa6": {"name": "yes-no-questions", "difficulty": "easy", "description": "Yes/No questions"},
    "qa7": {"name": "counting", "difficulty": "medium", "description": "Counting"},
    "qa8": {"name": "lists-sets", "difficulty": "medium", "description": "Lists/Sets"},
    "qa9": {"name": "simple-negation", "difficulty": "easy", "description": "Simple negation"},
    "qa10": {"name": "indefinite-knowledge", "difficulty": "medium", "description": "Indefinite knowledge"},
    "qa11": {"name": "basic-coreference", "difficulty": "easy", "description": "Basic coreference"},
    "qa12": {"name": "conjunction", "difficulty": "medium", "description": "Conjunction"},
    "qa13": {"name": "compound-coreference", "difficulty": "medium", "description": "Compound coreference"},
    "qa14": {"name": "time-reasoning", "difficulty": "hard", "description": "Time reasoning"},
    "qa15": {"name": "basic-deduction", "difficulty": "medium", "description": "Basic deduction"},
    "qa16": {"name": "basic-induction", "difficulty": "hard", "description": "Basic induction"},
    "qa17": {"name": "positional-reasoning", "difficulty": "medium", "description": "Positional reasoning"},
    "qa18": {"name": "size-reasoning", "difficulty": "medium", "description": "Size reasoning"},
    "qa19": {"name": "path-finding", "difficulty": "hard", "description": "Path finding"},
    "qa20": {"name": "agents-motivations", "difficulty": "hard", "description": "Agent's motivations"},
}


def parse_babi_file(filepath: Path) -> List[Dict]:
    """Parse a bAbI task file into QA examples."""
    examples = []
    current_context = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse line number and content
            match = re.match(r'^(\d+)\s+(.+)$', line)
            if not match:
                continue

            line_num = int(match.group(1))
            content = match.group(2)

            # Line num 1 starts a new story
            if line_num == 1:
                current_context = []

            # Check if this is a question (contains tab)
            if '\t' in content:
                parts = content.split('\t')
                question = parts[0]
                answer = parts[1]
                # supporting_facts = parts[2] if len(parts) > 2 else ""

                # Create example
                context = " ".join(current_context)
                examples.append({
                    "context": context,
                    "question": question,
                    "answer": answer
                })
            else:
                # This is a fact
                current_context.append(content)

    return examples


def convert_to_validation_format(examples: List[Dict], task_id: str, max_examples: int = 100) -> List[Dict]:
    """Convert parsed examples to our validation format."""
    task_info = TASK_INFO.get(task_id, {"difficulty": "medium", "description": task_id})

    import random
    if len(examples) > max_examples:
        examples = random.sample(examples, max_examples)

    converted = []
    for ex in examples:
        prompt = f"""Context: {ex['context']}

Question: {ex['question']}

Answer with just the answer word(s), nothing else."""

        converted.append({
            "skill": f"babi_{task_id}",
            "difficulty": task_info["difficulty"],
            "user_prompt": prompt,
            "expected_answer": ex["answer"],
            "metadata": {
                "source": "babi",
                "task": task_id,
                "task_description": task_info["description"]
            }
        })

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert bAbI to validation format")
    parser.add_argument("babi_dir", help="Path to bAbI en/ directory")
    parser.add_argument("--output", default="/path/to/training/data/validation/benchmarks",
                        help="Output directory")
    parser.add_argument("--max-examples", type=int, default=100, help="Max examples per task")
    parser.add_argument("--tasks", nargs="+", help="Specific tasks to convert (e.g., qa1 qa7)")

    args = parser.parse_args()
    babi_dir = Path(args.babi_dir)
    output_dir = Path(args.output)

    # Determine which tasks to process
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = list(TASK_INFO.keys())

    total_examples = 0

    for task_id in tasks:
        # Find test file
        test_files = list(babi_dir.glob(f"{task_id}_*_test.txt"))
        if not test_files:
            print(f"No test file found for {task_id}")
            continue

        test_file = test_files[0]
        print(f"Processing {task_id}: {test_file.name}")

        # Parse and convert
        examples = parse_babi_file(test_file)
        converted = convert_to_validation_format(examples, task_id, args.max_examples)

        if not converted:
            print(f"  No examples converted for {task_id}")
            continue

        # Save
        task_dir = output_dir / f"babi_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        difficulty = TASK_INFO.get(task_id, {}).get("difficulty", "medium")
        filepath = task_dir / f"val_babi_{task_id}_{difficulty}_{len(converted)}.jsonl"

        with open(filepath, 'w') as f:
            for ex in converted:
                f.write(json.dumps(ex) + '\n')

        print(f"  Saved {len(converted)} examples to {filepath.name}")
        total_examples += len(converted)

    print(f"\nTotal: {total_examples} examples across {len(tasks)} tasks")


if __name__ == "__main__":
    main()
