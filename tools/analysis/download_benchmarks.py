#!/usr/bin/env python3
"""
Download and convert standard benchmarks to our validation format.

Downloads relevant subsets from:
- BIG-Bench (Google): Diverse language tasks
- bAbI (Facebook): Reasoning tasks
- BIG-Bench Hard: Challenging subset

Converts to our standard format:
{
    "skill": "benchmark_name",
    "difficulty": "easy|medium|hard",
    "user_prompt": "...",
    "expected_answer": "...",
    "metadata": {"source": "bigbench", "task": "..."}
}

Usage:
    python3 download_benchmarks.py --all
    python3 download_benchmarks.py --bigbench
    python3 download_benchmarks.py --babi
    python3 download_benchmarks.py --list
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# TASK DEFINITIONS - Which tasks to download and how to process them
# =============================================================================

# BIG-Bench tasks relevant for transfer testing
BIGBENCH_TASKS = {
    # Arithmetic & Math
    "simple_arithmetic": {
        "description": "Basic arithmetic operations",
        "relevance": "Tests if counting training helps math",
        "difficulty_map": {"simple": "easy", "medium": "medium", "hard": "hard"},
    },
    "elementary_math_qa": {
        "description": "Elementary math word problems",
        "relevance": "Math reasoning transfer",
    },
    "multistep_arithmetic": {
        "description": "Multi-step arithmetic",
        "relevance": "Complex counting/math transfer",
    },

    # String & Sequence
    "word_sorting": {
        "description": "Sort words alphabetically",
        "relevance": "String manipulation transfer",
    },
    "word_unscrambling": {
        "description": "Unscramble letters to form words",
        "relevance": "Letter manipulation transfer",
    },

    # Logic & Reasoning
    "logical_deduction": {
        "description": "Deductive reasoning",
        "relevance": "General reasoning transfer",
    },
    "tracking_shuffled_objects": {
        "description": "Track object positions through shuffles",
        "relevance": "Sequence tracking transfer",
    },
    "boolean_expressions": {
        "description": "Evaluate boolean expressions",
        "relevance": "Logic operations transfer",
    },

    # Pattern & Sequence
    "date_understanding": {
        "description": "Date parsing and reasoning",
        "relevance": "Structured data understanding",
    },
    "navigate": {
        "description": "Follow navigation instructions",
        "relevance": "Sequential instruction following",
    },
}

# bAbI tasks - 20 reasoning tasks
BABI_TASKS = {
    "qa1": {"description": "Single supporting fact", "difficulty": "easy"},
    "qa2": {"description": "Two supporting facts", "difficulty": "medium"},
    "qa3": {"description": "Three supporting facts", "difficulty": "hard"},
    "qa4": {"description": "Two argument relations", "difficulty": "medium"},
    "qa5": {"description": "Three argument relations", "difficulty": "hard"},
    "qa6": {"description": "Yes/No questions", "difficulty": "easy"},
    "qa7": {"description": "Counting", "difficulty": "medium"},  # Very relevant!
    "qa8": {"description": "Lists/Sets", "difficulty": "medium"},  # Relevant!
    "qa9": {"description": "Simple negation", "difficulty": "easy"},
    "qa10": {"description": "Indefinite knowledge", "difficulty": "medium"},
    "qa11": {"description": "Basic coreference", "difficulty": "easy"},
    "qa12": {"description": "Conjunction", "difficulty": "medium"},
    "qa13": {"description": "Compound coreference", "difficulty": "medium"},
    "qa14": {"description": "Time reasoning", "difficulty": "hard"},
    "qa15": {"description": "Basic deduction", "difficulty": "medium"},
    "qa16": {"description": "Basic induction", "difficulty": "hard"},
    "qa17": {"description": "Positional reasoning", "difficulty": "medium"},
    "qa18": {"description": "Size reasoning", "difficulty": "medium"},
    "qa19": {"description": "Path finding", "difficulty": "hard"},
    "qa20": {"description": "Agent's motivations", "difficulty": "hard"},
}


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_bigbench_task(task_name: str, max_examples: int = 200) -> List[Dict]:
    """Download a BIG-Bench task from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return []

    logger.info(f"Downloading BIG-Bench task: {task_name}")

    try:
        # Try tasksource version first (easier dependencies)
        dataset = load_dataset("tasksource/bigbench", task_name, trust_remote_code=True)
    except Exception as e1:
        try:
            # Fall back to official version
            dataset = load_dataset("google/bigbench", task_name, trust_remote_code=True)
        except Exception as e2:
            logger.warning(f"Could not load {task_name}: {e1} / {e2}")
            return []

    examples = []

    # Get available split
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    data = dataset[split_name]

    # Sample if too large
    indices = list(range(len(data)))
    if len(indices) > max_examples:
        indices = random.sample(indices, max_examples)

    for idx in indices:
        item = data[idx]
        example = convert_bigbench_example(item, task_name)
        if example:
            examples.append(example)

    logger.info(f"  Downloaded {len(examples)} examples from {task_name}")
    return examples


def download_babi_task(task_num: str, max_examples: int = 200) -> List[Dict]:
    """Download a bAbI task from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return []

    logger.info(f"Downloading bAbI task: {task_num}")

    try:
        dataset = load_dataset("facebook/babi_qa", f"en-10k-{task_num}", trust_remote_code=True)
    except Exception as e:
        try:
            # Try alternative format
            dataset = load_dataset("facebook/babi_qa", f"en-{task_num}", trust_remote_code=True)
        except Exception as e2:
            logger.warning(f"Could not load bAbI {task_num}: {e} / {e2}")
            return []

    examples = []

    # Get test split preferably
    split_name = "test" if "test" in dataset else "train"
    data = dataset[split_name]

    # Sample if too large
    indices = list(range(len(data)))
    if len(indices) > max_examples:
        indices = random.sample(indices, max_examples)

    for idx in indices:
        item = data[idx]
        example = convert_babi_example(item, task_num)
        if example:
            examples.append(example)

    logger.info(f"  Downloaded {len(examples)} examples from bAbI {task_num}")
    return examples


def download_bigbench_hard(max_examples: int = 200) -> Dict[str, List[Dict]]:
    """Download BIG-Bench Hard tasks."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return {}

    logger.info("Downloading BIG-Bench Hard...")

    # BBH tasks
    bbh_tasks = [
        "boolean_expressions",
        "date_understanding",
        "logical_deduction_three_objects",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "multistep_arithmetic_two",
        "navigate",
        "tracking_shuffled_objects_three_objects",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "word_sorting",
    ]

    all_examples = {}

    for task in bbh_tasks:
        try:
            dataset = load_dataset("maveriq/bigbenchhard", task, trust_remote_code=True)
            split_name = "train" if "train" in dataset else list(dataset.keys())[0]
            data = dataset[split_name]

            examples = []
            indices = list(range(len(data)))
            if len(indices) > max_examples:
                indices = random.sample(indices, max_examples)

            for idx in indices:
                item = data[idx]
                example = convert_bbh_example(item, task)
                if example:
                    examples.append(example)

            if examples:
                all_examples[task] = examples
                logger.info(f"  {task}: {len(examples)} examples")

        except Exception as e:
            logger.warning(f"Could not load BBH {task}: {e}")

    return all_examples


# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================

def convert_bigbench_example(item: Dict, task_name: str) -> Optional[Dict]:
    """Convert a BIG-Bench example to our format."""
    try:
        # BIG-Bench format varies, but typically has:
        # - inputs: the question/prompt
        # - targets: the expected answer(s)
        # - multiple_choice_targets: for MC questions

        if "inputs" in item:
            prompt = item["inputs"]
        elif "input" in item:
            prompt = item["input"]
        elif "question" in item:
            prompt = item["question"]
        else:
            return None

        if "targets" in item:
            targets = item["targets"]
            if isinstance(targets, list):
                answer = targets[0] if targets else ""
            else:
                answer = str(targets)
        elif "target" in item:
            answer = str(item["target"])
        elif "answer" in item:
            answer = str(item["answer"])
        else:
            return None

        # Determine difficulty based on task or content length
        difficulty = estimate_difficulty(prompt, answer, task_name)

        return {
            "skill": f"bb_{task_name}",
            "difficulty": difficulty,
            "user_prompt": prompt.strip(),
            "expected_answer": answer.strip(),
            "metadata": {
                "source": "bigbench",
                "task": task_name,
                "original": {k: str(v)[:100] for k, v in item.items() if k not in ["inputs", "targets"]}
            }
        }
    except Exception as e:
        logger.debug(f"Could not convert example: {e}")
        return None


def convert_babi_example(item: Dict, task_num: str) -> Optional[Dict]:
    """Convert a bAbI example to our format."""
    try:
        # bAbI format:
        # - story: context/story text (may be list of sentences)
        # - question: the question
        # - answer: the answer

        story = item.get("story", item.get("passage", ""))
        if isinstance(story, list):
            story = " ".join(story)
        elif isinstance(story, dict):
            # Some versions have nested structure
            if "text" in story:
                if isinstance(story["text"], list):
                    story = " ".join(story["text"])
                else:
                    story = story["text"]

        question = item.get("question", "")
        answer = item.get("answer", "")

        if not story or not question or not answer:
            return None

        # Format as a single prompt
        prompt = f"Context: {story}\n\nQuestion: {question}\n\nAnswer with just the answer, nothing else."

        # Get difficulty from our mapping
        difficulty = BABI_TASKS.get(task_num, {}).get("difficulty", "medium")

        return {
            "skill": f"babi_{task_num}",
            "difficulty": difficulty,
            "user_prompt": prompt.strip(),
            "expected_answer": str(answer).strip(),
            "metadata": {
                "source": "babi",
                "task": task_num,
                "task_description": BABI_TASKS.get(task_num, {}).get("description", ""),
            }
        }
    except Exception as e:
        logger.debug(f"Could not convert bAbI example: {e}")
        return None


def convert_bbh_example(item: Dict, task_name: str) -> Optional[Dict]:
    """Convert a BIG-Bench Hard example to our format."""
    try:
        prompt = item.get("input", item.get("inputs", ""))
        answer = item.get("target", item.get("targets", ""))

        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        if not prompt or not answer:
            return None

        # BBH is all hard by definition
        difficulty = "hard"

        # Simplify task name
        simple_name = task_name.replace("_three_objects", "").replace("_five_objects", "").replace("_seven_objects", "")
        simple_name = simple_name.replace("_two", "")

        return {
            "skill": f"bbh_{simple_name}",
            "difficulty": difficulty,
            "user_prompt": prompt.strip(),
            "expected_answer": str(answer).strip(),
            "metadata": {
                "source": "bigbench_hard",
                "task": task_name,
            }
        }
    except Exception as e:
        logger.debug(f"Could not convert BBH example: {e}")
        return None


def estimate_difficulty(prompt: str, answer: str, task_name: str) -> str:
    """Estimate difficulty based on content."""
    # Simple heuristics
    prompt_len = len(prompt)
    answer_len = len(answer)

    # Check for numbers (more = harder for arithmetic)
    num_count = len(re.findall(r'\d+', prompt))

    if prompt_len < 50 and num_count <= 2:
        return "easy"
    elif prompt_len < 150 and num_count <= 4:
        return "medium"
    else:
        return "hard"


# =============================================================================
# SAVE FUNCTIONS
# =============================================================================

def save_benchmark_data(examples: List[Dict], skill_name: str, base_dir: Path):
    """Save benchmark examples to validation directory."""
    if not examples:
        return

    # Group by difficulty
    by_difficulty = {"easy": [], "medium": [], "hard": []}
    for ex in examples:
        diff = ex.get("difficulty", "medium")
        if diff in by_difficulty:
            by_difficulty[diff].append(ex)
        else:
            by_difficulty["medium"].append(ex)

    # Create directory
    skill_dir = base_dir / "data" / "validation" / "benchmarks" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Save each difficulty
    total_saved = 0
    for difficulty, diff_examples in by_difficulty.items():
        if not diff_examples:
            continue

        filepath = skill_dir / f"val_{skill_name}_{difficulty}_{len(diff_examples)}.jsonl"
        with open(filepath, 'w') as f:
            for ex in diff_examples:
                f.write(json.dumps(ex) + '\n')

        total_saved += len(diff_examples)
        logger.info(f"  Saved {len(diff_examples)} {difficulty} examples to {filepath.name}")

    return total_saved


# =============================================================================
# COMPARISON FUNCTIONS (for use in baseline runner)
# =============================================================================

def compare_benchmark(model_output: str, expected: str, skill_name: str) -> bool:
    """Compare model output to expected answer for benchmark tasks."""
    # Clean outputs
    model_clean = clean_answer(model_output)
    expected_clean = clean_answer(expected)

    # Exact match
    if model_clean == expected_clean:
        return True

    # Case-insensitive match
    if model_clean.lower() == expected_clean.lower():
        return True

    # Check if expected is contained in model output
    if expected_clean.lower() in model_clean.lower():
        return True

    # For yes/no questions
    if expected_clean.lower() in ["yes", "no", "true", "false"]:
        if expected_clean.lower() in model_clean.lower():
            return True

    # For numeric answers
    try:
        model_num = extract_number(model_clean)
        expected_num = extract_number(expected_clean)
        if model_num is not None and expected_num is not None:
            return abs(model_num - expected_num) < 0.001
    except:
        pass

    return False


def clean_answer(text: str) -> str:
    """Clean answer text for comparison."""
    # Remove think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove emojis
    text = re.sub(r'[🧠🎯🚦❌✓✗]+', '', text)
    # Get first line or first sentence
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        text = lines[0]
    # Remove common prefixes
    text = re.sub(r'^(answer:|the answer is|result:)\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def extract_number(text: str) -> Optional[float]:
    """Extract a number from text."""
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        return float(match.group())
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--all", action="store_true", help="Download all benchmarks")
    parser.add_argument("--bigbench", action="store_true", help="Download BIG-Bench tasks")
    parser.add_argument("--babi", action="store_true", help="Download bAbI tasks")
    parser.add_argument("--bbh", action="store_true", help="Download BIG-Bench Hard")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--max-examples", type=int, default=200, help="Max examples per task")
    parser.add_argument("--base-dir", default="/path/to/training")
    parser.add_argument("--task", help="Download specific task only")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    if args.list:
        print("\n=== AVAILABLE BENCHMARK TASKS ===\n")
        print("BIG-BENCH TASKS:")
        for name, info in BIGBENCH_TASKS.items():
            print(f"  - {name}: {info['description']}")
            print(f"      Relevance: {info.get('relevance', 'General')}")
        print("\nbAbI TASKS:")
        for name, info in BABI_TASKS.items():
            print(f"  - {name}: {info['description']} ({info['difficulty']})")
        print()
        return

    if not (args.all or args.bigbench or args.babi or args.bbh or args.task):
        parser.print_help()
        return

    total_downloaded = 0

    # Download specific task
    if args.task:
        if args.task.startswith("qa"):
            examples = download_babi_task(args.task, args.max_examples)
            if examples:
                total_downloaded += save_benchmark_data(examples, f"babi_{args.task}", base_dir)
        else:
            examples = download_bigbench_task(args.task, args.max_examples)
            if examples:
                total_downloaded += save_benchmark_data(examples, f"bb_{args.task}", base_dir)

    # Download BIG-Bench
    if args.all or args.bigbench:
        print("\n=== Downloading BIG-Bench Tasks ===\n")
        for task_name in BIGBENCH_TASKS.keys():
            examples = download_bigbench_task(task_name, args.max_examples)
            if examples:
                total_downloaded += save_benchmark_data(examples, f"bb_{task_name}", base_dir)

    # Download bAbI
    if args.all or args.babi:
        print("\n=== Downloading bAbI Tasks ===\n")
        # Prioritize most relevant tasks
        priority_tasks = ["qa7", "qa8", "qa6", "qa1", "qa2", "qa15", "qa17"]
        other_tasks = [t for t in BABI_TASKS.keys() if t not in priority_tasks]

        for task_num in priority_tasks + other_tasks:
            examples = download_babi_task(task_num, args.max_examples)
            if examples:
                total_downloaded += save_benchmark_data(examples, f"babi_{task_num}", base_dir)

    # Download BIG-Bench Hard
    if args.all or args.bbh:
        print("\n=== Downloading BIG-Bench Hard ===\n")
        bbh_examples = download_bigbench_hard(args.max_examples)
        for task_name, examples in bbh_examples.items():
            simple_name = task_name.replace("_three_objects", "").replace("_five_objects", "").replace("_seven_objects", "")
            simple_name = simple_name.replace("_two", "")
            total_downloaded += save_benchmark_data(examples, f"bbh_{simple_name}", base_dir)

    print(f"\n=== DOWNLOAD COMPLETE ===")
    print(f"Total examples downloaded: {total_downloaded}")
    print(f"Saved to: {base_dir}/data/validation/benchmarks/")


if __name__ == "__main__":
    main()
