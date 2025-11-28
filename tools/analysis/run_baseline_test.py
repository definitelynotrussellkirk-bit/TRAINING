#!/usr/bin/env python3
"""
Run baseline tests on a specific model.
Can load any model locally and test against validation sets.

Supports:
- syllable: Syllable counting task (trained skill)
- binary: Binary conversion task (trained skill)
- primitive skills: 26 primitive cognitive skills for transfer testing

Usage:
    # Test via API (uses currently loaded model)
    python3 run_baseline_test.py --tag trained_model --api-url http://192.168.x.x:8765

    # Test local model directly (requires GPU)
    python3 run_baseline_test.py --tag base_model --model-path /path/to/Qwen3-0.6B --local

    # Test primitive skills
    python3 run_baseline_test.py --tag base_model --model-path /path/to/model --local --skill letter_count
    python3 run_baseline_test.py --tag base_model --model-path /path/to/model --local --skill primitives

    # List available skills
    python3 run_baseline_test.py --list-skills
"""

import argparse
import json
import time
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

# Import primitive skills module
try:
    from primitive_skills import SKILL_CLASSES, ALL_SKILLS, SKILL_CATEGORIES, get_skill
    PRIMITIVES_AVAILABLE = True
except ImportError:
    PRIMITIVES_AVAILABLE = False

# All supported skills
TRAINED_SKILLS = ["syllable", "binary"]
SPECIAL_SKILLS = ["all", "trained", "primitives", "benchmarks", "babi", "bigbench"]


def get_benchmark_skills(base_dir: Path) -> List[str]:
    """Get list of available benchmark skills."""
    benchmark_dir = base_dir / "data" / "validation" / "benchmarks"
    if not benchmark_dir.exists():
        return []
    return [d.name for d in benchmark_dir.iterdir() if d.is_dir()]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_answer_syllable(text: str) -> Optional[str]:
    """Extract answer from SYLLABLE (SYLLO) format."""
    try:
        # Remove <think>...</think> blocks first
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove emoji patterns (🧠, 🎯, 🚦, ❌, etc.)
        text = re.sub(r'[🧠🎯🚦❌]+', '', text)
        # Find JSON object with solutions
        match = re.search(r'\{[^{}]*"solutions"[^{}]*\[.*?\]\s*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        # Fallback: find any JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
    except:
        pass
    return text.strip()


def extract_answer_binary(text: str) -> Optional[str]:
    """Extract answer from BINARY format."""
    # Remove <think>...</think> blocks first
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove emoji patterns
    text = re.sub(r'[🧠🎯🚦❌]+', '', text)
    text = text.strip()
    # Get first non-empty line as the answer
    for line in text.split('\n'):
        line = line.strip()
        if line:
            return line
    return text


def compare_syllable(model_output: str, expected: str) -> bool:
    """Compare SYLLABLE answers."""
    try:
        model_json = json.loads(extract_answer_syllable(model_output))
        expected_json = json.loads(extract_answer_syllable(expected))

        model_solutions = model_json.get("solutions", [])
        expected_solutions = expected_json.get("solutions", [])

        if len(model_solutions) != len(expected_solutions):
            return False

        for ms, es in zip(model_solutions, expected_solutions):
            if ms.get("answer", "").upper() != es.get("answer", "").upper():
                return False
        return True
    except:
        return model_output.strip().lower() == expected.strip().lower()


def compare_binary(model_output: str, expected: str) -> bool:
    """Compare BINARY answers."""
    model_ans = extract_answer_binary(model_output)
    expected_ans = extract_answer_binary(expected)

    # Normalize: remove extra spaces, lowercase
    model_norm = re.sub(r'\s+', ' ', model_ans).strip().lower()
    expected_norm = re.sub(r'\s+', ' ', expected_ans).strip().lower()

    # Check if they match or if the result part matches
    if model_norm == expected_norm:
        return True

    # Extract just the result (after =)
    try:
        model_result = model_norm.split('=')[-1].strip()
        expected_result = expected_norm.split('=')[-1].strip()
        return model_result == expected_result
    except:
        return False


def load_validation_data(base_dir: Path, skill: str) -> Dict[str, List[Dict]]:
    """Load validation data for a skill."""
    data = defaultdict(list)

    if skill == "syllable":
        val_dir = base_dir / "data" / "validation"
        for filepath in val_dir.glob("val_*.jsonl"):
            if "binary" in filepath.name:
                continue
            # Extract difficulty from filename: val_easy_200.jsonl -> easy
            parts = filepath.stem.split("_")
            if len(parts) >= 2:
                difficulty = parts[1]
            else:
                difficulty = "unknown"

            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        data[difficulty].append(json.loads(line))

    elif skill == "binary":
        val_dir = base_dir / "data" / "validation" / "binary"
        if val_dir.exists():
            for filepath in val_dir.glob("*.jsonl"):
                # val_binary_easy_100.jsonl -> easy
                parts = filepath.stem.split("_")
                if len(parts) >= 3:
                    difficulty = parts[2]
                else:
                    difficulty = "unknown"

                with open(filepath) as f:
                    for line in f:
                        if line.strip():
                            data[difficulty].append(json.loads(line))

    else:
        # Check primitives directory first
        val_dir = base_dir / "data" / "validation" / "primitives" / skill
        if not val_dir.exists():
            # Check benchmarks directory
            val_dir = base_dir / "data" / "validation" / "benchmarks" / skill

        if val_dir.exists():
            for filepath in val_dir.glob("*.jsonl"):
                # val_letter_count_easy_100.jsonl -> easy
                parts = filepath.stem.split("_")
                # Find difficulty (easy/medium/hard) in filename
                for part in parts:
                    if part in ["easy", "medium", "hard"]:
                        difficulty = part
                        break
                else:
                    difficulty = "unknown"

                with open(filepath) as f:
                    for line in f:
                        if line.strip():
                            data[difficulty].append(json.loads(line))

    return dict(data)


def compare_primitive(model_output: str, expected: str, skill_name: str) -> bool:
    """Compare model output to expected answer for primitive skill."""
    if not PRIMITIVES_AVAILABLE:
        return False

    skill = get_skill(skill_name)
    return skill.compare(model_output, expected)


def test_via_api(prompt: str, api_url: str) -> Dict:
    """Test via API endpoint."""
    import requests

    try:
        start = time.time()
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1
            },
            timeout=60
        )
        elapsed = time.time() - start

        if response.status_code != 200:
            return {"error": f"API error: {response.status_code}", "time": elapsed}

        result = response.json()
        output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"output": output, "time": elapsed}
    except Exception as e:
        return {"error": str(e), "time": 0}


def test_local_model(prompt: str, model, tokenizer, device: str) -> Dict:
    """Test using locally loaded model."""
    import torch

    try:
        start = time.time()

        # Format as chat messages and apply chat template
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template if available (required for trained models)
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        elapsed = time.time() - start
        return {"output": output, "time": elapsed}
    except Exception as e:
        return {"error": str(e), "time": 0}


def get_checkpoint_info(model_path: str) -> Dict:
    """Extract checkpoint info from model path or trainer_state.json."""
    info = {"checkpoint_step": None, "checkpoint_name": None}

    if not model_path:
        return info

    model_path = Path(model_path)

    # Try to extract step from path name (e.g., checkpoint-156000)
    match = re.search(r'checkpoint-(\d+)', str(model_path))
    if match:
        info["checkpoint_step"] = int(match.group(1))
        info["checkpoint_name"] = f"checkpoint-{match.group(1)}"
        return info

    # Try to read trainer_state.json
    trainer_state = model_path / "trainer_state.json"
    if trainer_state.exists():
        try:
            with open(trainer_state) as f:
                state = json.load(f)
                info["checkpoint_step"] = state.get("global_step")
                if info["checkpoint_step"]:
                    info["checkpoint_name"] = f"checkpoint-{info['checkpoint_step']}"
        except:
            pass

    return info


def run_evaluation(
    skill: str,
    tag: str,
    base_dir: Path,
    api_url: str = None,
    model_path: str = None,
    local: bool = False,
    max_per_difficulty: int = 50,
    checkpoint_step: int = None
) -> Dict:
    """Run evaluation for a skill."""
    logger.info(f"Running {skill.upper()} evaluation: {tag}")

    # Load validation data
    data = load_validation_data(base_dir, skill)
    if not data:
        logger.error(f"No validation data found for skill: {skill}")
        return {}

    # Get checkpoint info
    ckpt_info = get_checkpoint_info(model_path)
    if checkpoint_step:
        ckpt_info["checkpoint_step"] = checkpoint_step
        ckpt_info["checkpoint_name"] = f"checkpoint-{checkpoint_step}"

    if ckpt_info["checkpoint_step"]:
        logger.info(f"Checkpoint: {ckpt_info['checkpoint_name']} (step {ckpt_info['checkpoint_step']})")

    # Load model if local
    model, tokenizer, device = None, None, None
    if local and model_path:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model from {model_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        logger.info("Model loaded")

    results = {
        "skill": skill,
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "checkpoint_step": ckpt_info.get("checkpoint_step"),
        "checkpoint_name": ckpt_info.get("checkpoint_name"),
        "by_difficulty": {},
        "overall": {"total": 0, "correct": 0, "accuracy": 0.0}
    }

    total_correct = 0
    total_tested = 0

    for difficulty, examples in data.items():
        logger.info(f"Testing {difficulty}: {len(examples)} examples")

        import random
        if len(examples) > max_per_difficulty:
            examples = random.sample(examples, max_per_difficulty)

        diff_results = {"total": len(examples), "correct": 0, "errors": 0}
        times = []

        for i, example in enumerate(examples):
            # Get prompt and expected answer based on format
            if "messages" in example:
                # SYLLABLE format (chat messages)
                prompt = example["messages"][0]["content"]
                expected = example["messages"][1]["content"]
                compare_fn = compare_syllable
            elif "user_prompt" in example and "expected_answer" in example:
                # Primitive skill format
                prompt = example["user_prompt"]
                expected = example["expected_answer"]
                compare_fn = lambda out, exp: compare_primitive(out, exp, skill)
            else:
                # BINARY format (legacy)
                prompt = example.get("user_prompt", "")
                expected = example.get("assistant_response", "")
                compare_fn = compare_binary

            # Get model output
            if local and model:
                result = test_local_model(prompt, model, tokenizer, device)
            else:
                result = test_via_api(prompt, api_url)

            if "error" in result:
                diff_results["errors"] += 1
            else:
                correct = compare_fn(result["output"], expected)
                if correct:
                    diff_results["correct"] += 1
                    total_correct += 1
                times.append(result["time"])

            total_tested += 1

            if (i + 1) % 10 == 0:
                acc = diff_results["correct"] / (i + 1) * 100
                logger.info(f"  Progress: {i+1}/{len(examples)} | Accuracy: {acc:.1f}%")

        diff_results["accuracy"] = diff_results["correct"] / diff_results["total"] if diff_results["total"] > 0 else 0
        diff_results["avg_time"] = sum(times) / len(times) if times else 0
        results["by_difficulty"][difficulty] = diff_results

        logger.info(f"  {difficulty.upper()}: {diff_results['correct']}/{diff_results['total']} ({diff_results['accuracy']*100:.1f}%)")

    results["overall"]["total"] = total_tested
    results["overall"]["correct"] = total_correct
    results["overall"]["accuracy"] = total_correct / total_tested if total_tested > 0 else 0

    # Cleanup
    if model:
        import torch
        del model
        torch.cuda.empty_cache()

    return results


def list_skills(base_dir: Path = None):
    """Print all available skills."""
    if base_dir is None:
        base_dir = Path("/path/to/training")

    print("\n=== AVAILABLE SKILLS ===\n")
    print("TRAINED SKILLS (what we train on):")
    print("  - syllable: Syllable counting task")
    print("  - binary: Binary conversion task")
    print()
    print("SPECIAL OPTIONS:")
    print("  - all: Test syllable + binary (trained skills)")
    print("  - trained: Same as 'all'")
    print("  - primitives: Test all 26 primitive skills")
    print("  - benchmarks: Test all benchmark tasks (bAbI + BIG-Bench)")
    print("  - babi: Test all bAbI reasoning tasks")
    print("  - bigbench: Test all BIG-Bench tasks")
    print()

    if PRIMITIVES_AVAILABLE:
        print("PRIMITIVE SKILLS (for transfer testing):")
        for category, skills in SKILL_CATEGORIES.items():
            print(f"  {category.upper()}:")
            for skill in skills:
                skill_obj = get_skill(skill)
                print(f"    - {skill}: {skill_obj.description}")
        print(f"\n  Total: {len(ALL_SKILLS)} primitive skills")
    else:
        print("PRIMITIVE SKILLS: Not available (import error)")

    # List benchmark skills
    benchmark_skills = get_benchmark_skills(base_dir)
    if benchmark_skills:
        babi_skills = sorted([s for s in benchmark_skills if s.startswith("babi_")])
        bb_skills = sorted([s for s in benchmark_skills if s.startswith("bb_")])

        print("\nBENCHMARK SKILLS (external datasets):")
        if babi_skills:
            print(f"  bAbI Tasks ({len(babi_skills)}):")
            for skill in babi_skills[:5]:
                print(f"    - {skill}")
            if len(babi_skills) > 5:
                print(f"    ... and {len(babi_skills) - 5} more")

        if bb_skills:
            print(f"  BIG-Bench Tasks ({len(bb_skills)}):")
            for skill in bb_skills[:5]:
                print(f"    - {skill}")
            if len(bb_skills) > 5:
                print(f"    ... and {len(bb_skills) - 5} more")

        print(f"\n  Total benchmarks: {len(benchmark_skills)} tasks")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run baseline tests")
    parser.add_argument("--tag", help="Tag for this run")
    parser.add_argument("--skill", default="all", help="Skill to test (see --list-skills)")
    parser.add_argument("--base-dir", default="/path/to/training")
    parser.add_argument("--api-url", default="http://192.168.x.x:8765")
    parser.add_argument("--model-path", help="Path to model for local testing")
    parser.add_argument("--local", action="store_true", help="Load model locally instead of API")
    parser.add_argument("--max-per-difficulty", type=int, default=50)
    parser.add_argument("--checkpoint", type=int, help="Checkpoint step number (auto-detected if not provided)")
    parser.add_argument("--list-skills", action="store_true", help="List all available skills")
    parser.add_argument("--category", help="Test all skills in a category (counting, conversion, etc.)")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Handle list-skills
    if args.list_skills:
        list_skills(base_dir)
        return

    # Require tag for actual tests
    if not args.tag:
        parser.error("--tag is required for running tests")

    # Auto-detect checkpoint from model path
    ckpt_info = get_checkpoint_info(args.model_path) if args.model_path else {}
    checkpoint_step = args.checkpoint or ckpt_info.get("checkpoint_step")

    # Get available benchmark skills
    benchmark_skills = get_benchmark_skills(base_dir)
    babi_skills = [s for s in benchmark_skills if s.startswith("babi_")]
    bb_skills = [s for s in benchmark_skills if s.startswith("bb_")]

    # Determine which skills to test
    if args.skill == "all" or args.skill == "trained":
        skills = TRAINED_SKILLS
    elif args.skill == "primitives":
        if not PRIMITIVES_AVAILABLE:
            logger.error("Primitive skills not available (import error)")
            return
        skills = ALL_SKILLS
    elif args.skill == "benchmarks":
        skills = benchmark_skills
    elif args.skill == "babi":
        skills = babi_skills
    elif args.skill == "bigbench":
        skills = bb_skills
    elif args.category and PRIMITIVES_AVAILABLE:
        skills = SKILL_CATEGORIES.get(args.category, [])
        if not skills:
            logger.error(f"Unknown category: {args.category}")
            return
    elif PRIMITIVES_AVAILABLE and args.skill in ALL_SKILLS:
        skills = [args.skill]
    elif args.skill in TRAINED_SKILLS:
        skills = [args.skill]
    elif args.skill in benchmark_skills:
        skills = [args.skill]
    else:
        logger.error(f"Unknown skill: {args.skill}. Use --list-skills to see available options.")
        return

    all_results = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "checkpoint_step": checkpoint_step,
        "checkpoint_name": f"checkpoint-{checkpoint_step}" if checkpoint_step else None,
        "model_path": args.model_path,
        "skills": {}
    }

    for skill in skills:
        results = run_evaluation(
            skill=skill,
            tag=args.tag,
            base_dir=base_dir,
            api_url=args.api_url,
            model_path=args.model_path,
            local=args.local,
            max_per_difficulty=args.max_per_difficulty,
            checkpoint_step=checkpoint_step
        )
        if results:
            all_results["skills"][skill] = results

    # Save results
    results_dir = base_dir / "status" / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Include checkpoint in filename if available and not already in tag
    if checkpoint_step and f"ckpt{checkpoint_step}" not in args.tag and f"checkpoint-{checkpoint_step}" not in args.tag:
        filepath = results_dir / f"baseline_{args.tag}_ckpt{checkpoint_step}.json"
    else:
        filepath = results_dir / f"baseline_{args.tag}.json"

    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {filepath}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS: {args.tag}")
    if checkpoint_step:
        print(f"Checkpoint: {checkpoint_step}")
    print(f"{'='*60}")
    for skill, data in all_results.get("skills", {}).items():
        print(f"\n{skill.upper()}:")
        print(f"  Overall: {data['overall']['accuracy']*100:.1f}%")
        for diff, ddata in data.get("by_difficulty", {}).items():
            print(f"  {diff.capitalize()}: {ddata['accuracy']*100:.1f}% ({ddata['correct']}/{ddata['total']})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
