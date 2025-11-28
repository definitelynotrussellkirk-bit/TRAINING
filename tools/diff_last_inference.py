#!/usr/bin/env python3
"""
Inspect recent training prompts, goldens, models, and compute diffs.

Examples:
    python tools/diff_last_inference.py --count 3
    python tools/diff_last_inference.py --prompt-only --count 5
    python tools/diff_last_inference.py --compare-file my_answer.txt
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff recent inference previews.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("training_output.log"),
        help="Path to the training output log (default: training_output.log)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of most recent previews to display (default: 1)",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Show only the prompts for the requested previews.",
    )
    parser.add_argument(
        "--compare-file",
        type=Path,
        default=None,
        help="Path to a file containing your answer. If provided, diff against the golden output instead of the model output.",
    )
    return parser.parse_args()


def slice_block(lines: List[str], start: int, end: int) -> str:
    block = "".join(lines[start:end]).rstrip()
    return block + ("\n" if block and not block.endswith("\n") else "")


def extract_block(lines: List[str], start_idx: int, end_idx: int) -> Tuple[str, str, str]:
    """Return prompt, golden, model strings for the block."""
    golden_idx = next(
        (i for i in range(start_idx, end_idx) if lines[i].startswith("✅ GOLDEN")),
        None,
    )
    if golden_idx is None:
        raise ValueError("Could not find GOLDEN section inside block.")

    model_idx = next(
        (i for i in range(golden_idx, end_idx) if lines[i].startswith("🤖 MODEL")),
        None,
    )
    if model_idx is None:
        raise ValueError("Could not find MODEL section inside block.")

    prompt = slice_block(lines, start_idx, golden_idx)
    golden = slice_block(lines, golden_idx + 1, model_idx)

    model_end = model_idx + 1
    while model_end < end_idx and not (
        lines[model_end].startswith("✅") or lines[model_end].startswith("❌")
    ):
        model_end += 1
    model = slice_block(lines, model_idx + 1, model_end)
    return prompt, golden, model


def print_block(
    label: str,
    prompt: str,
    golden: str,
    model: str,
    compare_file: Optional[Path],
) -> None:
    print(f"=== {label} ===")
    print(">>> PROMPT")
    print(prompt.rstrip(), "\n", sep="")

    if compare_file is None:
        print(">>> GOLDEN")
        print(golden.rstrip(), "\n", sep="")

        print(">>> MODEL")
        print(model.rstrip(), "\n", sep="")

        left = golden
        right = model
        to_label = "model"
    else:
        mine = compare_file.read_text(encoding="utf-8", errors="ignore").rstrip()
        print(">>> GOLDEN")
        print(golden.rstrip(), "\n", sep="")
        print(f">>> YOUR ANSWER ({compare_file})")
        print(mine, "\n", sep="")
        left = golden
        right = mine + ("\n" if mine and not mine.endswith("\n") else "")
        to_label = f"{compare_file}"

    diff = difflib.unified_diff(
        left.splitlines(keepends=True),
        right.splitlines(keepends=True),
        fromfile="golden",
        tofile=to_label,
    )

    diff_list = list(diff)
    print(">>> DIFF")
    if diff_list:
        for line in diff_list:
            print(line.rstrip("\n"))
    else:
        print("No differences.")
    print()


def main() -> None:
    args = parse_args()
    if not args.log.exists():
        raise SystemExit(f"Log file not found: {args.log}")

    lines = args.log.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    prompt_indices = [
        idx for idx, line in enumerate(lines) if line.startswith("🔍 CURRENT TRAINING EXAMPLE")
    ]
    if not prompt_indices:
        raise SystemExit("No inference previews found in the log.")

    count = max(1, args.count)
    selected = prompt_indices[-count:]

    for position, start_idx in enumerate(selected, start=1):
        next_idx = (
            prompt_indices[prompt_indices.index(start_idx) + 1]
            if start_idx != prompt_indices[-1]
            else len(lines)
        )
        prompt, golden, model = extract_block(lines, start_idx, next_idx)
        label = lines[start_idx].strip()

        if args.prompt_only:
            print(f"=== {label} ===")
            print(prompt.rstrip(), "\n", sep="")
            continue

        print_block(label, prompt, golden, model, args.compare_file)


if __name__ == "__main__":
    main()
