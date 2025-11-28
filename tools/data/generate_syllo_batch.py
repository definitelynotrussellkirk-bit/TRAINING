#!/usr/bin/env python3
"""
Auto-generate SYLLO training batches via the local API server.

This script checks the queue depth and, when it drops below the configured
threshold, requests a new batch from skill_syllo_variant/api_server.py,
writes the results to inbox/, and immediately queues the file for training.
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib import request, error

from training_queue import TrainingQueue


# =============================================================================
# OUTPUT FORMAT VARIATIONS
# =============================================================================
# Vary output format to prevent overfitting to a single response pattern

OUTPUT_FORMATS = ["json", "json_summary_first", "plaintext", "yaml"]
FORMAT_WEIGHTS = [0.4, 0.2, 0.25, 0.15]  # JSON most common, but varied


def format_as_json_solutions_first(solutions: List[Dict], inventory: Dict, analysis: Dict) -> str:
    """Standard JSON with solutions first (original format)."""
    return json.dumps({
        "solutions": solutions,
        "inventory_check": inventory,
        "analysis": analysis
    }, ensure_ascii=False)


def format_as_json_summary_first(solutions: List[Dict], inventory: Dict, analysis: Dict) -> str:
    """JSON with summary/analysis first, then solutions."""
    return json.dumps({
        "analysis": analysis,
        "inventory_check": inventory,
        "solutions": solutions
    }, ensure_ascii=False)


def format_as_plaintext(solutions: List[Dict], inventory: Dict, analysis: Dict) -> str:
    """Plain text format with numbered answers."""
    lines = ["ANSWERS:"]
    for sol in solutions:
        syllables_str = " + ".join(sol["syllables"])
        lines.append(f"{sol['ans_num']}. {sol['answer']} ({syllables_str})")

    lines.append("")
    lines.append(f"SUMMARY: {inventory['status']}")

    if inventory.get("unused_tiles"):
        lines.append(f"Unused tiles: {', '.join(inventory['unused_tiles'])}")

    return "\n".join(lines)


def format_as_yaml(solutions: List[Dict], inventory: Dict, analysis: Dict) -> str:
    """YAML-style format."""
    lines = ["solutions:"]
    for sol in solutions:
        lines.append(f"  - num: {sol['ans_num']}")
        lines.append(f"    answer: {sol['answer']}")
        lines.append(f"    syllables: [{', '.join(sol['syllables'])}]")

    lines.append("")
    lines.append("inventory:")
    lines.append(f"  total_tiles: {inventory['total_tiles']}")
    lines.append(f"  status: \"{inventory['status']}\"")

    if inventory.get("unused_tiles"):
        lines.append(f"  unused: [{', '.join(inventory['unused_tiles'])}]")

    lines.append("")
    lines.append("analysis:")
    lines.append(f"  word_count: {analysis['word_count']}")
    lines.append(f"  unused_count: {analysis['unused_tile_count']}")

    return "\n".join(lines)


def format_response(solutions: List[Dict], inventory: Dict, analysis: Dict, format_type: str = None) -> str:
    """Format response in randomly chosen or specified format."""
    if format_type is None:
        format_type = random.choices(OUTPUT_FORMATS, weights=FORMAT_WEIGHTS, k=1)[0]

    if format_type == "json":
        return format_as_json_solutions_first(solutions, inventory, analysis)
    elif format_type == "json_summary_first":
        return format_as_json_summary_first(solutions, inventory, analysis)
    elif format_type == "plaintext":
        return format_as_plaintext(solutions, inventory, analysis)
    elif format_type == "yaml":
        return format_as_yaml(solutions, inventory, analysis)
    else:
        return format_as_json_solutions_first(solutions, inventory, analysis)


def get_queue_depth(base_dir: Path) -> int:
    queue_dir = base_dir / "queue"
    total = 0
    for priority in ("high", "normal", "low"):
        total += len(list((queue_dir / priority).glob("*.jsonl")))
    return total


def call_api(host: str, port: int, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"http://{host}:{port}/generate"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req = request.Request(url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API returned invalid JSON: {exc}") from exc

    if isinstance(parsed, dict):
        if "examples" in parsed:
            examples = parsed["examples"]
        elif "puzzles" in parsed:
            examples = parsed["puzzles"]
        else:
            # Allow servers that just return a dict of example -> ...
            examples = parsed.get("data", [])
    else:
        examples = parsed

    if not isinstance(examples, list) or not examples:
        raise RuntimeError("API response did not include any examples")

    return examples


def build_user_prompt(puzzle: Dict[str, Any]) -> str:
    puzzle_id = puzzle.get("puzzle_id", "syllo_autogen")
    rules = puzzle.get("rules", {})
    difficulty = rules.get("difficulty", "Unknown")
    word_count = rules.get("word_count", len(puzzle.get("words", [])))
    red_herring_count = max(0, len(puzzle.get("syllable_bank", [])) - sum(len(w.get("syllables", [])) for w in puzzle.get("words", [])))
    notes = rules.get("notes", "")

    header = [
        f"SYLLO Puzzle {puzzle_id}",
        "You must recover every hidden word by assigning syllable tiles to definitions.",
        f"Difficulty: {difficulty}",
        "Rules:",
        f"- {word_count} target words (always between 4 and 8).",
        "- Each word lists its syllable count via blank slots.",
        "- Syllable tiles may repeat across clues when the bank includes duplicates.",
        "- Return your answers as JSON with keys `solutions` and `inventory_check`.",
    ]

    slots = ["", "Word slots:"]
    for idx, word in enumerate(puzzle.get("words", []), 1):
        blanks = " ".join(["___"] * word.get("syllable_count", len(word.get("syllables", []))))
        clue = word.get("definition") or ", ".join(word.get("available_hints", [])[:1]) or word.get("label", "Unknown clue")
        slots.append(f"{idx}. {blanks} \u2014 {clue}")

    note_line = "Note: "
    if red_herring_count > 0:
        note_line += f"{red_herring_count} tile(s) in the bank are red herrings and do not belong to any answer."
    else:
        note_line += "All tiles belong to some answer."
    if notes:
        note_line += f" {notes}"

    syllable_bank = puzzle.get("syllable_bank", [])
    bank_line = ["Syllable bank (shuffled):", " | ".join(syllable_bank)]

    contract = [
        "",
        "Output contract:",
        "- Return a single JSON object.",
        "- Top-level keys: `solutions` (array) and `inventory_check` (object).",
        "- Each `solutions` entry contains `ans_num` (1-indexed clue number),",
        "  the ordered `syllables` you used, and the final UPPERCASE `answer`.",
        "- `inventory_check` must include `total_tiles`, a `usage` map of tile\u2192count,",
        "  the `used` counts per tile, and a short `status` string.",
        "- If red herrings exist, include them under `inventory_check.unused_tiles`.",
        "Do not include literal JSON examples or commentary outside the payload.",
        "- Format: include standard keys plus an `analysis` object listing `word_count`, `unused_tile_count`, and `tile_usage_span`.",
    ]

    return "\n".join(header + slots + ["", note_line, ""] + bank_line + contract)


def build_assistant_payload(puzzle: Dict[str, Any]) -> Dict[str, Any]:
    words = puzzle.get("words", [])
    syllable_bank = puzzle.get("syllable_bank", [])

    solutions = []
    used_counter = Counter()
    for idx, word in enumerate(words, 1):
        syllables = word.get("syllables", [])
        used_counter.update(syllables)
        solutions.append({
            "ans_num": idx,
            "syllables": syllables,
            "answer": word.get("label", "").upper()
        })

    bank_counter = Counter(syllable_bank)
    unused_tiles = []
    for tile, count in bank_counter.items():
        unused = count - used_counter.get(tile, 0)
        if unused > 0:
            unused_tiles.extend([tile] * unused)

    usage_map = dict(bank_counter)
    used_map = {tile: used_counter.get(tile, 0) for tile in bank_counter.keys() if used_counter.get(tile, 0) > 0}

    if unused_tiles:
        status = "All target words completed; red herrings unused: " + ", ".join(sorted(set(unused_tiles))) + "."
    else:
        status = "All target words completed; no unused tiles."

    analysis = {
        "word_count": len(words),
        "unused_tile_count": len(set(unused_tiles)),
        "unused_tiles": sorted(set(unused_tiles)),
        "tile_usage_span": [[tile, used_counter.get(tile, 0)] for tile in sorted(bank_counter.keys())],
    }

    return {
        "solutions": solutions,
        "inventory_check": {
            "total_tiles": len(syllable_bank),
            "usage": usage_map,
            "used": used_map,
            "unused_tiles": sorted(set(unused_tiles)),
            "status": status
        },
        "analysis": analysis
    }


def convert_puzzles_to_training(puzzles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries = []
    for puzzle in puzzles:
        user_prompt = build_user_prompt(puzzle)
        assistant_payload = build_assistant_payload(puzzle)
        assistant_text = json.dumps(assistant_payload, ensure_ascii=False)

        entry = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_text}
            ],
            "metadata": {
                "dataset": "syllo_api_autogen",
                "puzzle_id": puzzle.get("puzzle_id"),
                "word_count": len(puzzle.get("words", [])),
                "syllable_bank_size": len(puzzle.get("syllable_bank", [])),
                "rules": puzzle.get("rules", {}),
            }
        }
        entries.append(entry)
    return entries


def write_examples(file_path: Path, examples: List[Dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False))
            fh.write("\n")


def queue_file(base_dir: Path, jsonl_path: Path, priority: str) -> None:
    queue = TrainingQueue(str(base_dir))
    queue.add_to_queue(jsonl_path, priority)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SYLLO training batches via API")
    parser.add_argument("--base-dir", default=None, help="Repository root (auto-detected if not set)")
    parser.add_argument("--host", default="127.0.0.1", help="API server host")
    parser.add_argument("--port", type=int, default=8765, help="API server port")
    parser.add_argument("--count", type=int, default=20000, help="Examples to request")
    parser.add_argument("--seed", type=int, help="Seed forwarded to the API")
    parser.add_argument("--difficulty", default="EASY", help="Difficulty level (EASY/MEDIUM/HARD, defaults to EASY)")
    parser.add_argument("--payload", help="Extra JSON payload to merge into the request")
    parser.add_argument("--threshold", type=int, default=1, help="Generate if queued files <= threshold")
    parser.add_argument("--priority", choices=["high", "normal", "low"], default="normal", help="Queue priority for the generated batch")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions without hitting the API")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()

    queued_files = get_queue_depth(base_dir)
    if queued_files > args.threshold:
        print(f"Queue depth {queued_files} exceeds threshold {args.threshold}; nothing to do.")
        return 0

    payload: Dict[str, Any] = {"count": args.count}
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.difficulty:
        payload["difficulty"] = args.difficulty
    if args.payload:
        try:
            payload.update(json.loads(args.payload))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--payload must be valid JSON: {exc}") from exc

    if args.dry_run:
        print("Dry run: would call API with payload:")
        print(json.dumps(payload, indent=2))
        return 0

    puzzles = call_api(args.host, args.port, payload)
    training_entries = convert_puzzles_to_training(puzzles)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"syllo_autogen_{timestamp}_count{args.count}.jsonl"
    inbox_path = base_dir / "inbox" / filename

    write_examples(inbox_path, training_entries)
    queue_file(base_dir, inbox_path, args.priority)
    print(f"Generated {len(training_entries)} examples -> {inbox_path}")
    print("Queued file for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
