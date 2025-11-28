#!/usr/bin/env python3
"""
Sparring with the Trainers - Learn from combat mistakes

When DIO spars with skill trainers, every wrong answer generates 3 training examples:
1. IDENTIFY WRONG: "Is this correct?" → "It is incorrect."
2. CORRECT IT: "This is incorrect. Find the correct solution." → [golden answer]
3. CONFIRM RIGHT: [fresh problem + golden] "Is this correct?" → "It is correct."

Usage:
    python3 guild/sparring.py --skill binary --count 100
    python3 guild/sparring.py --skill sy --count 50 --checkpoint checkpoint-180000
"""

import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.training_queue import TrainingQueue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Generator identification for data lineage
GENERATOR_ID = "sparring"
GENERATOR_VERSION = "1.0.0"


@dataclass
class SparringResult:
    """Result from a single sparring round"""
    problem: str           # The original problem/prompt
    golden_answer: str     # The correct answer
    model_answer: str      # What the model said
    is_correct: bool       # Did model get it right?
    skill: str             # Which skill (binary, sy)
    level: int             # Difficulty level
    metadata: Dict = field(default_factory=dict)


@dataclass
class SparringSession:
    """A complete sparring session"""
    checkpoint: str
    skill: str
    level: int
    timestamp: str
    total_rounds: int = 0
    correct: int = 0
    incorrect: int = 0
    results: List[SparringResult] = field(default_factory=list)
    training_examples_generated: int = 0


class SparringTrainer:
    """
    Manages sparring sessions between DIO and skill trainers.

    Every wrong answer → 3 training examples:
    1. Identify incorrect answer
    2. Produce correct answer after seeing mistake
    3. Confirm a correct answer (fresh problem)
    """

    def __init__(
        self,
        base_dir: Path = None,
        inference_url: str = "http://192.168.x.x:8765",
        inference_key: str = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path("/path/to/training")
        self.inference_url = inference_url
        self.inference_key = inference_key or "admin123"

        # Output directories
        self.output_dir = self.base_dir / "guild" / "sparring_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.status_dir = self.base_dir / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)

        # Training queue for auto-queuing results
        self.queue = TrainingQueue(str(self.base_dir))

        # Cache of fresh problems for "confirm correct" examples
        self._fresh_problems_cache: Dict[str, List[Dict]] = {}

    def _get_skill_client(self, skill: str):
        """Get API client for a skill"""
        from data_manager.skill_api_client import SkillAPIClient
        return SkillAPIClient(skill)

    def _call_inference(
        self,
        prompt: str,
        system_prompt: str = None,
        checkpoint: str = None,
        max_tokens: int = 512,
    ) -> Tuple[str, bool]:
        """
        Call inference API to get model response.

        Returns:
            (response_text, success)
        """
        import requests

        headers = {"X-API-Key": self.inference_key}

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic for evaluation
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        if checkpoint:
            payload["model"] = checkpoint

        try:
            resp = requests.post(
                f"{self.inference_url}/generate",
                json=payload,
                headers=headers,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("text", ""), True
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return "", False

    def _fetch_problems(self, skill: str, level: int, count: int) -> List[Dict]:
        """Fetch problems from skill API"""
        client = self._get_skill_client(skill)

        try:
            response = client.generate(level=level, count=count)

            if skill == "binary":
                return response.get("samples", [])
            elif skill in ("sy", "syllo"):
                return response.get("puzzles", [])
            else:
                return response.get("samples", response.get("puzzles", []))
        except Exception as e:
            logger.error(f"Failed to fetch problems: {e}")
            return []

    def _format_problem(self, problem: Dict, skill: str) -> Tuple[str, str]:
        """
        Format a problem into (prompt, golden_answer).

        Returns:
            (prompt_text, golden_answer)
        """
        if skill == "binary":
            # Binary format
            prompt = problem.get("prompt", "")
            golden = problem.get("solution", problem.get("answer", ""))
            return prompt, golden

        elif skill in ("sy", "syllo"):
            # SYLLO format
            from data_manager.skill_api_client import syllo_to_training_format
            training_fmt = syllo_to_training_format(problem)
            messages = training_fmt.get("messages", [])

            # Find user prompt and assistant answer
            prompt = ""
            golden = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt = msg["content"]
                elif msg["role"] == "assistant":
                    golden = msg["content"]

            return prompt, golden

        # Generic fallback
        return str(problem.get("prompt", problem)), str(problem.get("answer", ""))

    def _check_answer(self, model_answer: str, golden_answer: str, skill: str) -> bool:
        """Check if model answer matches golden (skill-specific logic)"""
        # Normalize whitespace
        model_clean = " ".join(model_answer.split()).strip().lower()
        golden_clean = " ".join(golden_answer.split()).strip().lower()

        # Exact match (normalized)
        if model_clean == golden_clean:
            return True

        # For binary: check if key numbers/results match
        if skill == "binary":
            # Extract numbers from both
            import re
            model_nums = set(re.findall(r'-?\d+', model_answer))
            golden_nums = set(re.findall(r'-?\d+', golden_answer))
            if model_nums and golden_nums and model_nums == golden_nums:
                return True

        return False

    def _get_fresh_problem(self, skill: str, level: int) -> Tuple[str, str]:
        """
        Get a fresh problem (not from current batch) for "confirm correct" training.

        Returns:
            (prompt, golden_answer)
        """
        cache_key = f"{skill}_{level}"

        # Fetch more if cache is empty
        if cache_key not in self._fresh_problems_cache or not self._fresh_problems_cache[cache_key]:
            fresh = self._fetch_problems(skill, level, 50)
            self._fresh_problems_cache[cache_key] = fresh

        if not self._fresh_problems_cache[cache_key]:
            return "", ""

        # Pop one from cache
        problem = self._fresh_problems_cache[cache_key].pop()
        return self._format_problem(problem, skill)

    def spar(
        self,
        skill: str,
        level: int,
        count: int,
        checkpoint: str = None,
        system_prompt: str = None,
    ) -> SparringSession:
        """
        Run a sparring session.

        Args:
            skill: Skill to spar with (binary, sy)
            level: Difficulty level
            count: Number of rounds
            checkpoint: Specific checkpoint to test (None = current)
            system_prompt: Optional system prompt

        Returns:
            SparringSession with results
        """
        session = SparringSession(
            checkpoint=checkpoint or "current",
            skill=skill,
            level=level,
            timestamp=datetime.now().isoformat(),
        )

        logger.info(f"⚔️  Starting sparring session: {skill} L{level} x{count}")
        logger.info(f"   Checkpoint: {checkpoint or 'current'}")

        # Fetch problems
        problems = self._fetch_problems(skill, level, count)
        if not problems:
            logger.error("No problems fetched!")
            return session

        logger.info(f"   Fetched {len(problems)} problems")

        # Run each round
        for i, problem in enumerate(problems):
            prompt, golden = self._format_problem(problem, skill)

            if not prompt or not golden:
                continue

            # Get model's answer
            model_answer, success = self._call_inference(
                prompt=prompt,
                system_prompt=system_prompt,
                checkpoint=checkpoint,
            )

            if not success:
                continue

            # Check correctness
            is_correct = self._check_answer(model_answer, golden, skill)

            result = SparringResult(
                problem=prompt,
                golden_answer=golden,
                model_answer=model_answer,
                is_correct=is_correct,
                skill=skill,
                level=level,
                metadata={"problem_data": problem}
            )

            session.results.append(result)
            session.total_rounds += 1

            if is_correct:
                session.correct += 1
                logger.debug(f"   ✓ Round {i+1}: Correct")
            else:
                session.incorrect += 1
                logger.debug(f"   ✗ Round {i+1}: Wrong")

            # Progress update every 10 rounds
            if (i + 1) % 10 == 0:
                acc = session.correct / session.total_rounds * 100
                logger.info(f"   Progress: {i+1}/{len(problems)} ({acc:.1f}% accuracy)")

        # Summary
        if session.total_rounds > 0:
            acc = session.correct / session.total_rounds * 100
            logger.info(f"⚔️  Session complete: {session.correct}/{session.total_rounds} ({acc:.1f}%)")
            logger.info(f"   Wrong answers to learn from: {session.incorrect}")

        return session

    def generate_training_data(self, session: SparringSession) -> List[Dict]:
        """
        Generate training examples from wrong answers.

        Each wrong answer → 3 examples:
        1. Identify incorrect: "Is this correct?" → "It is incorrect."
        2. Correct it: "Find the correct solution." → [golden]
        3. Confirm correct: [fresh problem] "Is this correct?" → "It is correct."

        Returns:
            List of training examples in messages format
        """
        examples = []
        wrong_answers = [r for r in session.results if not r.is_correct]

        if not wrong_answers:
            logger.info("No wrong answers - no training data to generate")
            return examples

        logger.info(f"📝 Generating training data from {len(wrong_answers)} wrong answers...")

        for result in wrong_answers:
            # === Example 1: Identify Incorrect ===
            ex1 = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{result.problem}\n\n---\nProposed answer:\n{result.model_answer}\n---\n\nIs this answer correct?"
                    },
                    {
                        "role": "assistant",
                        "content": "It is incorrect."
                    }
                ],
                "type": "sparring_identify_incorrect",
                "skill": result.skill,
                "level": result.level,
            }
            examples.append(ex1)

            # === Example 2: Correct It ===
            ex2 = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{result.problem}\n\n---\nProposed answer:\n{result.model_answer}\n---\n\nThis answer is incorrect. Find the correct solution."
                    },
                    {
                        "role": "assistant",
                        "content": result.golden_answer
                    }
                ],
                "type": "sparring_correction",
                "skill": result.skill,
                "level": result.level,
            }
            examples.append(ex2)

            # === Example 3: Confirm Correct (fresh problem) ===
            fresh_prompt, fresh_golden = self._get_fresh_problem(result.skill, result.level)

            if fresh_prompt and fresh_golden:
                ex3 = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{fresh_prompt}\n\n---\nProposed answer:\n{fresh_golden}\n---\n\nIs this answer correct?"
                        },
                        {
                            "role": "assistant",
                            "content": "It is correct."
                        }
                    ],
                    "type": "sparring_confirm_correct",
                    "skill": result.skill,
                    "level": result.level,
                }
                examples.append(ex3)

        session.training_examples_generated = len(examples)
        logger.info(f"   Generated {len(examples)} training examples ({len(examples)//3} sets of 3)")

        return examples

    def save_and_queue(
        self,
        examples: List[Dict],
        session: SparringSession,
        auto_queue: bool = True,
        priority: str = "high",  # ALWAYS HIGH - sparring data is checkpoint-specific, becomes stale!
    ) -> Optional[Path]:
        """
        Save training examples and optionally queue for training.

        Returns:
            Path to saved file
        """
        if not examples:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sparring_{session.skill}_L{session.level}_{len(examples)}_{timestamp}.jsonl"

        # Save to output directory
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            for ex in examples:
                # Add lineage metadata
                ex["generator_id"] = GENERATOR_ID
                ex["generator_version"] = GENERATOR_VERSION
                ex["session_checkpoint"] = session.checkpoint
                ex["session_timestamp"] = session.timestamp
                f.write(json.dumps(ex) + '\n')

        logger.info(f"💾 Saved: {filepath}")

        # Queue for training
        if auto_queue:
            queue_path = self.base_dir / "inbox" / filename

            # Copy to inbox
            import shutil
            shutil.copy(filepath, queue_path)

            self.queue.add_to_queue(queue_path, priority)
            logger.info(f"📥 Queued: {filename} (priority: {priority})")

        # Save session status
        self._save_status(session)

        return filepath

    def _save_status(self, session: SparringSession):
        """Save session status for dashboard"""
        status_file = self.status_dir / "sparring.json"

        status = {
            "last_session": {
                "timestamp": session.timestamp,
                "checkpoint": session.checkpoint,
                "skill": session.skill,
                "level": session.level,
                "total_rounds": session.total_rounds,
                "correct": session.correct,
                "incorrect": session.incorrect,
                "accuracy": session.correct / session.total_rounds if session.total_rounds > 0 else 0,
                "training_examples": session.training_examples_generated,
            },
            "updated_at": datetime.now().isoformat(),
        }

        # Merge with existing history
        if status_file.exists():
            try:
                with open(status_file) as f:
                    existing = json.load(f)
                history = existing.get("history", [])
                history.append(status["last_session"])
                # Keep last 50 sessions
                status["history"] = history[-50:]
                status["total_sessions"] = len(history)
            except:
                status["history"] = [status["last_session"]]
                status["total_sessions"] = 1
        else:
            status["history"] = [status["last_session"]]
            status["total_sessions"] = 1

        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def run_session(
        self,
        skill: str,
        level: int = None,
        count: int = 100,
        checkpoint: str = None,
        auto_queue: bool = True,
        priority: str = "high",
    ) -> Tuple[SparringSession, Optional[Path]]:
        """
        Complete sparring workflow: spar → generate → save → queue.

        Args:
            skill: Skill to spar (binary, sy)
            level: Level (None = from curriculum)
            count: Number of problems
            checkpoint: Checkpoint to test
            auto_queue: Auto-queue results for training
            priority: Queue priority

        Returns:
            (session, output_path)
        """
        # Get level from curriculum if not specified
        if level is None:
            level = self._get_curriculum_level(skill)

        # Run sparring
        session = self.spar(
            skill=skill,
            level=level,
            count=count,
            checkpoint=checkpoint,
        )

        # Generate training data
        examples = self.generate_training_data(session)

        # Save and queue
        output_path = self.save_and_queue(
            examples=examples,
            session=session,
            auto_queue=auto_queue,
            priority=priority,
        )

        return session, output_path

    def _get_curriculum_level(self, skill: str) -> int:
        """Get current level from curriculum state"""
        state_file = self.base_dir / "data_manager" / "curriculum_state.json"

        try:
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)

                # Map skill names
                skill_key = skill
                if skill == "sy":
                    skill_key = "syllo"

                # Get mastered level, training level = mastered + 1
                mastered = state.get("skills", {}).get(skill_key, {}).get("current_level", 0)
                return mastered + 1
        except Exception as e:
            logger.warning(f"Could not read curriculum: {e}")

        return 1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sparring with the Trainers - Learn from combat mistakes"
    )
    parser.add_argument("--skill", required=True, choices=["binary", "sy", "syllo"],
                        help="Skill to spar with")
    parser.add_argument("--level", type=int, help="Level (default: from curriculum)")
    parser.add_argument("--count", type=int, default=100, help="Number of problems")
    parser.add_argument("--checkpoint", help="Specific checkpoint to test")
    parser.add_argument("--no-queue", action="store_true", help="Don't auto-queue results")
    parser.add_argument("--priority", default="high", choices=["high", "normal", "low"])
    parser.add_argument("--base-dir", default="/path/to/training")
    parser.add_argument("--inference-url", default="http://192.168.x.x:8765")

    args = parser.parse_args()

    trainer = SparringTrainer(
        base_dir=Path(args.base_dir),
        inference_url=args.inference_url,
    )

    session, output_path = trainer.run_session(
        skill=args.skill,
        level=args.level,
        count=args.count,
        checkpoint=args.checkpoint,
        auto_queue=not args.no_queue,
        priority=args.priority,
    )

    # Final summary
    print("\n" + "="*60)
    print("⚔️  SPARRING SESSION COMPLETE")
    print("="*60)
    print(f"Skill:      {session.skill} Level {session.level}")
    print(f"Checkpoint: {session.checkpoint}")
    print(f"Rounds:     {session.total_rounds}")
    print(f"Correct:    {session.correct} ({session.correct/session.total_rounds*100:.1f}%)" if session.total_rounds else "N/A")
    print(f"Wrong:      {session.incorrect}")
    print(f"Training:   {session.training_examples_generated} examples generated")
    if output_path:
        print(f"Output:     {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
