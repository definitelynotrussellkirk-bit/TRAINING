"""
Eval Runner - Processes evaluation queues.

Handles:
1. Skill evaluations (from checkpoint saves)
2. Passive evaluations (LITE priority, FULL manual)

Run as daemon or one-shot:
    # Daemon mode (continuous)
    python3 core/eval_runner.py --daemon --interval 60

    # One-shot (process all pending)
    python3 core/eval_runner.py --once

    # Process specific checkpoint
    python3 core/eval_runner.py --checkpoint 190000
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation_ledger import (
    get_eval_ledger,
    record_evaluation,
    pop_evaluation,
    get_pending_evaluations,
    queue_evaluation,
    queue_full_evaluation,
    EvalRecord,
    EvalQueueEntry,
)
from core.passives import (
    get_passives_ledger,
    get_passive_definitions,
    get_passive,
    pop_passive,
    get_pending_passives,
    PassiveResult,
    PassiveMode,
)
from core.battle_log import log_eval, format_eval_result
import re


def strip_thinking_tags(text: str) -> str:
    """Strip Qwen3 <think>...</think> tags from model output.

    Qwen3 models use thinking mode which wraps reasoning in <think></think> tags.
    We need to strip these for answer comparison while preserving the actual answer.
    """
    if not text:
        return text

    # Remove <think>...</think> blocks (including multiline)
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Clean up any leading/trailing whitespace from removal
    return stripped.strip()


def to_chat_problem(prompt: str, expected: str, metadata: dict = None) -> dict:
    """Convert prompt/expected to chat messages format."""
    problem = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": expected},
        ]
    }
    if metadata:
        problem["metadata"] = metadata
    return problem


def extract_prompt_expected(problem: dict) -> tuple:
    """Extract prompt and expected from various formats.

    Handles:
    - Chat format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    - Legacy format: {"prompt": ..., "expected": ...}
    - Validation format: {"prompt": ..., "expected_answer_format": ...}
    """
    if "messages" in problem:
        messages = problem["messages"]
        prompt = next((m["content"] for m in messages if m.get("role") == "user"), "")
        expected = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        return prompt, expected
    else:
        prompt = problem.get("prompt", "")
        # Try expected, then expected_answer_format (used in validation files)
        expected = problem.get("expected", "") or problem.get("expected_answer_format", "")
        return prompt, expected


class EvalRunner:
    """
    Runs evaluations against checkpoints.

    Uses static validation sets for skills, generates problems for passives.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        inference_host: Optional[str] = None,
        inference_port: Optional[int] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else PROJECT_ROOT

        # Get inference URL from hosts.json if not explicitly provided
        if inference_host and inference_port:
            self.inference_url = f"http://{inference_host}:{inference_port}"
        else:
            from core.hosts import get_service_url
            self.inference_url = get_service_url("inference") or "http://localhost:8765"
        self.eval_ledger = get_eval_ledger(self.base_dir)
        self.passives_ledger = get_passives_ledger(self.base_dir)

        # Load validation sets
        self.validation_dir = self.base_dir / "data" / "validation"

        # Load API key for inference server
        self.api_key = self._load_api_key()

        # Track currently loaded model
        self._current_model_id: Optional[str] = None

        # Cache of checkpoints known to be missing (avoid repeated lookups)
        self._missing_checkpoints: set = set()

    def _load_api_key(self) -> Optional[str]:
        """Load inference API key from secrets file or environment."""
        import os

        # Try environment variable first
        key = os.environ.get("INFERENCE_ADMIN_KEY", "")
        if key:
            return key

        # Try secrets file
        secrets_file = self.base_dir / ".secrets" / "inference.json"
        if secrets_file.exists():
            try:
                with open(secrets_file) as f:
                    secrets = json.load(f)
                return secrets.get("admin_key", "")
            except Exception as e:
                logger.warning(f"Failed to load inference secrets: {e}")

        return None

    def _checkpoint_exists(self, checkpoint_step: int) -> bool:
        """
        Check if checkpoint exists (uses cache to avoid repeated disk lookups).

        Returns True if checkpoint exists, False if missing.
        """
        # Check cache first
        if checkpoint_step in self._missing_checkpoints:
            return False

        # Check ledger and disk
        try:
            from core.checkpoint_ledger import get_ledger
            ledger = get_ledger()
            record = ledger.get(checkpoint_step)

            if not record:
                self._missing_checkpoints.add(checkpoint_step)
                return False

            if not Path(record.path).exists():
                self._missing_checkpoints.add(checkpoint_step)
                return False

            return True
        except Exception:
            # On error, assume exists (will fail properly in _load_checkpoint)
            return True

    def _load_skill_config(self, skill: str) -> Optional[dict]:
        """Load skill configuration from YAML."""
        import yaml
        skill_config_file = self.base_dir / "configs" / "skills" / f"{skill}.yaml"
        if not skill_config_file.exists():
            logger.warning(f"Skill config not found: {skill_config_file}")
            return None

        try:
            with open(skill_config_file) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load skill config: {e}")
            return None

    def _load_eval_set(self, skill: str, level: int) -> Optional[List[dict]]:
        """
        Load evaluation set for a skill at a specific level.

        Tries new format first (data/validation/{skill}/level_XX.json),
        falls back to legacy format ({skill}_validation.json).
        """
        # New format: per-level files
        new_format_file = self.validation_dir / skill / f"level_{level:02d}.json"
        if new_format_file.exists():
            try:
                with open(new_format_file) as f:
                    data = json.load(f)
                # New format has "problems" key
                problems = data.get("problems", [])
                if problems:
                    logger.debug(f"Loaded {len(problems)} problems from {new_format_file}")
                    return problems
            except Exception as e:
                logger.warning(f"Failed to load new format: {e}")

        # Legacy format: single file with level keys
        legacy_file = self.validation_dir / f"{skill}_validation.json"
        if legacy_file.exists():
            try:
                with open(legacy_file) as f:
                    data = json.load(f)
                level_key = str(level)
                if level_key in data:
                    logger.debug(f"Loaded from legacy format: {legacy_file}")
                    return data[level_key]
            except Exception as e:
                logger.warning(f"Failed to load legacy format: {e}")

        logger.error(f"No eval set found for {skill} L{level}")
        return None

    def run_skill_evaluation(
        self,
        checkpoint_step: int,
        skill: str,
        level: int,
        eval_type: str = "quick",
    ) -> Optional[EvalRecord]:
        """
        Run skill evaluation using static validation set.

        Args:
            checkpoint_step: Checkpoint to evaluate
            skill: Skill ID
            level: Skill level
            eval_type: "quick" or "full"

        Returns EvalRecord if successful, None if failed.
        """
        logger.info(f"Running {eval_type} skill eval: checkpoint-{checkpoint_step} {skill} L{level}")

        # Check if already done
        if self.eval_ledger.has_evaluation(checkpoint_step, skill, level):
            logger.info(f"Already evaluated, skipping")
            return self.eval_ledger.get(checkpoint_step, skill, level)

        # Load evaluation set
        problems = self._load_eval_set(skill, level)
        if not problems:
            return None

        # Load checkpoint on inference server
        if not self._load_checkpoint(checkpoint_step):
            logger.error(f"Failed to load checkpoint-{checkpoint_step}")
            return None

        # Run evaluation
        correct = 0
        total = len(problems)
        results = []

        for i, problem in enumerate(problems):
            prompt, expected = extract_prompt_expected(problem)

            # Get model response
            response = self._get_model_response(prompt)
            if response is None:
                logger.warning(f"No response for problem {i+1}")
                results.append({
                    "problem_idx": i,
                    "correct": False,
                    "expected": expected,
                    "got": None,
                    "error": "no_response",
                })
                continue

            # Check correctness
            is_correct = self._check_answer(expected, response, skill)
            if is_correct:
                correct += 1

            results.append({
                "problem_idx": i,
                "correct": is_correct,
                "prompt": prompt,
                "expected": expected,
                "got": response,
            })

        accuracy = correct / total if total > 0 else 0

        # Record result
        success = record_evaluation(
            checkpoint_step=checkpoint_step,
            skill=skill,
            level=level,
            accuracy=accuracy,
            correct=correct,
            total=total,
            problems=results,
            validation_type="static",
            eval_type=eval_type,
            base_dir=self.base_dir,
        )

        if success:
            logger.info(f"Recorded: {skill} L{level} = {accuracy:.1%} ({correct}/{total})")

            # Check for suspicious eval: 0% accuracy but low training loss
            # This is a canary for broken eval system (format mismatch, etc.)
            suspicious_eval = False
            checkpoint_loss = None
            try:
                from core.checkpoint_ledger import get_ledger
                ckpt_ledger = get_ledger()
                ckpt_record = ckpt_ledger.get(checkpoint_step)
                if ckpt_record and ckpt_record.train_loss is not None:
                    checkpoint_loss = ckpt_record.train_loss
                    # Flag: 0% accuracy but loss < 0.05 is suspicious
                    if accuracy == 0 and checkpoint_loss < 0.05:
                        suspicious_eval = True
                        logger.error(
                            f"🚨 SUSPICIOUS EVAL: {skill} L{level} = 0% accuracy "
                            f"but checkpoint loss={checkpoint_loss:.4f} < 0.05. "
                            f"Eval system may be broken!"
                        )
            except Exception as e:
                logger.debug(f"Could not check for suspicious eval: {e}")

            # Log to realm state for UI visibility
            try:
                from core.realm_store import emit_event

                # Determine severity based on accuracy
                if suspicious_eval:
                    severity = "error"  # Suspicious eval gets error severity
                elif accuracy >= 0.8:
                    severity = "success"
                elif accuracy >= 0.5:
                    severity = "info"
                else:
                    severity = "warning"

                # Build message - add warning if suspicious
                base_msg = f"Checkpoint {checkpoint_step}: {skill} L{level} = {accuracy:.1%} ({correct}/{total})"
                if suspicious_eval:
                    base_msg = f"🚨 SUSPICIOUS: {base_msg} (loss={checkpoint_loss:.4f})"

                emit_event(
                    kind="eval_result",
                    channel="eval",
                    message=base_msg,
                    severity=severity,
                    details={
                        "checkpoint_step": checkpoint_step,
                        "skill": skill,
                        "level": level,
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total,
                        "eval_type": eval_type,
                        "suspicious_eval": suspicious_eval,
                        "checkpoint_loss": checkpoint_loss,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to log to realm state: {e}")

            # Log to battle log for UI visibility
            try:
                message = format_eval_result(skill, level, accuracy)
                if suspicious_eval:
                    message = f"🚨 SUSPICIOUS: {message} (loss={checkpoint_loss:.4f})"
                log_eval(
                    message,
                    severity=severity,
                    details={
                        "checkpoint_step": checkpoint_step,
                        "skill": skill,
                        "level": level,
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total,
                        "eval_type": eval_type,
                        "suspicious_eval": suspicious_eval,
                        "checkpoint_loss": checkpoint_loss,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to log to battle log: {e}")

            return self.eval_ledger.get(checkpoint_step, skill, level)
        else:
            logger.warning(f"Failed to record evaluation")
            return None

    def run_full_skill_evaluation(
        self,
        checkpoint_step: int,
        skill: str,
        max_level: Optional[int] = None,
    ) -> List[EvalRecord]:
        """
        Run FULL evaluation - all levels for a skill.

        Args:
            checkpoint_step: Checkpoint to evaluate
            skill: Skill ID
            max_level: Max level to evaluate (default: from skill config)

        Returns list of EvalRecords for each level.
        """
        logger.info(f"Running FULL skill eval: checkpoint-{checkpoint_step} {skill}")

        # Get max level from skill config if not specified
        if max_level is None:
            config = self._load_skill_config(skill)
            if config:
                max_level = config.get("max_level", 10)
            else:
                max_level = 10  # Default fallback

        results = []
        for level in range(1, max_level + 1):
            result = self.run_skill_evaluation(
                checkpoint_step=checkpoint_step,
                skill=skill,
                level=level,
                eval_type="full",
            )
            if result:
                results.append(result)

        logger.info(f"FULL eval complete: {skill} {len(results)}/{max_level} levels")
        return results

    def run_passive_evaluation(
        self,
        checkpoint_step: int,
        passive_id: str,
        mode: str,
    ) -> Optional[PassiveResult]:
        """
        Run passive evaluation using modular passive system.

        Uses guild/passives/ modules for problem generation and answer checking.
        Records version for result comparability.
        """
        logger.info(f"Running passive eval: checkpoint-{checkpoint_step} {passive_id} ({mode})")

        # Get passive module (modular system)
        from core.passives import get_passive_module
        passive_module = get_passive_module(passive_id)

        if not passive_module:
            # Fall back to legacy method
            logger.warning(f"No module for {passive_id}, using legacy generator")
            return self._run_passive_legacy(checkpoint_step, passive_id, mode)

        # Get version info
        config = passive_module.get_config()
        version = config.version
        config_hash = config.config_hash()

        # Check if already done (same version)
        existing = self.passives_ledger.get(checkpoint_step, passive_id, mode)
        if existing and existing.version == version:
            logger.info(f"Already evaluated with v{version}, skipping")
            return existing

        # Determine problem count
        count = config.lite_count if mode == "lite" else config.full_count

        # Generate problems using module
        seed = checkpoint_step  # Reproducible per checkpoint
        problems = passive_module.generate_problems(count, seed=seed)
        if not problems:
            logger.error(f"Failed to generate problems for {passive_id}")
            return None

        # Load checkpoint
        if not self._load_checkpoint(checkpoint_step):
            logger.error(f"Failed to load checkpoint-{checkpoint_step}")
            return None

        # Run evaluation
        correct = 0
        total = len(problems)
        results = []

        for i, problem in enumerate(problems):
            prompt, expected = extract_prompt_expected(problem)

            response = self._get_model_response(prompt)
            if response is None:
                results.append({
                    "problem_idx": i,
                    "correct": False,
                    "prompt": prompt,
                    "expected": expected,
                    "got": None,
                    "error": "no_response",
                })
                continue

            # Use module's answer checker
            is_correct = passive_module.check_answer(expected, response)
            if is_correct:
                correct += 1

            results.append({
                "problem_idx": i,
                "correct": is_correct,
                "prompt": prompt,
                "expected": expected,
                "got": response,
            })

        accuracy = correct / total if total > 0 else 0

        # Record result with version
        result = PassiveResult(
            checkpoint_step=checkpoint_step,
            passive_id=passive_id,
            mode=mode,
            accuracy=accuracy,
            correct=correct,
            total=total,
            timestamp=datetime.now().isoformat(),
            version=version,
            config_hash=config_hash,
            problems=results,
        )

        if self.passives_ledger.record(result):
            logger.info(f"Recorded: {passive_id} v{version} ({mode}) = {accuracy:.1%} ({correct}/{total})")

            # Log to battle log for UI visibility
            try:
                # Determine severity
                if accuracy >= 0.8:
                    severity = "success"
                elif accuracy >= 0.5:
                    severity = "info"
                else:
                    severity = "warning"

                message = f"Passive {passive_id} ({mode}): {accuracy*100:.1f}% ({correct}/{total})"
                log_eval(
                    message,
                    severity=severity,
                    details={
                        "checkpoint_step": checkpoint_step,
                        "passive_id": passive_id,
                        "mode": mode,
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total,
                        "version": version
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to log passive to battle log: {e}")

            return result
        else:
            logger.warning(f"Failed to record passive result")
            return None

    def _run_passive_legacy(
        self,
        checkpoint_step: int,
        passive_id: str,
        mode: str,
    ) -> Optional[PassiveResult]:
        """Legacy passive evaluation (for passives without modules)."""
        passive_def = get_passive(passive_id)
        if not passive_def:
            logger.error(f"Unknown passive: {passive_id}")
            return None

        count = passive_def.lite_count if mode == "lite" else passive_def.full_count
        problems = self._generate_passive_problems(passive_id, count)

        if not problems or not self._load_checkpoint(checkpoint_step):
            return None

        correct = 0
        results = []
        for i, problem in enumerate(problems):
            prompt, expected = extract_prompt_expected(problem)
            response = self._get_model_response(prompt)

            if response is None:
                results.append({"problem_idx": i, "correct": False, "prompt": prompt, "expected": expected, "got": None})
                continue

            is_correct = self._check_passive_answer(passive_id, expected, response)
            if is_correct:
                correct += 1
            results.append({"problem_idx": i, "correct": is_correct, "prompt": prompt, "expected": expected, "got": response})

        accuracy = correct / len(problems) if problems else 0

        result = PassiveResult(
            checkpoint_step=checkpoint_step,
            passive_id=passive_id,
            mode=mode,
            accuracy=accuracy,
            correct=correct,
            total=len(problems),
            timestamp=datetime.now().isoformat(),
            version="legacy",
            problems=results,
        )

        self.passives_ledger.record(result)
        return result

    def _load_checkpoint(self, checkpoint_step: int) -> bool:
        """Load checkpoint on inference server using Ledger as source of truth."""
        import requests
        import subprocess

        # Remote server config from hosts.json
        from core.hosts import get_host, get_remote_path
        inference_host = get_host("3090")
        remote_host = inference_host.host if inference_host else "localhost"
        remote_models_dir = get_remote_path("3090")

        # Use the Checkpoint Ledger to find the path (single source of truth!)
        try:
            from core.checkpoint_ledger import get_ledger
            ledger = get_ledger()
            record = ledger.get(checkpoint_step)

            if not record:
                logger.error(f"Checkpoint {checkpoint_step} not in ledger")
                self._missing_checkpoints.add(checkpoint_step)
                return False

            local_path = record.path
            # Handle relative paths from ledger (recent entries use relative paths)
            if not Path(local_path).is_absolute():
                local_path = str(self.base_dir / local_path)

            if not Path(local_path).exists():
                logger.error(f"Checkpoint path from ledger doesn't exist: {local_path}")
                self._missing_checkpoints.add(checkpoint_step)
                return False

            logger.info(f"Loading {record.canonical_name} from ledger (path: {local_path})")

        except ImportError:
            # Fallback to glob if ledger not available
            logger.warning("Ledger not available, falling back to glob")
            models_dir = self.base_dir / "models" / "current_model"
            checkpoint_dirs = list(models_dir.glob(f"checkpoint-{checkpoint_step}*"))
            if not checkpoint_dirs:
                logger.error(f"Checkpoint directory not found for step {checkpoint_step}")
                return False
            local_path = str(checkpoint_dirs[0])

        # Remote path uses simple name (checkpoint-{step})
        checkpoint_name = f"checkpoint-{checkpoint_step}"
        remote_path = f"{remote_models_dir}/{checkpoint_name}"

        # Check if checkpoint exists on remote, if not sync it
        try:
            check_result = subprocess.run(
                ["ssh", remote_host, f"test -d {remote_path} && echo exists"],
                capture_output=True,
                text=True,
                timeout=10
            )
            checkpoint_exists = "exists" in check_result.stdout
        except Exception as e:
            logger.warning(f"Failed to check remote checkpoint: {e}")
            checkpoint_exists = False

        if not checkpoint_exists:
            # Clean up old checkpoints on remote before syncing (keep max 10)
            try:
                cleanup_cmd = f"cd {remote_models_dir} && ls -t | tail -n +11 | xargs -r rm -rf"
                cleanup_result = subprocess.run(
                    ["ssh", remote_host, cleanup_cmd],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if cleanup_result.returncode == 0:
                    logger.debug(f"Cleaned up old checkpoints on {remote_host}")
            except Exception as e:
                logger.debug(f"Remote cleanup skipped: {e}")

            logger.info(f"Syncing checkpoint-{checkpoint_step} to {remote_host}...")
            try:
                sync_cmd = [
                    "rsync", "-avz", "--delete", "--checksum",
                    str(local_path) + "/",
                    f"{remote_host}:{remote_path}/"
                ]
                sync_result = subprocess.run(
                    sync_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout for sync
                )
                # Exit code 24 = "Partial transfer due to vanished source files"
                # This can happen when retention cleanup runs during sync - treat as warning
                if sync_result.returncode == 24:
                    logger.warning(f"⚠️  Sync partial (files vanished during transfer): {sync_result.stderr.split(chr(10))[-2] if sync_result.stderr else 'unknown'}")
                    # Continue anyway - core model files likely synced successfully
                elif sync_result.returncode != 0:
                    logger.error(f"Sync failed: {sync_result.stderr}")
                    return False
                logger.info(f"Sync completed for checkpoint-{checkpoint_step}")
            except Exception as e:
                logger.error(f"Sync error: {e}")
                return False

        # Load checkpoint on inference server using remote path
        try:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key

            response = requests.post(
                f"{self.inference_url}/models/reload",
                json={"model_path": remote_path},
                headers=headers,
                timeout=120,
            )
            if response.status_code == 200:
                self._current_model_id = checkpoint_name
                logger.info(f"Loaded checkpoint-{checkpoint_step}")

                # Record usage for retention tracking
                try:
                    # Record usage on inference server (where model was loaded)
                    inference_device_id = inference_host.device_id if inference_host else "inference3090"
                    ledger.record_usage(checkpoint_step, inference_device_id)
                    logger.debug(f"Recorded usage: checkpoint {checkpoint_step} on {inference_device_id}")
                except Exception as e:
                    logger.debug(f"Failed to record usage: {e}")

                return True
            else:
                logger.error(f"Failed to load checkpoint: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False

    def _get_model_response(self, prompt: str) -> Optional[str]:
        """Get response from inference server."""
        import requests
        from core.output_validator import validate_model_output, classify_error, get_error_guidance

        if not self._current_model_id:
            logger.error("No model loaded")
            return None

        try:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key

            response = requests.post(
                f"{self.inference_url}/v1/chat/completions",
                json={
                    "model": self._current_model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0,  # Deterministic for evaluation
                },
                headers=headers,
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Check if response is actually an error message masquerading as output
                validated = validate_model_output(content, log_error=True)
                if validated is None:
                    # It was an error - provide debugging guidance
                    error_type = classify_error(content)
                    if error_type:
                        guidance = get_error_guidance(error_type)
                        logger.error(f"Error type: {error_type}. {guidance}")
                    return None

                return content
            else:
                logger.warning(f"Inference error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def _check_answer(self, expected: str, got: str, skill: str) -> bool:
        """Check if answer is correct for a skill evaluation."""
        # Guard: empty expected means we don't have an answer to check against
        if not expected or not expected.strip():
            logger.warning(f"Empty expected answer for {skill} - marking as incorrect")
            return False

        # Strip Qwen3 thinking tags from model output before comparison
        got = strip_thinking_tags(got)

        # Normalize
        expected_norm = expected.strip().lower()
        got_norm = got.strip().lower()

        # Skill-specific checking
        if skill in ("bin", "binary"):
            # Handle JSON format from validation files: {"answer": "①⓪①"}
            import json
            try:
                expected_data = json.loads(expected.strip())
                if isinstance(expected_data, dict) and "answer" in expected_data:
                    expected_answer = expected_data["answer"]
                    # Check if answer appears in response
                    return expected_answer in got
            except (json.JSONDecodeError, TypeError):
                pass

            # Legacy format: "decrement(①⓪) = ①" - check if result appears
            return expected_norm in got_norm or self._extract_binary_result(expected) in got

        elif skill in ("sy", "syllo"):
            # Try JSON format first (new format)
            try:
                import json
                expected_json = json.loads(expected.strip())
                got_json = json.loads(got.strip())

                # Normalize JSON for comparison (case-insensitive, whitespace-insensitive)
                def normalize_json(obj):
                    if isinstance(obj, dict):
                        return {k.lower(): normalize_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [normalize_json(item) for item in obj]
                    elif isinstance(obj, str):
                        return obj.upper().strip()  # Uppercase for word comparison
                    else:
                        return obj

                expected_normalized = normalize_json(expected_json)
                got_normalized = normalize_json(got_json)

                return expected_normalized == got_normalized
            except (json.JSONDecodeError, Exception):
                # Fall back to legacy word list format
                expected_words = set(w.strip().lower() for w in expected.split(",") if w.strip())
                # Extract words from response (look for comma-separated or newline-separated)
                got_words = set()
                for line in got.split("\n"):
                    for word in line.replace(",", " ").split():
                        cleaned = word.strip().lower()
                        if cleaned and len(cleaned) > 1:
                            got_words.add(cleaned)
                return expected_words == got_words

        else:
            # Default: exact match
            return expected_norm == got_norm

    def _extract_binary_result(self, expected: str) -> str:
        """Extract just the result from binary expected answer."""
        if "=" in expected:
            return expected.split("=")[-1].strip()
        return expected

    def _check_passive_answer(self, passive_id: str, expected: str, got: str) -> bool:
        """Check answer for passive evaluation."""
        # Strip Qwen3 thinking tags from model output before comparison
        got = strip_thinking_tags(got)

        expected_norm = expected.strip().lower()
        got_norm = got.strip().lower()

        # Most passives use simple containment or exact match
        if passive_id == "decimal_math":
            # Check if the numeric answer appears
            import re
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            got_nums = re.findall(r'-?\d+\.?\d*', got)
            if expected_nums:
                return expected_nums[-1] in got_nums

        elif passive_id == "instruction_following":
            # Check key parts of expected response appear
            return expected_norm in got_norm

        elif passive_id == "common_sense":
            # Multiple choice - check if correct letter/option appears
            return expected_norm in got_norm

        # Default: containment
        return expected_norm in got_norm

    def _generate_passive_problems(self, passive_id: str, count: int) -> List[Dict[str, Any]]:
        """Generate problems for a passive evaluation."""
        # For now, use static problems. Later: integrate with problem generators
        problems = []

        if passive_id == "decimal_math":
            import random
            random.seed(42)  # Reproducible
            for i in range(count):
                a = random.randint(1, 100)
                b = random.randint(1, 100)
                op = random.choice(["+", "-", "*"])
                if op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                else:
                    result = a * b
                problems.append(to_chat_problem(
                    f"Calculate: {a} {op} {b} = ?",
                    str(result),
                    {"passive": passive_id}
                ))

        elif passive_id == "instruction_following":
            # Simple instruction following
            instructions = [
                ("Say 'hello' three times", "hello hello hello"),
                ("Count from 1 to 5", "1 2 3 4 5"),
                ("List the colors of a rainbow", "red orange yellow green blue indigo violet"),
                ("Name the first three letters of the alphabet", "a b c"),
                ("What is the opposite of hot?", "cold"),
            ]
            for prompt, expected in instructions[:count]:
                problems.append(to_chat_problem(prompt, expected, {"passive": passive_id}))

        elif passive_id == "common_sense":
            # Common sense questions
            questions = [
                ("Water freezes at what temperature in Celsius? A) 0 B) 100 C) 50 D) -10", "A"),
                ("How many legs does a spider have? A) 4 B) 6 C) 8 D) 10", "C"),
                ("What comes after Monday? A) Sunday B) Tuesday C) Wednesday D) Friday", "B"),
                ("What do you use to cut paper? A) hammer B) scissors C) spoon D) pencil", "B"),
                ("Which is the largest ocean? A) Atlantic B) Indian C) Pacific D) Arctic", "C"),
            ]
            for prompt, expected in questions[:count]:
                problems.append(to_chat_problem(prompt, expected, {"passive": passive_id}))

        elif passive_id == "word_problems":
            word_problems = [
                ("If you have 5 apples and eat 2, how many are left?", "3"),
                ("A train travels 60 miles per hour. How far does it go in 2 hours?", "120"),
                ("You have 10 cookies and share them equally with a friend. How many does each person get?", "5"),
                ("If a book costs $15 and you have $50, how much change do you get?", "35"),
                ("There are 24 hours in a day. How many hours in 2 days?", "48"),
            ]
            for prompt, expected in word_problems[:count]:
                problems.append(to_chat_problem(prompt, expected, {"passive": passive_id}))

        elif passive_id == "text_completion":
            completions = [
                ("The sun rises in the ___", "east"),
                ("Water is made of hydrogen and ___", "oxygen"),
                ("The capital of France is ___", "paris"),
                ("Dogs bark and cats ___", "meow"),
                ("The opposite of up is ___", "down"),
            ]
            for prompt, expected in completions[:count]:
                problems.append(to_chat_problem(prompt, expected, {"passive": passive_id}))

        return problems

    def process_skill_queue(
        self,
        limit: int = 10,
        eval_type: Optional[str] = None,
    ) -> int:
        """
        Process pending skill evaluations.

        Args:
            limit: Max evaluations to process
            eval_type: Filter by type ("quick" or "full"), or None for any

        Returns count processed.
        """
        processed = 0

        for _ in range(limit):
            entry = pop_evaluation(eval_type=eval_type)
            if not entry:
                break

            # Handle FULL eval (all levels)
            if entry.eval_type == "full" and entry.level is None:
                results = self.run_full_skill_evaluation(
                    checkpoint_step=entry.checkpoint_step,
                    skill=entry.skill,
                )
                processed += len(results)
            else:
                # Quick eval (single level)
                result = self.run_skill_evaluation(
                    checkpoint_step=entry.checkpoint_step,
                    skill=entry.skill,
                    level=entry.level,
                    eval_type=entry.eval_type,
                )
                if result:
                    processed += 1

        return processed

    def scan_for_backfill(self, skills: Optional[List[str]] = None) -> int:
        """
        Scan checkpoint ledger for missing evaluations and queue them.

        Args:
            skills: List of skills to check (default: all from configs)

        Returns count of evaluations queued.
        """
        from core.checkpoint_ledger import get_ledger

        logger.info("Scanning for missing evaluations...")

        # Get skills from configs if not specified
        if skills is None:
            skills = []
            configs_dir = self.base_dir / "configs" / "skills"
            for config_file in configs_dir.glob("*.yaml"):
                if not config_file.name.startswith("_"):
                    skills.append(config_file.stem)

        if not skills:
            logger.warning("No skills found")
            return 0

        # Get checkpoint ledger
        ledger = get_ledger()
        all_checkpoints = ledger.list_all(limit=100)

        # Sort by step descending (newest first)
        all_checkpoints.sort(key=lambda r: r.step, reverse=True)

        queued = 0
        for skill in skills:
            config = self._load_skill_config(skill)
            if not config:
                continue

            max_level = config.get("max_level", 10)

            for checkpoint_record in all_checkpoints:
                step = checkpoint_record.step

                # Check each level for this skill
                for level in range(1, max_level + 1):
                    if not self.eval_ledger.has_evaluation(step, skill, level):
                        # Queue at low priority (backfill)
                        if queue_evaluation(
                            checkpoint_step=step,
                            skill=skill,
                            level=level,
                            eval_type="quick",
                            priority=3,  # Low priority for backfill
                        ):
                            queued += 1

        logger.info(f"Queued {queued} backfill evaluations")
        return queued

    def show_status(self) -> Dict[str, Any]:
        """Show current evaluation status."""
        pending = get_pending_evaluations()
        quick_pending = [e for e in pending if e.eval_type == "quick"]
        full_pending = [e for e in pending if e.eval_type == "full"]

        summary = self.eval_ledger.summary()

        status = {
            "queue": {
                "total": len(pending),
                "quick": len(quick_pending),
                "full": len(full_pending),
            },
            "ledger": summary,
            "next_up": [],
        }

        # Show next few items
        for entry in pending[:5]:
            level_str = f"L{entry.level}" if entry.level else "ALL"
            status["next_up"].append({
                "checkpoint": entry.checkpoint_step,
                "skill": entry.skill,
                "level": level_str,
                "type": entry.eval_type,
                "priority": entry.priority,
            })

        return status

    def process_passive_queue(self, limit: int = 10) -> int:
        """Process pending passive evaluations. Returns count processed."""
        processed = 0
        skipped = 0

        for _ in range(limit + 100):  # Allow extra iterations for skips
            item = pop_passive()
            if not item:
                break

            # Skip if checkpoint is known to be missing
            checkpoint_step = item["checkpoint_step"]
            if not self._checkpoint_exists(checkpoint_step):
                logger.debug(f"Skipping passive for missing checkpoint-{checkpoint_step}")
                skipped += 1
                continue

            result = self.run_passive_evaluation(
                checkpoint_step=checkpoint_step,
                passive_id=item["passive_id"],
                mode=item["mode"],
            )

            if result:
                processed += 1

            if processed >= limit:
                break

        if skipped > 0:
            logger.info(f"Skipped {skipped} passives for missing checkpoints")

        return processed

    def process_all(self) -> Dict[str, int]:
        """Process all pending evaluations."""
        skills_processed = self.process_skill_queue(limit=100)
        passives_processed = self.process_passive_queue(limit=100)

        return {
            "skills": skills_processed,
            "passives": passives_processed,
        }


def prune_stale_queues(base_dir: Path, also_clean_ledger: bool = False) -> dict:
    """
    Remove queue entries for checkpoints that no longer exist.

    Args:
        base_dir: Base directory for the training project
        also_clean_ledger: If True, also clean stale entries from checkpoint ledger

    Returns dict with counts of pruned items.
    """
    from core.checkpoint_ledger import get_ledger
    from vault.device_mapping import get_local_device_id

    pruned = {"skill_evals": 0, "passives": 0, "ledger_paths": 0, "ledger_entries": 0}

    # Optionally clean ledger first (this is the root cause of stale queue entries)
    if also_clean_ledger:
        try:
            ledger = get_ledger()
            local_device = get_local_device_id()

            # Remove this device from paths that don't exist
            paths_removed = ledger.verify_local_checkpoints(local_device, dry_run=False)
            if paths_removed:
                logger.info(f"Removed {local_device} from {paths_removed} non-existent checkpoint paths")
                pruned["ledger_paths"] = paths_removed

            # Remove entries with no valid locations left
            entries_removed = ledger.cleanup_stale_entries(dry_run=False)
            if entries_removed:
                logger.info(f"Removed {entries_removed} stale entries from checkpoint ledger")
                pruned["ledger_entries"] = entries_removed

        except Exception as e:
            logger.warning(f"Failed to clean ledger: {e}")

    # Get existing checkpoint steps
    try:
        ledger = get_ledger()
        existing_steps = set()
        for record in ledger.list_all():
            if Path(record.path).exists():
                existing_steps.add(record.step)

        logger.info(f"Found {len(existing_steps)} existing checkpoints on disk")
    except Exception as e:
        logger.error(f"Failed to get checkpoint list: {e}")
        return pruned

    # Prune skill eval queue - directly modify the queue file
    try:
        eval_queue_file = base_dir / "status" / "eval_queue.json"
        if eval_queue_file.exists():
            with open(eval_queue_file) as f:
                data = json.load(f)

            original_count = len(data.get("queue", []))
            # Filter out entries where checkpoint doesn't exist
            data["queue"] = [
                item for item in data.get("queue", [])
                if item.get("checkpoint_step") in existing_steps
            ]
            pruned_count = original_count - len(data["queue"])

            if pruned_count > 0:
                with open(eval_queue_file, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Pruned {pruned_count} stale skill evals from queue")
                pruned["skill_evals"] = pruned_count
    except Exception as e:
        logger.error(f"Failed to prune skill queue: {e}")

    # Prune passive queue - directly modify the queue file
    try:
        passive_queue_file = base_dir / "status" / "passive_queue.json"
        if passive_queue_file.exists():
            with open(passive_queue_file) as f:
                data = json.load(f)

            original_count = len(data.get("queue", []))
            data["queue"] = [
                item for item in data.get("queue", [])
                if item.get("checkpoint_step") in existing_steps
            ]
            pruned_count = original_count - len(data["queue"])

            if pruned_count > 0:
                with open(passive_queue_file, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Pruned {pruned_count} stale passives from queue")
                pruned["passives"] = pruned_count
    except Exception as e:
        logger.error(f"Failed to prune passive queue: {e}")

    return pruned


def run_daemon(
    base_dir: Path,
    interval: int = 60,
    inference_host: Optional[str] = None,
    inference_port: Optional[int] = None,
):
    """Run eval runner as daemon."""
    import os
    import random

    # Write PID file for service registry
    pid_file = base_dir / ".pids" / "eval_runner.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    logger.info(f"Starting eval runner daemon (interval={interval}s, pid={os.getpid()})")

    # Prune stale queue entries on startup (with full ledger cleanup)
    pruned = prune_stale_queues(base_dir, also_clean_ledger=True)
    if any(pruned.values()):
        logger.info(f"Startup cleanup: {pruned}")

    runner = EvalRunner(
        base_dir=base_dir,
        inference_host=inference_host,
        inference_port=inference_port,
    )  # EvalRunner uses hosts.json if host/port not provided

    # Periodic cleanup settings
    cycle_count = 0
    CLEANUP_INTERVAL = 10  # Full cleanup every N cycles
    RANDOM_CLEANUP_CHANCE = 0.15  # 15% chance per cycle for quick prune
    last_full_cleanup = time.time()
    FULL_CLEANUP_MIN_INTERVAL = 300  # Minimum 5 minutes between full cleanups

    while True:
        try:
            cycle_count += 1

            # Periodic random cleanup to prevent stale entry buildup
            do_full_cleanup = False
            do_quick_prune = False

            # Full cleanup: every N cycles OR random chance (but not too often)
            time_since_last = time.time() - last_full_cleanup
            if cycle_count % CLEANUP_INTERVAL == 0 and time_since_last >= FULL_CLEANUP_MIN_INTERVAL:
                do_full_cleanup = True
            elif random.random() < RANDOM_CLEANUP_CHANCE:
                do_quick_prune = True

            if do_full_cleanup:
                logger.info(f"[Cycle {cycle_count}] Running periodic full cleanup...")
                pruned = prune_stale_queues(base_dir, also_clean_ledger=True)
                if any(pruned.values()):
                    logger.info(f"Periodic cleanup: {pruned}")
                last_full_cleanup = time.time()
            elif do_quick_prune:
                # Quick prune: just queues, not ledger (faster)
                pruned = prune_stale_queues(base_dir, also_clean_ledger=False)
                if pruned["skill_evals"] or pruned["passives"]:
                    logger.info(f"[Cycle {cycle_count}] Quick prune: {pruned['skill_evals']} skills, {pruned['passives']} passives")

            # Check queues
            skill_pending = len(get_pending_evaluations())
            passive_pending = len(get_pending_passives())

            if skill_pending > 0 or passive_pending > 0:
                logger.info(f"Pending: {skill_pending} skills, {passive_pending} passives")
                results = runner.process_all()
                logger.info(f"Processed: {results['skills']} skills, {results['passives']} passives")
            else:
                logger.debug("No pending evaluations")

        except Exception as e:
            logger.error(f"Error in daemon loop: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Eval Runner - Process skill and passive evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 core/eval_runner.py --status                    # Show queue status
  python3 core/eval_runner.py --once                      # Process all pending
  python3 core/eval_runner.py --once --type quick         # Process only quick evals
  python3 core/eval_runner.py --once --type full          # Process only full evals
  python3 core/eval_runner.py --checkpoint 183000 --full  # Full eval for checkpoint
  python3 core/eval_runner.py --backfill                  # Queue missing evals
  python3 core/eval_runner.py --daemon                    # Run as daemon
        """
    )
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--inference-host", default=None, help="Inference host (default: from hosts.json)")
    parser.add_argument("--inference-port", type=int, default=None, help="Inference port (default: from hosts.json)")

    # Mode flags
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=60, help="Daemon check interval")
    parser.add_argument("--once", action="store_true", help="Process pending and exit")
    parser.add_argument("--status", action="store_true", help="Show detailed status")
    parser.add_argument("--backfill", action="store_true", help="Scan and queue missing evals")

    # Eval type filtering
    parser.add_argument("--type", choices=["quick", "full"], help="Filter by eval type")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (all levels)")

    # Specific evaluation
    parser.add_argument("--checkpoint", type=int, help="Evaluate specific checkpoint")
    parser.add_argument("--skill", help="Skill to evaluate (with --checkpoint)")
    parser.add_argument("--level", type=int, help="Level to evaluate (with --checkpoint)")

    args = parser.parse_args()

    runner = EvalRunner(
        base_dir=args.base_dir,
        inference_host=args.inference_host,
        inference_port=args.inference_port,
    )

    if args.daemon:
        run_daemon(
            base_dir=args.base_dir,
            interval=args.interval,
            inference_host=args.inference_host,
            inference_port=args.inference_port,
        )

    elif args.once:
        # Process with optional type filter
        skills_processed = runner.process_skill_queue(limit=100, eval_type=args.type)
        passives_processed = runner.process_passive_queue(limit=100)
        print(f"Processed: {skills_processed} skills, {passives_processed} passives")

    elif args.backfill:
        queued = runner.scan_for_backfill()
        print(f"Queued {queued} backfill evaluations")

    elif args.status:
        status = runner.show_status()
        print("\n  EVALUATION STATUS")
        print("  " + "=" * 50)
        print(f"  Queue: {status['queue']['total']} pending")
        print(f"    Quick: {status['queue']['quick']}")
        print(f"    Full:  {status['queue']['full']}")
        print(f"\n  Ledger: {status['ledger']['total_evaluations']} recorded")

        if status['ledger']['by_skill']:
            print("\n  By Skill:")
            for skill, info in status['ledger']['by_skill'].items():
                print(f"    {skill}: {info['count']} evals, best={info['best_accuracy']:.0%} @ step {info['best_checkpoint']}")

        if status['next_up']:
            print("\n  Next up:")
            for item in status['next_up']:
                print(f"    checkpoint-{item['checkpoint']} {item['skill']} {item['level']} ({item['type']}, pri={item['priority']})")
        print()

    elif args.checkpoint:
        if args.full:
            # Full evaluation (all levels)
            skill = args.skill
            if not skill:
                print("Specify --skill with --checkpoint --full")
                sys.exit(1)

            results = runner.run_full_skill_evaluation(args.checkpoint, skill)
            print(f"\nFull evaluation: {skill} @ checkpoint-{args.checkpoint}")
            print(f"Completed {len(results)} levels")
            for r in results:
                print(f"  L{r.level}: {r.accuracy:.0%} ({r.correct}/{r.total})")

        elif args.skill and args.level:
            # Single level evaluation
            result = runner.run_skill_evaluation(
                args.checkpoint, args.skill, args.level,
                eval_type="quick"
            )
            if result:
                print(f"Result: {result.accuracy:.1%} ({result.correct}/{result.total})")
            else:
                print("Evaluation failed")
        else:
            print("Specify --skill and --level, or --full with --checkpoint")
            sys.exit(1)

    else:
        # Default: show simple status
        pending = get_pending_evaluations()
        passive_pending = get_pending_passives()
        print(f"Pending skill evaluations: {len(pending)}")
        print(f"Pending passive evaluations: {len(passive_pending)}")

        if pending:
            print("\nSkill queue (sorted by priority):")
            for entry in pending[:5]:
                level_str = f"L{entry.level}" if entry.level else "ALL"
                print(f"  - checkpoint-{entry.checkpoint_step} {entry.skill} {level_str} ({entry.eval_type}, pri={entry.priority})")

        if passive_pending:
            print("\nPassive queue:")
            for item in passive_pending[:5]:
                print(f"  - checkpoint-{item['checkpoint_step']} {item['passive_id']} ({item['mode']})")


if __name__ == "__main__":
    main()
