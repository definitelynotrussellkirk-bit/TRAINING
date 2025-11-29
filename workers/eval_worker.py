"""
Eval Worker - Worker for evaluation and sparring jobs.

Handles:
- EVAL: Skill evaluations (requires inference server access)
- SPARRING: Self-correction sparring sessions

Usage:
    # Start on a machine with inference server access
    python3 -m workers.eval_worker --port 8900

    # With specific device ID
    python3 -m workers.eval_worker --device macmini_eval_1 --port 8900
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from workers.base_worker import BaseWorker, WorkerConfig

logger = logging.getLogger("eval_worker")


class EvalWorker(BaseWorker):
    """
    Worker for evaluation and sparring jobs.

    Requires access to an inference server for model generation.
    """

    SUPPORTED_TYPES = ["eval", "sparring", "inference"]

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        device_id: Optional[str] = None,
        inference_url: Optional[str] = None,
    ):
        """
        Initialize eval worker.

        Args:
            config: Worker configuration
            device_id: Device ID override
            inference_url: Inference server URL (default from INFERENCE_URL env)
        """
        super().__init__(config, device_id)

        # Get inference URL - use service discovery from hosts.json
        if inference_url:
            self.inference_url = inference_url
        else:
            from core.hosts import get_service_url
            self.inference_url = get_service_url("inference") or os.environ.get(
                "INFERENCE_URL", "http://localhost:8765"
            )

        # Verify inference connection
        self._check_inference()

    def _check_inference(self) -> bool:
        """Check inference server connectivity."""
        try:
            response = requests.get(f"{self.inference_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to inference server: {self.inference_url}")
                return True
        except Exception as e:
            logger.warning(f"Inference server not available: {e}")
        return False

    def get_supported_job_types(self) -> List[str]:
        """Get supported job types."""
        return self.SUPPORTED_TYPES

    def handle_job(self, job_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an eval/sparring/inference job.

        Args:
            job_id: Job identifier
            spec: Job specification

        Returns:
            Result dict
        """
        job_type = spec.get("job_type", "unknown")
        payload = spec.get("payload", {})

        logger.info(f"Handling {job_type} job {job_id}")

        # Verify context hash if provided (detects context drift)
        mismatch = self._verify_context_hash(payload)
        if mismatch:
            logger.warning(f"Context mismatch for job {job_id}: {mismatch}")
            # Log warning but continue - job payload has explicit identity

        if job_type == "eval":
            return self._handle_eval(payload)
        elif job_type == "sparring":
            return self._handle_sparring(payload)
        elif job_type == "inference":
            return self._handle_inference(payload)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    def _verify_context_hash(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        Verify context hash in payload matches current RunContext.

        This is a warning-only check - jobs with explicit model identity
        in their payload are still valid even if context has drifted.

        Returns:
            Mismatch description if detected, None if OK or no hash in payload
        """
        job_hash = payload.get("context_hash")
        if not job_hash:
            # No hash in payload - skip verification
            return None

        try:
            from core.run_context import get_run_context
            ctx = get_run_context()
            current_hash = ctx.context_hash()

            if job_hash != current_hash:
                return (
                    f"Job context_hash={job_hash[:8]}... does not match "
                    f"current={current_hash[:8]}... "
                    f"(job.hero={payload.get('hero_id')}, current.hero={ctx.hero_id})"
                )
        except Exception as e:
            logger.warning(f"Could not verify context hash: {e}")

        return None

    def _handle_eval(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle evaluation job.

        Payload:
            skill_id: Skill to evaluate
            level: Skill level
            batch_size: Number of problems
        """
        skill_id = payload.get("skill_id", "bin")
        level = payload.get("level", 1)
        batch_size = payload.get("batch_size", 100)

        logger.info(f"Running eval: skill={skill_id}, level={level}, batch={batch_size}")

        # Import skill engine
        try:
            from guild.skills import get_engine

            engine = get_engine()

            # Generate eval batch
            batch = engine.generate_eval_batch(skill_id, level=level, count=batch_size)

            if not batch or not batch.problems:
                return {
                    "success": False,
                    "error": f"Could not generate eval batch for {skill_id}",
                }

            # Get model answers via inference
            answers = []
            for problem in batch.problems:
                try:
                    answer = self._generate(problem.prompt)
                    answers.append(answer)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    answers.append("")

            # Score results
            result, state = engine.run_eval(skill_id, answers, level=level, count=batch_size)

            return {
                "success": True,
                "skill_id": skill_id,
                "level": level,
                "accuracy": result.accuracy,
                "correct": result.num_correct,
                "total": result.num_examples,
                "per_primitive": result.per_primitive_accuracy,
            }

        except ImportError as e:
            logger.error(f"Failed to import skill engine: {e}")
            return {
                "success": False,
                "error": f"Skill engine not available: {e}",
            }
        except Exception as e:
            logger.error(f"Eval failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _handle_sparring(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle sparring job.

        Payload:
            skill_id: Skill to spar with
            count: Number of problems
            checkpoint: Optional checkpoint to use
        """
        skill_id = payload.get("skill_id", "binary")
        count = payload.get("count", 100)
        checkpoint = payload.get("checkpoint")

        logger.info(f"Running sparring: skill={skill_id}, count={count}")

        try:
            from guild.sparring import run_sparring_session

            result = run_sparring_session(
                skill=skill_id,
                count=count,
                checkpoint=checkpoint,
                inference_url=self.inference_url,
            )

            return {
                "success": True,
                "skill_id": skill_id,
                "problems_attempted": result.get("attempted", 0),
                "correct": result.get("correct", 0),
                "training_examples": result.get("examples_generated", 0),
                "output_file": result.get("output_file"),
            }

        except ImportError as e:
            logger.error(f"Failed to import sparring module: {e}")
            return {
                "success": False,
                "error": f"Sparring module not available: {e}",
            }
        except Exception as e:
            logger.error(f"Sparring failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _handle_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle direct inference job.

        Payload:
            prompt: Text to generate from
            max_tokens: Max tokens to generate
        """
        prompt = payload.get("prompt", "")
        max_tokens = payload.get("max_tokens", 100)

        if not prompt:
            return {"success": False, "error": "No prompt provided"}

        try:
            response = self._generate(prompt, max_tokens=max_tokens)
            return {
                "success": True,
                "prompt": prompt,
                "response": response,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from inference server."""
        response = requests.post(
            f"{self.inference_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            headers={"X-API-Key": "admin123"},
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Inference failed: {response.status_code}")

        data = response.json()
        return data.get("text", data.get("response", ""))


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Eval Worker - Run evaluations and sparring")
    parser.add_argument("--port", type=int, default=8900, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--device", help="Device ID (default from TRAINING_DEVICE_ID)")
    parser.add_argument("--inference-url", help="Inference server URL")
    parser.add_argument("--max-concurrent", type=int, default=1, help="Max concurrent jobs")

    args = parser.parse_args()

    config = WorkerConfig(
        device_id=args.device or os.environ.get("TRAINING_DEVICE_ID", "eval_worker"),
        max_concurrent=args.max_concurrent,
    )

    worker = EvalWorker(
        config=config,
        inference_url=args.inference_url,
    )

    worker.run(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
