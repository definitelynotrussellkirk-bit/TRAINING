#!/usr/bin/env python3
"""
Remote Evaluator - Run evaluation jobs on RTX 3090

Handles model evaluation on the remote GPU server to avoid
interrupting training on the 4090.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from urllib import request, error

logger = logging.getLogger(__name__)


def _get_api_key() -> str:
    """Get the inference API key for remote server authentication."""
    try:
        from config.secrets_loader import get_inference_api_key
        return get_inference_api_key()
    except ImportError:
        return "admin123"  # Fallback default


class RemoteEvaluator:
    """
    Evaluator that runs on remote RTX 3090 server

    Decouples evaluation from training - sends eval jobs to remote GPU
    while training continues uninterrupted.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 timeout: int = 300):
        # Use host registry for defaults
        if host is None or port is None:
            try:
                from core.hosts import get_host
                inference_host = get_host("3090")
                if host is None:
                    host = inference_host.host
                if port is None:
                    port = inference_host.services.get("inference", {}).get("port", 8765)
            except Exception:
                # Fallback if host registry unavailable
                host = host or "inference.local"
                port = port or 8765

        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self.api_key = _get_api_key()

    def _get_headers(self, content_type: str = None) -> Dict[str, str]:
        """Get HTTP headers with API key authentication."""
        headers = {"X-API-Key": self.api_key}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _make_request(self, url: str, method: str = "GET",
                     data: bytes = None, timeout: int = 30) -> Any:
        """Make authenticated HTTP request."""
        headers = self._get_headers("application/json" if data else None)
        req = request.Request(url, data=data, headers=headers, method=method)
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def submit_eval_job(self, model_id: str, dataset_ref: str,
                       name: Optional[str] = None,
                       metrics: List[str] = None,
                       config: Optional[Dict] = None) -> str:
        """
        Submit evaluation job to remote server

        Args:
            model_id: Model identifier on remote server
            dataset_ref: Path to evaluation dataset
            name: Job name (auto-generated if None)
            metrics: List of metrics to compute (e.g., ["accuracy", "perplexity"])
            config: Additional eval configuration

        Returns:
            job_id: ID of submitted job
        """
        url = f"{self.base_url}/eval/jobs"

        if name is None:
            name = f"eval_{model_id}_{int(time.time())}"

        if metrics is None:
            metrics = ["accuracy", "loss"]

        payload = {
            "model_id": model_id,
            "name": name,
            "dataset_ref": dataset_ref,
            "metrics": metrics,
        }

        if config:
            payload["config"] = config

        data = json.dumps(payload).encode("utf-8")

        try:
            result = self._make_request(url, method="POST", data=data, timeout=30)
            job_id = result.get("job_id")

            logger.info(f"✅ Submitted eval job: {job_id}")
            logger.info(f"   Model: {model_id}")
            logger.info(f"   Dataset: {dataset_ref}")

            return job_id

        except error.URLError as e:
            logger.error(f"Failed to submit eval job: {e}")
            raise RuntimeError(f"Eval job submission failed: {e}") from e

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check status of evaluation job

        Returns:
            Job status dict with fields: status, progress, results, error
        """
        url = f"{self.base_url}/eval/jobs/{job_id}"

        try:
            return self._make_request(url, timeout=10)

        except error.URLError as e:
            logger.error(f"Failed to get job status: {e}")
            return {"status": "error", "error": str(e)}

    def wait_for_completion(self, job_id: str, max_wait: int = 600,
                           poll_interval: int = 10) -> Dict[str, Any]:
        """
        Wait for evaluation job to complete

        Args:
            job_id: Job ID to wait for
            max_wait: Maximum seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            Final job status with results
        """
        start_time = time.time()
        last_status = None

        while time.time() - start_time < max_wait:
            status = self.get_job_status(job_id)
            current_status = status.get("status", "unknown")

            if current_status != last_status:
                logger.info(f"Eval job {job_id}: {current_status}")
                last_status = current_status

            if current_status in ["completed", "failed", "error"]:
                return status

            time.sleep(poll_interval)

        # Timeout
        logger.warning(f"Eval job {job_id} timed out after {max_wait}s")
        return {"status": "timeout", "job_id": job_id}

    def quick_eval(self, model_id: str, dataset_ref: str,
                  wait: bool = True) -> Optional[Dict[str, Any]]:
        """
        Convenience method: submit eval and optionally wait for results

        Args:
            model_id: Model to evaluate
            dataset_ref: Evaluation dataset
            wait: If True, wait for completion and return results

        Returns:
            Results dict if wait=True, else None
        """
        job_id = self.submit_eval_job(model_id, dataset_ref)

        if wait:
            result = self.wait_for_completion(job_id)
            return result.get("results")
        else:
            logger.info(f"Eval job {job_id} submitted (not waiting)")
            return None

    def register_checkpoint(self, checkpoint_path: str, model_id: str,
                           tags: Optional[str] = None) -> bool:
        """
        Register a checkpoint with remote server

        Args:
            checkpoint_path: Path to checkpoint on remote server
            model_id: ID to assign
            tags: Optional tags (e.g., "step5000,math")

        Returns:
            True if successful
        """
        url = f"{self.base_url}/models/register"

        payload = {
            "id": model_id,
            "path": checkpoint_path,
            "source": "4090_training"
        }

        if tags:
            payload["tags"] = tags

        data = json.dumps(payload).encode("utf-8")

        try:
            self._make_request(url, method="POST", data=data, timeout=30)
            logger.info(f"✅ Registered checkpoint: {model_id}")
            return True

        except error.URLError as e:
            logger.error(f"Failed to register checkpoint: {e}")
            return False

    def list_jobs(self, status: Optional[str] = None) -> List[Dict]:
        """
        List evaluation jobs

        Args:
            status: Filter by status (e.g., "running", "completed")

        Returns:
            List of job dicts
        """
        url = f"{self.base_url}/eval/jobs"

        if status:
            url += f"?status={status}"

        try:
            return self._make_request(url, timeout=10)

        except error.URLError as e:
            logger.error(f"Failed to list jobs: {e}")
            return []

    def get_active_model(self) -> Optional[str]:
        """Get currently active model on remote server"""
        url = f"{self.base_url}/models/active"

        try:
            result = self._make_request(url, timeout=10)
            return result.get("model_id")

        except error.URLError as e:
            logger.error(f"Failed to get active model: {e}")
            return None

    def set_active_model(self, model_id: str) -> bool:
        """Set active model on remote server"""
        url = f"{self.base_url}/models/set_active"

        payload = {"id": model_id}
        data = json.dumps(payload).encode("utf-8")

        try:
            self._make_request(url, method="POST", data=data, timeout=30)
            logger.info(f"✅ Set active model: {model_id}")
            return True

        except error.URLError as e:
            logger.error(f"Failed to set active model: {e}")
            return False


def main():
    """CLI for Remote Evaluator"""
    import argparse

    parser = argparse.ArgumentParser(description="Remote Evaluator - Run eval on RTX 3090")
    parser.add_argument('--host', default=None, help='Remote server host (auto-detect from host registry)')
    parser.add_argument('--port', type=int, default=None, help='Remote server port (auto-detect from host registry)')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Submit eval
    eval_parser = subparsers.add_parser('eval', help='Submit evaluation job')
    eval_parser.add_argument('model_id', help='Model ID')
    eval_parser.add_argument('dataset', help='Dataset path')
    eval_parser.add_argument('--wait', action='store_true', help='Wait for completion')
    eval_parser.add_argument('--metrics', nargs='+', default=['accuracy', 'loss'],
                            help='Metrics to compute')

    # Check job
    status_parser = subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument('job_id', help='Job ID')

    # List jobs
    subparsers.add_parser('list', help='List evaluation jobs')

    # Register checkpoint
    register_parser = subparsers.add_parser('register', help='Register checkpoint')
    register_parser.add_argument('checkpoint_path', help='Path on remote server')
    register_parser.add_argument('model_id', help='Model ID to assign')
    register_parser.add_argument('--tags', help='Tags (e.g., step5000,math)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    evaluator = RemoteEvaluator(args.host, args.port)

    if args.command == 'eval':
        job_id = evaluator.submit_eval_job(
            model_id=args.model_id,
            dataset_ref=args.dataset,
            metrics=args.metrics
        )

        print(f"\n✅ Eval job submitted: {job_id}")

        if args.wait:
            print("\nWaiting for completion...")
            result = evaluator.wait_for_completion(job_id)

            print(f"\nStatus: {result.get('status')}")
            if result.get('results'):
                print("\nResults:")
                print(json.dumps(result['results'], indent=2))

    elif args.command == 'status':
        status = evaluator.get_job_status(args.job_id)
        print(json.dumps(status, indent=2))

    elif args.command == 'list':
        jobs = evaluator.list_jobs()
        print(f"\nTotal jobs: {len(jobs)}\n")
        for job in jobs:
            print(f"  {job.get('id')}: {job.get('status')}")

    elif args.command == 'register':
        success = evaluator.register_checkpoint(
            checkpoint_path=args.checkpoint_path,
            model_id=args.model_id,
            tags=args.tags
        )

        if success:
            print(f"\n✅ Registered: {args.model_id}")
        else:
            print(f"\n❌ Failed to register checkpoint")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
