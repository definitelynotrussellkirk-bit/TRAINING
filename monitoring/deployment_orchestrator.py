#!/usr/bin/env python3
"""
Deployment Orchestrator - Automated checkpoint deployment to 3090

Uses checkpoint ledger to find best checkpoint and deploy to 3090.

Architecture:
- Runs on 4090 (training machine)
- Uses ledger get_best() for checkpoint selection
- rsync checkpoints to 3090
- Triggers reload via API
- Logs all deployments
"""

import json
import time
import logging
import subprocess
import hashlib
import sys
import os
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# API keys from environment
ADMIN_KEY = os.environ.get("INFERENCE_ADMIN_KEY", "")
READ_KEY = os.environ.get("INFERENCE_READ_KEY", "")

# Protected directories - never delete (base models needed for PEFT loading)
PROTECTED_MODEL_DIRS = {
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-4B-Instruct-2507",
    "Qwen3-8B",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
}  # Never delete base models - needed for PEFT adapter loading

def _get_remote_models_dir() -> str:
    """Get remote models directory from hosts.json."""
    try:
        from core.hosts import get_host
        host = get_host("3090")
        if host and host.models_dir:
            return host.models_dir
    except Exception:
        pass
    raise RuntimeError(
        "Cannot determine remote models_dir. "
        "Ensure 3090 host has models_dir configured in hosts.json."
    )

def _get_cleanup_paths() -> list:
    """Get cleanup paths based on 3090 configuration."""
    try:
        from core.hosts import get_host
        host = get_host("3090")
        if host and host.ssh_user:
            home = f"/home/{host.ssh_user}"
            return [
                (f"{home}/trained_adapter", "old adapter files"),
                (f"{home}/3090_server_backup_*.tar.gz", "old backup archives"),
                (f"{home}/3090_status_backup_*.tar.gz", "old status backups"),
            ]
    except Exception:
        pass
    return []  # No cleanup paths if we can't determine them

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentOrchestrator:
    """
    Orchestrates automated checkpoint deployment from 4090 to 3090.

    Smart Push Logic:
    - Only push if checkpoint is significantly different from last pushed
    - Push immediately if it's a NEW best (better loss than last pushed)
    - Minimum steps between pushes to avoid spam (unless new best)
    - Persist state across restarts via deployment log
    """

    def __init__(
        self,
        base_dir: str = None,
        remote_host: str = None,
        remote_api_url: str = None,
        check_interval: int = 600,  # 10 minutes
        min_steps_between_push: int = 5000,  # Don't push more often than this (unless new best)
        push_on_improvement: bool = True,  # Always push if loss improved
    ):
        if base_dir is None:
            from core.paths import require_base_dir
            base_dir = str(require_base_dir())
        if remote_host is None:
            try:
                from core.hosts import get_host
                remote_host = get_host("3090").host
            except (ImportError, Exception):
                remote_host = "inference.local"
        if remote_api_url is None:
            try:
                from core.hosts import get_service_url
                remote_api_url = get_service_url("inference")
            except (ImportError, Exception):
                remote_api_url = "http://inference.local:8765"
        self.base_dir = Path(base_dir)
        self.remote_host = remote_host
        self.remote_api_url = remote_api_url.rstrip('/')
        self.check_interval = check_interval

        # Smart push settings
        self.min_steps_between_push = min_steps_between_push
        self.push_on_improvement = push_on_improvement

        # Paths
        self.deployment_log = self.base_dir / "status" / "deployment_status.json"
        self.checkpoint_dir = self.base_dir / "models" / "current_model"

        # State - restored from deployment log for persistence
        self.last_deployed_step = None
        self.last_deployed_loss = None
        self._restore_last_deployment()

        logger.info("Deployment Orchestrator initialized")
        logger.info(f"Base dir: {self.base_dir}")
        logger.info(f"Remote: {self.remote_host}")
        logger.info(f"API: {self.remote_api_url}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"Min steps between push: {self.min_steps_between_push}")
        logger.info(f"Push on improvement: {self.push_on_improvement}")
        if self.last_deployed_step:
            logger.info(f"Last deployed: step {self.last_deployed_step} (loss: {self.last_deployed_loss})")

    def _restore_last_deployment(self):
        """Restore last deployment state from log file for persistence across restarts."""
        if not self.deployment_log.exists():
            return

        try:
            with open(self.deployment_log) as f:
                log_data = json.load(f)

            if log_data and isinstance(log_data, list):
                # Get most recent successful deployment
                for entry in reversed(log_data):
                    if entry.get("status") == "success":
                        self.last_deployed_step = entry.get("step")
                        self.last_deployed_loss = entry.get("metrics", {}).get("train_loss")
                        break
        except Exception as e:
            logger.warning(f"Could not restore last deployment state: {e}")

    def should_deploy(self, candidate: Dict[str, Any]) -> tuple[bool, str]:
        """
        Decide whether to deploy a checkpoint using smart push logic.

        Args:
            candidate: Checkpoint info dict with 'step' and 'loss' keys

        Returns:
            Tuple of (should_deploy, reason)
        """
        step = candidate.get('step')
        loss = candidate.get('loss')

        # First deployment - always deploy
        if self.last_deployed_step is None:
            return True, "first_deployment"

        # Same checkpoint - skip
        if step == self.last_deployed_step:
            return False, "already_deployed"

        # Check if this is a new best (better loss)
        is_new_best = False
        if loss is not None and self.last_deployed_loss is not None:
            is_new_best = loss < self.last_deployed_loss

        # Always push if it's a new best and push_on_improvement is enabled
        if self.push_on_improvement and is_new_best:
            return True, f"new_best_loss ({loss:.4f} < {self.last_deployed_loss:.4f})"

        # Check minimum steps between pushes
        steps_since_last = step - self.last_deployed_step
        if steps_since_last < self.min_steps_between_push:
            return False, f"too_soon ({steps_since_last} < {self.min_steps_between_push} steps)"

        # Enough steps have passed - deploy
        return True, f"step_threshold ({steps_since_last} >= {self.min_steps_between_push})"

    def get_active_hero_context(self):
        """Get the currently active hero context."""
        try:
            from core.eval_dynamics import get_active_hero_context
            return get_active_hero_context(self.base_dir)
        except Exception as e:
            logger.warning(f"Could not get active hero context: {e}")
            return None

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get best checkpoint from ledger (by lowest train_loss) for the active hero.

        Returns:
            {
                'step': 156000,
                'checkpoint_path': Path('/path/to/checkpoint-156000'),
                'checkpoint_name': 'checkpoint-156000',
                'loss': 0.12,
                'hero_id': 'ojas-qwen3-8b',
            }
            or None if no valid checkpoint found
        """
        try:
            from core.checkpoint_ledger import get_ledger

            # Get active hero context
            hero_ctx = self.get_active_hero_context()
            hero_id = hero_ctx.hero_id if hero_ctx else None

            if hero_id:
                logger.info(f"Looking for best checkpoint for hero: {hero_id}")

            ledger = get_ledger()
            best = ledger.get_best(metric="train_loss", hero_id=hero_id)

            if not best:
                # Try path-based search for the hero's checkpoints
                if hero_ctx and hero_ctx.checkpoint_dir.exists():
                    from core.eval_dynamics import get_latest_checkpoint_for_hero
                    latest = get_latest_checkpoint_for_hero(hero_id, self.base_dir)
                    if latest:
                        logger.info(f"Using latest checkpoint from hero's campaign: {latest}")
                        return {
                            'step': int(latest.name.split('-')[1].split('-')[0]),
                            'checkpoint_path': latest,
                            'checkpoint_name': latest.name,
                            'loss': None,  # Not in ledger
                            'hero_id': hero_id,
                        }

                logger.debug("No checkpoints in ledger for active hero")
                return None

            # Find checkpoint path - check hero's campaign dir first
            checkpoint_path = None
            if hero_ctx and hero_ctx.checkpoint_dir.exists():
                # Check in hero's checkpoint dir
                checkpoint_path = hero_ctx.checkpoint_dir / f"checkpoint-{best.step}"
                if not checkpoint_path.exists() and best.canonical_name:
                    checkpoint_path = hero_ctx.checkpoint_dir / best.canonical_name

            # Fall back to legacy location
            if not checkpoint_path or not checkpoint_path.exists():
                checkpoint_path = self.checkpoint_dir / f"checkpoint-{best.step}"
                if not checkpoint_path.exists() and best.canonical_name:
                    checkpoint_path = self.checkpoint_dir / best.canonical_name

            # Last resort: use the path from the ledger record
            if not checkpoint_path or not checkpoint_path.exists():
                checkpoint_path = Path(best.path)

            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: step {best.step}")
                return None

            return {
                'step': best.step,
                'checkpoint_path': checkpoint_path,
                'checkpoint_name': checkpoint_path.name,
                'loss': best.train_loss,
                'hero_id': hero_id,
            }

        except Exception as e:
            logger.error(f"Error getting best checkpoint from ledger: {e}")
            return None

    def get_deployment_metadata(self) -> Dict[str, str]:
        """Collect system metadata for deployment records"""
        metadata = {}

        # Config hash
        try:
            config_file = self.base_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                config_str = json.dumps(config, sort_keys=True)
                metadata['config_hash'] = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"Could not read config: {e}")
            metadata['config_hash'] = "unknown"

        # Git commit
        try:
            result = subprocess.run(
                ["git", "-C", str(self.base_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                metadata['git_commit'] = result.stdout.strip()[:8]
            else:
                metadata['git_commit'] = "unknown"
        except Exception as e:
            logger.warning(f"Could not get git commit: {e}")
            metadata['git_commit'] = "unknown"

        # Python/torch versions
        metadata['python_version'] = sys.version.split()[0]

        try:
            import torch
            metadata['torch_version'] = torch.__version__
        except:
            metadata['torch_version'] = "unknown"

        return metadata

    def rsync_checkpoint(self, ckpt_info: Dict[str, Any]) -> tuple[bool, float]:
        """
        rsync checkpoint to 3090 models/checkpoint-NNNNNN/ (preserves checkpoint name)

        Returns:
            (success: bool, duration: float)
        """
        local_path = ckpt_info['checkpoint_path']
        checkpoint_name = ckpt_info['checkpoint_name']  # e.g., checkpoint-175000
        remote_models_dir = _get_remote_models_dir()
        remote_target = f"{self.remote_host}:{remote_models_dir}/{checkpoint_name}/"

        logger.info(f"📦 Syncing {checkpoint_name} to 3090 as models/{checkpoint_name}/...")

        cmd = [
            "rsync",
            "-avz",
            "--delete",
            "--checksum",
            str(local_path) + "/",  # Trailing slash = sync contents
            remote_target
        ]

        start = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            duration = time.time() - start

            if result.returncode == 0:
                logger.info(f"✅ Synced in {duration:.1f}s")
                return True, duration
            else:
                logger.error(f"❌ rsync failed: {result.stderr}")
                return False, duration

        except subprocess.TimeoutExpired:
            duration = time.time() - start
            logger.error(f"❌ rsync timeout after {duration:.1f}s")
            return False, duration
        except Exception as e:
            duration = time.time() - start
            logger.error(f"❌ rsync error: {e}")
            return False, duration

    def reload_remote_model(self, ckpt_info: Dict[str, Any]) -> tuple[bool, Optional[Dict]]:
        """
        Call 3090 /models/reload API with explicit checkpoint path

        Requires INFERENCE_ADMIN_KEY environment variable.

        Returns:
            (success: bool, response: Optional[Dict])
        """
        url = f"{self.remote_api_url}/models/reload"
        checkpoint_name = ckpt_info['checkpoint_name']
        remote_models_dir = _get_remote_models_dir()
        model_path = f"{remote_models_dir}/{checkpoint_name}"

        logger.info(f"🔄 Loading {checkpoint_name} on 3090...")

        # Build headers with admin API key
        headers = {"Content-Type": "application/json"}
        if ADMIN_KEY:
            headers['X-API-Key'] = ADMIN_KEY
        else:
            logger.warning("⚠️  INFERENCE_ADMIN_KEY not set - reload may fail")

        # Send model_path in request body
        payload = {"model_path": model_path}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ Loaded {result.get('model_id', checkpoint_name)} from {result.get('loaded_from', model_path)}")

            return True, result

        except requests.exceptions.Timeout:
            logger.error("❌ Reload timeout")
            return False, None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Reload failed: {e}")
            return False, None
        except Exception as e:
            logger.error(f"❌ Reload error: {e}")
            return False, None

    def verify_deployment(self, expected_step: int) -> bool:
        """
        Verify 3090 is serving expected checkpoint

        Requires INFERENCE_READ_KEY or INFERENCE_ADMIN_KEY environment variable.

        Returns:
            True if deployment verified
        """
        url = f"{self.remote_api_url}/models/info"

        logger.info(f"🔍 Verifying deployment (expecting step {expected_step})...")

        # Build headers with API key (read key or admin key)
        headers = {}
        api_key = READ_KEY or ADMIN_KEY
        if api_key:
            headers['X-API-Key'] = api_key

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            info = response.json()

            # Check if model is loaded
            if not info.get('loaded'):
                logger.error("❌ Model not loaded")
                return False

            # Check checkpoint step (may be None if not extractable)
            actual_step = info.get('checkpoint_step')

            if actual_step == expected_step:
                logger.info(f"✅ Verified: step {expected_step}, model_id={info.get('model_id')}")
                return True
            elif actual_step is None:
                # Step not extractable, check model_id looks like a checkpoint
                model_id = info.get('model_id', '')
                if model_id.startswith('checkpoint-') or model_id == f'checkpoint-{expected_step}':
                    logger.warning(f"⚠️  Step not extractable, but model_id={model_id} - assuming success")
                    return True
                else:
                    logger.error(f"❌ Model ID is {model_id}, expected checkpoint-{expected_step}")
                    return False
            else:
                logger.error(f"❌ Step mismatch: expected {expected_step}, got {actual_step}")
                return False

        except Exception as e:
            logger.error(f"❌ Verification error: {e}")
            return False

    def cleanup_remote_models(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up old model directories and files on the remote 3090.

        Removes:
        - Old model directories in ~/llm/models/ (except protected ones)
        - Old backup archives
        - Old adapter files

        Returns:
            Summary dict with cleanup results
        """
        logger.info("🧹 Running remote cleanup on 3090...")

        results = {
            "models_deleted": [],
            "files_deleted": [],
            "bytes_freed": 0,
            "errors": []
        }

        # Step 1: Find and remove old model directories
        try:
            # List all directories in models folder
            cmd = f"ssh {self.remote_host} 'ls -1 {REMOTE_MODELS_DIR}/ 2>/dev/null'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                dirs = [d.strip() for d in result.stdout.strip().split('\n') if d.strip()]

                for dir_name in dirs:
                    if dir_name in PROTECTED_MODEL_DIRS:
                        logger.debug(f"Protected: {dir_name}")
                        continue

                    dir_path = f"{REMOTE_MODELS_DIR}/{dir_name}"

                    # Get size before deletion
                    size_cmd = f"ssh {self.remote_host} 'du -sb {dir_path} 2>/dev/null | cut -f1'"
                    size_result = subprocess.run(size_cmd, shell=True, capture_output=True, text=True, timeout=30)
                    size_bytes = int(size_result.stdout.strip()) if size_result.returncode == 0 and size_result.stdout.strip() else 0

                    if dry_run:
                        logger.info(f"  Would delete: {dir_name} ({size_bytes / (1024**3):.1f}GB)")
                        results["models_deleted"].append({"name": dir_name, "size_bytes": size_bytes})
                        results["bytes_freed"] += size_bytes
                    else:
                        logger.info(f"  Deleting: {dir_name} ({size_bytes / (1024**3):.1f}GB)")
                        del_cmd = f"ssh {self.remote_host} 'rm -rf {dir_path}'"
                        del_result = subprocess.run(del_cmd, shell=True, capture_output=True, text=True, timeout=300)

                        if del_result.returncode == 0:
                            results["models_deleted"].append({"name": dir_name, "size_bytes": size_bytes})
                            results["bytes_freed"] += size_bytes
                        else:
                            results["errors"].append(f"Failed to delete {dir_name}: {del_result.stderr}")
        except Exception as e:
            results["errors"].append(f"Model cleanup error: {e}")

        # Step 2: Clean up other known paths
        for path_pattern, description in CLEANUP_PATHS:
            try:
                # Check if path/pattern exists
                check_cmd = f"ssh {self.remote_host} 'ls -d {path_pattern} 2>/dev/null'"
                check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=30)

                if check_result.returncode == 0 and check_result.stdout.strip():
                    paths = check_result.stdout.strip().split('\n')

                    for path in paths:
                        path = path.strip()
                        if not path:
                            continue

                        # Get size
                        size_cmd = f"ssh {self.remote_host} 'du -sb {path} 2>/dev/null | cut -f1'"
                        size_result = subprocess.run(size_cmd, shell=True, capture_output=True, text=True, timeout=30)
                        size_bytes = int(size_result.stdout.strip()) if size_result.returncode == 0 and size_result.stdout.strip() else 0

                        if dry_run:
                            logger.info(f"  Would delete: {path} - {description} ({size_bytes / (1024**3):.1f}GB)")
                            results["files_deleted"].append({"path": path, "description": description, "size_bytes": size_bytes})
                            results["bytes_freed"] += size_bytes
                        else:
                            logger.info(f"  Deleting: {path} - {description} ({size_bytes / (1024**3):.1f}GB)")
                            del_cmd = f"ssh {self.remote_host} 'rm -rf {path}'"
                            del_result = subprocess.run(del_cmd, shell=True, capture_output=True, text=True, timeout=300)

                            if del_result.returncode == 0:
                                results["files_deleted"].append({"path": path, "description": description, "size_bytes": size_bytes})
                                results["bytes_freed"] += size_bytes
                            else:
                                results["errors"].append(f"Failed to delete {path}: {del_result.stderr}")
            except Exception as e:
                results["errors"].append(f"Cleanup error for {path_pattern}: {e}")

        # Summary
        total_gb = results["bytes_freed"] / (1024**3)
        mode = "Would free" if dry_run else "Freed"
        logger.info(f"🧹 Cleanup complete: {mode} {total_gb:.1f}GB")

        if results["errors"]:
            for err in results["errors"]:
                logger.warning(f"  ⚠️  {err}")

        return results

    def log_deployment(self, record: Dict[str, Any]):
        """Append deployment record to log file"""
        history = []

        if self.deployment_log.exists():
            try:
                with open(self.deployment_log) as f:
                    history = json.load(f)
            except:
                logger.warning("Could not read deployment log, starting fresh")
                history = []

        history.append(record)

        # Keep last 100 deployments
        history = history[-100:]

        try:
            with open(self.deployment_log, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"📝 Logged deployment: {record['deployment_id']}")
        except Exception as e:
            logger.error(f"Failed to write deployment log: {e}")

    def deploy_full(self, ckpt_info: Dict[str, Any]) -> bool:
        """
        Run full deployment pipeline:
        1. rsync checkpoint
        2. Reload model
        3. Verify deployment
        4. Log result

        Returns:
            True if deployment successful
        """
        deployment_id = f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        logger.info("="*80)
        logger.info(f"🚀 STARTING DEPLOYMENT: {deployment_id}")
        logger.info(f"Checkpoint: {ckpt_info['checkpoint_name']}")
        logger.info(f"Step: {ckpt_info['step']}")
        logger.info(f"Score: {ckpt_info['score']:.3f}")
        logger.info(f"Loss: {ckpt_info['loss']:.3f}")
        logger.info(f"Accuracy: {ckpt_info['accuracy']:.1%}")
        logger.info("="*80)

        # Initialize record
        record = {
            'deployment_id': deployment_id,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_step': ckpt_info['step'],
            'checkpoint_name': ckpt_info['checkpoint_name'],
            'checkpoint_path': str(ckpt_info['checkpoint_path']),
            'score': ckpt_info['score'],
            'loss': ckpt_info['loss'],
            'accuracy': ckpt_info['accuracy'],
            **self.get_deployment_metadata(),
            'status': 'in_progress'
        }

        # Step 1: rsync
        success, rsync_duration = self.rsync_checkpoint(ckpt_info)
        record['rsync_duration_sec'] = rsync_duration

        if not success:
            record['status'] = 'failed_rsync'
            self.log_deployment(record)
            logger.error(f"❌ DEPLOYMENT FAILED: rsync error")
            return False

        # Step 2: reload
        success, reload_result = self.reload_remote_model(ckpt_info)

        if not success:
            record['status'] = 'failed_reload'
            self.log_deployment(record)
            logger.error(f"❌ DEPLOYMENT FAILED: reload error")
            return False

        if reload_result:
            record['reload_vram_gb'] = reload_result.get('vram_usage_gb')
            record['reload_timestamp'] = reload_result.get('loaded_at')

        # Step 3: verify
        if not self.verify_deployment(ckpt_info['step']):
            record['status'] = 'failed_verification'
            self.log_deployment(record)
            logger.error(f"❌ DEPLOYMENT FAILED: verification error")
            return False

        # Success!
        record['status'] = 'success'
        record['verified_at'] = datetime.now().isoformat()
        self.log_deployment(record)

        # Record in ledger that checkpoint now exists on inference server
        try:
            from core.checkpoint_ledger import get_ledger
            ledger = get_ledger()
            ledger.record_usage(ckpt_info['step'], "inference3090")
            logger.info(f"📖 Ledger updated: checkpoint {ckpt_info['step']} → inference3090")
        except Exception as e:
            logger.warning(f"Failed to update ledger: {e}")

        logger.info("="*80)
        logger.info(f"✅ DEPLOYMENT SUCCESSFUL: {deployment_id}")
        logger.info("="*80)

        # Step 4: Cleanup old models on remote
        logger.info("Running post-deployment cleanup...")
        cleanup_results = self.cleanup_remote_models(dry_run=False)
        record['cleanup'] = cleanup_results

        return True

    def run_continuous(self):
        """Run continuous deployment loop with smart push logic."""
        logger.info("="*80)
        logger.info("🚀 DEPLOYMENT ORCHESTRATOR - STARTING (Smart Push Mode)")
        logger.info("="*80)

        while True:
            try:
                # Get best checkpoint
                best = self.get_best_checkpoint()

                if best:
                    step = best['step']
                    loss = best.get('loss')
                    loss_str = f"{loss:.4f}" if loss else "N/A"

                    logger.info(f"📊 Best checkpoint: step {step}, loss {loss_str}")

                    # Use smart push logic to decide whether to deploy
                    should_push, reason = self.should_deploy(best)

                    if should_push:
                        logger.info(f"📦 Deploying: {reason}")

                        # Deploy
                        success = self.deploy_full(best)

                        if success:
                            self.last_deployed_step = step
                            self.last_deployed_loss = loss
                            logger.info(f"✓ Deployed step {step}")
                        else:
                            logger.error("Deployment failed, will retry next cycle")
                    else:
                        logger.info(f"⏭️  Skipping deployment: {reason}")
                else:
                    logger.info("⏳ No valid checkpoint available yet")

                # Sleep until next check
                logger.info(f"💤 Next check in {self.check_interval}s...")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("\n🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in orchestrator loop: {e}", exc_info=True)
                logger.info(f"⏳ Retrying in {self.check_interval}s...")
                time.sleep(self.check_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment Orchestrator")
    parser.add_argument('--base-dir', default=None,
                       help="Base training directory (auto-detect if not set)")
    parser.add_argument('--remote-host', default=None,
                       help="Remote inference server hostname (from hosts.json if not set)")
    parser.add_argument('--api-url', default=None,
                       help="Remote API URL (from hosts.json if not set)")
    parser.add_argument('--interval', type=int, default=600,
                       help="Check interval in seconds (default: 600)")

    # Smart push options
    parser.add_argument('--min-steps', type=int, default=5000,
                       help="Minimum steps between pushes unless new best (default: 5000)")
    parser.add_argument('--no-push-on-improvement', action='store_true',
                       help="Don't push immediately on loss improvement (respect min-steps)")

    # Cleanup options
    parser.add_argument('--cleanup', action='store_true',
                       help="Run remote cleanup only (no deployment)")
    parser.add_argument('--cleanup-dry-run', action='store_true',
                       help="Show what would be cleaned up (no deletion)")

    args = parser.parse_args()

    orchestrator = DeploymentOrchestrator(
        base_dir=args.base_dir,
        remote_host=args.remote_host,
        remote_api_url=args.api_url,
        check_interval=args.interval,
        min_steps_between_push=args.min_steps,
        push_on_improvement=not args.no_push_on_improvement,
    )

    if args.cleanup or args.cleanup_dry_run:
        # Run cleanup only
        results = orchestrator.cleanup_remote_models(dry_run=args.cleanup_dry_run)
        print(f"\nCleanup {'preview' if args.cleanup_dry_run else 'complete'}:")
        print(f"  Models deleted: {len(results['models_deleted'])}")
        print(f"  Files deleted: {len(results['files_deleted'])}")
        print(f"  Space freed: {results['bytes_freed'] / (1024**3):.1f}GB")
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")
    else:
        orchestrator.run_continuous()
