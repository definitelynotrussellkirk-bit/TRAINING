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

# Protected directories - never delete
PROTECTED_MODEL_DIRS = {"Qwen3-0.6B"}  # Never delete base model

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
    """

    def __init__(
        self,
        base_dir: str = None,
        remote_host: str = None,
        remote_api_url: str = None,
        check_interval: int = 600  # 10 minutes
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

        # Paths
        self.deployment_log = self.base_dir / "status" / "deployment_status.json"
        self.checkpoint_dir = self.base_dir / "models" / "current_model"

        # State
        self.last_deployed_step = None

        logger.info("Deployment Orchestrator initialized")
        logger.info(f"Base dir: {self.base_dir}")
        logger.info(f"Remote: {self.remote_host}")
        logger.info(f"API: {self.remote_api_url}")
        logger.info(f"Check interval: {self.check_interval}s")

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get best checkpoint from ledger (by lowest train_loss).

        Returns:
            {
                'step': 156000,
                'checkpoint_path': Path('/path/to/checkpoint-156000'),
                'checkpoint_name': 'checkpoint-156000',
                'loss': 0.12,
            }
            or None if no valid checkpoint found
        """
        try:
            from core.checkpoint_ledger import get_ledger
            ledger = get_ledger()
            best = ledger.get_best(metric="train_loss")

            if not best:
                logger.debug("No checkpoints in ledger")
                return None

            # Find checkpoint path
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{best.step}"
            if not checkpoint_path.exists():
                # Try canonical name format
                if best.canonical_name:
                    checkpoint_path = self.checkpoint_dir / best.canonical_name
                if not checkpoint_path.exists():
                    logger.warning(f"Checkpoint not found: step {best.step}")
                    return None

            return {
                'step': best.step,
                'checkpoint_path': checkpoint_path,
                'checkpoint_name': checkpoint_path.name,
                'loss': best.train_loss,
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

        logger.info("="*80)
        logger.info(f"✅ DEPLOYMENT SUCCESSFUL: {deployment_id}")
        logger.info("="*80)

        # Step 4: Cleanup old models on remote
        logger.info("Running post-deployment cleanup...")
        cleanup_results = self.cleanup_remote_models(dry_run=False)
        record['cleanup'] = cleanup_results

        return True

    def run_continuous(self):
        """Run continuous deployment loop"""
        logger.info("="*80)
        logger.info("🚀 DEPLOYMENT ORCHESTRATOR - STARTING")
        logger.info("="*80)

        while True:
            try:
                # Get best checkpoint
                best = self.get_best_checkpoint()

                if best:
                    step = best['step']
                    score = best['score']

                    logger.info(f"📊 Best checkpoint: step {step}, score {score:.3f}")

                    # Check if already deployed
                    if step == self.last_deployed_step:
                        logger.info("✓ Already deployed")
                    else:
                        logger.info(f"📦 New best detected - deploying...")

                        # Deploy
                        success = self.deploy_full(best)

                        if success:
                            self.last_deployed_step = step
                        else:
                            logger.error("Deployment failed, will retry next cycle")
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
        check_interval=args.interval
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
