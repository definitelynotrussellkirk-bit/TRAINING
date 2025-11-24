#!/usr/bin/env python3
"""
Checkpoint Auto-Deployment - Copies best checkpoint to 3090 for inference
"""

import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CheckpointAutoDeployment:
    def __init__(
        self,
        base_dir: str = "/path/to/training",
        remote_host: str = "192.168.x.x",
        interval: int = 600
    ):
        self.base_dir = Path(base_dir)
        self.remote_host = remote_host
        self.interval = interval
        self.last_deployed_checkpoint = None

    def get_best_checkpoint(self) -> dict:
        """Get best checkpoint from model comparison engine"""
        comparison_file = self.base_dir / "status/model_comparisons.json"
        
        if not comparison_file.exists():
            return None
        
        with open(comparison_file) as f:
            data = json.load(f)
        
        if 'comparisons' in data and len(data['comparisons']) > 0:
            # Get best by composite score
            best = max(data['comparisons'], key=lambda x: x.get('composite_score', 0))
            return best
        
        return None

    def deploy_checkpoint(self, checkpoint_info: dict):
        """Deploy checkpoint to 3090"""
        checkpoint_name = checkpoint_info['checkpoint']
        checkpoint_path = self.base_dir / "models/checkpoints" / checkpoint_name
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        logger.info(f"Deploying {checkpoint_name} (score: {checkpoint_info.get('composite_score', 0):.3f})")
        
        # rsync to 3090
        remote_path = f"{self.remote_host}:~/TRAINING/models/deployed_checkpoint"
        
        cmd = [
            "rsync", "-avz", "--delete",
            str(checkpoint_path) + "/",
            remote_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Deployed to {self.remote_host}")
            
            # Save deployment record
            deployment_record = {
                'timestamp': datetime.now().isoformat(),
                'checkpoint': checkpoint_name,
                'composite_score': checkpoint_info.get('composite_score'),
                'metrics': checkpoint_info
            }
            
            deployment_file = self.base_dir / "status/last_deployment.json"
            with open(deployment_file, 'w') as f:
                json.dump(deployment_record, f, indent=2)
            
            self.last_deployed_checkpoint = checkpoint_name
            return True
        else:
            logger.error(f"Deployment failed: {result.stderr}")
            return False

    def run_continuous(self):
        """Run continuous auto-deployment loop"""
        logger.info("🚀 CHECKPOINT AUTO-DEPLOYMENT - STARTING")
        logger.info(f"Remote host: {self.remote_host}")
        logger.info(f"Check interval: {self.interval}s")
        logger.info("=" * 80)
        
        while True:
            try:
                best = self.get_best_checkpoint()
                
                if best:
                    checkpoint_name = best['checkpoint']
                    score = best.get('composite_score', 0)
                    
                    logger.info(f"Best checkpoint: {checkpoint_name} (score: {score:.3f})")
                    
                    if checkpoint_name != self.last_deployed_checkpoint:
                        logger.info(f"📦 New best checkpoint detected - deploying...")
                        self.deploy_checkpoint(best)
                    else:
                        logger.info("Already deployed")
                else:
                    logger.info("No checkpoint comparisons available yet")
                
                logger.info(f"Next check in {self.interval}s...")
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(self.interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='/path/to/training')
    parser.add_argument('--remote-host', default='192.168.x.x')
    parser.add_argument('--interval', type=int, default=600)
    args = parser.parse_args()
    
    daemon = CheckpointAutoDeployment(args.base_dir, args.remote_host, args.interval)
    daemon.run_continuous()
