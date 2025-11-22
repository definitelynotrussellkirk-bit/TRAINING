# Remote Model Deployment System

**Date:** 2025-11-16
**Remote Server:** 192.168.x.x (RTX 3090 24GB)
**Purpose:** Auto-deploy trained models for testing/inference

---

## 🎯 GOALS

1. **Automatic Deployment:** Push new model versions to remote server
2. **Parallel Operation:** Train locally, test remotely simultaneously
3. **Backup Location:** Remote server = additional backup
4. **GPU Utilization:** Use 3090 for inference while training continues
5. **Easy Testing:** Always have latest model ready on remote

---

## 🏗️ ARCHITECTURE

### Local Server (Training)
```
Desktop/TRAINING/
├── models/versions/
│   ├── v001_baseline/
│   ├── v002_improved/
│   └── latest -> v002_improved
├── deployment/
│   ├── deploy_config.json
│   ├── deploy_log.txt
│   └── .last_deployed
```

### Remote Server (192.168.x.x)
```
/home/user/models/
├── deployed/
│   ├── v001_baseline/
│   ├── v002_improved/
│   └── latest -> v002_improved
├── inference/
│   ├── serve_model.py
│   └── test_model.py
└── logs/
```

---

## 🚀 DEPLOYMENT WORKFLOW

### Automatic Deployment Triggers

```python
# Deploy automatically when:
DEPLOY_TRIGGERS = {
    "on_consolidation": True,      # After consolidating model
    "on_version_complete": True,   # After training completes
    "on_manual_request": True,     # User requests deployment
    "on_schedule": "daily_03:30"   # Daily at 3:30 AM
}
```

### Deployment Process

```python
def deploy_model_to_remote(version_id, deploy_config):
    """
    Deploy model version to remote server
    """
    # Step 1: Pre-deployment checks
    check_ssh_connection()
    check_remote_disk_space()
    check_local_model_validity(version_id)

    # Step 2: Prepare deployment package
    package = prepare_deployment_package(version_id)
    # Includes: adapter, metadata, base model (if needed)

    # Step 3: Compress for transfer
    compressed = compress_package(package)
    # Use tar.gz for efficient transfer

    # Step 4: Transfer to remote
    rsync_to_remote(compressed, remote_path)
    # Use rsync for resume capability

    # Step 5: Remote extraction & setup
    ssh_execute("extract_and_setup.sh", version_id)

    # Step 6: Verify deployment
    verify_remote_model(version_id)

    # Step 7: Update remote 'latest' symlink
    ssh_execute("ln -sf deployed/{version_id} latest")

    # Step 8: Log deployment
    log_deployment(version_id, timestamp, status="success")
```

---

## 📋 DEPLOYMENT CONFIG

```json
// deployment/deploy_config.json
{
  "remote": {
    "host": "192.168.x.x",
    "user": "user",
    "base_path": "/home/user/models",
    "ssh_key": "~/.ssh/id_rsa",
    "port": 22
  },
  "deployment": {
    "auto_deploy": true,
    "triggers": {
      "on_consolidation": true,
      "on_version_complete": true,
      "on_schedule": "03:30"
    },
    "transfer": {
      "method": "rsync",
      "compression": true,
      "bandwidth_limit": "50M",
      "resume": true
    },
    "versions_to_keep": 5,
    "include_base_model": false
  },
  "verification": {
    "check_md5": true,
    "test_inference": true,
    "max_retry": 3
  }
}
```

---

## 🔄 DEPLOYMENT STRATEGIES

### Strategy 1: Adapter Only (Fast)
```bash
# Deploy just the LoRA adapter (1.4 GB)
# Assumes base model already on remote
# Transfer time: ~30 seconds

Deploy: adapter + metadata
Remote: Merges with existing base model
```

### Strategy 2: Full Model (Slow but Safe)
```bash
# Deploy complete merged model (16 GB)
# Doesn't depend on remote having base
# Transfer time: ~5 minutes

Deploy: Complete model
Remote: Ready to use immediately
```

### Strategy 3: Incremental (Efficient)
```bash
# Deploy only changes from last version
# Uses rsync for efficient delta transfer
# Transfer time: Varies

Deploy: Diff from last version
Remote: Applies patch to previous version
```

---

## 🛠️ DEPLOYMENT SCRIPTS

### Local: `bin/deploy_to_remote.py`

```python
#!/usr/bin/env python3
"""Deploy model version to remote server"""

import argparse
import subprocess
import json
from pathlib import Path

def deploy_version(version_id, config_path="deployment/deploy_config.json"):
    """Deploy a specific version to remote"""

    # Load config
    config = json.load(open(config_path))
    remote = config['remote']

    # Paths
    local_version = f"models/versions/{version_id}"
    remote_path = f"{remote['user']}@{remote['host']}:{remote['base_path']}/deployed/"

    print(f"🚀 Deploying {version_id} to {remote['host']}")

    # Step 1: Verify local model exists
    if not Path(local_version).exists():
        raise FileNotFoundError(f"Version {version_id} not found locally")

    # Step 2: Check remote connection
    ssh_test = subprocess.run(
        ["ssh", f"{remote['user']}@{remote['host']}", "echo OK"],
        capture_output=True
    )
    if ssh_test.returncode != 0:
        raise ConnectionError("Cannot connect to remote server")

    # Step 3: Create remote directory
    subprocess.run([
        "ssh", f"{remote['user']}@{remote['host']}",
        f"mkdir -p {remote['base_path']}/deployed/{version_id}"
    ])

    # Step 4: Rsync transfer
    print("📦 Transferring files...")
    rsync_cmd = [
        "rsync", "-avz", "--progress",
        f"{local_version}/",
        f"{remote_path}{version_id}/"
    ]

    if config['deployment']['transfer']['bandwidth_limit']:
        rsync_cmd.extend(["--bwlimit", config['deployment']['transfer']['bandwidth_limit']])

    result = subprocess.run(rsync_cmd)

    if result.returncode != 0:
        raise Exception("Transfer failed")

    # Step 5: Update 'latest' symlink
    print("🔗 Updating latest symlink...")
    subprocess.run([
        "ssh", f"{remote['user']}@{remote['host']}",
        f"cd {remote['base_path']} && ln -sf deployed/{version_id} latest"
    ])

    # Step 6: Verify
    print("✅ Verifying deployment...")
    verify_cmd = [
        "ssh", f"{remote['user']}@{remote['host']}",
        f"ls -lh {remote['base_path']}/deployed/{version_id}/adapter_model.safetensors"
    ]
    subprocess.run(verify_cmd)

    # Step 7: Log
    log_deployment(version_id, "success")

    print(f"✅ Deployment complete: {version_id}")
    print(f"   Remote path: {remote['base_path']}/deployed/{version_id}")
    print(f"   Latest: {remote['base_path']}/latest")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version_id", help="Version to deploy (e.g., v002_improved)")
    parser.add_argument("--config", default="deployment/deploy_config.json")
    args = parser.parse_args()

    deploy_version(args.version_id, args.config)
```

### Remote: `~/models/inference/serve_model.py`

```python
#!/usr/bin/env python3
"""Serve model for inference on remote server"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_latest_model():
    """Load the latest deployed model"""

    # Paths
    base_model_path = "/path/to/training/consolidated_models/20251119_152444"
    adapter_path = "/home/user/models/latest"

    print("🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("🔄 Loading adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print("✅ Model loaded successfully")
    return model, tokenizer

def run_inference(prompt, model, tokenizer):
    """Run inference on prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_latest_model()

    # Test
    test_prompt = "What is 2+2?"
    response = run_inference(test_prompt, model, tokenizer)
    print(f"\nPrompt: {test_prompt}")
    print(f"Response: {response}")
```

---

## 🎛️ DEPLOYMENT CONTROLS

### Manual Deployment
```bash
# Deploy specific version
./bin/deploy_to_remote.py v002_improved

# Deploy latest version
./bin/deploy_to_remote.py $(readlink models/versions/latest)

# Deploy with custom config
./bin/deploy_to_remote.py v002_improved --config custom_deploy.json
```

### Control Files
```bash
# In deployment/ directory

.deploy_now         # Touch to trigger immediate deployment
.deploy_on_next     # Deploy after next training completes
.skip_deploy        # Skip next auto-deployment
```

### Web UI Integration
```
┌─ DEPLOYMENT STATUS ───────────────────────────────────────┐
│                                                            │
│ Remote Server: 192.168.x.x (RTX 3090) ✓ Connected     │
│                                                            │
│ Deployed Versions:                                        │
│  ✓ v002_improved (Latest) - Deployed 2h ago              │
│  ✓ v001_baseline          - Deployed yesterday           │
│                                                            │
│ Auto-Deploy: ✓ Enabled                                   │
│  Triggers: After consolidation, Daily 3:30 AM            │
│                                                            │
│ Last Deployment:                                          │
│  Version: v002_improved                                   │
│  Time: 2025-11-16 14:30:00                               │
│  Status: ✅ Success                                       │
│  Transfer: 1.4 GB in 28 seconds                          │
│                                                            │
│ [Deploy Latest Now] [Deploy Specific...] [Settings]     │
└────────────────────────────────────────────────────────────┘
```

---

## 📊 DEPLOYMENT MONITORING

### Deployment Log

```json
// deployment/deploy_log.json
{
  "deployments": [
    {
      "id": "deploy_20251116_143000",
      "version": "v002_improved",
      "timestamp": "2025-11-16T14:30:00Z",
      "trigger": "on_consolidation",
      "status": "success",
      "transfer": {
        "size_mb": 1400,
        "duration_seconds": 28,
        "method": "rsync",
        "bandwidth_mbps": 50
      },
      "verification": {
        "md5_match": true,
        "inference_test": "passed"
      }
    }
  ]
}
```

### Health Checks

```bash
# Check remote server status
ssh user@xxx.xxx.88.149 "nvidia-smi && df -h /home/user/models"

# Verify latest deployment
ssh user@xxx.xxx.88.149 "ls -lh /home/user/models/latest"

# Test inference
ssh user@xxx.xxx.88.149 "cd /home/user/models && python3 inference/serve_model.py"
```

---

## 🔐 SECURITY & RELIABILITY

### SSH Key Authentication
```bash
# Setup passwordless SSH (one-time)
ssh-copy-id user@xxx.xxx.88.149

# Verify
ssh user@xxx.xxx.88.149 "echo 'Success'"
```

### Deployment Verification
```python
def verify_deployment(version_id):
    """Verify remote deployment is valid"""

    # Check 1: Files exist
    check_files_exist(version_id)

    # Check 2: MD5 checksums match
    verify_checksums(version_id)

    # Check 3: Model can be loaded
    test_model_loading(version_id)

    # Check 4: Inference works
    test_inference(version_id)

    return all_checks_passed
```

### Error Recovery
```python
if deployment_fails():
    # Retry with exponential backoff
    for attempt in range(max_retries):
        wait_time = 2 ** attempt
        time.sleep(wait_time)

        if retry_deployment():
            break
    else:
        # Alert user
        send_notification("Deployment failed after retries")
        log_error(deployment_details)
```

---

## 🚀 INTEGRATION WITH MASTER PLAN

### Add to Phase 2: Model Versioning
- When new version created → auto-deploy to remote
- Version metadata includes deployment status

### Add to Phase 3: Control System
- Add `.deploy_now` control file
- Add deployment status to queue UI

### New Phase 3.5: Remote Deployment (3 hours)
1. **Create deployment scripts** (1 hour)
2. **Setup remote structure** (30 min)
3. **Integrate with versioning** (1 hour)
4. **Add UI controls** (30 min)

---

## 📝 USAGE EXAMPLES

### Example 1: Auto-Deploy After Training
```bash
# Train completes → v003_math_fixes created
# System automatically:
1. Compresses adapter (1.4 GB → 0.8 GB)
2. Rsyncs to remote (~30 seconds)
3. Extracts on remote
4. Updates 'latest' symlink
5. Verifies deployment
6. Logs success
```

### Example 2: Manual Deploy for Testing
```bash
# You want to test a specific version
./bin/deploy_to_remote.py v002_improved

# SSH to remote and test
ssh user@xxx.xxx.88.149
cd /home/user/models
python3 inference/serve_model.py
```

### Example 3: Rollback
```bash
# Deploy previous version
./bin/deploy_to_remote.py v001_baseline

# Or on remote, just update symlink
ssh user@xxx.xxx.88.149 "cd /home/user/models && ln -sf deployed/v001_baseline latest"
```

---

## ✅ SUCCESS CRITERIA

- [ ] Can deploy version to remote with one command
- [ ] Auto-deployment after consolidation works
- [ ] Remote 'latest' symlink always points to newest
- [ ] Deployment verified with checksums
- [ ] Can test inference on remote
- [ ] Deployment status visible in UI
- [ ] Error handling and retry logic works
- [ ] Logs all deployments

---

## 🎯 BENEFITS

### Parallel Operations
- Train locally on your GPU
- Test/serve on remote 3090 simultaneously
- No downtime for testing

### Additional Backup
- Remote server = another backup location
- Physical separation from training machine
- Different network segment

### Easy Testing
- Always have latest model ready
- Just SSH in and test
- No manual file transfers

### GPU Utilization
- Your GPU: Training
- Remote 3090: Inference/testing
- Maximize hardware usage

---

**Ready to implement?**

This adds automatic deployment to remote server as part of the training workflow. Every time you consolidate or complete training, the model automatically deploys to your 3090 machine for testing.
