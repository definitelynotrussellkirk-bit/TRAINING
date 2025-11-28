# Troubleshooting Guide

Common problems and solutions for the training system.

---

## CRITICAL: Model Outputs Garbage

### Symptom: Model outputs random text instead of answers

**Example outputs:**
- "You are happy. You enjoy helping others."
- "| | | | | | | | | | | | | | |"
- "Output contract: Return your answer..."
- Random Greek letters, dates, instruction fragments

**Root Cause:** Model was trained on instructions, not just responses.

**Diagnosis:**
```bash
# Check masking ratio in training logs
grep "Masking verification" logs/training_daemon.log | tail -5

# Should show 30-70% masked. If < 20%, training is broken!
```

**The Bug (discovered 2025-11-27):**
When packing is enabled (default), multiple examples are combined:
```
[user1 instruction][assistant response1][user2 instruction][assistant response2]...
```
The old collator only masked the FIRST instruction, so the model learned:
- User instructions ("What is 2+2?")
- System prompts ("You are a helpful assistant")
- Response templates ("Output contract: ...")

**Fix Applied:**
1. `core/custom_collator.py` now masks ALL instruction segments
2. `core/masking_validators.py` validates masking before training
3. `core/train.py` aborts if masking < 25%

**If you see this issue:**
```bash
# 1. Stop training
python3 core/training_controller.py stop

# 2. Clear Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 3. Restart daemon (picks up fixed collator)
pkill -f training_daemon
nohup python3 core/training_daemon.py >> logs/training_daemon.log 2>&1 &

# 4. Verify masking is correct (should be 30-70% masked)
sleep 30
grep "Masking verification" logs/training_daemon.log | tail -3
```

**Prevention:**
The masking validators now check:
1. Masking ratio (30-85% expected)
2. Response template count matches masked regions
3. Trained tokens don't contain instruction markers
4. Packed sequences have proper mask/train alternation

---

## Training Daemon Issues

### Daemon Not Running

**Symptom:** No training happening, `ps aux | grep training_daemon` shows nothing

**Check:**
```bash
ps aux | grep training_daemon | grep -v grep
```

**Solution:**
```bash
# Restart daemon
nohup python3 core/training_daemon.py > training_output.log 2>&1 &

# Verify it started
sleep 2
ps aux | grep training_daemon | grep -v grep
```

**Root causes:**
- Daemon crashed (check `logs/daemon_YYYYMMDD.log`)
- Manually killed
- System reboot

### Daemon Crashes Immediately

**Symptom:** Daemon starts but exits within seconds

**Check logs:**
```bash
tail -100 logs/daemon_$(date +%Y%m%d).log
```

**Common causes:**

1. **Config file missing/invalid:**
   ```bash
   # Verify config exists
   cat config.json

   # Validate JSON
   python3 -m json.tool config.json
   ```

2. **Model path doesn't exist:**
   ```bash
   # Check model_path from config
   cat config.json | jq -r .model_path
   ls -lh $(cat config.json | jq -r .model_path)
   ```

3. **Permission issues:**
   ```bash
   # Check write permissions
   touch status/test.json && rm status/test.json
   touch logs/test.log && rm logs/test.log
   ```

### Multiple Daemon Processes Running

**Symptom:** OOM errors, slow training, duplicate processing

**Check:**
```bash
ps aux | grep "python3.*training_daemon" | grep -v grep | wc -l
# Should return 1, not 2+
```

**Solution:**
```bash
# Kill all daemons
pkill -f training_daemon

# Wait for processes to die
sleep 3

# Start single daemon
nohup python3 core/training_daemon.py > training_output.log 2>&1 &
```

## Out of Memory (OOM) Errors

### CUDA OOM During Training

**Symptom:** Training crashes with "CUDA out of memory" error

**Check current batch size:**
```bash
cat config.json | jq .batch_size
```

**Solution 1: Reduce batch size**
```bash
# Conservative (14GB VRAM)
python3 tools/config/edit_config.py batch_size 16

# Very conservative (10GB VRAM)
python3 tools/config/edit_config.py batch_size 12

# Emergency (7GB VRAM)
python3 tools/config/edit_config.py batch_size 8
```

**Solution 2: Check for multiple processes**
```bash
# Kill all training processes
pkill -f training_daemon
pkill -f "python.*train.py"

# Clear GPU memory
sleep 5

# Restart daemon
nohup python3 core/training_daemon.py > training_output.log 2>&1 &
```

**Solution 3: Reduce max_length (if data allows)**
```bash
# Check current max_length
cat config.json | jq .max_length

# Validate data token lengths
python3 tools/data/validate_data.py --file inbox/your_data.jsonl

# If most examples < 2048 tokens, reduce:
python3 tools/config/edit_config.py max_length 2048
```

### OOM During Eval Steps

**Symptom:** Training stable, crashes every N steps (eval_steps)

**This should be FIXED** (2025-11-22 update to `core/train.py`)

**Verify fix:**
```bash
# Check if torch.cuda.empty_cache() present after model.generate()
grep -A 2 "model.generate" core/train.py | grep "empty_cache"
```

**If not fixed:**
```bash
# Reduce eval frequency
python3 tools/config/edit_config.py eval_steps 100

# Reduce eval samples
python3 tools/config/edit_config.py num_eval_samples 2
```

## Queue Issues

### Files Stuck in Processing Queue

**Symptom:** File in `queue/processing/` for hours, no training progress

**Check:**
```bash
ls -lh queue/processing/
cat status/training_status.json | jq .current_file
```

**Solution:**
```bash
# Check if daemon is running
ps aux | grep training_daemon | grep -v grep

# If daemon running but stuck, kill it
pkill -f training_daemon

# Move file back to normal queue
mv queue/processing/* queue/normal/

# Restart daemon
nohup python3 core/training_daemon.py > training_output.log 2>&1 &
```

### Files Stuck in Failed Queue

**Symptom:** Files in `queue/failed/`, not retrying

**Check failure reason:**
```bash
# Check queue metadata
cat queue/queue_metadata.json | jq '.failed'

# Check daemon logs
tail -200 logs/daemon_$(date +%Y%m%d).log | grep -A 10 "Failed:"
```

**Common reasons:**
1. **OOM errors** - Reduce batch_size
2. **Data validation errors** - Fix data format
3. **Timeout** - Increase timeout or reduce file size

**Solution:**
```bash
# Fix underlying issue (batch size, data format, etc.)

# Move back to normal queue for retry
mv queue/failed/* queue/normal/

# Or retry with high priority
for f in queue/failed/*; do
  python3 core/training_queue.py add "$f" --priority high
done
rm queue/failed/*
```

### Empty Queue But Training Not Starting

**Symptom:** Files in `inbox/`, but not moving to queue

**Check:**
```bash
ls -lh inbox/
cat logs/daemon_$(date +%Y%m%d).log | grep "inbox"
```

**Possible causes:**

1. **Daemon not polling:**
   ```bash
   # Check daemon running
   ps aux | grep training_daemon | grep -v grep

   # Restart if needed
   pkill -f training_daemon
   nohup python3 core/training_daemon.py > training_output.log 2>&1 &
   ```

2. **File permissions:**
   ```bash
   # Check if daemon can read files
   ls -lh inbox/
   chmod 644 inbox/*.jsonl
   ```

3. **Invalid filename:**
   ```bash
   # Ensure .jsonl extension
   for f in inbox/*; do echo "$f"; done
   ```

## Model Loading Issues

### Model Not Found

**Symptom:** Error "model not found" or "no such file or directory"

**Check model path:**
```bash
# From config
cat config.json | jq -r .model_path
ls -lh $(cat config.json | jq -r .model_path)

# Check current_model
ls -lh models/current_model/
```

**Solution:**
```bash
# If current_model/ empty, copy base model
cp -r models/Qwen3-0.6B/* models/current_model/

# Or update config to point to base model
python3 tools/config/edit_config.py model_path "models/Qwen3-0.6B"
```

### Checkpoint Resume Fails

**Symptom:** Training starts from scratch instead of resuming

**Check checkpoints:**
```bash
ls -lh models/current_model/checkpoint-*/
```

**Verify resume logic:**
```bash
python3 safety/verify_checkpoint_resume.py
```

**Solution:**
```bash
# If checkpoints corrupt, restore from backup
python3 management/model_versioner.py list
python3 management/model_versioner.py restore vXXX

# Or start fresh from base model
rm -rf models/current_model/*
cp -r models/Qwen3-0.6B/* models/current_model/
```

## Monitoring Issues

### Web UI Not Loading

**Symptom:** http://localhost:8080 not accessible

**Check monitor server:**
```bash
ps aux | grep live_monitor | grep -v grep
```

**Check port:**
```bash
netstat -tuln | grep 8080
# OR
ss -tuln | grep 8080
```

**Solution:**
```bash
# Restart monitor
pkill -f live_monitor
nohup python3 monitoring/servers/launch_live_monitor.py > /dev/null 2>&1 &

# Verify started
sleep 2
curl -s http://localhost:8080/status/training_status.json | jq .current_step
```

### Status JSON Not Updating

**Symptom:** Web UI shows stale data, no updates

**Check status file:**
```bash
ls -lh status/training_status.json
cat status/training_status.json | jq .
```

**Check write permissions:**
```bash
touch status/test.json && rm status/test.json
```

**Solution:**
```bash
# If daemon running but status not updating, restart
pkill -f training_daemon
nohup python3 core/training_daemon.py > training_output.log 2>&1 &
```

## Disk Space Issues

### Out of Disk Space

**Symptom:** Training crashes with "No space left on device"

**Check disk usage:**
```bash
df -h .
du -sh models/ logs/ queue/
```

**Quick fix:**
```bash
# Delete old logs
find logs/ -name "*.log" -mtime +30 -delete

# Delete completed queue files
rm -rf queue/recently_completed/*

# Clean old checkpoints (keeps latest 5)
python3 management/checkpoint_retention.py --keep 5
```

**Long-term solution:**
```bash
# Ensure auto disk manager running
ps aux | grep auto_disk_manager | grep -v grep

# Start if not running
nohup python3 management/auto_disk_manager.py > /dev/null 2>&1 &
```

### Checkpoints Filling Disk

**Symptom:** `models/current_model/` has 50+ checkpoint directories

**Check:**
```bash
ls -ld models/current_model/checkpoint-*/ | wc -l
du -sh models/current_model/
```

**Solution:**
```bash
# Safe cleanup (only if training NOT running)
python3 core/training_controller.py status
# If "idle" or "stopped":
python3 management/safe_checkpoint_cleanup.py --keep 5

# If training running, wait for pause
python3 core/training_controller.py pause
python3 management/safe_checkpoint_cleanup.py --keep 5
python3 core/training_controller.py resume
```

## Configuration Issues

### Config Changes Not Taking Effect

**Symptom:** Changed `config.json` but training uses old values

**Cause:** Config locked during training

**Check:**
```bash
ls -lh .config_lock.json
cat .config_lock.json
```

**Solution:**
```bash
# Stop training
python3 core/training_controller.py stop

# Wait for daemon to exit
sleep 10

# Verify lock released
ls .config_lock.json  # Should not exist

# Edit config
nano config.json

# Restart training
nohup python3 core/training_daemon.py > training_output.log 2>&1 &
```

### Invalid Config JSON

**Symptom:** Daemon won't start, error about config

**Validate JSON:**
```bash
python3 -m json.tool config.json
# If error, shows line number of problem
```

**Common issues:**
- Missing comma
- Trailing comma
- Unquoted strings
- Wrong data type (string instead of number)

**Fix:**
```bash
# Edit config
nano config.json

# Validate
python3 -m json.tool config.json

# If still broken, restore from backup
cp archive/configs/config_*.json config.json
```

## Data Issues

### Token Length Exceeds max_length

**Symptom:** Warning about truncation, poor training results

**Check:**
```bash
python3 tools/data/validate_data.py --file inbox/your_file.jsonl
```

**Solution 1: Increase max_length (if VRAM allows)**
```bash
# Check current
cat config.json | jq .max_length

# Increase (requires more VRAM, may need to reduce batch_size)
python3 tools/config/edit_config.py max_length 8192
python3 tools/config/edit_config.py batch_size 12  # Reduce to compensate
```

**Solution 2: Filter/split data**
```bash
# Filter out long examples
python3 tools/data/filter_by_length.py \
  --input long_data.jsonl \
  --output filtered_data.jsonl \
  --max-tokens 4096
```

### Invalid JSONL Format

**Symptom:** Daemon rejects file, validation fails

**Check format:**
```bash
head -1 inbox/your_file.jsonl | python3 -m json.tool
```

**Common issues:**
1. **Not JSON Lines** - Multiple objects in one line
2. **Missing messages field**
3. **Wrong role names** (should be "system", "user", "assistant")
4. **Missing content field**

**Fix:**
```bash
# Validate each line
python3 tools/data/validate_jsonl.py inbox/your_file.jsonl

# Auto-fix common issues
python3 tools/data/fix_jsonl.py --input broken.jsonl --output fixed.jsonl
```

## Performance Issues

### Training Very Slow

**Symptom:** < 1 step per second, GPU underutilized

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Common causes:**

1. **Batch size too small:**
   ```bash
   cat config.json | jq .batch_size
   # If < 16 and VRAM available, increase
   python3 tools/config/edit_config.py batch_size 24
   ```

2. **Eval too frequent:**
   ```bash
   cat config.json | jq .eval_steps
   # If < 100, increase
   python3 tools/config/edit_config.py eval_steps 200
   ```

3. **CPU bottleneck:**
   ```bash
   htop  # Check CPU usage
   # If 100% on one core, data loading issue
   # Reduce num_workers in train.py (not configurable currently)
   ```

### High Memory Usage (RAM, not VRAM)

**Symptom:** System slow, swap usage high

**Check:**
```bash
free -h
htop
```

**Solution:**
```bash
# Reduce eval samples
python3 tools/config/edit_config.py num_eval_samples 2

# Restart daemon to clear memory
pkill -f training_daemon
sleep 5
nohup python3 core/training_daemon.py > training_output.log 2>&1 &
```

## Remote Connection Issues

See `REMOTE_INFERENCE.md` for remote server troubleshooting.

**Quick checks:**
```bash
# Test connection (use your configured inference host)
ping "${INFERENCE_HOST}"

# Test SSH
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" 'echo "Connected"'

# Check remote GPU
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" 'nvidia-smi'
```

## Emergency Procedures

### Complete System Reset

**When:** Multiple issues, system unstable

```bash
# 1. Stop everything
pkill -f training_daemon
pkill -f live_monitor
pkill -f memory_stats
pkill -f auto_disk_manager

# 2. Backup current state
cp -r models/current_model models/current_model_backup_$(date +%Y%m%d_%H%M%S)

# 3. Clear locks and temp files
rm -f .stop .pause .config_lock.json .daemon.pid

# 4. Clear stuck queues
mv queue/processing/* queue/normal/ 2>/dev/null
mv queue/failed/* queue/normal/ 2>/dev/null

# 5. Restart system
scripts/start_all.sh

# 6. Verify
python3 safety/comprehensive_health_check.py
```

### Restore from Backup

**When:** Model corrupted, need to rollback

```bash
# List available versions
python3 management/model_versioner.py list

# Restore specific version
python3 management/model_versioner.py restore v003

# Or quick rollback to previous
python3 management/model_versioner.py rollback
```

## Getting Help

### Check System Health

```bash
# Comprehensive check
python3 safety/comprehensive_health_check.py

# State tracker
python3 tools/analysis/state_tracker.py --check

# Quick health check
scripts/check_health.sh
```

### Collect Diagnostic Info

```bash
# System state
python3 tools/analysis/state_tracker.py --check > diagnostic_$(date +%Y%m%d).txt

# Recent logs
tail -500 logs/daemon_$(date +%Y%m%d).log >> diagnostic_$(date +%Y%m%d).txt

# Config
cat config.json >> diagnostic_$(date +%Y%m%d).txt

# Queue status
python3 core/training_queue.py status >> diagnostic_$(date +%Y%m%d).txt
```

## Common Error Messages

### "No module named 'transformers'"

**Solution:**
```bash
pip install transformers datasets accelerate peft
```

### "RuntimeError: CUDA out of memory"

See "Out of Memory (OOM) Errors" section above.

### "FileNotFoundError: config.json"

**Solution:**
```bash
# Verify you're in TRAINING directory
pwd

# Navigate to correct directory
cd "${TRAINING_BASE_DIR:-/path/to/TRAINING}"
```

### "PermissionError: [Errno 13] Permission denied"

**Solution:**
```bash
# Fix permissions
chmod -R u+w logs/ status/ queue/ models/current_model/
```

### "JSONDecodeError: Expecting value"

**Invalid JSON in config or data file**

**Solution:**
```bash
# Validate config
python3 -m json.tool config.json

# Validate data
python3 tools/data/validate_data.py --file yourfile.jsonl
```
