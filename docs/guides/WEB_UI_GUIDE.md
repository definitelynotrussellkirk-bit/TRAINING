# Training Control Center - Web UI Guide

## Launch the Web UI

```bash
cd /path/to/training
./launch_web_ui.sh
```

**Access at:** http://localhost:7860

---

## Features

### 📊 Status Dashboard

Real-time monitoring of your training system:
- **Daemon Status**: Shows if daemon is running (🟢 Active | 🟡 Idle | 🔴 Stopped)
- **GPU Stats**: Utilization, VRAM usage, temperature
- **Auto-refresh**: Updates every 5 seconds

### ⚙️ Daemon Control

Start and stop the training daemon:
- **▶️ Start Daemon**: Launches continuous training system
- **⏹️ Stop Daemon**: Stops background training

**How it works:**
- Daemon watches `inbox/` for new JSONL files
- Automatically trains on each file (1 epoch)
- Deletes data after processing
- Runs 24/7 in background

### 📊 Training Profiles

Switch between training configurations:

**Balanced** (default):
- LR: 2e-4, LoRA r=64
- Good for general use
- ~18-20GB VRAM

**Aggressive**:
- LR: 5e-4, LoRA r=128
- Faster training, higher capacity
- ~22-23GB VRAM (tight!)

**Conservative**:
- LR: 1e-4, LoRA r=32
- Stable, safe training
- ~15-17GB VRAM

**To switch:**
1. Select profile from radio buttons
2. Click "Apply Profile"
3. Daemon will use new settings for next batch

### 📥 Inbox Monitor

View files waiting to be processed:
- Shows all `.jsonl` files in inbox
- Displays file size and modification time
- Refresh to see new files

### ⚡ Quick Actions

**Convert LEO Data:**
1. Enter input file path (LEO training_samples.jsonl)
2. Enter output name (e.g., `batch1_100k.jsonl`)
3. Click "Convert & Copy to Inbox"
4. File is converted and moved to inbox automatically

---

## Typical Workflow

### Step 1: Launch Web UI
```bash
cd /path/to/training
./launch_web_ui.sh
```

### Step 2: Start Daemon
- Go to "⚙️ Daemon Control" tab
- Click "▶️ Start Daemon"
- Verify status shows 🟡 "Daemon Running (Idle)"

### Step 3: Feed Data

**Option A: Use Quick Actions**
- Go to "⚡ Quick Actions" tab
- Convert and add LEO batches directly

**Option B: Command Line**
```bash
# Convert LEO batch
python3 convert_leo_data.py \
    /path/to/leo/batch.jsonl \
    /path/to/training/inbox/batch_name.jsonl
```

### Step 4: Monitor
- Check "📥 Inbox Monitor" to see queued files
- Watch "📊 Status Dashboard" for:
  - Status changes to 🟢 "Training Active"
  - GPU utilization ~95-100%
  - VRAM usage

### Step 5: Switch Profiles (Optional)
- If training too slow → Use "Aggressive"
- If OOM errors → Use "Conservative"
- If stable → Keep "Balanced"

---

## Monitoring Tips

### GPU Usage
**Healthy training:**
- GPU: 95-100% utilization
- VRAM: 18-20GB / 24GB (balanced)
- Temp: <85°C

**Problems:**
- GPU: 0% → Training not running
- VRAM: 24GB → OOM imminent, switch to conservative
- Temp: >85°C → Check cooling

### Daemon Status

**🟢 Training Active**
- Daemon found a file and is training
- GPU should be at ~100%
- Check inbox monitor - file should be processing

**🟡 Daemon Running (Idle)**
- Daemon is waiting for files
- GPU at 0%
- Drop files in inbox to start training

**🔴 Daemon Stopped**
- Not running
- Click "▶️ Start Daemon"

---

## Command Reference

### Start Web UI
```bash
cd /path/to/training
./launch_web_ui.sh
# Access: http://localhost:7860
```

### Start Daemon (Alternative)
```bash
cd /path/to/training
python3 training_daemon.py --base-dir /path/to/training &
```

### Stop Daemon (Alternative)
```bash
pkill -f training_daemon.py
```

### Manual Training (Without Daemon)
```bash
cd /home/user/ultimate_trainer
python3 train.py \
    --dataset /path/to/training/inbox/file.jsonl \
    --model /path/to/training/model \
    --output-dir /path/to/training/output \
    --epochs 1 \
    --batch-size 1 \
    --gradient-accumulation 16 \
    --learning-rate 2e-4 \
    --lora-r 64 \
    --lora-alpha 32
```

---

## Troubleshooting

### Web UI won't start

**Port already in use:**
```bash
# Check if port 7860 is in use
lsof -i :7860

# Kill existing process and free port
pkill -f training_web_ui.py
fuser -k 7860/tcp

# Wait a moment
sleep 2

# Try again
./launch_web_ui.sh
```

**Gradio compatibility error:**
If you see `TypeError: EventListener._setup.<locals>.event_trigger() got an unexpected keyword argument 'every'`:

This is a Gradio version compatibility issue that has been fixed. The web UI now uses Gradio 5.x compatible syntax with `gr.Timer()` for auto-refresh.

**File:** `training_web_ui.py` (already fixed as of Nov 2025)
- Uses `gr.Timer(value=5)` for periodic updates
- Auto-refresh every 5 seconds works correctly
- Compatible with Gradio 5.49.1+

If you still see this error, ensure you're using the latest version of the file.

### Daemon won't start
- Check if already running: `pgrep -f training_daemon.py`
- Check logs: `tail -f logs/daemon_*.log`
- Verify config exists: `cat config.json`

### Training not starting
- Check inbox has `.jsonl` files (not `.json`)
- Verify files are in messages format (not prompt/response)
- Check daemon is running (status should be 🟡 or 🟢)
- Look for errors in daemon logs

### Can't switch profiles
- Verify profile files exist:
  - `config.json` (balanced)
  - `config_aggressive.json`
  - `config_conservative.json`
- Check file permissions: `ls -l config*.json`

---

## Advanced: Remote Access

To access web UI from other machines on your network:

```bash
# Edit training_web_ui.py
# Change: server_name="0.0.0.0"
# Already set by default!

# Find your IP
ip addr show | grep inet

# Access from other machine:
# http://YOUR_IP:7860
```

---

## Pro Tips

1. **Keep web UI open**: Real-time monitoring helps catch issues early
2. **Start daemon first**: Then feed data as you generate it
3. **Use Quick Actions**: Faster than command line for batch conversion
4. **Monitor first hour**: Watch closely when starting new batches
5. **Check inbox regularly**: Ensure files are being processed
6. **Profile switching**: Change profile between batches, not during training

---

## Summary

The web UI gives you full control over your continuous training system:
- ✅ Start/stop daemon with one click
- ✅ Switch training profiles on the fly
- ✅ Monitor GPU and training status
- ✅ Convert and queue batches easily
- ✅ Track inbox activity

Access it anytime at http://localhost:7860 after launching!

---

## Technical Notes

### Gradio Version Compatibility

**Issue Fixed:** Nov 2025
- **Problem:** Original code used deprecated Gradio 3.x syntax for auto-refresh
- **Error:** `TypeError: EventListener._setup.<locals>.event_trigger() got an unexpected keyword argument 'every'`

**Solution Implemented:**
```python
# OLD (deprecated):
app.load(fn=update_func, outputs=display, every=5)

# NEW (Gradio 5.x compatible):
timer = gr.Timer(value=5)
timer.tick(fn=update_func, outputs=display)
```

**Changes in `training_web_ui.py`:**
1. Created `update_status()` helper function for code reuse
2. Added `gr.Timer(value=5)` component for 5-second intervals
3. Connected timer to status display with `timer.tick()`
4. Removed deprecated `app.load(..., every=5)` syntax

**Compatibility:**
- Requires: Gradio 5.x+ (tested with 5.49.1)
- Virtual env: `/home/user/ultimate_trainer/web_ui_venv`
- Python: 3.12+

**Auto-refresh features:**
- Status dashboard updates every 5 seconds
- Shows daemon status, GPU stats, timestamp
- Manual refresh button also available
- No page reload required

### Launch Script

**File:** `launch_web_ui.sh`
```bash
#!/bin/bash
cd "$(dirname "$0")"
/home/user/ultimate_trainer/web_ui_venv/bin/python3 training_web_ui.py
```

**Key points:**
- Uses dedicated venv with Gradio 5.49.1
- Runs on port 7860 (configurable in code)
- Server binds to 0.0.0.0 for network access
- No auto-restart mechanism (manual relaunch if crashes)
