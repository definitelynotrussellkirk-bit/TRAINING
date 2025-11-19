# Training Monitor Guide

Complete guide to all monitoring dashboards and how to use them.

## Quick Reference

| Port | Name | Best For | URL |
|------|------|----------|-----|
| 8082 | Enhanced Monitor | **Real-time metrics & charts** | http://localhost:8082 |
| 8081 | Detailed Monitor | Token-by-token analysis | http://localhost:8081 |
| 8080 | Basic Monitor | Simple status | http://localhost:8080 |

## Port 8082 - Enhanced Monitor ⭐ RECOMMENDED

### Overview
The best dashboard for real-time training monitoring with beautiful visualizations.

### Features
- **Live Metrics Cards**
  - Training Loss (current value)
  - Step counter (X / 12,488)
  - Epoch progress
  - Learning rate (scientific notation)
  - GPU memory usage
  - ETA (time remaining)
  - Speed (seconds per step)

- **Progress Bar**
  - Visual percentage bar
  - Steps completed
  - Time remaining estimate

- **Live Charts**
  - Training loss over time (line chart)
  - Learning rate schedule (line chart)
  - Automatically updates every 2 seconds
  - Shows last 200 data points

### Update Frequency
- **Dashboard refresh**: Every 2 seconds
- **Data collection**: Every 10 steps (after fix)
- **Charts**: Smooth animation, no flickering

### Usage
```bash
# Monitor is already running in background
# Just open in browser:
open http://localhost:8082
```

### What You'll See

**During Training**:
- Status badge: "TRAINING" (green)
- All metrics updating in real-time
- Charts showing loss decreasing
- Progress bar advancing

**Waiting for Training**:
- Status badge: "Waiting for Training" (yellow)
- All metrics show N/A or 0
- Charts are empty

### Technical Details
- **File**: `/path/to/training/enhanced_monitor.py`
- **Data Source**: `/path/to/training/current_model/status/training_detail.json`
- **Framework**: Flask + Chart.js
- **Port**: 8082

## Port 8081 - Detailed Monitor

### Overview
Advanced dashboard showing actual prompts, predictions, and token-by-token comparison.

### Features
- **Complete Prompt Context**
  - System message
  - User message
  - Assistant response (expected)

- **Token-by-Token Comparison**
  - Green highlights: Matching tokens
  - Red highlights: Mismatched tokens
  - Yellow highlights: Missing tokens

- **Accuracy Metrics**
  - Match percentage
  - Token diff count
  - Sample index

### Endpoints

#### Main Dashboard: `/`
Interactive UI with color-coded comparisons

#### Raw JSON Stream: `/json`
```bash
curl http://localhost:8081/json
```
Shows raw JSON with auto-refresh every 2 seconds

#### API Endpoint: `/api/detail`
```bash
curl http://localhost:8081/api/detail
```
Programmatic access to training details

### Update Frequency
- **Updates**: Every 625 steps (will change to 10 after fix)
- **Shows**: Detailed inference on validation examples

### Usage
```bash
# View in browser
open http://localhost:8081

# Watch JSON stream
watch -n 2 curl -s http://localhost:8081/api/detail
```

### Technical Details
- **File**: `/path/to/training/detailed_monitor.py`
- **Port**: 8081
- **Data**: Same as Port 8082

## Port 8080 - Basic Monitor

### Overview
Simple status display (legacy monitor).

### Features
- Basic step/loss info
- Minimal UI
- Fast and lightweight

### Usage
```bash
open http://localhost:8080
```

## Status JSON File

All monitors read from the same source:
```
/path/to/training/current_model/status/training_detail.json
```

### JSON Structure
```json
{
  "status": "training",
  "step": 500,
  "epoch": 0.04,
  "train_loss": 0.567,
  "eval_loss": null,
  "learning_rate": 0.000194,
  "gpu_memory": "8.12GB",
  "timestamp": "2025-11-07T01:04:39.884167"
}
```

### Status Values
- `"initialized"` - Training not started yet
- `"training"` - Actively training
- `"completed"` - Training finished
- `"error"` - Something went wrong

## Monitor Management

### Start All Monitors
```bash
cd /path/to/training

# Enhanced Monitor (Port 8082)
nohup python3 enhanced_monitor.py > /tmp/enhanced_monitor.log 2>&1 &

# Detailed Monitor (Port 8081)
nohup python3 detailed_monitor.py > /tmp/detailed_monitor.log 2>&1 &

# Basic Monitor (Port 8080)
nohup python3 live_monitor.py > /tmp/live_monitor.log 2>&1 &
```

### Check Monitor Status
```bash
# See what's running
ps aux | grep monitor

# Check ports
netstat -tlnp | grep -E "8080|8081|8082"

# Check logs
tail -f /tmp/enhanced_monitor.log
tail -f /tmp/detailed_monitor.log
```

### Stop Monitors
```bash
# Kill all monitors
pkill -f "monitor.py"

# Or kill specific ports
lsof -ti:8082 | xargs kill
lsof -ti:8081 | xargs kill
lsof -ti:8080 | xargs kill
```

### Restart Monitors
```bash
# Stop first
pkill -f "monitor.py"

# Wait a moment
sleep 2

# Start again
cd /path/to/training
nohup python3 enhanced_monitor.py > /tmp/enhanced_monitor.log 2>&1 &
nohup python3 detailed_monitor.py > /tmp/detailed_monitor.log 2>&1 &
```

## Troubleshooting

### Monitor Shows "Waiting" But Training is Running

**Cause**: Status JSON file not being updated

**Fix**:
```bash
# Check if training_detail.json exists
cat /path/to/training/current_model/status/training_detail.json

# Check daemon is running
ps aux | grep training_daemon

# Check logs
tail -f /tmp/daemon.log
```

### "Address Already in Use" Error

**Cause**: Monitor already running on that port

**Fix**:
```bash
# Kill the process on that port
lsof -ti:8082 | xargs kill

# Or kill all monitors and restart
pkill -f "monitor.py"
```

### Charts Not Updating

**Cause**: Browser cache or connection issue

**Fix**:
- Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
- Clear browser cache
- Try different browser
- Check browser console for errors (F12)

### Monitor Shows Old Data

**Cause**: Training crashed or stopped

**Fix**:
```bash
# Check if daemon is running
ps aux | grep training_daemon

# Check last update time in JSON
cat /path/to/training/current_model/status/training_detail.json | grep timestamp

# Restart daemon if needed
cd /path/to/training
nohup python3 training_daemon.py --base-dir /path/to/training > /tmp/daemon.log 2>&1 &
```

## Best Practices

1. **Keep Port 8082 Open** - Best overall view
2. **Check Port 8081 Occasionally** - See actual predictions
3. **Monitor nvidia-smi** - GPU usage and temperature
4. **Watch Logs** - `tail -f /tmp/daemon.log`
5. **Take Screenshots** - Document your training progress

## Advanced Usage

### Export Metrics to CSV
```bash
# Scrape metrics every 10 seconds
while true; do
    curl -s http://localhost:8082/api/status | \
    jq -r '[.step, .train_loss, .learning_rate, .gpu_memory] | @csv' >> metrics.csv
    sleep 10
done
```

### Alert on Completion
```bash
# Email when training completes
while true; do
    status=$(curl -s http://localhost:8082/api/status | jq -r '.status')
    if [ "$status" = "completed" ]; then
        echo "Training completed!" | mail -s "Training Done" your@email.com
        break
    fi
    sleep 60
done
```

### Grafana Integration
The JSON API at port 8082 can be scraped by Grafana using the JSON datasource plugin for professional dashboards.

---

**Quick Start**: Just open http://localhost:8082 and watch your model train! 🚀
