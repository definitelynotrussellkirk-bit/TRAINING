# Troubleshooting Session Summary - 2025-11-15

## Issue Reported
Monitor not showing evaluation data despite evaluations appearing in terminal logs.

## Root Causes Identified

### 1. **Missing Memory Stats API Server (Primary Issue)**
- The live monitor requires **TWO servers** on separate ports:
  - Port 8080: Main UI (launch_live_monitor.py)
  - Port 8081: Memory stats API (memory_stats_api.py)
- The memory API was not running, causing the monitor to fail loading completely

### 2. **Training Daemon Stopped**
- Training daemon had crashed/stopped, leaving system at step 16
- No new evaluations were being generated
- Status JSON was stale

### 3. **Browser Caching**
- Browser may have cached old version of UI with errors
- Hard refresh needed after fixes

## Solutions Implemented

### 1. Documentation Updates (CLAUDE.md)
- Added comprehensive "Troubleshooting Monitors" section
- Documented the two-server requirement prominently
- Added common issues and fixes
- Updated Quick Start to mention both servers
- Created test page reference

### 2. Health Check Script (check_health.sh)
- Automated diagnostic for all system components
- Checks:
  - Process status (daemon, monitors, APIs)
  - API endpoint responses
  - Training status and evaluations
  - Data sources (inbox files)
  - GPU utilization
- Provides clear error messages and fix commands
- Exit codes: 0 = success, 1 = errors found

### 3. Startup Script (start_all.sh)
- One-command startup for all services
- Starts in correct order with delays
- Runs health check automatically
- Simplifies deployment

### 4. Test Page (test_display.html)
- Simple diagnostic page for troubleshooting
- Minimal JavaScript for easy debugging
- Shows raw data from status JSON
- Helps isolate UI vs backend issues

## Key Learnings for Future Claude Instances

### Critical Requirements
1. **TWO servers required:** Ports 8080 AND 8081 must be running
2. **Training can stop silently:** Always check daemon process status
3. **Browser caching:** Hard refresh (Ctrl+Shift+R) often needed after fixes

### Diagnostic Steps
1. Run `./check_health.sh` first
2. Check if evaluations exist: `cat status/training_status.json | jq .total_evals`
3. Test simple page: http://localhost:8080/test_display.html
4. Check browser console for JavaScript errors
5. Verify both monitor servers running

### Common Fixes
- Missing memory API: `nohup python3 memory_stats_api.py > /dev/null 2>&1 &`
- Stopped daemon: `nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &`
- Stale browser: Hard refresh with Ctrl+Shift+R

## Files Created/Modified

### New Files
- `check_health.sh` - System health diagnostic
- `start_all.sh` - One-command startup
- `test_display.html` - Simple diagnostic page
- `TROUBLESHOOTING_SESSION_2025-11-15.md` - This file

### Modified Files
- `CLAUDE.md` - Added troubleshooting section, updated quick start
  - New section: "🔧 TROUBLESHOOTING MONITORS"
  - Updated: "Start Monitors" with two-server requirement
  - Added: Latest updates entry for 2025-11-15

## System Status After Fixes

```
✅ Training daemon: Running
✅ Live monitor (8080): Running
✅ Memory API (8081): Running
✅ Enhanced monitor (8082): Optional, not critical
✅ All API endpoints: Responding
✅ Training: Active (step 61+, 6+ evaluations)
✅ GPU: 99% utilization, 64°C
✅ Evaluations: Displaying correctly in monitor
```

## Recommendations

### For Users
1. Use `./start_all.sh` to start all services
2. Run `./check_health.sh` to verify system health
3. Bookmark the test page for quick diagnostics
4. Hard refresh browser after any server restarts

### For Future Development
1. Consider combining ports 8080 and 8081 into single server
2. Add auto-restart logic for crashed services
3. Add visual indicator in UI when memory API is unreachable
4. Consider service supervisor (systemd, supervisor, etc.)

## Training Data
- File: `no_think_tags_20k.jsonl` (21.4 MB)
- Examples: 20,000
- Total steps: 2,487
- Eval frequency: Every 10 steps
- Current progress: Active training ongoing
