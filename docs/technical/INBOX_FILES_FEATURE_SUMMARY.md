# Inbox Files Indicator Feature
**Added:** 2025-11-08

## ✅ What Was Added

### Backend API (`launch_live_monitor.py`)
- **New Endpoint:** `/api/inbox_files`
- **Purpose:** Returns list of `.jsonl` files in `inbox/` directory
- **Response Format:**
```json
{
  "count": 2,
  "files": [
    {"name": "leo_10k.jsonl", "size_mb": 38.5, "modified": 1699548715.5},
    {"name": "data2.jsonl", "size_mb": 12.3, "modified": 1699548610.2}
  ]
}
```
- **Features:**
  - Only scans root of inbox (not subdirectories)
  - Sorted by modification time (newest first)
  - Returns file size in MB
  - Error handling (returns empty list if inbox doesn't exist)

### Frontend UI (`live_monitor_ui.html`)
- **New Indicator:** Top-right corner (below refresh status)
- **Display:** "📁 Queued Files: [N]" badge
- **Behavior:**
  - Only shows when `count > 0`
  - Hovering shows tooltip with file names and sizes
  - Auto-updates every 2 seconds (with other stats)
- **Styling:**
  - Blue border (`#00d9ff`)
  - Badge with file count
  - Hover effect for interactivity

## 📊 How It Works

1. Every 2 seconds, `fetchStatus()` is called
2. Fetches from `/api/inbox_files` endpoint
3. If files exist, shows indicator with count
4. Tooltip displays file details on hover
5. Hides automatically when inbox is empty

## 🎯 User Experience

**Before training starts:**
- Shows "📁 Queued Files: 3" if 3 files in inbox
- User can hover to see which files are queued

**During training:**
- Count decreases as daemon processes files
- Updates in real-time

**After training:**
- Indicator disappears when inbox is empty

## 🔧 Technical Details

### CSS Classes Added
```css
.files-indicator      /* Container styling */
.files-indicator:hover /* Hover effect */
.files-badge         /* Count badge */
```

### JavaScript Functions Added
```javascript
updateInboxFiles(data)  /* Updates UI with file count */
```

## 🚀 To Activate

The server needs to be restarted for changes to take effect:
```bash
# Kill old server
ps aux | grep launch_live_monitor | grep -v grep | awk '{print $2}' | xargs kill

# Start new server
nohup python3 launch_live_monitor.py > /dev/null 2>&1 &
```

Then refresh the browser at: http://localhost:8080/live_monitor_ui.html

## 📝 Notes

- **Location-aware:** Only shows files directly in `inbox/`, not subdirectories
- **Lightweight:** Minimal performance impact (just counts files)
- **Auto-updating:** No manual refresh needed
- **Non-intrusive:** Only shows when relevant (files exist)

