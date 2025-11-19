# Flagged Examples System - Complete Guide

**Implementation Date:** 2025-11-12
**Status:** ✅ Fully integrated and ready to use

---

## 🎯 What It Does

Automatically tracks training examples that need review:
- **Mismatches**: Model output ≠ golden answer
- **High Loss**: Loss > 3.0 threshold
- **Errors**: Runtime errors during training
- **Manual**: Manually flagged by you (future feature)

---

## 🔧 How It Works

### **Automatic Flagging**
During training (train.py:779-792), every evaluation example is checked:
1. If model output doesn't match golden answer → **FLAGGED** as "mismatch"
2. If loss > 3.0 → **FLAGGED** as "high_loss"
3. Stored to `flagged_examples/flagged_examples.json`

### **Data Storage**
- **Location**: `/path/to/training/flagged_examples/`
- **File**: `flagged_examples.json`
- **Max entries**: 1,000 (keeps most recent)
- **Auto-managed**: Old entries pruned automatically

### **API Endpoint**
- **URL**: `http://localhost:8080/api/flagged_examples`
- **Returns**: JSON with examples + statistics
- **Serves**: Flagged examples to UI

---

## 📊 Floating Panel UI

### **Opening the Panel**
Click the **🚩 Flagged (N)** button in the control panel (top right)

### **Panel Features**

**📈 Statistics Dashboard:**
- Total Flagged
- Mismatches count
- High Loss count
- Average Loss

**🔍 Filter & Sort:**
- **Filter**: All / Mismatches Only / High Loss Only / Errors Only
- **Sort**: Recent First / Oldest First / Highest Loss / Lowest Loss
- **Refresh**: Manual refresh button

**📋 Example Cards:**
Each flagged example shows:
- **Step number** and **file name**
- **Flag reason** (color-coded)
- **Loss value** and **match status**
- **Full prompt** (scrollable)
- **Golden answer** vs **Model output** (side-by-side)
- **Timestamp** of when flagged
- **Notes** explaining why flagged

**Color Coding:**
- 🔴 Mismatch (red border)
- 🟠 High Loss (orange border)
- 🟡 Error (yellow-orange border)
- 🔵 Manual (cyan border)

---

## 🚀 Usage Examples

### **Review Mismatches During Training**
```bash
# Start training
python3 train.py ...

# Open live monitor in browser
http://localhost:8080/live_monitor_ui.html

# Click "🚩 Flagged (N)" button
# → Floating panel opens
# → See all mismatches in real-time
```

### **Find Problematic Examples**
1. Click "🚩 Flagged" button
2. Filter: "Mismatches Only"
3. Sort: "Highest Loss First"
4. → See worst-performing examples first
5. Review prompts/answers to find data issues

### **Track Progress**
- Badge shows count: **🚩 Flagged (42)**
- Updates every 10 seconds
- Watch count decrease as model improves

---

## 🔄 Integration Points

### **Backend (Python)**

**1. flagged_examples.py**
- Core tracking module
- Auto-flagging logic
- JSON storage management

**2. train.py**
- Integrated at line 779-792
- Flags examples during eval steps
- Passes flagged_tracker to callback

**3. launch_live_monitor.py**
- API endpoint at `/api/flagged_examples`
- Serves JSON data to frontend

### **Frontend (HTML/JS)**

**1. Floating Panel**
- Modal overlay (lines 33-85)
- Statistics grid
- Filter/sort controls
- Scrollable example list

**2. JavaScript**
- Load/render functions (lines 2433-2552)
- Filter/sort logic
- Event handlers
- Auto-refresh badge (every 10s)

---

## 📐 Customization

### **Change High Loss Threshold**
Edit `flagged_examples.py:72`:
```python
self.high_loss_threshold = 3.0  # Change to your preferred threshold
```

### **Change Max Flagged Examples**
Edit `train.py:353`:
```python
self.flagged_tracker = create_flagged_tracker(
    base_dir=Path(self.args.output_dir).parent,
    max_flagged=1000  # Change to keep more/fewer
)
```

### **Add Manual Flagging**
You can manually flag examples:
```python
from flagged_examples import create_flagged_tracker

tracker = create_flagged_tracker(base_dir="/path/to/training")
tracker.flag_example(
    step=12345,
    prompt="Your prompt",
    golden_answer="Expected output",
    model_output="Actual output",
    loss=2.5,
    matches=False,
    reason="manual",
    notes="Needs review - edge case"
)
```

---

## 🐛 Troubleshooting

### **Button shows "Flagged (0)" but examples exist**
- Click the button to refresh
- Check `flagged_examples/flagged_examples.json` exists
- Verify API endpoint: `curl http://localhost:8080/api/flagged_examples`

### **Panel is empty**
- No examples flagged yet (model is perfect!)
- Or training hasn't reached first eval step
- Check filter settings (not filtering everything out)

### **Examples not being flagged**
- Check `flagged_tracker` initialized in logs
- Look for "📋 Flagged Examples Tracker initialized" message
- Verify eval_steps in config.json (must evaluate to flag)

### **Panel won't close**
- Click "✕ Close" button
- Or press **Escape** key
- Refresh browser if stuck

---

## 📊 Example Output

**Console During Training:**
```
📋 Flagged Examples Tracker initialized: 0 existing flags
...
🔍 CURRENT TRAINING EXAMPLE - Step 125
❌ NO MATCH
📉 LOSS ON THIS EXAMPLE: 4.2315
→ Auto-flagged as 'mismatch'
```

**Floating Panel:**
```
🚩 Flagged Training Examples

[Statistics]
Total: 42  |  Mismatches: 38  |  High Loss: 4  |  Avg Loss: 3.125

[Example Card]
Step 125  |  🚩 MISMATCH  |  📁 training_data.jsonl
Loss: 4.2315  |  ✗ No Match

💡 Model output does not match golden answer

📝 Prompt:
What is the capital of France?

✅ Golden Answer:        🤖 Model Output:
Paris                    The capital is Paris, France

🕒 11/12/2025, 8:45:30 PM
```

---

## ✅ Benefits

**🔍 Debug Training Issues:**
- See exactly which examples fail
- Identify patterns in failures
- Find data quality problems

**📈 Track Model Progress:**
- Watch flagged count decrease over time
- See improvement in problematic areas
- Know when model has "learned" difficult cases

**🎯 Improve Data Quality:**
- Find ambiguous prompts
- Catch incorrect golden answers
- Identify edge cases

**💡 Make Informed Decisions:**
- "42 mismatches at step 5K, down from 200"
- "High loss examples all involve math - add more math data"
- "Model plateaued but still 15 mismatches - investigate those first"

---

## 🚀 Next Steps

**Immediate:**
1. Start training to populate flagged examples
2. Click "🚩 Flagged" button to view panel
3. Review mismatches and high-loss examples

**Future Enhancements:**
1. Add manual flagging UI (click to flag current example)
2. Export flagged examples to JSONL
3. Re-train on just flagged examples
4. Add "mark as resolved" feature
5. Visualize flagging trends over time

---

## 📁 Files Added/Modified

**New Files:**
- `flagged_examples.py` (285 lines)
- `FLAGGED_EXAMPLES_GUIDE.md` (this file)

**Modified Files:**
- `train.py` (+25 lines)
- `launch_live_monitor.py` (+45 lines)
- `live_monitor_ui.html` (+175 lines)

**Total Impact:** ~530 lines of production-ready code

---

## 🎉 Summary

✅ **Automatic flagging** during training
✅ **Beautiful floating panel** UI
✅ **Filter, sort, search** capabilities
✅ **Real-time updates** every 10 seconds
✅ **Color-coded** by flag reason
✅ **Side-by-side comparison** of golden vs model output
✅ **Persistent storage** with auto-pruning
✅ **Zero configuration** required - works out of the box

**The system is fully integrated and ready to help you debug training!** 🚀
