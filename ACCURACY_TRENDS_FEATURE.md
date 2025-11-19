# Accuracy Trends Feature - Added 2025-11-12

## 🎯 What's New

Added granular accuracy trend analysis to the live monitor UI that shows if your model is **improving, stable, or regressing** over time.

## 📊 New Metrics Display

In the **Running Accuracy** panel, you now see:

```
🎯 Running Accuracy
Overall: 64.3% (9 / 14)

📈 Accuracy Trends
├─ Last 20 Evaluations: 75.0% (15/20)     [GREEN if > overall]
├─ Last 50 Evaluations: 68.2% (34/50)     [GREEN if > overall]
└─ Trend Analysis: ↑ Improving (+10.7%)    [Shows direction]
```

## 🎨 What Each Metric Shows

### 1. **Overall Accuracy** (top, big number)
- All-time accuracy since training started
- Baseline for comparison
- What you've always had

### 2. **Last 20 Evaluations** (NEW!)
- **Short-term performance**
- Most recent 20 evaluation examples
- Reacts quickly to changes
- **Green:** > overall (model improving!)
- **Red:** < overall (model regressing)
- **White:** ≈ overall (stable)

### 3. **Last 50 Evaluations** (NEW!)
- **Medium-term trend**
- More stable than Last 20
- Shows sustained improvement
- Same color coding as Last 20

### 4. **Trend Analysis** (NEW!)
- **↑ Improving (+X%):** Recent > overall - model getting better! 🎉
- **→ Stable:** Recent ≈ overall - model plateaued 📊
- **↓ Regressing (-X%):** Recent < overall - investigate 😟

## 🎯 How to Read It

### Example 1: Model Improving
```
Overall: 60.0%
Last 20: 75.0% (GREEN) ← Recent performance better!
Last 50: 65.0% (GREEN)
Trend: ↑ Improving (+15.0%)
```
**Meaning:** Model is learning! Recent performance significantly better than overall average.

### Example 2: Model Plateaued
```
Overall: 80.0%
Last 20: 82.0% (WHITE) ← About the same
Last 50: 81.0% (WHITE)
Trend: → Stable
```
**Meaning:** Model has reached a plateau. May need more data or different approach.

### Example 3: Model Regressing
```
Overall: 70.0%
Last 20: 55.0% (RED) ← Recent performance worse!
Last 50: 60.0% (RED)
Trend: ↓ Regressing (-15.0%)
```
**Meaning:** Something's wrong! Check for:
- Data quality issues in recent batches
- Learning rate too high (loss unstable)
- Overfitting on earlier data

## 🔍 Color Coding Thresholds

### Last 20 Evaluations:
- **Green (success):** > overall + 5%
- **White (neutral):** within ±5% of overall
- **Red (danger):** < overall - 5%

### Last 50 Evaluations:
- **Green (success):** > overall + 3%
- **White (neutral):** within ±3% of overall
- **Red (danger):** < overall - 3%

### Trend Analysis:
- **↑ Green:** Difference > +3%
- **→ White:** Difference within ±3%
- **↓ Red:** Difference < -3%

## 🎓 Understanding the Insights

### What "Improving" Means:
Your model is **actively learning** and getting better at the task. Recent predictions are more accurate than your overall average. This is what you want to see!

### What "Stable" Means:
Your model has **reached a plateau**. It's not getting worse, but not improving either. This could mean:
- Model has learned what it can from current data
- Need more diverse examples
- May need to adjust hyperparameters

### What "Regressing" Means:
Your model's **recent performance is worse** than overall. Investigate:
- Recent data quality (bad batches?)
- Loss stability (learning rate too high?)
- Overfitting indicators
- System issues (OOM, throttling?)

## 📈 Tracking History

The UI tracks up to **100 recent evaluations** in memory:
- Stored as: `{step: number, correct: boolean}`
- Updates automatically from `recent_examples` in status JSON
- Exported with training data for analysis

## 💾 Data Export Enhanced

When you click **Export** (or press **E**), now includes:
```json
{
  "accuracy": "64.3%",
  "accuracyLast20": "75.0% (15/20)",
  "accuracyLast50": "68.2% (34/50)",
  "accuracyTrend": "↑ Improving (+10.7%)",
  "accuracyHistory": [
    {"step": 15825, "correct": true},
    {"step": 15850, "correct": true},
    ...
  ]
}
```

## 🎯 Practical Use Cases

### Use Case 1: Verify Learning
**Watch:** Last 20 evaluations
**Look for:** Green color, upward trend
**Action:** Continue training confidently

### Use Case 2: Detect Quality Issues
**Watch:** Sudden drop in Last 20
**Look for:** Red color, downward trend
**Action:** Check recent data files for corruption

### Use Case 3: Find Optimal Stopping Point
**Watch:** All three metrics converge (become similar)
**Look for:** Stable trend
**Action:** Model may be fully trained, consider stopping

### Use Case 4: Compare Approaches
**Watch:** Trend after changing learning rate/batch size
**Look for:** Improvement in Last 20/50
**Action:** Keep changes if metrics improve

## 🛠️ Technical Details

### Calculation Method:
```javascript
// Last 20 accuracy
last20 = accuracyHistory.slice(-20)
correct20 = last20.filter(h => h.correct).length
accuracy20 = (correct20 / last20.length) * 100

// Compare to overall
diff = accuracy20 - overallAccuracy
if (diff > 5%) → GREEN
else if (diff < -5%) → RED
else → WHITE
```

### When Metrics Appear:
- **Last 20:** Shows after 10+ evaluations (partial until 20)
- **Last 50:** Shows after 50+ evaluations (uses all available until then)
- **Trend:** Shows after 10+ evaluations

### Update Frequency:
- Refreshes every 2 seconds (with all other metrics)
- Updates immediately when new evaluation completes

## 📱 Where to Find It

**Live Monitor:** http://localhost:8080/live_monitor_ui.html

**Location in UI:**
1. Scroll to "🎯 EVALUATION RESULTS" section
2. Look in "🎯 Running Accuracy" panel
3. See new "📈 Accuracy Trends" subsection

## 🎁 Bonus Features

### Tooltips:
Hover over ℹ️ icons for detailed explanations:
- What each metric means
- How to interpret values
- What actions to take

### Visual Feedback:
- Colors change instantly based on thresholds
- Numbers update every 2 seconds
- Clear labels show trend direction

### Accessibility:
- Screen reader friendly
- Clear visual indicators
- Keyboard navigation compatible

## 🔗 Related Metrics

This complements the existing **Loss Trend** metric:
- **Loss Trend:** Shows if loss decreasing/increasing
- **Accuracy Trend:** Shows if predictions getting better/worse

Both together give complete picture of training health!

## ✅ Current Status

Your training right now:
- **Overall Accuracy:** 64.3% (9/14 correct)
- **Recent Examples:** 5 in history
- **Status:** Collecting data... (need 10+ for trends)

After a few more evaluation steps, you'll see:
- Last 20 accuracy
- Last 50 accuracy
- Trend analysis with colors

## 🎉 Benefits

1. **Early Warning:** Catch regression before wasting hours
2. **Confidence:** See improvement in real-time
3. **Optimization:** Know when to stop training
4. **Debugging:** Isolate problem batches quickly
5. **Comparison:** Test different approaches objectively

## 🚀 Next Steps

1. **Refresh your browser** at http://localhost:8080/live_monitor_ui.html
2. **Wait for 10+ evaluations** (happens every 25 steps)
3. **Watch the trends** appear and update
4. **Use colors** as quick visual indicators
5. **Export data** for historical analysis

---

**Enjoy the new insights into your model's learning progress!** 📊🎯
