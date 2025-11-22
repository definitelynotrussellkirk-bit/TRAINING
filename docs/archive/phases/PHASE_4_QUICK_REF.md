# PHASE 4 QUICK REFERENCE

**Validation System + Modular UI - 2025-11-16**

---

## 🎯 Validation Metrics

### Quick Check:
```bash
cat status/training_status.json | jq '{
  step: .current_step,
  train_loss: .loss,
  val_loss: .validation_loss,
  gap: .val_train_gap,
  think_pct: .think_tag_percent,
  accuracy: .accuracy_percent
}'
```

### Interpret Gap:
- **< 0.3:** ✅ Excellent - model generalizing well
- **0.3 - 0.5:** ⚠️ Warning - monitor closely
- **> 0.5:** 🚨 Overfitting - consider stopping

### Interpret Think Tag %:
- **100%:** Model adding `<think>` tags (base model behavior)
- **< 20%:** ✅ Good - learning clean format
- **0%:** 🎯 Perfect - no unwanted tags

---

## 🖥️ New Modular Monitor

**URL:** http://localhost:8080/live_monitor_ui_v2.html

**Features:**
- ✅ Large prompt/response display
- ✅ Train vs Val loss chart
- ✅ Think tag percentage chart
- ✅ Training data browser
- ✅ Expandable detail modals
- ✅ Color-coded metrics

---

## 📁 Modular File Structure

```
live_monitor_ui_v2.html  (200 lines) - HTML structure
css/live_monitor.css     (400 lines) - Styling
js/live_monitor.js       (200 lines) - Main logic
js/chart_manager.js      (150 lines) - Charts
js/metrics_display.js    (120 lines) - Prompts
js/data_browser.js       (130 lines) - Data browser
```

**To edit:**
- Prompt display → `js/metrics_display.js`
- Charts → `js/chart_manager.js`
- Colors/layout → `css/live_monitor.css`
- Data browser → `js/data_browser.js`

---

## 📚 Documentation

- `VALIDATION_SYSTEM_DOCS.md` - Validation system details
- `REFACTOR_COMPLETE.md` - Modular UI details
- `VALIDATION_SYSTEM_SUMMARY.md` - Overview
- `CLAUDE.md` - Main reference (updated)

---

## 🔑 Key Code Locations

**Validation:**
- `train.py:754-802` - Validation loss computation
- `training_status.py:95-101` - Status fields
- `data/validation/syllo_validation_1000.jsonl` - Validation set

**Think Tags:**
- `training_status.py:216-220` - Think tag tracking
- `js/chart_manager.js:80-120` - Think tag chart

---

**Current Status (Step 1617+):**
- Train Loss: 0.132
- Val Loss: 0.127
- Gap: -0.005 ✅ (excellent!)
- Think %: 100% (will decrease)
