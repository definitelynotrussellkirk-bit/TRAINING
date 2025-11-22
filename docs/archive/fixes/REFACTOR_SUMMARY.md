# Live Monitor Refactor Summary

**Date:** 2025-11-16
**Status:** In Progress

---

## What We're Building

A **modular, prompt-focused training monitor** with:

### 🎯 Core Features:

1. **Prompt/Response Viewer (MAIN FOCUS)**
   - Large, readable display of current training example
   - System prompt visible
   - Side-by-side Golden vs Model comparison
   - Click to expand full details

2. **Validation Dashboard**
   - Train vs Val loss chart
   - Think tag percentage trend
   - Gap monitoring with color-coded alerts

3. **Training Data Browser**
   - Browse current training file
   - See upcoming examples in queue
   - Interactive viewer showing system prompts + golden answers
   - Search and filter capabilities

4. **Expandable Detail Modals**
   - Full conversation view
   - Token-level comparisons
   - Loss breakdown
   - Metadata display

---

## New File Structure:

```
/path/to/training/
├── live_monitor_ui_v2.html       # New modular HTML (200 lines)
├── live_monitor_ui.html.backup   # Original backup (2716 lines)
├── css/
│   └── live_monitor.css          # All styling (complete ✅)
└── js/
    ├── live_monitor.js           # Main logic (TODO)
    ├── chart_manager.js          # Chart updates (TODO)
    ├── metrics_display.js        # Metric rendering (TODO)
    └── data_browser.js           # Training data browser (TODO)
```

---

## Benefits:

### For Editing:
- **Want to add validation loss chart?** → Edit `chart_manager.js` only (~100 lines)
- **Change prompt display?** → Edit `metrics_display.js` only (~150 lines)
- **Tweak colors/layout?** → Edit `css/live_monitor.css` only (~400 lines)
- **Add data browser features?** → Edit `data_browser.js` only (~200 lines)

### For Development:
- Each module is 100-200 lines instead of 2700 line monolith
- Clear separation of concerns
- Easy to test individual components
- Can add features without breaking existing code

---

## Next Steps:

1. **Create JavaScript modules** (4 files, ~600 lines total)
2. **Test with current training data**
3. **Add validation loss display**
4. **Implement think tag percentage chart**
5. **Build training data browser**
6. **Add expandable detail modals**

---

## Current Status:

- ✅ HTML structure created (`live_monitor_ui_v2.html`)
- ✅ CSS complete (`css/live_monitor.css`)
- 🔄 JavaScript modules (in progress)

**Estimated completion:** 30-40 minutes for all JS modules

---

Would you like me to:
1. **Finish the JavaScript modules now** (complete the refactor)
2. **Create a minimal working version first** (just show prompts/responses)
3. **Focus on specific feature** (e.g., just the data browser)

Let me know and I'll build it!
