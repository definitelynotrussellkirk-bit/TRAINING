# Live Monitor UI Modularization Plan

**Date:** 2025-11-21
**Current State:** 2757 lines, 1595 lines inline script
**Existing Modules:** monitor_metrics.js (26K), monitor_charts.js (6K), monitor_improvements.js (18K)

---

## Current Problems

1. **Massive Inline Script** (1595 lines) - Hard to maintain, debug, and test
2. **Global State Pollution** - 40+ global variables scattered throughout
3. **Tight Coupling** - Functions directly reference DOM elements and global state
4. **No Clear Separation** - Data fetching, business logic, and UI updates mixed together
5. **Hard to Test** - Cannot easily unit test individual components
6. **Duplicate Code** - Similar patterns repeated (escapeHTML appears twice)
7. **Large HTML File** - Difficult to navigate and edit

---

## Proposed Architecture

### Module Structure

```
/js/
├── core/
│   ├── config.js           # Constants, API endpoints, limits
│   ├── state.js            # Central state management
│   └── events.js           # Event bus for component communication
├── services/
│   ├── api.js              # All API calls (status, GPU, memory, etc.)
│   ├── data-processor.js   # Data transformation and calculations
│   └── storage.js          # LocalStorage helpers
├── ui/
│   ├── status-bar.js       # Top status bar component
│   ├── training-stats.js   # Training statistics panel
│   ├── gpu-monitor.js      # GPU/Memory monitoring
│   ├── accuracy-trends.js  # Accuracy tracking panel
│   ├── queue-panel.js      # Queue preview panel
│   ├── flagged-panel.js    # Flagged examples modal
│   └── modals.js           # All modal dialogs
├── utils/
│   ├── formatters.js       # Time, number, HTML formatting
│   ├── animations.js       # Value change animations, deltas
│   └── audio.js            # Sound notifications
└── main.js                 # Application initialization & orchestration

/css/
├── base.css                # Reset, variables, typography
├── components.css          # Reusable components (panels, buttons)
├── layout.css              # Grid, flexbox, responsive
└── themes.css              # Dark/light theme definitions

/components/
├── status-bar.html         # Status bar template
├── training-panel.html     # Training stats template
├── gpu-panel.html          # GPU monitoring template
└── modals.html             # Modal templates
```

---

## Phase 1: Extract Core Infrastructure (Priority: HIGH)

**Goal:** Set up foundation for modular architecture

### 1.1 Create Core Modules

**File: `js/core/config.js`**
- Extract all constants (STATUS_FILE, APIs, LIMITS, etc.)
- Export single config object
- Remove hardcoded values

**File: `js/core/state.js`**
- Centralized state management using reactive pattern
- Replace 40+ global variables with state object
- Implement state change notifications
- Provide getters/setters with validation

**File: `js/core/events.js`**
- Simple event bus for component communication
- Subscribe/publish pattern
- Decouple components

### 1.2 Create Service Layer

**File: `js/services/api.js`**
- Extract all fetch calls into single module
- Implement retry logic, backoff
- Error handling
- Cache management
- Functions: `fetchStatus()`, `fetchGPU()`, `fetchMemory()`, `fetchInbox()`, `fetchQueue()`

**File: `js/services/data-processor.js`**
- Data transformation logic
- Calculations (throughput, ETA, accuracy trends)
- Separate business logic from UI

**File: `js/services/storage.js`**
- Wrap localStorage access
- Type-safe get/set
- Default values

---

## Phase 2: Extract UI Components (Priority: HIGH)

**Goal:** Break down monolithic updateStatusBar() and other UI functions

### 2.1 Status Bar Component

**File: `js/ui/status-bar.js`**
```javascript
class StatusBar {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
        this.subscribe();
    }

    update(data) {
        this.updateHealth(data.status);
        this.updateLoss(data.loss);
        this.updateProgress(data.current_step, data.total_steps);
        this.updateGPU(data.gpuTemp);
        this.updateThroughput(data.throughput);
        this.updateRAM(data.ramUsed);
        this.updateQueue(data.batch_queue_size);
    }

    updateHealth(status) { /* ... */ }
    updateLoss(loss) { /* ... */ }
    // ... other methods
}
```

Extract from: Lines 1411-1498 (updateStatusBar function)

### 2.2 Training Stats Component

**File: `js/ui/training-stats.js`**
```javascript
class TrainingStats {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }

    update(data) {
        this.updateBasicMetrics(data);
        this.updateProgress(data);
        this.updateTimers(data);
        this.updatePromptDisplay(data);
    }
}
```

Extract from: Lines 1536-1811 (updateTrainingStats function)

### 2.3 GPU/Memory Monitor Component

**File: `js/ui/gpu-monitor.js`**
```javascript
class GPUMonitor {
    update(gpuData) { /* ... */ }
}

class MemoryMonitor {
    update(memData) { /* ... */ }
}
```

Extract from: Lines 1884-2008

### 2.4 Accuracy Trends Component

**File: `js/ui/accuracy-trends.js`**
```javascript
class AccuracyTrends {
    constructor() {
        this.history = [];
    }

    update(accuracy) { /* ... */ }
    calculateStreak() { /* ... */ }
    renderChart() { /* ... */ }
}
```

Extract from: Lines 1812-1883

### 2.5 Queue Panel Component

**File: `js/ui/queue-panel.js`**
```javascript
class QueuePanel {
    async fetchSamples() { /* ... */ }
    render(samples) { /* ... */ }
}
```

Extract from: Lines 2330-2430

### 2.6 Flagged Examples Component

**File: `js/ui/flagged-panel.js`**
```javascript
class FlaggedPanel {
    async load() { /* ... */ }
    applyFilters() { /* ... */ }
    render() { /* ... */ }
}
```

Extract from: Lines 2576-2730

---

## Phase 3: Extract Utilities (Priority: MEDIUM)

### 3.1 Formatters

**File: `js/utils/formatters.js`**
```javascript
export const formatTime = (seconds) => { /* ... */ };
export const formatTimeHHMMSS = (seconds) => { /* ... */ };
export const escapeHTML = (str) => { /* ... */ };
export const formatNumber = (num, decimals) => { /* ... */ };
export const formatPercent = (value) => { /* ... */ };
```

Extract from: Lines 1204-1236

### 3.2 Animations

**File: `js/utils/animations.js`**
```javascript
export const animateValueChange = (elementId) => { /* ... */ };
export const updateDelta = (elementId, current, previous) => { /* ... */ };
export const showToast = (message, type) => { /* ... */ };
```

Extract from: Lines 1313-1351

### 3.3 Audio

**File: `js/utils/audio.js`**
```javascript
export const playCompletionSound = () => { /* ... */ };
export const initAudio = () => { /* ... */ };
```

Extract from: Lines 1291-1312

---

## Phase 4: Main Application Controller (Priority: HIGH)

**File: `js/main.js`**
```javascript
class TrainingMonitor {
    constructor() {
        this.statusBar = new StatusBar('statusBar');
        this.trainingStats = new TrainingStats('trainingStatsPanel');
        this.gpuMonitor = new GPUMonitor();
        this.memoryMonitor = new MemoryMonitor();
        this.accuracyTrends = new AccuracyTrends();
        this.queuePanel = new QueuePanel();
        this.flaggedPanel = new FlaggedPanel();

        this.api = new APIService();
        this.processor = new DataProcessor();

        this.initializeEventListeners();
        this.startPolling();
    }

    async poll() {
        const data = await this.api.fetchStatus();
        const processed = this.processor.process(data);

        this.statusBar.update(processed);
        this.trainingStats.update(processed);
        this.gpuMonitor.update(processed.gpu);
        this.memoryMonitor.update(processed.memory);
        this.accuracyTrends.update(processed.accuracy);
    }
}

// Initialize on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    window.monitor = new TrainingMonitor();
});
```

---

## Phase 5: HTML Templates (Priority: MEDIUM)

### 5.1 Split HTML into Templates

Instead of one 2757-line file, create:

**`live_monitor_ui.html`** (main shell, ~100 lines)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Training Monitor</title>
    <link rel="stylesheet" href="css/base.css">
    <link rel="stylesheet" href="css/components.css">
    <link rel="stylesheet" href="css/layout.css">
    <link rel="stylesheet" href="css/themes.css">
</head>
<body>
    <div id="statusBar"></div>
    <div id="mainContainer"></div>

    <!-- Load templates -->
    <script type="module" src="js/main.js"></script>
</body>
</html>
```

**`components/*.html`** - Individual templates
- Each component loads its own template
- Use template literals or fetch() to load

---

## Phase 6: CSS Refactoring (Priority: LOW)

### 6.1 Split monitor_styles.css (28K)

**`css/base.css`** - CSS reset, variables, typography
**`css/components.css`** - Buttons, panels, cards, modals
**`css/layout.css`** - Grid system, responsive breakpoints
**`css/themes.css`** - Dark/light theme definitions

---

## Migration Strategy

### Incremental Approach (Recommended)

**Step 1:** Create new module files alongside existing code
**Step 2:** Test each module in isolation
**Step 3:** Replace inline code with module imports one at a time
**Step 4:** Remove old inline code after verification
**Step 5:** Update documentation

### Testing Plan

1. **Manual Testing:** Load page, verify all features work
2. **Console Logs:** Add logging to track component lifecycle
3. **Fallback:** Keep backup of working version
4. **Gradual Rollout:** Start with utils, then services, then UI

---

## Benefits After Refactoring

✅ **Maintainability** - Each module has single responsibility
✅ **Testability** - Components can be unit tested
✅ **Reusability** - Components can be used in other projects
✅ **Performance** - Load only what's needed
✅ **Collaboration** - Multiple devs can work on different modules
✅ **Debugging** - Easier to isolate issues
✅ **Documentation** - Each module self-documenting
✅ **Scalability** - Easy to add new features

---

## Timeline Estimate

**Phase 1 (Core):** 3-4 hours
**Phase 2 (UI Components):** 6-8 hours
**Phase 3 (Utils):** 2-3 hours
**Phase 4 (Main Controller):** 3-4 hours
**Phase 5 (HTML Templates):** 2-3 hours
**Phase 6 (CSS):** 2-3 hours

**Total:** 18-25 hours for complete refactor
**Incremental approach:** Can be done in smaller chunks

---

## Quick Wins (Start Here)

1. **Extract formatters** - Low risk, high impact (15 min)
2. **Extract config** - Centralize constants (30 min)
3. **Extract API service** - Isolate network layer (1 hour)
4. **Create StatusBar class** - Most visible component (2 hours)

---

## Notes

- Use ES6 modules (`import`/`export`)
- Consider using a build tool (Rollup/Webpack) for production
- Add TypeScript for type safety (optional but recommended)
- Consider using a reactive framework (Vue/React/Svelte) for complex components
- Keep backward compatibility during migration
- Document all public APIs with JSDoc comments

---

## Next Steps

1. Review and approve plan
2. Choose starting phase (recommend: Quick Wins)
3. Create feature branch for refactor
4. Begin implementation
5. Test thoroughly before merging

---

**Status:** 📋 DRAFT - Ready for review
**Owner:** Claude + User
**Priority:** MEDIUM (improves maintainability, not urgent)
