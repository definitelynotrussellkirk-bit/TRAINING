# UI Architecture - Before & After

## Current Architecture (Monolithic)

```
┌─────────────────────────────────────────────────────────────┐
│                  live_monitor_ui.html (2757 lines)          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  <style> - All inline CSS                              │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  HTML Structure - All UI markup                        │ │
│  │  - Status bar                                          │ │
│  │  - Training panels                                     │ │
│  │  - GPU monitoring                                      │ │
│  │  - Modals (shortcuts, flagged, etc.)                  │ │
│  │  - Queue panel                                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  <script> - MASSIVE inline JS (1595 lines)            │ │
│  │                                                         │ │
│  │  • 40+ global variables                               │ │
│  │  • 30+ functions (fetch, update, render)              │ │
│  │  • State management scattered                         │ │
│  │  • Event listeners mixed in                           │ │
│  │  • Business logic + UI updates intertwined            │ │
│  │                                                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  External modules (partially refactored)               │ │
│  │  - monitor_metrics.js                                  │ │
│  │  - monitor_charts.js                                   │ │
│  │  - monitor_improvements.js                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

Problems:
❌ Hard to maintain (everything in one file)
❌ Hard to test (tight coupling)
❌ Hard to debug (no clear boundaries)
❌ Hard to scale (adding features makes it worse)
❌ Global state pollution
❌ No separation of concerns
```

---

## Proposed Architecture (Modular)

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Entry Point                   │
│                  live_monitor_ui.html (~100 lines)          │
│                                                              │
│  - Minimal HTML shell                                       │
│  - Load CSS modules                                         │
│  - Load main.js (entry point)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer (js/core/)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  config.js   │  │   state.js   │  │  events.js   │     │
│  │              │  │              │  │              │     │
│  │ • Constants  │  │ • Central    │  │ • Event bus  │     │
│  │ • API URLs   │  │   state mgmt │  │ • Pub/sub    │     │
│  │ • Limits     │  │ • Reactive   │  │ • Decouple   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Services Layer (js/services/)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    api.js    │  │ processor.js │  │  storage.js  │     │
│  │              │  │              │  │              │     │
│  │ • fetchStatus│  │ • Transform  │  │ • localStorage│     │
│  │ • fetchGPU   │  │ • Calculate  │  │ • Type-safe  │     │
│  │ • fetchMem   │  │ • Aggregate  │  │ • Defaults   │     │
│  │ • Error      │  │ • Business   │  │              │     │
│  │   handling   │  │   logic      │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    UI Components (js/ui/)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │ status-bar.js │  │ training-     │  │ gpu-monitor.js│  │
│  │               │  │ stats.js      │  │               │  │
│  │ StatusBar     │  │               │  │ GPUMonitor    │  │
│  │ class         │  │ TrainingStats │  │ class         │  │
│  │               │  │ class         │  │               │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
│                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │ accuracy-     │  │ queue-panel.js│  │ flagged-      │  │
│  │ trends.js     │  │               │  │ panel.js      │  │
│  │               │  │ QueuePanel    │  │               │  │
│  │ AccuracyTrends│  │ class         │  │ FlaggedPanel  │  │
│  │ class         │  │               │  │ class         │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
│                                                              │
│  ┌───────────────┐                                          │
│  │  modals.js    │  Each component:                        │
│  │               │  • Self-contained                       │
│  │ Modal manager │  • Single responsibility                │
│  │               │  • Testable                             │
│  └───────────────┘  • Reusable                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Utilities Layer (js/utils/)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │formatters.js │  │animations.js │  │   audio.js   │     │
│  │              │  │              │  │              │     │
│  │ • formatTime │  │ • animate    │  │ • play       │     │
│  │ • formatNum  │  │ • deltas     │  │   sounds     │     │
│  │ • escapeHTML │  │ • transitions│  │ • init       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Main Controller (js/main.js)                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  class TrainingMonitor {                                    │
│    constructor() {                                          │
│      // Initialize all components                           │
│      this.statusBar = new StatusBar();                     │
│      this.trainingStats = new TrainingStats();             │
│      this.gpuMonitor = new GPUMonitor();                   │
│      // ...                                                 │
│                                                              │
│      // Initialize services                                 │
│      this.api = new APIService();                          │
│      this.processor = new DataProcessor();                 │
│                                                              │
│      // Set up event listeners                              │
│      this.initializeEventListeners();                      │
│                                                              │
│      // Start polling                                       │
│      this.startPolling();                                  │
│    }                                                         │
│                                                              │
│    async poll() {                                           │
│      const data = await this.api.fetchStatus();            │
│      const processed = this.processor.process(data);       │
│                                                              │
│      // Notify all components                               │
│      this.statusBar.update(processed);                     │
│      this.trainingStats.update(processed);                 │
│      // ...                                                 │
│    }                                                         │
│  }                                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
       │ (Poll every 2s)
       ▼
┌─────────────────────┐
│   APIService        │ ◄─── Handles all HTTP requests
│   (js/services/)    │      Retry logic, error handling
└──────┬──────────────┘
       │
       │ Raw JSON data
       ▼
┌─────────────────────┐
│  DataProcessor      │ ◄─── Transforms, calculates
│  (js/services/)     │      Business logic here
└──────┬──────────────┘
       │
       │ Processed data
       ▼
┌─────────────────────┐
│  State Manager      │ ◄─── Central state storage
│  (js/core/state.js) │      Reactive updates
└──────┬──────────────┘
       │
       │ State change events
       ▼
┌─────────────────────┐
│  Event Bus          │ ◄─── Pub/sub notifications
│  (js/core/events.js)│      Decouple components
└──────┬──────────────┘
       │
       │ UI update events
       ▼
┌─────────────────────────────────────────────┐
│          UI Components                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │StatusBar │  │Training  │  │   GPU    │ │
│  └──────────┘  │  Stats   │  │ Monitor  │ │
│                └──────────┘  └──────────┘ │
│                                             │
│  Each component:                            │
│  1. Subscribes to state changes            │
│  2. Updates own DOM elements               │
│  3. Publishes own events                   │
└─────────────────────────────────────────────┘
```

---

## Benefits Visualization

### Before (Monolithic)
```
┌────────────────────────────────────┐
│  Everything in one giant file      │
│                                     │
│  Change anything → Risk breaking   │
│                    everything       │
│                                     │
│  Want to add feature? → Add to     │
│                         1500+ lines │
│                                     │
│  Want to test? → Can't isolate     │
│                  anything           │
└────────────────────────────────────┘
```

### After (Modular)
```
┌──────────┐  ┌──────────┐  ┌──────────┐
│Component │  │Component │  │Component │
│    A     │  │    B     │  │    C     │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     │    Independent modules   │
     │    Can test separately   │
     │    Can reuse elsewhere   │
     │                           │
     └───────────┬───────────────┘
                 │
         ┌───────▼────────┐
         │  Event Bus     │
         │  coordinates   │
         └────────────────┘
```

---

## Key Principles

1. **Single Responsibility** - Each module does one thing well
2. **Loose Coupling** - Components don't directly depend on each other
3. **High Cohesion** - Related code stays together
4. **Dependency Injection** - Pass dependencies, don't hardcode
5. **Open/Closed** - Open for extension, closed for modification

---

## Example: Status Bar Before vs After

### Before (Inline)
```javascript
// Deep in 1500+ line script...
function updateStatusBar(data, gpu, throughput, ram, ramPct, mem) {
    // 100 lines of direct DOM manipulation
    document.getElementById('summaryStatus').textContent = data.status;
    document.getElementById('lossValue').textContent = data.loss;
    // ... 95 more lines ...
    // Directly accesses global variables
    // Tightly coupled to DOM structure
    // Hard to test
}
```

### After (Modular)
```javascript
// js/ui/status-bar.js
class StatusBar {
    constructor(elementId, state) {
        this.element = document.getElementById(elementId);
        this.state = state;

        // Subscribe to state changes
        this.state.on('update', (data) => this.update(data));
    }

    update(data) {
        this.updateHealth(data.status);
        this.updateLoss(data.loss);
        this.updateProgress(data);
        this.updateGPU(data.gpu);
        this.updateQueue(data.queue);
    }

    updateHealth(status) {
        // Clear, focused, testable
        const el = this.element.querySelector('.health-label');
        el.textContent = status.toUpperCase();
        el.className = `health-label ${this.getHealthClass(status)}`;
    }

    getHealthClass(status) {
        return {
            'training': 'health-good',
            'idle': 'health-warning',
            'error': 'health-danger'
        }[status] || 'health-unknown';
    }
}

// Easy to test!
describe('StatusBar', () => {
    it('should update health indicator', () => {
        const statusBar = new StatusBar('test-element', mockState);
        statusBar.updateHealth('training');
        expect(element.textContent).toBe('TRAINING');
    });
});
```

---

## Implementation Priority

```
HIGH PRIORITY (Do First)
├── js/core/config.js          ← Extract constants (15 min)
├── js/utils/formatters.js     ← Extract formatters (15 min)
├── js/services/api.js         ← Isolate API calls (1 hour)
└── js/ui/status-bar.js        ← Most visible component (2 hours)

MEDIUM PRIORITY
├── js/core/state.js           ← State management (2 hours)
├── js/ui/training-stats.js    ← Large component (3 hours)
├── js/ui/gpu-monitor.js       ← Separate concern (1 hour)
└── js/services/data-processor.js ← Business logic (2 hours)

LOW PRIORITY (Polish)
├── js/core/events.js          ← Event bus (2 hours)
├── js/ui/queue-panel.js       ← Nice to have (1 hour)
├── css/ splitting             ← Maintainability (2 hours)
└── HTML templates             ← Advanced (3 hours)
```

---

**Total Refactor Time:** ~20 hours
**Quick Wins (first 4):** ~4 hours gets 60% of benefits!

Start with Quick Wins, see immediate improvement, then continue incrementally.
