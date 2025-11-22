# UI Refactoring Implementation Checklist

**Project:** Live Monitor Modularization
**Started:** [Date]
**Completed:** [Date]

---

## Pre-Implementation

- [ ] Backup current `live_monitor_ui.html`
  ```bash
  cp live_monitor_ui.html live_monitor_ui.html.backup
  ```

- [ ] Create branch (if using git)
  ```bash
  git checkout -b refactor/modular-ui
  ```

- [ ] Create directory structure
  ```bash
  mkdir -p js/{core,services,ui,utils}
  mkdir -p css
  ```

- [ ] Read all documentation
  - [ ] UI_MODULARIZATION_PLAN.md
  - [ ] UI_ARCHITECTURE_DIAGRAM.md
  - [ ] UI_REFACTOR_SUMMARY.md

---

## QUICK WINS (Start Here)

### 1. Extract Formatters (15 min)

- [ ] Create `js/utils/formatters.js`
- [ ] Extract these functions:
  - [ ] `formatTime(seconds)`
  - [ ] `formatTimeHHMMSS(seconds)`
  - [ ] `escapeHTML(str)`
  - [ ] `formatNumber(num, decimals)`
- [ ] Add exports
- [ ] Import in main HTML
- [ ] Replace inline calls
- [ ] Test all formatting works
- [ ] Remove inline code

**Test:** All times, numbers, and HTML display correctly

### 2. Extract Config (15 min)

- [ ] Create `js/core/config.js`
- [ ] Extract constants:
  - [ ] STATUS_FILE
  - [ ] GPU_STATS_API
  - [ ] MEMORY_STATS_API
  - [ ] INBOX_FILES_API
  - [ ] CONFIG_API
  - [ ] BASE_REFRESH_MS
  - [ ] EFFECTIVE_BATCH_SIZE
  - [ ] EMA_ALPHA
  - [ ] LIMITS object
- [ ] Export as single config object
- [ ] Import in main HTML
- [ ] Replace hardcoded references
- [ ] Remove inline constants

**Test:** API endpoints still work, refresh rate correct

### 3. Extract API Service (1 hour)

- [ ] Create `js/services/api.js`
- [ ] Create APIService class
- [ ] Implement methods:
  - [ ] `fetchStatus()`
  - [ ] `fetchGPU()`
  - [ ] `fetchMemory()`
  - [ ] `fetchInbox()`
  - [ ] `fetchQueueSamples()`
- [ ] Add error handling
- [ ] Add retry logic
- [ ] Add exponential backoff
- [ ] Export service
- [ ] Import in main HTML
- [ ] Replace inline fetch calls
- [ ] Test all API calls work
- [ ] Remove inline fetch code

**Test:** All data still loads, errors handled gracefully

### 4. Create StatusBar Component (2 hours)

- [ ] Create `js/ui/status-bar.js`
- [ ] Create StatusBar class
- [ ] Implement constructor
- [ ] Implement `update(data)` method
- [ ] Implement individual update methods:
  - [ ] `updateHealth(status)`
  - [ ] `updateLoss(loss)`
  - [ ] `updateProgress(step, total)`
  - [ ] `updateGPU(temp)`
  - [ ] `updateThroughput(value)`
  - [ ] `updateRAM(used, percent)`
  - [ ] `updateQueue(size)`
- [ ] Add color coding logic
- [ ] Add animation support
- [ ] Export class
- [ ] Import in main HTML
- [ ] Instantiate in main script
- [ ] Replace `updateStatusBar()` calls
- [ ] Test status bar updates
- [ ] Remove inline code

**Test:** Status bar displays all metrics correctly, colors work

---

## CHECKPOINT: Quick Wins Complete

**At this point you should have:**
- [ ] ~400 lines removed from main file
- [ ] 4 new module files
- [ ] All features still working
- [ ] Easier to read and maintain

**Decide:** Continue to full refactor or stop here?

---

## PHASE 1: Core Infrastructure

### 1.1 Create State Manager

- [ ] Create `js/core/state.js`
- [ ] Implement State class
- [ ] Add reactive properties
- [ ] Add getter/setter methods
- [ ] Add change notification
- [ ] Move global variables to state
- [ ] Test state updates
- [ ] Test change notifications

**Test:** State changes trigger updates

### 1.2 Create Event Bus

- [ ] Create `js/core/events.js`
- [ ] Implement EventBus class
- [ ] Add `on(event, callback)` method
- [ ] Add `emit(event, data)` method
- [ ] Add `off(event, callback)` method
- [ ] Test event publishing
- [ ] Test event subscribing

**Test:** Components can communicate via events

### 1.3 Create Data Processor

- [ ] Create `js/services/data-processor.js`
- [ ] Create DataProcessor class
- [ ] Implement `process(rawData)` method
- [ ] Extract calculation logic:
  - [ ] Throughput calculations
  - [ ] ETA calculations
  - [ ] Accuracy trends
  - [ ] Loss trends
- [ ] Test calculations
- [ ] Test data transformations

**Test:** All metrics calculated correctly

### 1.4 Create Storage Service

- [ ] Create `js/services/storage.js`
- [ ] Wrap localStorage access
- [ ] Add type-safe getters
- [ ] Add default values
- [ ] Move localStorage calls to service
- [ ] Test storage operations

**Test:** Settings persist across page loads

---

## PHASE 2: UI Components

### 2.1 TrainingStats Component

- [ ] Create `js/ui/training-stats.js`
- [ ] Create TrainingStats class
- [ ] Implement update methods
- [ ] Extract from `updateTrainingStats()`
- [ ] Test component
- [ ] Remove inline code

### 2.2 GPUMonitor Component

- [ ] Create `js/ui/gpu-monitor.js`
- [ ] Create GPUMonitor class
- [ ] Implement update methods
- [ ] Extract from `updateGPUStats()`
- [ ] Test component
- [ ] Remove inline code

### 2.3 MemoryMonitor Component

- [ ] Add to `js/ui/gpu-monitor.js`
- [ ] Create MemoryMonitor class
- [ ] Implement update methods
- [ ] Extract from `updateMemoryStats()`
- [ ] Test component
- [ ] Remove inline code

### 2.4 AccuracyTrends Component

- [ ] Create `js/ui/accuracy-trends.js`
- [ ] Create AccuracyTrends class
- [ ] Implement update methods
- [ ] Extract from `updateAccuracyTrends()`
- [ ] Test component
- [ ] Remove inline code

### 2.5 QueuePanel Component

- [ ] Create `js/ui/queue-panel.js`
- [ ] Create QueuePanel class
- [ ] Implement fetch methods
- [ ] Implement render methods
- [ ] Extract from `fetchAndDisplayQueueSamples()`
- [ ] Test component
- [ ] Remove inline code

### 2.6 FlaggedPanel Component

- [ ] Create `js/ui/flagged-panel.js`
- [ ] Create FlaggedPanel class
- [ ] Implement load methods
- [ ] Implement filter methods
- [ ] Implement render methods
- [ ] Extract from flagged panel code
- [ ] Test component
- [ ] Remove inline code

### 2.7 Modals Component

- [ ] Create `js/ui/modals.js`
- [ ] Create ModalManager class
- [ ] Implement show/hide methods
- [ ] Extract modal code
- [ ] Test modals
- [ ] Remove inline code

---

## PHASE 3: Utilities

### 3.1 Animations

- [ ] Create `js/utils/animations.js`
- [ ] Extract animation functions
- [ ] Extract delta functions
- [ ] Test animations

### 3.2 Audio

- [ ] Create `js/utils/audio.js`
- [ ] Extract audio functions
- [ ] Test sounds

---

## PHASE 4: Main Controller

- [ ] Create `js/main.js`
- [ ] Create TrainingMonitor class
- [ ] Implement constructor
- [ ] Initialize all components
- [ ] Implement `poll()` method
- [ ] Implement event listeners
- [ ] Move initialization code
- [ ] Test main controller
- [ ] Remove inline initialization

---

## PHASE 5: HTML Templates (Optional)

- [ ] Create component templates
- [ ] Extract HTML to separate files
- [ ] Implement template loading
- [ ] Test template rendering

---

## PHASE 6: CSS Refactoring (Optional)

- [ ] Create `css/base.css`
- [ ] Create `css/components.css`
- [ ] Create `css/layout.css`
- [ ] Create `css/themes.css`
- [ ] Split `monitor_styles.css`
- [ ] Update HTML imports
- [ ] Test all styles work

---

## Final Steps

- [ ] Run full manual test suite
  - [ ] Page loads without errors
  - [ ] Status bar updates
  - [ ] All panels display data
  - [ ] GPU/memory stats show
  - [ ] Accuracy trends work
  - [ ] Queue panel displays
  - [ ] Flagged panel works
  - [ ] Modals open/close
  - [ ] Keyboard shortcuts work
  - [ ] Theme toggle works
  - [ ] Compact mode works
  - [ ] Export data works
  - [ ] Sound alerts work
  - [ ] Refresh works
  - [ ] All links work

- [ ] Check console for errors
- [ ] Test in different browsers
  - [ ] Chrome
  - [ ] Firefox
  - [ ] Safari
  - [ ] Edge

- [ ] Performance check
  - [ ] Page load time
  - [ ] Update speed
  - [ ] Memory usage

- [ ] Clean up
  - [ ] Remove commented code
  - [ ] Remove console.logs
  - [ ] Remove unused functions
  - [ ] Format code
  - [ ] Add JSDoc comments

- [ ] Documentation
  - [ ] Update README
  - [ ] Document new modules
  - [ ] Update inline comments
  - [ ] Create API docs

- [ ] Commit changes
  ```bash
  git add .
  git commit -m "refactor: modularize UI into separate components"
  ```

- [ ] Create backup of old version
  ```bash
  mkdir -p archive
  mv live_monitor_ui.html.backup archive/live_monitor_ui_pre_refactor.html
  ```

---

## Rollback Plan (If Needed)

If something breaks:

```bash
# Restore backup
cp live_monitor_ui.html.backup live_monitor_ui.html

# Or revert git commit
git revert HEAD
```

---

## Success Criteria

✅ All features work as before
✅ No console errors
✅ No visual regressions
✅ Code is more maintainable
✅ Components are testable
✅ Documentation is updated

---

**Notes:**

- Check off items as you complete them
- Test after each major step
- Don't rush - quality over speed
- Take breaks between phases
- Ask for help if stuck

---

**Started:** __________
**Completed:** __________
**Time Spent:** __________
**Issues Encountered:** __________
