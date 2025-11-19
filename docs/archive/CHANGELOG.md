# Changelog

All notable changes to the Ultimate Trainer System.

---

## [2025-11-12] - Memory Management & Accuracy Trends

### Added
- **System RAM Monitoring**
  - New memory stats API on port 8081
  - RAM usage panel in live UI with gauges
  - Training process memory tracking
  - OOM (Out of Memory) risk indicator
  - Color-coded alerts (green/yellow/red)
  - Added RAM to persistent status bar
  - Memory metrics in data export

- **Accuracy Trends Analysis**
  - Last 20 evaluations metric (short-term)
  - Last 50 evaluations metric (medium-term)
  - Trend analysis (improving/stable/regressing)
  - Color-coded performance indicators
  - Comparison to overall accuracy baseline
  - Accuracy history in data export

- **Documentation**
  - `MEMORY_LEAK_FIX.md` - Technical fix details
  - `MEMORY_MONITORING_QUICKREF.md` - Quick reference
  - `UI_IMPROVEMENTS_SUMMARY.md` - UI changes summary
  - `ACCURACY_TRENDS_FEATURE.md` - Trends guide
  - `DOCS_INDEX.md` - Complete documentation index
  - Updated `README.md` with comprehensive overview
  - Updated `CLAUDE.md` with latest features

- **Organization**
  - Created `docs/` directory structure
  - Moved guides to `docs/guides/`
  - Moved technical docs to `docs/technical/`
  - Archived session summaries to `docs/archive/`
  - Created `.gitignore` for repository cleanliness

### Fixed
- **Memory Leak in Dataset Tokenization** (train.py:394-410)
  - Added chunked processing (batch_size=1000)
  - Disabled caching to prevent accumulation
  - Added explicit garbage collection
  - Reduced peak memory usage by ~70% (50GB → 15GB)

- **Memory Leak in Dataset Preparation** (train.py:346-350)
  - Added cleanup of intermediate data structures
  - Explicit deletion of large lists
  - Garbage collection after processing

### Changed
- Updated health indicator to prioritize RAM issues
- Enhanced system health monitoring with memory thresholds
- Improved export data to include memory and accuracy metrics
- Reorganized documentation for better navigation

### Performance
- Memory usage during tokenization: 50-60GB → 15-20GB
- Training process stable at 6-10GB (was growing to 40GB+)
- OOM crashes: Frequent → Should not occur

---

## [2025-11-11] - Continuous Training Fix

### Fixed
- **Continuous Training Checkpoint Preservation**
  - Changed `save_total_limit=None` (was 3) to keep all checkpoints
  - Reduced `save_steps` to 100 (was 1250) for frequent checkpoints
  - Removed manual checkpoint deletion code
  - Preserved optimizer state between batches
  - Fixed global_step counter to never reset

### Added
- **Auto-Flattening Inbox**
  - Daemon automatically moves `.jsonl` files from subdirectories to root
  - Handles naming collisions with counters
  - Cleans up empty subdirectories
  - Can now copy entire LEO output directories

- **Queue Time Estimation**
  - Throughput calculation (MB/hour)
  - Queue completion time prediction
  - Shows in live UI when files queued

### Changed
- Checkpoint frequency: 1250 → 100 steps
- Checkpoint retention: Last 3 → All (manual cleanup)
- Training is now truly continuous across batches

### Documentation
- Created `CONTINUOUS_TRAINING_GUIDE.md`
- Updated `CLAUDE.md` with continuous training details
- Added inbox flattening documentation

---

## [2025-11-08] - Monitoring Improvements

### Added
- UTF-8 encoding support for emojis in web UI
- Enhanced monitor tooltips and help text
- Evaluation frequency configuration (25 steps)

### Fixed
- Character encoding issues in web monitors
- Display of special characters in training examples

---

## [Earlier] - Initial System

### Core Features
- QLoRA (4-bit) training for Qwen 2.5 7B
- Auto-ingestion daemon for continuous training
- Live monitoring UI (port 8080)
- Enhanced Gradio monitor (port 8082)
- GPU stats monitoring
- Real-time loss and accuracy tracking
- Prompt/answer evaluation display
- Daily consolidation (3 AM)
- Automatic checkpoint saving
- JSONL data format support

### Monitoring
- Real-time training status
- GPU temperature, utilization, memory
- Loss sparkline and trend analysis
- Running accuracy tracking
- ETA calculations
- Progress bars
- Recent examples display

### Configuration
- `config.json` for all settings
- Adjustable LoRA rank and alpha
- Configurable batch size and learning rate
- Customizable evaluation frequency
- System prompt configuration

---

## Version Numbering

We don't use formal version numbers. Changes are tracked by date.
- **2025-11-12:** Memory management + accuracy trends
- **2025-11-11:** Continuous training fix
- **Earlier:** Initial development

---

## Future Roadmap

### Planned Features
- [ ] Automatic learning rate scheduling
- [ ] Advanced loss analysis (validation loss tracking)
- [ ] Model comparison tools
- [ ] Automated hyperparameter tuning
- [ ] Multi-GPU support
- [ ] Distributed training
- [ ] Cloud deployment guide

### Monitoring Enhancements
- [ ] Historical accuracy charts
- [ ] Loss derivative tracking
- [ ] Gradient norm monitoring
- [ ] Training stability scores
- [ ] Automated anomaly detection

### UX Improvements
- [ ] Mobile-responsive UI
- [ ] Dark/light theme persistence
- [ ] Customizable dashboard layouts
- [ ] Email/Slack notifications
- [ ] Training completion webhooks

---

## Breaking Changes

None yet! System is backwards compatible with all previous training runs.

---

## Deprecations

None yet.

---

## Security

No security issues identified. System runs locally only.

---

## Contributors

- System design and implementation: Russ
- AI assistance and documentation: Claude (Anthropic)

---

**For detailed changes, see git commit history and session summaries in `docs/archive/`**
