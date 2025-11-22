#!/bin/bash
# Phase 4: Directory Reorganization Script
# Safely moves files to new organized structure

set -e  # Exit on error

echo "🏗️  Phase 4: Directory Reorganization"
echo "======================================"
echo ""

# Core training scripts → bin/
echo "📦 Moving core training scripts to bin/..."
mv -v training_daemon_integrated.py bin/
mv -v training_daemon.py bin/
mv -v train.py bin/
mv -v training_controller.py bin/
mv -v training_queue.py bin/
mv -v model_versioner.py bin/
mv -v backup_manager.py bin/
mv -v evolution_tracker.py bin/
mv -v consolidate_model.py bin/

# Utility scripts → bin/
echo "📦 Moving utility scripts to bin/..."
mv -v compare_models.py bin/ 2>/dev/null || true
mv -v convert_format.py bin/ 2>/dev/null || true
mv -v convert_leo_data.py bin/ 2>/dev/null || true
mv -v add_system_prompt.py bin/ 2>/dev/null || true
mv -v flagged_examples.py bin/ 2>/dev/null || true
mv -v fixed_eval.py bin/ 2>/dev/null || true
mv -v daily_report.py bin/ 2>/dev/null || true
mv -v pattern_tracker.py bin/ 2>/dev/null || true
mv -v streaming_metrics.py bin/ 2>/dev/null || true
mv -v throughput_monitor.py bin/ 2>/dev/null || true

# Shell scripts → bin/
echo "📦 Moving shell scripts to bin/..."
mv -v check_health.sh bin/
mv -v start_all.sh bin/ 2>/dev/null || true
mv -v restart_qwen3.sh bin/ 2>/dev/null || true
mv -v cleanup_checkpoints.sh bin/ 2>/dev/null || true

# Monitoring files → monitoring/
echo "📦 Moving monitoring files to monitoring/ui/..."
mv -v launch_live_monitor.py monitoring/servers/
mv -v live_monitor_ui.html monitoring/ui/
mv -v evolution_viewer.html monitoring/ui/
mv -v evolution_viewer.js monitoring/assets/js/
mv -v monitor_styles.css monitoring/assets/css/
mv -v monitor_*.js monitoring/assets/js/ 2>/dev/null || true
mv -v lora_monitor.py monitoring/servers/ 2>/dev/null || true
mv -v smart_alerts.py monitoring/servers/ 2>/dev/null || true
mv -v detailed_monitor.py monitoring/servers/ 2>/dev/null || true
mv -v desktop_notifier.py monitoring/servers/ 2>/dev/null || true
mv -v detail_collector.py monitoring/servers/ 2>/dev/null || true

# Documentation → docs/
echo "📦 Moving documentation to docs/..."
mv -v *_COMPLETE.md docs/ 2>/dev/null || true
mv -v INTEGRATION_COMPLETE.md docs/
mv -v QUICK_START_INTEGRATED.md docs/
mv -v SESSION_SUMMARY_FINAL_2025-11-16.md docs/
mv -v *_PLAN.md docs/technical/ 2>/dev/null || true
mv -v *_GUIDE.md docs/guides/ 2>/dev/null || true
mv -v ACCURACY_TRENDS_FEATURE.md docs/technical/ 2>/dev/null || true
mv -v CUDA_MULTIPROCESSING_FIX.md docs/technical/
mv -v CATASTROPHIC_LOSS_POSTMORTEM.md docs/archive/ 2>/dev/null || true

echo ""
echo "✅ File reorganization complete!"
echo ""
echo "📁 New structure:"
echo "  bin/               - All executables"
echo "  monitoring/        - Web UI and monitoring servers"
echo "  docs/              - All documentation"
echo ""
