#!/usr/bin/env python3
"""
Daily Training Report Generator

Analyzes last 24 hours of training and creates summary report.
Run via cron: 0 8 * * * python3 daily_report.py
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import argparse

class DailyReporter:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.status_file = self.base_dir / "status" / "training_status.json"
        self.log_dir = self.base_dir / "logs"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def read_current_status(self):
        """Read current training status"""
        try:
            with open(self.status_file) as f:
                return json.load(f)
        except:
            return None

    def analyze_logs(self):
        """Analyze daemon logs from last 24 hours"""
        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()

        # Read logs from yesterday and today
        logs_to_check = [
            self.log_dir / f"daemon_{yesterday.strftime('%Y%m%d')}.log",
            self.log_dir / f"daemon_{today.strftime('%Y%m%d')}.log"
        ]

        stats = {
            'files_processed': 0,
            'errors': [],
            'steps_completed': 0,
            'training_time': 0
        }

        for log_file in logs_to_check:
            if not log_file.exists():
                continue

            try:
                with open(log_file) as f:
                    for line in f:
                        if 'Successfully trained on' in line:
                            stats['files_processed'] += 1
                        elif 'ERROR' in line or 'Error' in line:
                            stats['errors'].append(line.strip())
            except:
                pass

        return stats

    def analyze_anomalies(self):
        """Check anomaly snapshots from last 24 hours"""
        yesterday = datetime.now() - timedelta(days=1)

        anomalies = {
            'best_models': [],
            'loss_spikes': [],
            'accuracy_drops': [],
            'divergences': [],
            'prediction_anomalies': [],
            'z_score_anomalies': []
        }

        if not self.snapshots_dir.exists():
            return anomalies

        for snapshot_dir in self.snapshots_dir.glob("anomaly_*"):
            # Check if created in last 24 hours
            created_time = datetime.fromtimestamp(snapshot_dir.stat().st_mtime)
            if created_time < yesterday:
                continue

            # Read metadata
            metadata_file = snapshot_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file) as f:
                    meta = json.load(f)

                triggers = meta.get('triggers', [])
                for trigger in triggers:
                    if 'best_model' in trigger:
                        anomalies['best_models'].append(meta)
                    elif 'loss_spike' in trigger:
                        anomalies['loss_spikes'].append(meta)
                    elif 'accuracy_drop' in trigger:
                        anomalies['accuracy_drops'].append(meta)
                    elif 'divergence' in trigger:
                        anomalies['divergences'].append(meta)
                    elif 'zscore' in trigger:
                        anomalies['z_score_anomalies'].append(meta)
                    elif 'loss' in trigger or 'answer' in trigger:
                        anomalies['prediction_anomalies'].append(meta)
            except:
                pass

        return anomalies

    def check_disk_memory(self):
        """Check disk and memory status"""
        import shutil
        import psutil

        disk = shutil.disk_usage(self.base_dir)
        mem = psutil.virtual_memory()

        return {
            'disk_used_gb': disk.used / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100,
            'ram_used_gb': mem.used / (1024**3),
            'ram_percent': mem.percent,
            'checkpoints': len(list((self.base_dir / "current_model").glob("checkpoint-*"))) if (self.base_dir / "current_model").exists() else 0
        }

    def generate_report(self):
        """Generate comprehensive daily report"""
        status = self.read_current_status()
        log_stats = self.analyze_logs()
        anomalies = self.analyze_anomalies()
        resources = self.check_disk_memory()

        # Build report
        report_lines = []
        report_lines.append("# Daily Training Report")
        report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_lines.append(f"**Period:** Last 24 hours\n")

        # Training Status
        report_lines.append("## 📊 Training Status\n")
        if status:
            progress_pct = (status.get('current_step', 0) / status.get('total_steps', 1)) * 100
            report_lines.append(f"- **Status:** {status.get('status', 'Unknown')}")
            report_lines.append(f"- **Progress:** {status.get('current_step', 0):,} / {status.get('total_steps', 0):,} steps ({progress_pct:.1f}%)")
            report_lines.append(f"- **Current Loss:** {status.get('loss', 0):.4f}")
            report_lines.append(f"- **Accuracy:** {status.get('accuracy_percent', 0):.1f}%")
            report_lines.append(f"- **Learning Rate:** {status.get('learning_rate', 0):.6f}")
        else:
            report_lines.append("- **Status:** No status available")
        report_lines.append("")

        # Activity Summary
        report_lines.append("## 📈 Activity (Last 24 Hours)\n")
        report_lines.append(f"- **Files Processed:** {log_stats['files_processed']}")
        report_lines.append(f"- **Errors:** {len(log_stats['errors'])}")
        if log_stats['errors']:
            report_lines.append("\n**Recent Errors:**")
            for error in log_stats['errors'][-3:]:
                report_lines.append(f"  - {error[:100]}...")
        report_lines.append("")

        # Anomaly Summary
        report_lines.append("## 🔍 Anomaly Detection\n")
        total_anomalies = sum(len(v) for v in anomalies.values())
        report_lines.append(f"**Total Anomalies Detected:** {total_anomalies}\n")

        if anomalies['best_models']:
            report_lines.append(f"### 🏆 Best Models ({len(anomalies['best_models'])})")
            for meta in anomalies['best_models'][-3:]:
                report_lines.append(f"- Step {meta['step']:,}: Loss {meta['loss']:.4f}")
            report_lines.append("")

        if anomalies['loss_spikes']:
            report_lines.append(f"### 🔥 Loss Spikes ({len(anomalies['loss_spikes'])})")
            for meta in anomalies['loss_spikes'][-3:]:
                triggers = ', '.join(meta.get('triggers', []))
                report_lines.append(f"- Step {meta['step']:,}: {triggers}")
            report_lines.append("")

        if anomalies['accuracy_drops']:
            report_lines.append(f"### 📉 Accuracy Drops ({len(anomalies['accuracy_drops'])})")
            for meta in anomalies['accuracy_drops'][-3:]:
                triggers = ', '.join(meta.get('triggers', []))
                report_lines.append(f"- Step {meta['step']:,}: {triggers}")
            report_lines.append("")

        if anomalies['prediction_anomalies']:
            report_lines.append(f"### ⚠️ Prediction Anomalies ({len(anomalies['prediction_anomalies'])})")
            for meta in anomalies['prediction_anomalies'][-3:]:
                triggers = ', '.join(meta.get('triggers', []))
                report_lines.append(f"- Step {meta['step']:,}: {triggers}")
            report_lines.append("")

        if anomalies['divergences']:
            report_lines.append(f"### 🚨 Divergences ({len(anomalies['divergences'])})")
            for meta in anomalies['divergences']:
                report_lines.append(f"- Step {meta['step']:,}: Training diverged!")
            report_lines.append("")

        # Resource Status
        report_lines.append("## 💾 Resource Status\n")
        report_lines.append(f"- **Disk Used:** {resources['disk_used_gb']:.1f} GB")
        report_lines.append(f"- **Disk Free:** {resources['disk_free_gb']:.1f} GB ({100-resources['disk_percent']:.1f}% free)")
        report_lines.append(f"- **RAM Usage:** {resources['ram_used_gb']:.1f} GB ({resources['ram_percent']:.1f}%)")
        report_lines.append(f"- **Checkpoints:** {resources['checkpoints']}")
        report_lines.append("")

        # Health Summary
        report_lines.append("## ✅ Health Summary\n")
        health_issues = []

        if resources['disk_percent'] > 85:
            health_issues.append("⚠️ **Disk usage critical** (>85%)")
        if resources['ram_percent'] > 85:
            health_issues.append("⚠️ **RAM usage high** (>85%)")
        if anomalies['divergences']:
            health_issues.append("🚨 **Training divergence detected!**")
        if len(log_stats['errors']) > 10:
            health_issues.append(f"⚠️ **Many errors** ({len(log_stats['errors'])} in 24h)")

        if health_issues:
            for issue in health_issues:
                report_lines.append(f"- {issue}")
        else:
            report_lines.append("- ✅ **All systems healthy**")

        report_lines.append("")

        # Recommendations
        report_lines.append("## 💡 Recommendations\n")
        recs = []

        if resources['checkpoints'] > 50:
            recs.append("- Run `./cleanup_checkpoints.sh` to free disk space")
        if anomalies['best_models']:
            best = max(anomalies['best_models'], key=lambda x: x['step'])
            recs.append(f"- Consider using best model from step {best['step']:,} (loss {best['loss']:.4f})")
        if anomalies['divergences']:
            recs.append("- **Urgent:** Review divergence anomalies, training may need to rollback")
        if status and status.get('status') == 'idle':
            recs.append("- Training idle - add data files to `inbox/` to continue")

        if not recs:
            recs.append("- Continue monitoring - everything looks good!")

        for rec in recs:
            report_lines.append(rec)

        report_lines.append("")
        report_lines.append("---")
        report_lines.append(f"*Generated automatically by daily_report.py*")

        return "\n".join(report_lines)

    def save_report(self, report: str):
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_file = self.reports_dir / f"daily_report_{timestamp}.md"

        with open(report_file, 'w') as f:
            f.write(report)

        # Also save as "latest"
        latest_file = self.reports_dir / "latest_report.md"
        with open(latest_file, 'w') as f:
            f.write(report)

        return report_file

def main():
    parser = argparse.ArgumentParser(description="Generate daily training report")
    parser.add_argument('--base-dir', default='/path/to/training', help='Base training directory')
    parser.add_argument('--print', action='store_true', help='Print to console')
    parser.add_argument('--email', help='Email address to send report to (optional)')

    args = parser.parse_args()

    reporter = DailyReporter(args.base_dir)
    report = reporter.generate_report()
    report_file = reporter.save_report(report)

    if args.print:
        print(report)

    print(f"✓ Report saved to: {report_file}")
    print(f"✓ Also available at: {reporter.reports_dir / 'latest_report.md'}")

    # TODO: Email support
    if args.email:
        print(f"Note: Email feature not yet implemented. Report saved to {report_file}")

if __name__ == '__main__':
    main()
