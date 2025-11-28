#!/usr/bin/env python3
"""
Training Progress Analyzer

Analyzes training logs and generates interactive visualizations
showing loss curves, learning rates, and training metrics over time.

Usage:
    python3 analyze_training.py
    python3 analyze_training.py --log logs/daemon_20251116.log
    python3 analyze_training.py --output reports/training_analysis.html
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import subprocess

def parse_training_log(log_file: str) -> Dict[str, List]:
    """Extract training metrics from log file."""

    data = {
        'steps': [],
        'losses': [],
        'learning_rates': [],
        'timestamps': [],
        'examples_per_sec': []
    }

    # Patterns to match
    step_pattern = r"'global_step': (\d+)"
    loss_pattern = r"'loss': ([\d.]+)"
    lr_pattern = r"'learning_rate': ([\d.e-]+)"

    print(f"Parsing log file: {log_file}")

    with open(log_file) as f:
        for line in f:
            # Look for training step info
            if "'global_step':" in line and "'loss':" in line:
                step_match = re.search(step_pattern, line)
                loss_match = re.search(loss_pattern, line)
                lr_match = re.search(lr_pattern, line)

                if step_match and loss_match:
                    step = int(step_match.group(1))
                    loss = float(loss_match.group(1))
                    lr = float(lr_match.group(1)) if lr_match else 0.0

                    data['steps'].append(step)
                    data['losses'].append(loss)
                    data['learning_rates'].append(lr)

    print(f"Found {len(data['steps'])} training steps")
    return data

def calculate_statistics(data: Dict) -> Dict:
    """Calculate training statistics."""

    if not data['steps']:
        return {}

    losses = data['losses']

    stats = {
        'total_steps': len(data['steps']),
        'start_loss': losses[0] if losses else 0,
        'end_loss': losses[-1] if losses else 0,
        'min_loss': min(losses) if losses else 0,
        'max_loss': max(losses) if losses else 0,
        'avg_loss': sum(losses) / len(losses) if losses else 0,
        'improvement': losses[0] - losses[-1] if len(losses) > 1 else 0,
        'improvement_pct': ((losses[0] - losses[-1]) / losses[0] * 100) if len(losses) > 1 and losses[0] > 0 else 0
    }

    return stats

def generate_html_report(data: Dict, stats: Dict, output_file: str):
    """Generate interactive HTML report with charts."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Training Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #0d1117;
            color: #c9d1d9;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #161b22;
            border-radius: 8px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #161b22;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }}
        .stat-label {{
            font-size: 12px;
            color: #8b949e;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #58a6ff;
        }}
        .chart {{
            background: #161b22;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #30363d;
        }}
        h1 {{ color: #58a6ff; }}
        h2 {{ color: #8b949e; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Training Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Total Steps</div>
            <div class="stat-value">{stats.get('total_steps', 0):,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Start Loss</div>
            <div class="stat-value">{stats.get('start_loss', 0):.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">End Loss</div>
            <div class="stat-value">{stats.get('end_loss', 0):.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Best Loss</div>
            <div class="stat-value">{stats.get('min_loss', 0):.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Improvement</div>
            <div class="stat-value">{stats.get('improvement', 0):.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Improvement %</div>
            <div class="stat-value">{stats.get('improvement_pct', 0):.1f}%</div>
        </div>
    </div>

    <div class="chart" id="loss-chart"></div>
    <div class="chart" id="lr-chart"></div>

    <script>
        // Loss curve
        var lossTrace = {{
            x: {data['steps']},
            y: {data['losses']},
            mode: 'lines',
            name: 'Training Loss',
            line: {{ color: '#58a6ff', width: 2 }}
        }};

        var lossLayout = {{
            title: 'Training Loss Over Time',
            xaxis: {{ title: 'Step', color: '#8b949e', gridcolor: '#30363d' }},
            yaxis: {{ title: 'Loss', color: '#8b949e', gridcolor: '#30363d' }},
            plot_bgcolor: '#0d1117',
            paper_bgcolor: '#161b22',
            font: {{ color: '#c9d1d9' }}
        }};

        Plotly.newPlot('loss-chart', [lossTrace], lossLayout);

        // Learning rate curve
        var lrTrace = {{
            x: {data['steps']},
            y: {data['learning_rates']},
            mode: 'lines',
            name: 'Learning Rate',
            line: {{ color: '#3fb950', width: 2 }}
        }};

        var lrLayout = {{
            title: 'Learning Rate Schedule',
            xaxis: {{ title: 'Step', color: '#8b949e', gridcolor: '#30363d' }},
            yaxis: {{ title: 'Learning Rate', color: '#8b949e', gridcolor: '#30363d' }},
            plot_bgcolor: '#0d1117',
            paper_bgcolor: '#161b22',
            font: {{ color: '#c9d1d9' }}
        }};

        Plotly.newPlot('lr-chart', [lrTrace], lrLayout);
    </script>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ Report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze training progress")
    parser.add_argument("--log", help="Path to training log file")
    parser.add_argument("--output", default="reports/training_analysis.html",
                       help="Output HTML file")

    args = parser.parse_args()

    # Find most recent log if not specified
    if not args.log:
        log_dir = Path("logs")
        if log_dir.exists():
            logs = sorted(log_dir.glob("daemon_*.log"), reverse=True)
            if logs:
                args.log = str(logs[0])
                print(f"Using most recent log: {args.log}")
            else:
                print("❌ No log files found in logs/")
                return
        else:
            print("❌ logs/ directory not found")
            return

    if not Path(args.log).exists():
        print(f"❌ Log file not found: {args.log}")
        return

    # Parse log
    data = parse_training_log(args.log)

    if not data['steps']:
        print("❌ No training data found in log file")
        return

    # Calculate stats
    stats = calculate_statistics(data)

    # Print stats
    print("\n" + "=" * 80)
    print("TRAINING STATISTICS")
    print("=" * 80)
    print(f"Total steps: {stats['total_steps']:,}")
    print(f"Start loss: {stats['start_loss']:.4f}")
    print(f"End loss: {stats['end_loss']:.4f}")
    print(f"Best loss: {stats['min_loss']:.4f}")
    print(f"Improvement: {stats['improvement']:.4f} ({stats['improvement_pct']:.1f}%)")
    print("=" * 80)

    # Generate report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate_html_report(data, stats, args.output)

    print(f"\n📊 View report: file://{Path(args.output).absolute()}")

if __name__ == "__main__":
    main()
