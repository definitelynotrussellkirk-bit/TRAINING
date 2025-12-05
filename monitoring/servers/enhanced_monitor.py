#!/usr/bin/env python3
"""
Enhanced Training Monitor - Real-time training dashboard
Updates every 10 steps with comprehensive metrics and visualizations
"""

from flask import Flask, render_template_string
import json
from pathlib import Path
from collections import deque
import time

app = Flask(__name__)

# Path to training status
STATUS_FILE = Path(__file__).parent / "current_model" / "status" / "training_detail.json"

# Store history for charts (last 200 data points)
loss_history = deque(maxlen=200)
lr_history = deque(maxlen=200)
speed_history = deque(maxlen=50)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Training Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            font-size: 36px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .header .subtitle {
            color: #666;
            font-size: 14px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }

        .metric-card:hover {
            transform: translateY(-5px);
        }

        .metric-label {
            font-size: 11px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 8px;
            font-weight: 600;
            letter-spacing: 1px;
        }

        .metric-value {
            font-size: 32px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-subtext {
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }

        .progress-section {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .progress-bar-container {
            background: #eee;
            height: 40px;
            border-radius: 20px;
            overflow: hidden;
            margin: 20px 0;
            position: relative;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 15px;
        }

        .progress-text {
            color: white;
            font-weight: bold;
            font-size: 14px;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .chart-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .chart-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }

        canvas {
            max-height: 300px;
        }

        .status-badge {
            display: inline-block;
            padding: 6px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }

        .status-training {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }

        .status-waiting {
            background: #ffd93d;
            color: #333;
        }

        .timestamp {
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            font-size: 12px;
            margin-top: 20px;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #38ef7d;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced Training Monitor</h1>
            <div class="subtitle">
                <span class="live-indicator"></span>
                Real-time training metrics • Updates every 10 steps
            </div>
        </div>

        <div id="content">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Status</div>
                    <span class="status-badge status-waiting">Initializing</span>
                </div>
            </div>
        </div>

        <div class="timestamp" id="timestamp"></div>
    </div>

    <script>
        let lossChart, lrChart;
        let lossData = [];
        let lrData = [];
        let steps = [];

        function initCharts() {
            const commonOptions = {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Step'
                        }
                    }
                }
            };

            const lossCtx = document.getElementById('lossChart');
            if (lossCtx) {
                lossChart = new Chart(lossCtx, {
                    type: 'line',
                    data: {
                        labels: steps,
                        datasets: [{
                            label: 'Training Loss',
                            data: lossData,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        ...commonOptions,
                        scales: {
                            ...commonOptions.scales,
                            y: {
                                title: {
                                    display: true,
                                    text: 'Loss'
                                }
                            }
                        }
                    }
                });
            }

            const lrCtx = document.getElementById('lrChart');
            if (lrCtx) {
                lrChart = new Chart(lrCtx, {
                    type: 'line',
                    data: {
                        labels: steps,
                        datasets: [{
                            label: 'Learning Rate',
                            data: lrData,
                            borderColor: '#764ba2',
                            backgroundColor: 'rgba(118, 75, 162, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        ...commonOptions,
                        scales: {
                            ...commonOptions.scales,
                            y: {
                                title: {
                                    display: true,
                                    text: 'Learning Rate'
                                }
                            }
                        }
                    }
                });
            }
        }

        function updateCharts(data) {
            if (!data || !data.step) return;

            steps.push(data.step);
            lossData.push(data.train_loss || 0);
            lrData.push(data.learning_rate || 0);

            // Keep only last 200 points
            if (steps.length > 200) {
                steps.shift();
                lossData.shift();
                lrData.shift();
            }

            if (lossChart) {
                lossChart.data.labels = steps;
                lossChart.data.datasets[0].data = lossData;
                lossChart.update('none');
            }

            if (lrChart) {
                lrChart.data.labels = steps;
                lrChart.data.datasets[0].data = lrData;
                lrChart.update('none');
            }
        }

        function formatDuration(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            if (h > 0) return `${h}h ${m}m`;
            if (m > 0) return `${m}m ${s}s`;
            return `${s}s`;
        }

        function updateDisplay(data) {
            if (!data || data.status === 'initialized') {
                document.getElementById('content').innerHTML = `
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Status</div>
                            <span class="status-badge status-waiting">Waiting for Training</span>
                        </div>
                    </div>
                `;
                return;
            }

            const step = data.step || data.current_step || 0;
            const totalSteps = data.total_steps || data.batch_total_steps || step || 1;
            const progress = (step / totalSteps * 100).toFixed(2);
            const loss = data.train_loss?.toFixed(4) || 'N/A';
            const lr = data.learning_rate?.toExponential(2) || 'N/A';
            const gpuMem = data.gpu_memory || 'N/A';
            const epoch = data.epoch?.toFixed(3) || '0';

            // Calculate speed and ETA
            const stepsRemaining = totalSteps - step;
            const avgSpeed = 3.5; // seconds per step (estimate)
            const eta = stepsRemaining * avgSpeed;

            let html = `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Status</div>
                        <span class="status-badge status-training">Training</span>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Training Loss</div>
                        <div class="metric-value">${loss}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Step</div>
                        <div class="metric-value">${step.toLocaleString()}</div>
                        <div class="metric-subtext">of ${totalSteps.toLocaleString()}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Epoch</div>
                        <div class="metric-value">${epoch}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Learning Rate</div>
                        <div class="metric-value" style="font-size: 20px;">${lr}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">GPU Memory</div>
                        <div class="metric-value" style="font-size: 24px;">${gpuMem}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">ETA</div>
                        <div class="metric-value" style="font-size: 22px;">${formatDuration(eta)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Speed</div>
                        <div class="metric-value" style="font-size: 22px;">${avgSpeed.toFixed(1)}s</div>
                        <div class="metric-subtext">per step</div>
                    </div>
                </div>

                <div class="progress-section">
                    <div class="chart-title">Overall Progress</div>
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width: ${progress}%">
                            <span class="progress-text">${progress}%</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; color: #666; font-size: 13px;">
                        <span>Step ${step.toLocaleString()} / ${totalSteps.toLocaleString()}</span>
                        <span>~${formatDuration(eta)} remaining</span>
                    </div>
                </div>

                <div class="charts-grid">
                    <div class="chart-card">
                        <div class="chart-title">Training Loss Over Time</div>
                        <canvas id="lossChart"></canvas>
                    </div>
                    <div class="chart-card">
                        <div class="chart-title">Learning Rate Schedule</div>
                        <canvas id="lrChart"></canvas>
                    </div>
                </div>
            `;

            document.getElementById('content').innerHTML = html;

            // Initialize charts after DOM update
            setTimeout(() => {
                if (!lossChart) initCharts();
                updateCharts(data);
            }, 100);
        }

        function fetchData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateDisplay(data);
                    document.getElementById('timestamp').textContent =
                        `Last updated: ${new Date().toLocaleTimeString()} • Auto-refreshes every 2 seconds`;
                })
                .catch(error => console.error('Error:', error));
        }

        // Initial fetch
        fetchData();

        // Auto-refresh every 2 seconds
        setInterval(fetchData, 2000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'r') as f:
                data = json.load(f)
                return data
        else:
            return {
                'status': 'initialized',
                'message': 'Waiting for training to start'
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 ENHANCED TRAINING MONITOR")
    print("=" * 80)
    print()
    print("🌐 Server starting at: http://localhost:8082")
    print()
    print("✨ Features:")
    print("  • Real-time metrics dashboard")
    print("  • Live loss and learning rate charts")
    print("  • Progress bars and ETA calculations")
    print("  • GPU memory monitoring")
    print("  • Updates every 2 seconds")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()

    app.run(host='0.0.0.0', port=8082, debug=False, threaded=True)
