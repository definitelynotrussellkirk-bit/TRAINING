#!/usr/bin/env python3
"""
Detailed Training Monitor - Full Visibility Dashboard

Shows during training:
- Current loss / eval loss
- Complete prompt (system + user + assistant-so-far)
- Golden assistant response (expected output)
- Current model guess (predicted tokens)
- Token-by-token comparison with color coding

Runs at http://localhost:8081
"""

from flask import Flask, render_template_string, jsonify
import json
from pathlib import Path
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Path to detailed training data
DETAIL_FILE = Path(__file__).parent / "current_model" / "status" / "training_detail.json"
LIVE_INFERENCE_FILE = Path(__file__).parent / "current_model" / "status" / "live_inference.json"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Detailed Training Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Monaco', 'Consolas', monospace;
            background: #0a0e1a;
            color: #e0e0e0;
            padding: 20px;
            font-size: 14px;
            line-height: 1.6;
        }

        .container {
            max-width: 1800px;
            margin: 0 auto;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }

        .header h1 {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .header .subtitle {
            font-size: 16px;
            opacity: 0.9;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: #1a1f35;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }

        .metric-label {
            font-size: 12px;
            text-transform: uppercase;
            opacity: 0.6;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }

        .metric-value.loss {
            color: #f093fb;
        }

        .metric-value.eval {
            color: #4facfe;
        }

        .section {
            background: #1a1f35;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #2a3150;
        }

        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #667eea;
            border-bottom: 2px solid #2a3150;
            padding-bottom: 10px;
        }

        .message-block {
            background: #0f1420;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }

        .message-block.system {
            border-left-color: #f093fb;
        }

        .message-block.user {
            border-left-color: #4facfe;
        }

        .message-block.assistant {
            border-left-color: #43e97b;
        }

        .message-role {
            font-size: 11px;
            text-transform: uppercase;
            font-weight: bold;
            margin-bottom: 8px;
            opacity: 0.7;
        }

        .message-content {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 13px;
            line-height: 1.8;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .golden {
            background: #0f1420;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #43e97b;
        }

        .predicted {
            background: #0f1420;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #f093fb;
        }

        .comparison-title {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 12px;
            text-transform: uppercase;
        }

        .golden .comparison-title {
            color: #43e97b;
        }

        .predicted .comparison-title {
            color: #f093fb;
        }

        .token {
            display: inline-block;
            padding: 2px 6px;
            margin: 2px;
            border-radius: 4px;
            font-size: 12px;
        }

        .token.match {
            background: #43e97b33;
            color: #43e97b;
        }

        .token.mismatch {
            background: #ff6b6b33;
            color: #ff6b6b;
        }

        .token.missing {
            background: #ffd93d33;
            color: #ffd93d;
        }

        .stats {
            margin-top: 15px;
            padding: 12px;
            background: #0a0e1a;
            border-radius: 6px;
            font-size: 12px;
        }

        .stats-item {
            display: inline-block;
            margin-right: 20px;
        }

        .stats-label {
            opacity: 0.6;
        }

        .stats-value {
            font-weight: bold;
            color: #667eea;
        }

        .no-data {
            text-align: center;
            padding: 60px 20px;
            opacity: 0.5;
        }

        .timestamp {
            text-align: center;
            opacity: 0.4;
            font-size: 11px;
            margin-top: 20px;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        .loading {
            animation: pulse 2s infinite;
        }

        @media (max-width: 768px) {
            .comparison-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Detailed Training Monitor</h1>
            <div class="subtitle">Real-time training visibility with token-by-token comparison</div>
        </div>

        <div id="content">
            <div class="no-data loading">
                <h2>⏳ Waiting for training data...</h2>
                <p>Start training to see detailed metrics</p>
            </div>
        </div>

        <div id="live-inference-content"></div>

        <div class="timestamp" id="timestamp"></div>
    </div>

    <script>
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function formatContent(content) {
            if (typeof content === 'string') {
                return escapeHtml(content);
            } else if (typeof content === 'object') {
                return escapeHtml(JSON.stringify(content, null, 2));
            }
            return escapeHtml(String(content));
        }

        function tokenize(text) {
            // Simple tokenization - split on whitespace and punctuation
            return text.match(/\\S+/g) || [];
        }

        function compareTokens(golden, predicted) {
            const goldenTokens = tokenize(golden);
            const predictedTokens = tokenize(predicted);

            const maxLen = Math.max(goldenTokens.length, predictedTokens.length);
            let matches = 0;

            let goldenHtml = '';
            let predictedHtml = '';

            for (let i = 0; i < maxLen; i++) {
                const g = goldenTokens[i];
                const p = predictedTokens[i];

                if (g && p) {
                    if (g === p) {
                        goldenHtml += `<span class="token match">${escapeHtml(g)}</span> `;
                        predictedHtml += `<span class="token match">${escapeHtml(p)}</span> `;
                        matches++;
                    } else {
                        goldenHtml += `<span class="token mismatch">${escapeHtml(g)}</span> `;
                        predictedHtml += `<span class="token mismatch">${escapeHtml(p)}</span> `;
                    }
                } else if (g) {
                    goldenHtml += `<span class="token missing">${escapeHtml(g)}</span> `;
                } else if (p) {
                    predictedHtml += `<span class="token missing">${escapeHtml(p)}</span> `;
                }
            }

            const accuracy = goldenTokens.length > 0 ?
                (matches / goldenTokens.length * 100).toFixed(1) : 0;

            return {
                goldenHtml,
                predictedHtml,
                matches,
                total: goldenTokens.length,
                accuracy
            };
        }

        function updateDisplay(data) {
            if (!data || data.status === 'no_training') {
                return;
            }

            let html = '';

            // Metrics
            html += '<div class="metrics-grid">';
            html += `
                <div class="metric-card">
                    <div class="metric-label">Training Loss</div>
                    <div class="metric-value loss">${data.train_loss?.toFixed(4) || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Eval Loss</div>
                    <div class="metric-value eval">${data.eval_loss?.toFixed(4) || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Step</div>
                    <div class="metric-value">${data.step || 0}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Epoch</div>
                    <div class="metric-value">${data.epoch?.toFixed(2) || 0}</div>
                </div>
            `;
            html += '</div>';

            // Complete Prompt Section
            if (data.prompt) {
                html += '<div class="section">';
                html += '<div class="section-title">📝 Complete Prompt Context</div>';

                if (data.prompt.messages) {
                    for (const msg of data.prompt.messages) {
                        const roleClass = msg.role || 'unknown';
                        html += `
                            <div class="message-block ${roleClass}">
                                <div class="message-role">${msg.role || 'unknown'}</div>
                                <div class="message-content">${formatContent(msg.content)}</div>
                            </div>
                        `;
                    }
                }
                html += '</div>';
            }

            // Comparison Section
            if (data.golden && data.predicted) {
                const comparison = compareTokens(data.golden, data.predicted);

                html += '<div class="section">';
                html += '<div class="section-title">🎯 Golden vs Predicted Comparison</div>';
                html += '<div class="comparison-grid">';

                // Golden
                html += '<div class="golden">';
                html += '<div class="comparison-title">✓ Golden (Expected)</div>';
                html += `<div class="message-content">${comparison.goldenHtml}</div>`;
                html += '</div>';

                // Predicted
                html += '<div class="predicted">';
                html += '<div class="comparison-title">🤖 Predicted (Model Output)</div>';
                html += `<div class="message-content">${comparison.predictedHtml}</div>`;
                html += '</div>';

                html += '</div>';

                // Stats
                html += `
                    <div class="stats">
                        <span class="stats-item">
                            <span class="stats-label">Accuracy:</span>
                            <span class="stats-value">${comparison.accuracy}%</span>
                        </span>
                        <span class="stats-item">
                            <span class="stats-label">Matches:</span>
                            <span class="stats-value">${comparison.matches}/${comparison.total}</span>
                        </span>
                        <span class="stats-item">
                            <span class="stats-label">Token Diff:</span>
                            <span class="stats-value">${comparison.total - comparison.matches}</span>
                        </span>
                    </div>
                `;

                html += '</div>';
            }

            // Additional Info
            if (data.sample_idx !== undefined) {
                html += '<div class="section">';
                html += '<div class="section-title">ℹ️ Sample Information</div>';
                html += `
                    <div class="stats">
                        <span class="stats-item">
                            <span class="stats-label">Sample Index:</span>
                            <span class="stats-value">${data.sample_idx}</span>
                        </span>
                        <span class="stats-item">
                            <span class="stats-label">Learning Rate:</span>
                            <span class="stats-value">${data.learning_rate?.toExponential(2) || 'N/A'}</span>
                        </span>
                        <span class="stats-item">
                            <span class="stats-label">GPU Memory:</span>
                            <span class="stats-value">${data.gpu_memory || 'N/A'}</span>
                        </span>
                    </div>
                `;
                html += '</div>';
            }

            document.getElementById('content').innerHTML = html;

            // Update timestamp
            const now = new Date();
            document.getElementById('timestamp').textContent =
                `Last updated: ${now.toLocaleTimeString()} | Auto-refreshes every 2 seconds`;
        }

        function fetchData() {
            fetch('/api/detail')
                .then(response => response.json())
                .then(data => updateDisplay(data))
                .catch(error => console.error('Error fetching data:', error));
        }

        function updateLiveInference(data) {
            if (!data || data.status === 'no_inference') {
                return;
            }

            // Save state of which details are open
            const openDetails = new Set();
            document.querySelectorAll('#live-inference-content details[open]').forEach((detail, index) => {
                openDetails.add(index);
            });

            let html = '';
            let detailsIndex = 0;

            // Header section with metrics
            html += '<div class="section" style="margin-top: 30px;">';
            html += '<div class="section-title">🔍 Live Inference Results</div>';

            // Metrics row
            html += '<div class="metrics-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));">';
            html += `
                <div class="metric-card">
                    <div class="metric-label">Step</div>
                    <div class="metric-value">${data.step || 0} / ${data.total_steps || 0}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Progress</div>
                    <div class="metric-value">${data.progress_percent || 0}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value" style="color: ${data.accuracy >= 80 ? '#43e97b' : data.accuracy >= 50 ? '#ffd93d' : '#ff6b6b'};">
                        ${data.accuracy || 0}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Examples</div>
                    <div class="metric-value">${data.num_examples || 0}</div>
                </div>
            `;
            html += '</div>';

            // Accuracy trend if available
            if (data.accuracy_trend && data.accuracy_trend.length > 0) {
                html += '<div style="margin-top: 20px; padding: 15px; background: #0f1420; border-radius: 8px;">';
                html += '<div style="font-size: 14px; font-weight: bold; margin-bottom: 10px; color: #667eea;">Accuracy Trend</div>';
                for (const trend of data.accuracy_trend) {
                    const bars = '█'.repeat(Math.floor(trend.accuracy / 5));
                    html += `<div style="margin: 5px 0; font-size: 12px;">`;
                    html += `Step ${trend.step}: <span style="color: #43e97b;">${bars}</span> ${trend.accuracy.toFixed(1)}%`;
                    html += `</div>`;
                }
                html += '</div>';
            }

            // Display each example
            if (data.examples && data.examples.length > 0) {
                for (const example of data.examples) {
                    const statusIcon = example.match ? '✅' : '❌';
                    const statusColor = example.match ? '#43e97b' : '#ff6b6b';

                    html += `<div style="margin-top: 20px; padding: 20px; background: #0f1420; border-radius: 8px; border-left: 4px solid ${statusColor};">`;
                    html += `<div style="font-size: 16px; font-weight: bold; margin-bottom: 15px; color: ${statusColor};">`;
                    html += `${statusIcon} Example ${example.example_id + 1}`;
                    html += `</div>`;

                    // System prompt if present
                    if (example.system_prompt) {
                        html += '<div class="message-block system" style="margin-bottom: 10px;">';
                        html += '<div class="message-role">System Prompt</div>';
                        html += `<div class="message-content" style="max-height: 150px; overflow-y: auto;">${formatContent(example.system_prompt)}</div>`;
                        html += '</div>';
                    }

                    // User input
                    html += '<div class="message-block user" style="margin-bottom: 10px;">';
                    html += '<div class="message-role">User Input</div>';
                    html += `<div class="message-content" style="max-height: 200px; overflow-y: auto;">${formatContent(example.user_input)}</div>`;
                    html += '</div>';

                    // Formatted prompt (collapsed by default)
                    if (example.formatted_prompt) {
                        const wasOpen = openDetails.has(detailsIndex);
                        html += `<details style="margin-bottom: 10px;" ${wasOpen ? 'open' : ''}>`;
                        html += '<summary style="cursor: pointer; color: #667eea; font-size: 12px; margin-bottom: 5px;">Show Full Formatted Prompt</summary>';
                        html += '<div class="message-block" style="margin-top: 10px;">';
                        html += '<div class="message-role">Formatted Prompt (Tokenized)</div>';
                        html += `<div class="message-content" style="max-height: 300px; overflow-y: auto; font-size: 11px;">${formatContent(example.formatted_prompt)}</div>`;
                        html += '</div>';
                        html += '</details>';
                        detailsIndex++;
                    }

                    // Expected vs Predicted comparison
                    html += '<div class="comparison-grid" style="margin-top: 15px;">';

                    // Expected
                    html += '<div class="golden">';
                    html += '<div class="comparison-title">✓ Expected</div>';
                    html += `<div class="message-content" style="max-height: 200px; overflow-y: auto;">${formatContent(example.expected)}</div>`;
                    html += '</div>';

                    // Predicted
                    html += '<div class="predicted">';
                    html += '<div class="comparison-title">🤖 Model Output</div>';
                    html += `<div class="message-content" style="max-height: 200px; overflow-y: auto;">${formatContent(example.predicted)}</div>`;
                    html += '</div>';

                    html += '</div>';
                    html += '</div>';
                }
            }

            html += '</div>';

            document.getElementById('live-inference-content').innerHTML = html;
        }

        function fetchLiveInference() {
            fetch('/api/live_inference')
                .then(response => response.json())
                .then(data => updateLiveInference(data))
                .catch(error => console.error('Error fetching live inference:', error));
        }

        // Initial fetch
        fetchData();
        fetchLiveInference();

        // Auto-refresh every 2 seconds
        setInterval(fetchData, 2000);
        setInterval(fetchLiveInference, 2000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detail')
def api_detail():
    """API endpoint for detailed training data"""
    try:
        if DETAIL_FILE.exists():
            with open(DETAIL_FILE, 'r') as f:
                data = json.load(f)
                return jsonify(data)
        else:
            return jsonify({
                'status': 'no_training',
                'message': 'No training data available yet'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/live_inference')
def api_live_inference():
    """API endpoint for live inference data"""
    try:
        if LIVE_INFERENCE_FILE.exists():
            with open(LIVE_INFERENCE_FILE, 'r') as f:
                data = json.load(f)
                return jsonify(data)
        else:
            return jsonify({
                'status': 'no_inference',
                'message': 'No live inference data available yet'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/json')
def raw_json():
    """Raw JSON streaming endpoint - shows live JSON data"""
    try:
        if DETAIL_FILE.exists():
            with open(DETAIL_FILE, 'r') as f:
                data = json.load(f)
                return f"<html><head><title>Training JSON Stream</title><meta http-equiv='refresh' content='2'><style>body{{background:#0a0e1a;color:#e0e0e0;font-family:monospace;padding:20px;}}pre{{background:#1a1f35;padding:20px;border-radius:8px;overflow-x:auto;}}</style></head><body><h2>Training Detail JSON (Auto-refresh every 2s)</h2><pre>{json.dumps(data, indent=2)}</pre></body></html>"
        else:
            return f"<html><head><title>Training JSON Stream</title><meta http-equiv='refresh' content='2'><style>body{{background:#0a0e1a;color:#e0e0e0;font-family:monospace;padding:20px;}}</style></head><body><h2>⏳ Waiting for training data...</h2><p>File: {DETAIL_FILE}</p></body></html>"
    except Exception as e:
        return f"<html><body><h2>Error</h2><pre>{str(e)}</pre></body></html>"

def run_server():
    """Run Flask server"""
    print("=" * 80)
    print("🔬 DETAILED TRAINING MONITOR")
    print("=" * 80)
    print()
    print("Server starting at: http://localhost:8081")
    print()
    print("Features:")
    print("  ✓ Real-time loss metrics")
    print("  ✓ Complete prompt context (system + user + assistant)")
    print("  ✓ Golden vs predicted comparison")
    print("  ✓ Token-by-token analysis with color coding")
    print("  ✓ Accuracy metrics")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()

    app.run(host='0.0.0.0', port=8081, debug=False, threaded=True)

if __name__ == '__main__':
    run_server()
