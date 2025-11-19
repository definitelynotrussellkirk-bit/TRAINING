// ===== MAIN LIVE MONITOR LOGIC =====
// Handles data fetching, updates, and coordination between modules

// Prefer full-precision display for losses; avoid truncating with toFixed().
function formatFloat(value) {
    if (value === null || value === undefined) return '--';
    const num = Number(value);
    if (!Number.isFinite(num)) return '--';
    return num.toString();
}

class LiveMonitor {
    constructor() {
        this.updateInterval = 2000; // 2 seconds
        this.statusUrl = '/status/training_status.json';
        this.currentData = null;
        this.chartManager = null;
        this.metricsDisplay = null;
        this.dataBrowser = null;
    }

    async init() {
        console.log('🚀 Initializing Live Monitor...');

        // Initialize modules
        this.chartManager = new ChartManager();
        this.metricsDisplay = new MetricsDisplay();
        this.dataBrowser = new DataBrowser();

        await this.chartManager.init();

        // Start update loop
        this.startUpdating();
    }

    startUpdating() {
        this.update();
        setInterval(() => this.update(), this.updateInterval);
    }

    async update() {
        try {
            const response = await fetch(this.statusUrl);
            const data = await response.json();

            if (!data) return;

            this.currentData = data;

            // Update all components
            this.updateStatusBar(data);
            this.metricsDisplay.updatePromptDisplay(data);
            this.chartManager.updateCharts(data);
            this.updateRecentExamples(data);
            this.updateFileInfo(data);

        } catch (error) {
            console.error('Update failed:', error);
            this.updateHealthIndicator('error');
        }
    }

    updateStatusBar(data) {
        // Health indicator
        const status = data.status || 'idle';
        this.updateHealthIndicator(status);

        // Metrics
        document.getElementById('currentStep').textContent = (data.current_step || 0).toLocaleString();
        document.getElementById('currentLoss').textContent = formatFloat(data.loss);

        // Validation loss
        const valLoss = data.validation_loss;
        const valLossEl = document.getElementById('valLoss');
        if (valLoss !== null && valLoss !== undefined) {
            valLossEl.textContent = formatFloat(valLoss);
            valLossEl.style.color = 'var(--accent-blue)';
        } else {
            valLossEl.textContent = '--';
            valLossEl.style.color = 'var(--text-secondary)';
        }

        // Gap
        const gap = data.val_train_gap;
        const gapEl = document.getElementById('lossGap');
        if (gap !== null && gap !== undefined) {
            const gapText = formatFloat(gap);
            gapEl.textContent = (gap >= 0 ? '+' : '') + gapText;
            // Color code based on gap size
            if (Math.abs(gap) < 0.3) {
                gapEl.style.color = 'var(--accent-green)';
            } else if (Math.abs(gap) < 0.5) {
                gapEl.style.color = 'var(--accent-yellow)';
            } else {
                gapEl.style.color = 'var(--accent-red)';
            }
        } else {
            gapEl.textContent = '--';
            gapEl.style.color = 'var(--text-secondary)';
        }

        // Think tag percentage
        const thinkPct = data.think_tag_percent;
        const thinkEl = document.getElementById('thinkPct');
        if (thinkPct !== null && thinkPct !== undefined) {
            thinkEl.textContent = thinkPct.toFixed(1) + '%';
            // Color code: green = low, red = high
            if (thinkPct < 20) {
                thinkEl.style.color = 'var(--accent-green)';
            } else if (thinkPct < 60) {
                thinkEl.style.color = 'var(--accent-yellow)';
            } else {
                thinkEl.style.color = 'var(--accent-red)';
            }
        } else {
            thinkEl.textContent = '--';
            thinkEl.style.color = 'var(--text-secondary)';
        }

        // Accuracy
        const accuracy = data.accuracy_percent || 0;
        document.getElementById('accuracy').textContent = accuracy.toFixed(1) + '%';
    }

    updateHealthIndicator(status) {
        const dot = document.getElementById('healthDot');
        const label = document.getElementById('healthLabel');

        if (status === 'training') {
            dot.style.background = 'var(--accent-green)';
            label.textContent = 'TRAINING';
            label.style.color = 'var(--accent-green)';
        } else if (status === 'idle') {
            dot.style.background = 'var(--accent-yellow)';
            label.textContent = 'IDLE';
            label.style.color = 'var(--accent-yellow)';
        } else if (status === 'crashed' || status === 'error') {
            dot.style.background = 'var(--accent-red)';
            label.textContent = 'ERROR';
            label.style.color = 'var(--accent-red)';
        } else {
            dot.style.background = 'var(--text-secondary)';
            label.textContent = status.toUpperCase();
            label.style.color = 'var(--text-secondary)';
        }
    }

    updateRecentExamples(data) {
        const recentList = document.getElementById('recentList');
        const examples = data.recent_examples || [];

        if (examples.length === 0) {
            recentList.innerHTML = '<div style="text-align: center; padding: 20px; color: var(--text-secondary);">No recent examples</div>';
            return;
        }

        recentList.innerHTML = examples.reverse().map(ex => `
            <div class="recent-item ${ex.matches ? 'match' : 'no-match'}" onclick="monitor.showExampleDetail(${ex.step})">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: var(--accent-blue);">Step ${ex.step}</span>
                    <span style="color: ${ex.matches ? 'var(--accent-green)' : 'var(--accent-red)'};">
                        ${ex.matches ? '✅' : '❌'}
                    </span>
                </div>
                <div style="font-size: 0.85em; color: var(--text-secondary);">
                    Loss: ${formatFloat(ex.loss)}
                </div>
            </div>
        `).join('');
    }

    updateFileInfo(data) {
        const fileEl = document.getElementById('currentFile');
        const progressEl = document.getElementById('fileProgress');

        fileEl.textContent = data.current_file || '--';

        if (data.batch_step !== null && data.batch_total_steps !== null) {
            progressEl.textContent = `${data.batch_step} / ${data.batch_total_steps}`;
        } else {
            progressEl.textContent = '-- / --';
        }
    }

    showExampleDetail(step) {
        // TODO: Implement detailed view modal
        console.log('Show detail for step:', step);
    }
}

// ===== GLOBAL FUNCTIONS =====

function toggleSection(id) {
    const element = document.getElementById(id);
    if (element.style.display === 'none') {
        element.style.display = 'block';
    } else {
        element.style.display = 'none';
    }
}

function expandPrompt() {
    const modal = document.getElementById('expandModal');
    modal.classList.add('active');

    // Populate modal with current data
    if (monitor.currentData) {
        const content = document.getElementById('expandedContent');
        const data = monitor.currentData;

        content.innerHTML = `
            <div style="margin-bottom: 20px;">
                <h3 style="color: var(--accent-blue); margin-bottom: 10px;">System Prompt</h3>
                <div class="content-box">
                    <pre>${escapeHtml(data.current_prompt || 'N/A')}</pre>
                </div>
            </div>
            <div style="margin-bottom: 20px;">
                <h3 style="color: var(--accent-green); margin-bottom: 10px;">Golden Answer</h3>
                <div class="content-box">
                    <pre>${escapeHtml(data.golden_answer || 'N/A')}</pre>
                </div>
            </div>
            <div>
                <h3 style="color: var(--accent-blue); margin-bottom: 10px;">Model Output</h3>
                <div class="content-box">
                    <pre>${escapeHtml(data.model_answer || 'N/A')}</pre>
                </div>
            </div>
        `;

        document.getElementById('modalStep').textContent = data.current_step || '--';
    }
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

function openDataBrowser() {
    const modal = document.getElementById('dataBrowserModal');
    modal.classList.add('active');
    monitor.dataBrowser.loadExamples();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== INITIALIZATION =====

let monitor;

document.addEventListener('DOMContentLoaded', () => {
    monitor = new LiveMonitor();
    monitor.init();
});
