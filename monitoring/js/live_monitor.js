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
            this.updateAdvancedAnalytics(data);

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
        const modelLabel = data.model_name || '--';
        const modelEl = document.getElementById('currentModelName');
        if (modelEl) {
            modelEl.textContent = modelLabel;
            modelEl.title = modelLabel;
        }
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

        // Queue Size
        const queueSize = data.batch_queue_size;
        const queueEl = document.getElementById('queueSize');
        if (queueSize !== null && queueSize !== undefined) {
            queueEl.textContent = queueSize.toString();
            // Color code based on queue size
            if (queueSize === 0) {
                queueEl.style.color = 'var(--accent-red)';
            } else if (queueSize < 5) {
                queueEl.style.color = 'var(--accent-yellow)';
            } else {
                queueEl.style.color = 'var(--accent-green)';
            }
        } else {
            queueEl.textContent = '--';
            queueEl.style.color = 'var(--text-secondary)';
        }
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

        const ordered = [...examples].reverse();

        recentList.innerHTML = ordered.map(ex => `
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
        const queueCountEl = document.getElementById('queueCount');

        fileEl.textContent = data.current_file || '--';

        if (data.batch_step !== null && data.batch_total_steps !== null) {
            progressEl.textContent = `${data.batch_step} / ${data.batch_total_steps}`;
        } else {
            progressEl.textContent = '-- / --';
        }

        // Update queue count
        const queueSize = data.batch_queue_size;
        if (queueCountEl) {
            if (queueSize !== null && queueSize !== undefined) {
                queueCountEl.textContent = queueSize.toString() + ' files';
                // Color code
                if (queueSize === 0) {
                    queueCountEl.style.color = 'var(--accent-red)';
                } else if (queueSize < 5) {
                    queueCountEl.style.color = 'var(--accent-yellow)';
                } else {
                    queueCountEl.style.color = 'var(--accent-green)';
                }
            } else {
                queueCountEl.textContent = '--';
                queueCountEl.style.color = 'var(--text-secondary)';
            }
        }
    }

    updateAdvancedAnalytics(data) {
        if (!data) return;
        this.renderPenaltyStats(data.logit_penalty_stats);
        this.renderQueueVelocity(data.queue_velocity);
        this.renderLengthCoverage(data.length_bin_staleness);
        this.renderPatternLoss(data.pattern_loss_trend);
        this.renderPatternLayerLinks(data.pattern_layer_correlation);
        this.renderLayerStability(data.layer_stability_summary);
        this.renderVramScatter(data.throughput_vram_samples);
    }

    renderPenaltyStats(stats) {
        const container = document.getElementById('penaltyStats');
        if (!container) return;
        if (!stats || (!stats.totals && !stats.per_file)) {
            container.textContent = 'No penalties recorded yet.';
            return;
        }
        const totals = Object.entries(stats.totals || {}).sort((a, b) => (b[1] || 0) - (a[1] || 0));
        const perFileEntries = Object.entries(stats.per_file || {}).map(([file, labels]) => {
            const sum = Object.values(labels).reduce((acc, val) => acc + val, 0);
            return { file, sum, labels };
        }).sort((a, b) => b.sum - a.sum).slice(0, 3);

        let html = '';
        if (totals.length) {
            html += '<div><strong>Totals</strong></div>';
            totals.forEach(([label, hits]) => {
                html += `<div>${label.toUpperCase()}: ${hits.toLocaleString()} hits</div>`;
            });
        }
        if (perFileEntries.length) {
            html += '<div style="margin-top:8px;"><strong>Top Files</strong></div>';
            perFileEntries.forEach(entry => {
                const labelSummary = Object.entries(entry.labels)
                    .map(([label, count]) => `${label}:${count}`)
                    .join(', ');
                html += `<div>${entry.file}: <span class="muted">${labelSummary}</span></div>`;
            });
        }
        container.innerHTML = html;
    }

    renderQueueVelocity(queue) {
        const container = document.getElementById('queueVelocityStats');
        if (!container) return;
        if (!queue) {
            container.textContent = 'Waiting for throughput samples...';
            return;
        }
        const samplesSec = Number.isFinite(queue.samples_per_sec) ? queue.samples_per_sec : 0;
        const samplesHour = Number.isFinite(queue.samples_per_hour) ? queue.samples_per_hour : 0;
        container.innerHTML = `
            <div>${samplesSec.toFixed(1)} samples/sec</div>
            <div>${Math.round(samplesHour).toLocaleString()} samples/hour</div>
            <div class="muted small-text">Effective batch: ${queue.effective_batch}</div>
        `;
    }

    renderLengthCoverage(staleness) {
        const container = document.getElementById('lengthCoverageList');
        if (!container) return;
        if (!staleness || Object.keys(staleness).length === 0) {
            container.textContent = 'No coverage data yet.';
            return;
        }
        const sorted = Object.entries(staleness)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);
        const html = sorted.map(([bin, seconds]) => {
            return `<div>${bin}: <span class="muted">${this.formatDurationShort(seconds)}</span></div>`;
        }).join('');
        container.innerHTML = html;
    }

    renderPatternLoss(trend) {
        const container = document.getElementById('patternLossList');
        if (!container) return;
        if (!trend || Object.keys(trend).length === 0) {
            container.textContent = 'No losses recorded yet.';
            return;
        }
        const items = Object.entries(trend)
            .map(([pattern, info]) => ({
                pattern,
                avg: info.avg_loss,
                recent: info.recent_loss if info.recent_loss != null else info.avg_loss,
                samples: info.samples ?? 0
            }))
            .sort((a, b) => b.recent - a.recent)
            .slice(0, 4);
        const html = items.map(item => `
            <div>
                <strong>${item.pattern}</strong> · recent ${this.formatLossValue(item.recent)}
                <span class="muted">avg ${this.formatLossValue(item.avg)} (${item.samples})</span>
            </div>
        `).join('');
        container.innerHTML = html;
    }

    renderPatternLayerLinks(correlation) {
        const container = document.getElementById('patternLayerList');
        if (!container) return;
        if (!correlation || Object.keys(correlation).length === 0) {
            container.textContent = 'No layer correlation yet.';
            return;
        }
        const entries = Object.entries(correlation).map(([pattern, layers]) => {
            const ordered = Object.entries(layers || {}).sort(
                (a, b) =>
                    ((b[1]?.count) || 0) - ((a[1]?.count) || 0) ||
                    ((b[1]?.cumulative_delta) || 0) - ((a[1]?.cumulative_delta) || 0)
            );
            const topLayer = ordered[0];
            const stats = topLayer?.[1] || {};
            return {
                pattern,
                layer: topLayer ? topLayer[0] : 'n/a',
                count: stats.count || 0,
                delta: stats.cumulative_delta || 0
            };
        }).slice(0, 4);
        const html = entries.map(entry => `
            <div>
                <strong>${entry.pattern}</strong> → ${entry.layer}
                <span class="muted">(${entry.count} hits · ΣΔ ${entry.delta.toFixed(4)})</span>
            </div>
        `).join('');
        container.innerHTML = html;
    }

    renderLayerStability(summary) {
        const container = document.getElementById('layerStabilityList');
        if (!container) return;
        if (!summary || !summary.top || summary.top.length === 0) {
            container.textContent = 'Waiting for stability snapshots...';
            return;
        }
        const html = summary.top.map(entry => `
            <div>
                ${entry.name}
                <span class="muted">σ ${entry.stability.toExponential(2)}</span>
            </div>
        `).join('');
        container.innerHTML = html;
    }

    renderVramScatter(samples) {
        const canvas = document.getElementById('vramScatter');
        const statusEl = document.getElementById('vramScatterStatus');
        if (!canvas || !statusEl) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!samples || samples.length < 2) {
            statusEl.textContent = 'Waiting for throughput samples...';
            return;
        }

        const valid = samples.filter(s => Number.isFinite(s.tokens_per_sec) && Number.isFinite(s.vram_gb));
        if (valid.length < 2) {
            statusEl.textContent = 'Need more data to plot.';
            return;
        }

        const padding = 30;
        const width = canvas.width - padding * 2;
        const height = canvas.height - padding * 2;
        const maxTokens = Math.max(...valid.map(s => s.tokens_per_sec));
        const minTokens = Math.min(...valid.map(s => s.tokens_per_sec));
        const maxVram = Math.max(...valid.map(s => s.vram_gb));
        const minVram = Math.min(...valid.map(s => s.vram_gb));

        // Axes
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, padding + height);
        ctx.lineTo(padding + width, padding + height);
        ctx.stroke();

        let penaltyCount = 0;
        valid.forEach(point => {
            const xNorm = (point.tokens_per_sec - minTokens) / Math.max(maxTokens - minTokens, 1);
            const yNorm = (point.vram_gb - minVram) / Math.max(maxVram - minVram, 1);
            const x = padding + xNorm * width;
            const y = padding + height - yNorm * height;
            const hasPenalty = point.penalty && Object.values(point.penalty).some(v => v > 0);
            if (hasPenalty) penaltyCount += 1;
            ctx.fillStyle = hasPenalty ? '#ff8844' : '#00d9ff';
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
        });

        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '12px monospace';
        ctx.fillText(`${minTokens.toFixed(0)} tok/s`, padding, padding + height + 15);
        ctx.fillText(`${maxTokens.toFixed(0)} tok/s`, padding + width - 80, padding + height + 15);

        ctx.fillStyle = '#00d9ff';
        ctx.fillRect(padding + 10, padding + 10, 10, 10);
        ctx.fillStyle = '#fff';
        ctx.fillText('Normal', padding + 25, padding + 18);
        ctx.fillStyle = '#ff8844';
        ctx.fillRect(padding + 90, padding + 10, 10, 10);
        ctx.fillStyle = '#fff';
        ctx.fillText('Penalty', padding + 105, padding + 18);

        statusEl.textContent = `Samples: ${valid.length} · Penalty hits: ${penaltyCount} · tokens/sec ${minTokens.toFixed(0)}–${maxTokens.toFixed(0)} · VRAM ${minVram.toFixed(1)}–${maxVram.toFixed(1)} GB`;
    }

    formatDurationShort(seconds) {
        if (!Number.isFinite(seconds)) return '--';
        if (seconds < 60) return `${Math.round(seconds)}s ago`;
        if (seconds < 3600) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            return `${mins}m ${secs}s ago`;
        }
        const hours = Math.floor(seconds / 3600);
        const mins = Math.round((seconds % 3600) / 60);
        return `${hours}h ${mins}m ago`;
    }

    formatLossValue(value) {
        if (!Number.isFinite(value)) return '--';
        return value.toFixed(3);
    }

    showExampleDetail(step) {
        // Find the example in recent_examples
        const examples = this.currentData?.recent_examples || [];
        const example = examples.find(ex => ex.step === step);

        if (!example) {
            alert(`No data found for step ${step}`);
            return;
        }

        // Show available data (full prompt/answer not stored in recent_examples)
        const modal = document.getElementById('expandModal');
        const content = document.getElementById('expandedContent');

        content.innerHTML = `
            <div style="margin-bottom: 20px;">
                <h3 style="color: var(--accent-blue); margin-bottom: 10px;">Step ${step} Summary</h3>
                <div class="content-box">
                    <p><strong>Result:</strong> ${example.matches ? '✅ Correct' : '❌ Incorrect'}</p>
                    <p><strong>Loss:</strong> ${example.loss?.toFixed(4) || 'N/A'}</p>
                </div>
            </div>
            <div style="color: var(--text-secondary); font-style: italic;">
                Note: Full prompt/answer data is only available for the current step.
            </div>
        `;

        document.getElementById('modalStep').textContent = step;
        modal.classList.add('active');
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
