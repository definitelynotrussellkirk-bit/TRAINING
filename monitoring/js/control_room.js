/**
 * Control Room V2 - Client-Side Logic
 *
 * Handles:
 * - Multi-rate polling (1-2s, 2-5s, 10-30s, 5-10s)
 * - Component rendering and updates
 * - Charts (Chart.js)
 * - Preview history table with pagination
 * - Modal interactions
 * - Tab switching
 */

// ===== STATE =====
const state = {
    liveData: null,
    previewData: null,
    evalsData: null,
    systemData: null,

    // History pagination
    historyPage: 0,
    historyFilters: {
        regime: '',
        source: '',
        exactMatch: ''
    },

    // Charts
    lossChart: null,
    emTrendChart: null,
    throughputChart: null,

    // Loss history for sparkline
    lossHistory: [],
};

// ===== POLLING =====

/**
 * Poll /api/status/live every 2 seconds
 */
async function pollLiveStatus() {
    try {
        const response = await fetch('/api/status/live');
        const data = await response.json();
        state.liveData = data;
        renderLiveStatus(data);
        updateIndicator('trainingUpdate');
    } catch (error) {
        console.error('Failed to poll live status:', error);
    }
}

/**
 * Poll /api/status/preview every 3 seconds
 */
async function pollPreviewStatus() {
    try {
        const response = await fetch('/api/status/preview');
        const data = await response.json();
        state.previewData = data;
        renderPreviewStatus(data);
        updateIndicator('previewUpdate');
    } catch (error) {
        console.error('Failed to poll preview status:', error);
    }
}

/**
 * Poll /api/status/evals every 15 seconds
 */
async function pollEvalsStatus() {
    try {
        const response = await fetch('/api/status/evals');
        const data = await response.json();
        state.evalsData = data;
        renderEvalsStatus(data);
    } catch (error) {
        console.error('Failed to poll evals status:', error);
    }
}

/**
 * Poll /api/status/system every 7 seconds
 */
async function pollSystemStatus() {
    try {
        const response = await fetch('/api/status/system');
        const data = await response.json();
        state.systemData = data;
        renderSystemStatus(data);
    } catch (error) {
        console.error('Failed to poll system status:', error);
    }
}

/**
 * Start all polling intervals
 */
function startPolling() {
    // Initial fetch
    pollLiveStatus();
    pollPreviewStatus();
    pollEvalsStatus();
    pollSystemStatus();

    // Set up intervals
    setInterval(pollLiveStatus, 2000);     // 2s
    setInterval(pollPreviewStatus, 3000);  // 3s
    setInterval(pollEvalsStatus, 15000);   // 15s
    setInterval(pollSystemStatus, 7000);   // 7s
}

// ===== RENDERING FUNCTIONS =====

/**
 * Render live training status
 */
function renderLiveStatus(data) {
    try {
        // Status badge
        const statusBadge = document.getElementById('statusBadge');
        statusBadge.textContent = data.status.toUpperCase();
        statusBadge.className = `status-badge status-${data.status}`;

        // Model info
        document.getElementById('modelName').textContent = data.current_model_name || '-';
        document.getElementById('checkpointId').textContent = data.current_checkpoint_id || '-';

        // Progress bar
        const progress = data.total_steps > 0 ? (data.current_step / data.total_steps) * 100 : 0;
        document.getElementById('progressFill').style.width = `${progress}%`;
        document.getElementById('progressText').textContent =
            `Step ${data.current_step.toLocaleString()} / ${data.total_steps.toLocaleString()} (${progress.toFixed(1)}%)`;
        document.getElementById('epochText').textContent = `Epoch ${data.epoch.toFixed(1)}`;

        // Health indicators
        updateHealthIndicator('healthLoss', data.loss_trend);
        updateHealthIndicator('healthThroughput', data.throughput_trend);
        updateHealthIndicator('healthGPU', data.gpu_4090.temp_c > 80 ? 'critical' : 'good');
    } catch (error) {
        console.error('Error in status/progress section:', error);
    }

    try {

    // Training progress card
    document.getElementById('stepValue').textContent =
        `${data.current_step.toLocaleString()} / ${data.total_steps.toLocaleString()}`;
    document.getElementById('epochValue').textContent = data.epoch.toFixed(1);

    // Loss
    document.getElementById('lossValue').textContent = data.loss.toFixed(4);
    document.getElementById('streamingCE').textContent =
        data.streaming_ce !== null ? data.streaming_ce.toFixed(4) : '-';
    document.getElementById('lossVariance').textContent = data.loss_variance.toFixed(4);

    const lossTrend = document.getElementById('lossTrend');
    lossTrend.textContent = getTrendArrow(data.loss_trend);
    lossTrend.className = `trend-indicator trend-${data.loss_trend}`;

    // Val/train gap
    const valTrainGap = document.getElementById('valTrainGap');
    if (data.val_loss !== null && data.train_loss !== null) {
        const gap = data.val_loss - data.train_loss;
        valTrainGap.textContent = gap.toFixed(3);
        valTrainGap.className = gap > 0.5 ? 'metric-value gap-overfitting' :
                              gap > 0.3 ? 'metric-value gap-warning' :
                              'metric-value gap-good';
    } else {
        valTrainGap.textContent = '-';
        valTrainGap.className = 'metric-value';
    }

    // Throughput
    document.getElementById('tokensPerSec').textContent = data.tokens_per_sec.toLocaleString();
    document.getElementById('tokensPerSecAvg').textContent = data.tokens_per_sec_avg.toLocaleString();
    document.getElementById('tokensPerSecBaseline').textContent = data.tokens_per_sec_baseline.toLocaleString();
    } catch (error) {
        console.error('Error in loss/throughput section:', error);
    }

    try {
    // Queue & current file
    document.getElementById('currentFile').textContent = data.current_file || '-';
    document.getElementById('batchProgress').textContent = `${data.batch_step} / ${data.batch_total_steps}`;
    const batchProgress = data.batch_total_steps > 0 ? (data.batch_step / data.batch_total_steps) * 100 : 0;
    document.getElementById('batchProgressFill').style.width = `${batchProgress}%`;

    document.getElementById('queueHigh').textContent = `${data.batch_queue_size} files`;
    document.getElementById('queueNormal').textContent = '-';
    document.getElementById('queueLow').textContent = '-';

    document.getElementById('etaFile').textContent = data.eta_current_file || '-';
    document.getElementById('etaAll').textContent = data.eta_overall || '-';
    } catch (error) {
        console.error('Error in queue section:', error);
    }

    try {
    // Hardware - 4090
    const gpu4090 = data.gpu_4090;
    const temp4090 = document.getElementById('gpu4090Temp');
    temp4090.textContent = `${gpu4090.temp_c}°C`;
    temp4090.className = gpu4090.temp_c > 80 ? 'temp-value temp-hot' :
                        gpu4090.temp_c > 70 ? 'temp-value temp-warm' :
                        'temp-value temp-normal';

    document.getElementById('gpu4090VRAM').textContent =
        `${gpu4090.vram_used_gb.toFixed(1)} / ${gpu4090.vram_total_gb.toFixed(0)} GB`;
    document.getElementById('gpu4090VRAMFill').style.width = `${gpu4090.vram_pct}%`;
    document.getElementById('gpu4090Util').textContent = `${gpu4090.util_pct}%`;
    document.getElementById('gpu4090Power').textContent =
        `${gpu4090.power_w} / ${gpu4090.power_limit_w} W`;

    // Hardware - 3090
    const gpu3090 = data.gpu_3090;
    const status3090 = document.getElementById('gpu3090Status');
    status3090.className = gpu3090.online ? 'status-dot online' : 'status-dot offline';

    const temp3090 = document.getElementById('gpu3090Temp');
    temp3090.textContent = `${gpu3090.temp_c}°C`;
    temp3090.className = gpu3090.temp_c > 70 ? 'temp-value temp-hot' :
                        gpu3090.temp_c > 60 ? 'temp-value temp-warm' :
                        'temp-value temp-normal';

    document.getElementById('gpu3090VRAM').textContent =
        `${gpu3090.vram_used_gb.toFixed(1)} / ${gpu3090.vram_total_gb.toFixed(0)} GB`;
    const vram3090Pct = (gpu3090.vram_used_gb / gpu3090.vram_total_gb) * 100;
    document.getElementById('gpu3090VRAMFill').style.width = `${vram3090Pct}%`;
    document.getElementById('gpu3090Profile').textContent = gpu3090.power_profile || '-';

    // System resources
    const ram = data.ram;
    document.getElementById('ramUsage').textContent =
        `${ram.used_gb.toFixed(1)} / ${ram.total_gb.toFixed(0)} GB`;

    // Disk (if available in system data)
    if (state.systemData) {
        const disk = state.systemData.system_4090;
        document.getElementById('diskUsage').textContent =
            `${disk.disk_used_gb.toFixed(1)} / ${disk.disk_total_gb.toFixed(1)} TB`;
    }

    // Update loss chart
    updateLossChart(data.loss);
    } catch (error) {
        console.error('Error in hardware/chart section:', error);
    }
}

/**
 * Render preview status
 */
function renderPreviewStatus(data) {
    // Latest preview
    if (data.latest_preview) {
        renderLatestPreview(data.latest_preview);
    } else {
        document.getElementById('previewContent').innerHTML =
            '<div class="preview-empty">No preview data available</div>';
    }

    // Preview stats
    document.getElementById('emLast20').textContent = formatPercent(data.preview_em_last_20);
    document.getElementById('emLast50').textContent = formatPercent(data.preview_em_last_50);
    document.getElementById('emLast100').textContent = formatPercent(data.preview_em_last_100);

    document.getElementById('emLast20Fill').style.width = `${data.preview_em_last_20 * 100}%`;
    document.getElementById('emLast50Fill').style.width = `${data.preview_em_last_50 * 100}%`;
    document.getElementById('emLast100Fill').style.width = `${data.preview_em_last_100 * 100}%`;

    // Regime stats
    renderRegimeStats(data.regime_stats);

    // Source stats
    renderSourceStats(data.domain_stats);

    // EM trend chart
    if (data.em_trend && data.em_trend.length > 0) {
        updateEmTrendChart(data.em_trend);
    }
}

/**
 * Render latest preview
 */
function renderLatestPreview(preview) {
    // Get first result from results array
    if (!preview.results || preview.results.length === 0) {
        document.getElementById('previewContent').innerHTML =
            '<div class="preview-empty">No preview results available</div>';
        return;
    }

    const result = preview.results[0];
    const em_rate = preview.metrics.exact_match_rate;

    const html = `
        <div class="preview-meta">
            <div class="preview-tag">Step ${preview.training_step}</div>
            <div class="preview-tag">${preview.checkpoint_id || 'Training'}</div>
            <div class="preview-tag">${result.puzzle_id}</div>
            <div class="preview-tag regime">${result.difficulty}</div>
            <div class="preview-tag">EM: ${(em_rate * 100).toFixed(0)}%</div>
        </div>

        <div class="preview-outcome">
            <div class="outcome-badge ${result.exact_match ? 'outcome-match' : 'outcome-no-match'}">
                ${result.exact_match ? '✓ Exact Match' : '✗ No Match'}
            </div>
        </div>

        <div class="preview-section">
            <div class="preview-section-header" onclick="togglePreviewSection(this)">
                <div class="preview-section-title">Prompt</div>
                <div>▼</div>
            </div>
            <div class="preview-section-content">${escapeHtml(result.prompt)}</div>
        </div>

        <div class="preview-section">
            <div class="preview-section-header" onclick="togglePreviewSection(this)">
                <div class="preview-section-title">Expected Answer</div>
                <div>▼</div>
            </div>
            <div class="preview-section-content">${escapeHtml(result.expected)}</div>
        </div>

        <div class="preview-section">
            <div class="preview-section-header" onclick="togglePreviewSection(this)">
                <div class="preview-section-title">Model Answer</div>
                <div>▼</div>
            </div>
            <div class="preview-section-content ${result.exact_match ? '' : 'outcome-no-match'}">${escapeHtml(result.generated)}</div>
        </div>

        <div class="preview-metrics">
            <div class="preview-metric">
                <div class="preview-metric-label">Samples Tested</div>
                <div class="preview-metric-value">${preview.metrics.samples_tested}</div>
            </div>
            <div class="preview-metric">
                <div class="preview-metric-label">EM Rate</div>
                <div class="preview-metric-value">${(em_rate * 100).toFixed(1)}%</div>
            </div>
            <div class="preview-metric">
                <div class="preview-metric-label">Avg Time</div>
                <div class="preview-metric-value">${preview.metrics.avg_inference_time_ms.toFixed(0)}ms</div>
            </div>
        </div>
    `;

    document.getElementById('previewContent').innerHTML = html;
}

/**
 * Render regime stats
 */
function renderRegimeStats(regimeStats) {
    const container = document.getElementById('regimeStats');

    if (!regimeStats || Object.keys(regimeStats).length === 0) {
        container.innerHTML = '<div class="empty-state">No data</div>';
        return;
    }

    const html = Object.entries(regimeStats).map(([regime, stats]) => `
        <div class="regime-item">
            <div class="regime-name">${regime}</div>
            <div class="regime-stats-value">
                ${formatPercent(stats.em_rate)} (${stats.total} samples)
            </div>
        </div>
    `).join('');

    container.innerHTML = html;
}

/**
 * Render source stats
 */
function renderSourceStats(sourceStats) {
    const container = document.getElementById('sourceStats');

    if (!sourceStats || Object.keys(sourceStats).length === 0) {
        container.innerHTML = '<div class="empty-state">No data</div>';
        return;
    }

    const html = Object.entries(sourceStats).map(([source, stats]) => `
        <div class="source-item">
            <div class="source-name">${source}</div>
            <div class="source-stats-value">
                ${formatPercent(stats.em_rate)} (${stats.total} samples)
            </div>
        </div>
    `).join('');

    container.innerHTML = html;
}

/**
 * Render evals status
 */
function renderEvalsStatus(data) {
    // Fixed eval
    document.getElementById('fixedEvalEM').textContent =
        data.fixed_eval_em !== null ? formatPercent(data.fixed_eval_em) : '-';
    document.getElementById('fixedEvalCE').textContent =
        data.fixed_eval_ce !== null ? data.fixed_eval_ce.toFixed(3) : '-';
    document.getElementById('fixedEvalECE').textContent =
        data.fixed_eval_ece !== null ? data.fixed_eval_ece.toFixed(3) : '-';

    // Snapshots
    renderSnapshots(data.snapshots);
}

/**
 * Render snapshots list
 */
function renderSnapshots(snapshots) {
    const container = document.getElementById('snapshotsList');

    if (!snapshots || snapshots.length === 0) {
        container.innerHTML = '<div class="empty-state">No snapshots</div>';
        return;
    }

    const html = snapshots.map(snapshot => `
        <div class="snapshot-item">
            <div class="snapshot-date">${snapshot.date}</div>
            <div class="snapshot-metrics">
                EM: ${formatPercent(snapshot.fixed_eval_em || 0)}
                ${snapshot.tags && snapshot.tags.includes('best_so_far') ?
                    '<span class="snapshot-tag">BEST</span>' : ''}
            </div>
        </div>
    `).join('');

    container.innerHTML = html;
}

/**
 * Render system status (if needed for queues)
 */
function renderSystemStatus(data) {
    // Could update queue counts here if we have them
}

// ===== PREVIEW HISTORY TABLE =====

/**
 * Load preview history
 */
async function loadPreviewHistory() {
    try {
        const params = new URLSearchParams({
            limit: 100,
            offset: state.historyPage * 100,
            ...state.historyFilters
        });

        // Remove empty filters
        for (const [key, value] of [...params.entries()]) {
            if (!value) params.delete(key);
        }

        const response = await fetch(`/api/status/preview?${params}`);
        const data = await response.json();

        renderPreviewHistoryTable(data.previews);
        updatePagination(data.count);
    } catch (error) {
        console.error('Failed to load preview history:', error);
    }
}

/**
 * Render preview history table
 */
function renderPreviewHistoryTable(previews) {
    const tbody = document.getElementById('historyTableBody');

    if (!previews || previews.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No preview history available</td></tr>';
        return;
    }

    const html = previews.map(preview => {
        const time = new Date(preview.ts).toLocaleTimeString();
        const emBadge = preview.exact_match ?
            '<span class="em-badge em-yes">✓</span>' :
            '<span class="em-badge em-no">✗</span>';

        return `
            <tr onclick='showPreviewDetail(${JSON.stringify(preview)})'>
                <td>${time}</td>
                <td>${preview.step}</td>
                <td>${preview.example_id}</td>
                <td>${preview.regime}</td>
                <td>${preview.source_pool}</td>
                <td>${emBadge}</td>
                <td>${preview.ce !== null ? preview.ce.toFixed(3) : '-'}</td>
                <td>${preview.failure_mode || '-'}</td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = html;
}

/**
 * Update pagination controls
 */
function updatePagination(count) {
    const totalPages = Math.ceil(count / 100);
    const currentPage = state.historyPage + 1;

    document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages}`;
    document.getElementById('prevPage').disabled = state.historyPage === 0;
    document.getElementById('nextPage').disabled = currentPage >= totalPages;
}

/**
 * Show preview detail modal
 */
function showPreviewDetail(preview) {
    const modal = document.getElementById('previewModal');
    const body = document.getElementById('modalBody');

    body.innerHTML = `
        <div class="preview-meta">
            <div class="preview-tag">Step ${preview.step}</div>
            <div class="preview-tag">${preview.checkpoint_id}</div>
            <div class="preview-tag">${preview.example_id}</div>
            <div class="preview-tag regime">${preview.regime}</div>
            <div class="preview-tag source">${preview.source_pool}</div>
        </div>

        <div class="preview-section">
            <div class="preview-section-title">Prompt</div>
            <div class="preview-section-content expanded">${escapeHtml(preview.prompt)}</div>
        </div>

        <div class="preview-section">
            <div class="preview-section-title">Golden Answer</div>
            <div class="preview-section-content expanded">${escapeHtml(preview.golden)}</div>
        </div>

        <div class="preview-section">
            <div class="preview-section-title">Model Answer</div>
            <div class="preview-section-content expanded">${escapeHtml(preview.model_answer)}</div>
        </div>

        <div class="preview-metrics">
            <div class="preview-metric">
                <div class="preview-metric-label">Match</div>
                <div class="preview-metric-value">${preview.exact_match ? '✓ Yes' : '✗ No'}</div>
            </div>
            <div class="preview-metric">
                <div class="preview-metric-label">CE</div>
                <div class="preview-metric-value">${preview.ce !== null ? preview.ce.toFixed(3) : '-'}</div>
            </div>
            <div class="preview-metric">
                <div class="preview-metric-label">Tokens</div>
                <div class="preview-metric-value">${preview.prompt_tokens} → ${preview.model_tokens}</div>
            </div>
        </div>
    `;

    modal.classList.add('active');
}

// ===== CHARTS =====

/**
 * Initialize charts
 */
function initCharts() {
    // Loss sparkline
    const lossCtx = document.getElementById('lossSparkline').getContext('2d');
    state.lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: true, grid: { color: '#30363d' } }
            }
        }
    });

    // EM trend chart
    const emTrendCtx = document.getElementById('emTrendChart').getContext('2d');
    state.emTrendChart = new Chart(emTrendCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: '#3fb950',
                backgroundColor: 'rgba(63, 185, 80, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 2,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { min: 0, max: 1, display: true, grid: { color: '#30363d' } }
            }
        }
    });

    // Throughput chart
    const throughputCtx = document.getElementById('throughputChart').getContext('2d');
    state.throughputChart = new Chart(throughputCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Tokens/sec vs VRAM',
                data: [],
                backgroundColor: '#58a6ff',
                borderColor: '#58a6ff',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: '#e6edf3' } }
            },
            scales: {
                x: {
                    title: { display: true, text: 'VRAM (GB)', color: '#e6edf3' },
                    grid: { color: '#30363d' },
                    ticks: { color: '#8b949e' }
                },
                y: {
                    title: { display: true, text: 'Tokens/sec', color: '#e6edf3' },
                    grid: { color: '#30363d' },
                    ticks: { color: '#8b949e' }
                }
            }
        }
    });
}

/**
 * Update loss chart
 */
function updateLossChart(loss) {
    state.lossHistory.push(loss);
    if (state.lossHistory.length > 50) {
        state.lossHistory.shift();
    }

    state.lossChart.data.labels = state.lossHistory.map((_, i) => i);
    state.lossChart.data.datasets[0].data = state.lossHistory;
    state.lossChart.update('none');
}

/**
 * Update EM trend chart
 */
function updateEmTrendChart(emTrend) {
    state.emTrendChart.data.labels = emTrend.map((_, i) => i);
    state.emTrendChart.data.datasets[0].data = emTrend;
    state.emTrendChart.update('none');
}

/**
 * Load and update throughput chart
 */
async function loadThroughputData() {
    try {
        const response = await fetch('/api/throughput/samples?limit=200');
        const data = await response.json();

        const points = data.samples.map(s => ({
            x: s.vram_used_gb,
            y: s.tokens_per_sec
        }));

        state.throughputChart.data.datasets[0].data = points;
        state.throughputChart.update();
    } catch (error) {
        console.error('Failed to load throughput data:', error);
    }
}

// ===== UI INTERACTIONS =====

/**
 * Toggle preview section expand/collapse
 */
function togglePreviewSection(header) {
    const content = header.nextElementSibling;
    const arrow = header.querySelector('div:last-child');

    if (content.classList.contains('expanded')) {
        content.classList.remove('expanded');
        arrow.textContent = '▼';
    } else {
        content.classList.add('expanded');
        arrow.textContent = '▲';
    }
}

/**
 * Update health indicator
 */
function updateHealthIndicator(id, status) {
    const element = document.getElementById(id);
    element.className = 'health-indicator';

    if (status === 'improving' || status === 'normal' || status === 'good') {
        element.classList.add('health-good');
    } else if (status === 'stable' || status === 'degraded' || status === 'warning') {
        element.classList.add('health-warning');
    } else if (status === 'rising' || status === 'critical') {
        element.classList.add('health-critical');
    }
}

/**
 * Update indicator animation
 */
function updateIndicator(id) {
    const indicator = document.getElementById(id);
    indicator.style.background = '#3fb950';
    setTimeout(() => {
        indicator.style.background = '';
    }, 200);
}

/**
 * Get trend arrow
 */
function getTrendArrow(trend) {
    switch (trend) {
        case 'improving': return '↓';
        case 'stable': return '→';
        case 'rising': return '↑';
        default: return '–';
    }
}

/**
 * Format percentage
 */
function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== EVENT LISTENERS =====

document.addEventListener('DOMContentLoaded', () => {
    // Initialize charts
    initCharts();

    // Start polling
    startPolling();

    // Load preview history
    loadPreviewHistory();

    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active from all tabs
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // Add active to clicked tab
            tab.classList.add('active');
            const tabName = tab.dataset.tab;
            document.getElementById(`${tabName}Tab`).classList.add('active');

            // Load tab-specific data
            if (tabName === 'throughput') {
                loadThroughputData();
            }
        });
    });

    // Filter controls
    document.getElementById('regimeFilter').addEventListener('change', (e) => {
        state.historyFilters.regime = e.target.value;
        state.historyPage = 0;
        loadPreviewHistory();
    });

    document.getElementById('sourceFilter').addEventListener('change', (e) => {
        state.historyFilters.source = e.target.value;
        state.historyPage = 0;
        loadPreviewHistory();
    });

    document.getElementById('matchFilter').addEventListener('change', (e) => {
        state.historyFilters.exactMatch = e.target.value;
        state.historyPage = 0;
        loadPreviewHistory();
    });

    document.getElementById('refreshHistory').addEventListener('click', () => {
        loadPreviewHistory();
    });

    // Pagination
    document.getElementById('prevPage').addEventListener('click', () => {
        if (state.historyPage > 0) {
            state.historyPage--;
            loadPreviewHistory();
        }
    });

    document.getElementById('nextPage').addEventListener('click', () => {
        state.historyPage++;
        loadPreviewHistory();
    });

    // Modal close
    document.querySelector('.modal-close').addEventListener('click', () => {
        document.getElementById('previewModal').classList.remove('active');
    });

    // Modal background close
    document.getElementById('previewModal').addEventListener('click', (e) => {
        if (e.target.id === 'previewModal') {
            e.target.classList.remove('active');
        }
    });
});
