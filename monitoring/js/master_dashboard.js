/**
 * Master Monitoring Dashboard
 * Fetches and displays data from all 11 monitoring plugins via unified API
 */

// Configuration
const CONFIG = {
    // Auto-detect API URL - use current hostname or localhost
    apiUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8081/api/unified'
        : `http://${window.location.hostname}:8081/api/unified`,
    refreshInterval: 5000, // 5 seconds
    errorRetryDelay: 10000 // 10 seconds on error
};

// State
let refreshTimer = null;
let lastUpdateTime = null;
let errorCount = 0;

/**
 * Initialize dashboard
 */
function init() {
    console.log('Master Dashboard initializing...');
    fetchData();
    startAutoRefresh();
}

/**
 * Fetch data from unified API
 */
async function fetchData() {
    try {
        updateRefreshIndicator(true);

        const response = await fetch(CONFIG.apiUrl);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        console.log('Unified API response:', data);

        // Update all displays
        updateSystemHealth(data);
        updateQuickMetrics(data);
        updateTrainingStatus(data);
        updateGPU4090(data);
        updateGPU3090(data);
        updateCurriculum(data);
        updateRegression(data);
        updateModelComparison(data);
        updateTesting(data);
        updateAdversarial(data);
        updateConfidence(data);
        updateSelfCorrection(data);
        updateCheckpointSync(data);

        // Update timestamp
        lastUpdateTime = new Date();
        updateLastUpdateDisplay();
        errorCount = 0;

        updateRefreshIndicator(false);
    } catch (error) {
        console.error('Error fetching data:', error);
        errorCount++;
        handleError(error);
        updateRefreshIndicator(false);
    }
}

/**
 * Update system-wide health indicator
 */
function updateSystemHealth(data) {
    const healthDot = document.getElementById('systemHealthDot');
    const healthLabel = document.getElementById('systemHealthLabel');

    const systemStatus = data.summary?.system_status || 'unknown';

    healthDot.className = 'health-dot';

    if (systemStatus === 'healthy') {
        healthDot.classList.add('healthy');
        healthLabel.textContent = 'HEALTHY';
        healthLabel.style.color = 'var(--color-success)';
    } else if (systemStatus === 'degraded') {
        healthDot.classList.add('degraded');
        healthLabel.textContent = 'DEGRADED';
        healthLabel.style.color = 'var(--color-warning)';
    } else if (systemStatus === 'critical') {
        healthDot.classList.add('critical');
        healthLabel.textContent = 'CRITICAL';
        healthLabel.style.color = 'var(--color-danger)';
    } else {
        healthLabel.textContent = 'UNKNOWN';
        healthLabel.style.color = 'var(--text-muted)';
    }
}

/**
 * Update quick metrics in header
 */
function updateQuickMetrics(data) {
    const training = data.summary?.training || {};
    const hardware = data.summary?.hardware || {};

    // Training step
    const currentStep = training.current_step || '--';
    document.getElementById('quickTrainingStep').textContent =
        currentStep !== '--' ? `Step ${currentStep}` : '--';

    // Loss
    const loss = training.loss;
    document.getElementById('quickLoss').textContent =
        loss ? loss.toFixed(3) : '--';

    // 4090 VRAM
    const vram4090 = hardware['4090']?.vram_percent;
    document.getElementById('quick4090VRAM').textContent =
        vram4090 !== undefined ? `${vram4090.toFixed(0)}%` : '--';

    // 3090 VRAM
    const vram3090 = hardware['3090']?.vram_percent;
    document.getElementById('quick3090VRAM').textContent =
        vram3090 !== undefined ? `${vram3090.toFixed(0)}%` : '--';

    // Plugin health
    const healthy = data.summary?.plugins_healthy || 0;
    const total = data.summary?.plugins_total || 11;
    document.getElementById('quickPlugins').textContent = `${healthy}/${total}`;
}

/**
 * Update training status card (4090)
 */
function updateTrainingStatus(data) {
    const source = data.sources?.training_status;
    if (!source || source.status !== 'ok') {
        setCardOffline('trainingStatusIndicator');
        return;
    }

    setCardOnline('trainingStatusIndicator');
    const training = source.data;

    // Status
    setText('trainingStatus', training.status || '--');

    // Step
    const currentStep = training.current_step || 0;
    const totalSteps = training.total_steps || 0;
    setText('trainingStep', `${currentStep} / ${totalSteps}`);

    // Progress
    const progress = training.progress_percent || 0;
    setText('trainingProgress', `${progress.toFixed(1)}%`);
    setProgressBar('trainingProgressBar', progress);

    // Loss
    const loss = training.loss;
    setText('trainingLoss', loss ? loss.toFixed(4) : '--');

    // Val Loss
    const valLoss = training.validation_loss;
    setText('trainingValLoss', valLoss ? valLoss.toFixed(4) : '--');

    // Gap
    const gap = training.val_train_gap;
    setText('trainingGap', gap ? gap.toFixed(3) : '--');

    // Accuracy
    const acc = training.accuracy_percent;
    setText('trainingAccuracy', acc ? `${acc.toFixed(1)}%` : '--');

    // Tokens/sec
    const tokens = training.tokens_per_sec;
    setText('trainingTokens', tokens ? tokens.toFixed(0) : '--');
}

/**
 * Update GPU 4090 card
 */
function updateGPU4090(data) {
    const source = data.sources?.gpu_4090;
    if (!source || source.status !== 'ok') {
        setCardOffline('gpu4090Indicator');
        return;
    }

    setCardOnline('gpu4090Indicator');
    const gpu = source.data;

    // VRAM
    const vramUsed = gpu.vram_used_gb;
    const vramTotal = gpu.vram_total_gb;
    const vramPercent = gpu.vram_percent;

    setText('gpu4090VRAMUsed', vramUsed ? `${vramUsed.toFixed(1)} GB` : '--');
    setText('gpu4090VRAMTotal', vramTotal ? `${vramTotal.toFixed(1)} GB` : '--');
    setText('gpu4090Util', gpu.utilization_percent !== undefined ?
        `${gpu.utilization_percent}%` : '--');
    setText('gpu4090Temp', gpu.temperature_c !== undefined ?
        `${gpu.temperature_c}°C` : '--');

    // VRAM bar
    if (vramPercent !== undefined) {
        setVRAMBar('vram4090Bar', 'vram4090Label', vramPercent);
    }
}

/**
 * Update GPU 3090 card
 */
function updateGPU3090(data) {
    const source = data.sources?.gpu_3090;
    if (!source || source.status !== 'ok') {
        setCardOffline('gpu3090Indicator');
        return;
    }

    setCardOnline('gpu3090Indicator');
    const gpu = source.data;

    // VRAM
    const vramUsed = gpu.vram_used_gb;
    const vramTotal = gpu.vram_total_gb;
    const vramPercent = gpu.vram_percent;

    setText('gpu3090VRAMUsed', vramUsed ? `${vramUsed.toFixed(1)} GB` : '--');
    setText('gpu3090VRAMTotal', vramTotal ? `${vramTotal.toFixed(1)} GB` : '--');
    setText('gpu3090Util', gpu.utilization_percent !== undefined ?
        `${gpu.utilization_percent}%` : '--');
    setText('gpu3090Temp', gpu.temperature_c !== undefined ?
        `${gpu.temperature_c}°C` : '--');

    // VRAM bar
    if (vramPercent !== undefined) {
        setVRAMBar('vram3090Bar', 'vram3090Label', vramPercent);
    }
}

/**
 * Update curriculum optimization card
 */
function updateCurriculum(data) {
    const source = data.sources?.curriculum_optimization;
    if (!source || source.status !== 'ok') {
        setCardOffline('curriculumIndicator');
        return;
    }

    setCardOnline('curriculumIndicator');
    const curriculum = source.data;
    const latest = curriculum.latest_summary;

    if (latest) {
        const acc = latest.accuracies || {};
        setText('curriculumEasy', acc.easy ? `${(acc.easy * 100).toFixed(0)}%` : '--');
        setText('curriculumMedium', acc.medium ? `${(acc.medium * 100).toFixed(0)}%` : '--');
        setText('curriculumHard', acc.hard ? `${(acc.hard * 100).toFixed(0)}%` : '--');
        setText('curriculumStep', latest.step || '--');
    } else {
        setText('curriculumEasy', '--');
        setText('curriculumMedium', '--');
        setText('curriculumHard', '--');
        setText('curriculumStep', '--');
    }
}

/**
 * Update regression monitoring card
 */
function updateRegression(data) {
    const source = data.sources?.regression_monitoring;
    if (!source || source.status !== 'ok') {
        setCardOffline('regressionIndicator');
        return;
    }

    setCardOnline('regressionIndicator');
    const regression = source.data;
    const latest = regression.latest_summary;

    if (latest) {
        const detected = latest.regression_detected ? 'YES' : 'NO';
        setText('regressionDetected', detected);
        document.getElementById('regressionDetected').style.color =
            latest.regression_detected ? 'var(--color-danger)' : 'var(--color-success)';

        setText('regressionLossInc', latest.loss_increase ?
            `${latest.loss_increase.toFixed(1)}%` : '--');
        setText('regressionAccDrop', latest.accuracy_drop ?
            `${latest.accuracy_drop.toFixed(1)}%` : '--');
    } else {
        setText('regressionDetected', '--');
        setText('regressionLossInc', '--');
        setText('regressionAccDrop', '--');
    }

    setText('regressionTotal', regression.total_regressions || 0);
}

/**
 * Update model comparison card
 */
function updateModelComparison(data) {
    const source = data.sources?.model_comparison;
    if (!source || source.status !== 'ok') {
        setCardOffline('modelCompIndicator');
        return;
    }

    setCardOnline('modelCompIndicator');
    const comparison = source.data;
    const latest = comparison.latest_summary;

    if (latest) {
        setText('modelCompBest', latest.best_checkpoint || '--');
        setText('modelCompScore', latest.best_score ?
            latest.best_score.toFixed(3) : '--');
        setText('modelCompTotal', latest.total_compared || 0);

        // Top 3 models
        const container = document.getElementById('topModelsContainer');
        container.innerHTML = '';

        if (latest.top_3 && latest.top_3.length > 0) {
            latest.top_3.forEach(model => {
                const div = document.createElement('div');
                div.className = 'top-model-item';
                div.innerHTML = `
                    <span class="model-rank">#${model.rank}</span>
                    <span class="model-name">${model.checkpoint}</span>
                    <span class="model-score">${model.score.toFixed(3)}</span>
                `;
                container.appendChild(div);
            });
        }
    } else {
        setText('modelCompBest', '--');
        setText('modelCompScore', '--');
        setText('modelCompTotal', '--');
    }
}

/**
 * Update automated testing card
 */
function updateTesting(data) {
    const source = data.sources?.automated_testing;
    if (!source || source.status !== 'ok') {
        setCardOffline('testingIndicator');
        return;
    }

    setCardOnline('testingIndicator');
    const testing = source.data;
    const latest = testing.latest_summary;

    if (latest) {
        setText('testingPassRate', latest.pass_rate ?
            `${(latest.pass_rate * 100).toFixed(1)}%` : '--');
        setText('testingTotal', latest.total_tests || 0);
        setText('testingPassed', latest.passed || 0);
        setText('testingFailed', latest.failed || 0);
    } else {
        setText('testingPassRate', '--');
        setText('testingTotal', '--');
        setText('testingPassed', '--');
        setText('testingFailed', '--');
    }
}

/**
 * Update adversarial mining card
 */
function updateAdversarial(data) {
    const source = data.sources?.adversarial_mining;
    if (!source || source.status !== 'ok') {
        setCardOffline('advIndicator');
        return;
    }

    setCardOnline('advIndicator');
    const adv = source.data;
    const latest = adv.latest_summary;

    if (latest) {
        setText('advTotal', latest.total_examples || 0);

        const categories = latest.categories || {};
        const catCount = Object.keys(categories).length;
        setText('advCategories', catCount);

        // Display categories
        const container = document.getElementById('advCategoriesContainer');
        container.innerHTML = '';

        for (const [name, cat] of Object.entries(categories)) {
            const div = document.createElement('div');
            div.className = 'adv-category';
            div.innerHTML = `
                <span class="category-name">${name}</span>
                <span class="category-count">${cat.count} (avg loss: ${cat.avg_loss.toFixed(3)})</span>
            `;
            container.appendChild(div);
        }
    } else {
        setText('advTotal', '--');
        setText('advCategories', '--');
    }
}

/**
 * Update confidence calibration card
 */
function updateConfidence(data) {
    const source = data.sources?.confidence_calibration;
    if (!source || source.status !== 'ok') {
        setCardOffline('confIndicator');
        return;
    }

    setCardOnline('confIndicator');
    const conf = source.data;
    const latest = conf.latest_summary;

    if (latest) {
        setText('confECE', latest.ece ? latest.ece.toFixed(4) : '--');
        setText('confBins', latest.num_bins || '--');
        setText('confOverconf', latest.overconfident ?
            latest.overconfident.toFixed(3) : '--');
    } else {
        setText('confECE', '--');
        setText('confBins', '--');
        setText('confOverconf', '--');
    }
}

/**
 * Update self-correction loop card
 */
function updateSelfCorrection(data) {
    const source = data.sources?.self_correction;
    if (!source || source.status !== 'ok') {
        setCardOffline('selfCorrIndicator');
        return;
    }

    setCardOnline('selfCorrIndicator');
    const selfCorr = source.data;
    const latest = selfCorr.latest_summary;

    if (latest) {
        setText('selfCorrErrors', latest.errors_captured || 0);
        setText('selfCorrPatterns', latest.patterns_found || 0);
        setText('selfCorrGen', latest.corrections_generated || 0);
    } else {
        setText('selfCorrErrors', '--');
        setText('selfCorrPatterns', '--');
        setText('selfCorrGen', '--');
    }

    setText('selfCorrTotal', selfCorr.total_corrections || 0);
}

/**
 * Update checkpoint sync card
 */
function updateCheckpointSync(data) {
    const source = data.sources?.checkpoint_sync;
    if (!source || source.status !== 'ok') {
        setCardOffline('checkpointIndicator');
        return;
    }

    setCardOnline('checkpointIndicator');
    const checkpoint = source.data;
    const summary = checkpoint.summary;

    if (summary) {
        setText('checkpointStatus', summary.status || '--');
        setText('checkpointLatest', summary.latest_checkpoint || '--');
        setText('checkpointSynced', summary.total_synced || 0);
        setText('checkpointFailures', summary.failures || 0);
    } else {
        setText('checkpointStatus', '--');
        setText('checkpointLatest', '--');
        setText('checkpointSynced', '--');
        setText('checkpointFailures', '--');
    }
}

/**
 * Helper: Set text content with animation
 */
function setText(elementId, value) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const newValue = String(value);
    if (el.textContent !== newValue) {
        el.textContent = newValue;
        el.classList.add('updated');
        setTimeout(() => el.classList.remove('updated'), 500);
    }
}

/**
 * Helper: Set progress bar width
 */
function setProgressBar(elementId, percent) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

/**
 * Helper: Set VRAM bar
 */
function setVRAMBar(barId, labelId, percent) {
    const bar = document.getElementById(barId);
    const label = document.getElementById(labelId);

    if (bar) {
        const clamped = Math.min(100, Math.max(0, percent));
        bar.style.width = `${clamped}%`;

        // Color based on usage
        if (clamped > 90) {
            bar.style.background = 'linear-gradient(90deg, var(--color-danger), var(--color-warning))';
        } else if (clamped > 70) {
            bar.style.background = 'linear-gradient(90deg, var(--color-warning), var(--color-success))';
        } else {
            bar.style.background = 'linear-gradient(90deg, var(--color-success), var(--color-info))';
        }
    }

    if (label) {
        label.textContent = `${percent.toFixed(1)}%`;
    }
}

/**
 * Helper: Set card online
 */
function setCardOnline(indicatorId) {
    const el = document.getElementById(indicatorId);
    if (!el) return;
    el.className = 'status-indicator online';
}

/**
 * Helper: Set card offline
 */
function setCardOffline(indicatorId) {
    const el = document.getElementById(indicatorId);
    if (!el) return;
    el.className = 'status-indicator offline';
}

/**
 * Update last update time display
 */
function updateLastUpdateDisplay() {
    if (!lastUpdateTime) return;

    const el = document.getElementById('lastUpdate');
    if (!el) return;

    const now = new Date();
    const seconds = Math.floor((now - lastUpdateTime) / 1000);

    if (seconds < 60) {
        el.textContent = `${seconds}s ago`;
    } else {
        const minutes = Math.floor(seconds / 60);
        el.textContent = `${minutes}m ago`;
    }
}

/**
 * Update refresh indicator
 */
function updateRefreshIndicator(active) {
    const el = document.getElementById('refreshIndicator');
    if (!el) return;

    if (active) {
        el.style.color = 'var(--color-info)';
        el.style.animation = 'blink 0.5s ease-in-out infinite';
    } else {
        el.style.color = 'var(--color-success)';
        el.style.animation = 'blink 1s ease-in-out infinite';
    }
}

/**
 * Handle fetch errors
 */
function handleError(error) {
    console.error('Dashboard error:', error);

    const healthLabel = document.getElementById('systemHealthLabel');
    const healthDot = document.getElementById('systemHealthDot');

    if (healthLabel && healthDot) {
        healthLabel.textContent = `ERROR (${errorCount})`;
        healthLabel.style.color = 'var(--color-danger)';
        healthDot.className = 'health-dot critical';
    }
}

/**
 * Start auto-refresh
 */
function startAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
    }

    refreshTimer = setInterval(() => {
        fetchData();
        updateLastUpdateDisplay();
    }, CONFIG.refreshInterval);

    // Update "ago" time more frequently
    setInterval(updateLastUpdateDisplay, 1000);
}

/**
 * Stop auto-refresh
 */
function stopAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }
}

// Initialize on load
window.addEventListener('DOMContentLoaded', init);

// Handle visibility changes (pause when tab is hidden)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Dashboard hidden, pausing refresh');
        stopAutoRefresh();
    } else {
        console.log('Dashboard visible, resuming refresh');
        fetchData();
        startAutoRefresh();
    }
});
