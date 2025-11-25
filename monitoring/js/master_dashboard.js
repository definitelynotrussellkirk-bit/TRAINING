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

// Error card state
let activeErrorCards = new Map(); // id -> { issue, element, firstSeen }
let dismissedErrors = new Set(); // Track manually dismissed errors (clears on page reload)
let previousHealthState = null; // Track previous state for change detection

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
        updateSkillMetrics(data);
        updateTransferLearning(data);
        updateLayerDrift(data);
        updateParameterStability(data);

        // Update error cards based on system health
        updateErrorCards(data);

        // Fetch queue data separately (different endpoint)
        fetchQueueData();

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
    document.getElementById('quickLoss').textContent = formatNum(loss, 3);

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
    setText('trainingLoss', formatNum(loss, 4));

    // Val Loss
    const valLoss = training.validation_loss;
    setText('trainingValLoss', formatNum(valLoss, 4));

    // Gap
    const gap = training.val_train_gap;
    setText('trainingGap', formatNum(gap, 3));

    // Accuracy
    const acc = training.accuracy_percent;
    setText('trainingAccuracy', acc !== null && acc !== undefined ? `${formatNum(acc, 1)}%` : '--');

    // Tokens/sec
    const tokens = training.tokens_per_sec;
    setText('trainingTokens', formatNum(tokens, 0));
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

    setText('gpu4090VRAMUsed', vramUsed !== undefined ? `${formatNum(vramUsed, 1)} GB` : '--');
    setText('gpu4090VRAMTotal', vramTotal !== undefined ? `${formatNum(vramTotal, 1)} GB` : '--');
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

    setText('gpu3090VRAMUsed', vramUsed !== undefined ? `${formatNum(vramUsed, 1)} GB` : '--');
    setText('gpu3090VRAMTotal', vramTotal !== undefined ? `${formatNum(vramTotal, 1)} GB` : '--');
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

// Store skill versions data globally
let skillVersionsData = {};
let baseVersionData = null;  // Always keep base reference
let selectedSkillVersion = null;

// Constants
const BASE_VERSION_PREFIX = 'base_';
const IMPROVEMENT_THRESHOLD = 0.5;  // Minimum change to show as positive/negative

/**
 * Update skill metrics cards (SYLLABLE and BINARY)
 * Single entry point for all skill metric updates
 */
function updateSkillMetrics(data) {
    const source = data.sources?.skill_metrics;
    if (!source || source.status !== 'ok') {
        setCardOffline('syllableIndicator');
        setCardOffline('binaryIndicator');
        return;
    }

    const availableVersions = source.data?.available_versions || [];
    const versions = source.data?.versions || {};

    // Store versions and extract base reference
    skillVersionsData = versions;
    baseVersionData = findBaseVersion(versions, availableVersions);

    // Update version selector
    updateVersionSelector(availableVersions);

    // Update checkpoint info display
    updateCheckpointInfo(skillVersionsData[selectedSkillVersion]);

    // Update both skill cards using unified function
    const currentVersion = skillVersionsData[selectedSkillVersion];
    ['syllable', 'binary'].forEach(skill => {
        updateSkillCardUnified(skill, currentVersion, baseVersionData);
    });
}

/**
 * Find the base model version from available versions
 */
function findBaseVersion(versions, availableVersions) {
    const baseTag = availableVersions.find(v => v.startsWith(BASE_VERSION_PREFIX));
    return baseTag ? versions[baseTag] : null;
}

/**
 * Update the version selector dropdown
 */
function updateVersionSelector(availableVersions) {
    const select = document.getElementById('skillVersionSelect');
    if (!select || availableVersions.length === 0) return;

    const currentValue = select.value;
    select.innerHTML = '';

    // Populate options with formatted labels
    availableVersions.forEach(tag => {
        const option = document.createElement('option');
        option.value = tag;
        option.textContent = formatVersionLabel(tag, skillVersionsData[tag]);
        select.appendChild(option);
    });

    // Restore or set default selection
    if (currentValue && availableVersions.includes(currentValue)) {
        select.value = currentValue;
    } else {
        // Default: prefer trained/checkpoint over base
        const preferred = availableVersions.find(v =>
            v.startsWith('trained_') || v.includes('ckpt') || v.includes('checkpoint')
        );
        select.value = preferred || availableVersions[availableVersions.length - 1];
    }
    selectedSkillVersion = select.value;

    // Update count info
    const info = document.getElementById('skillVersionInfo');
    if (info) {
        info.textContent = `${availableVersions.length} version(s)`;
    }
}

/**
 * Format version label for dropdown (include checkpoint if available)
 */
function formatVersionLabel(tag, versionData) {
    if (!versionData?.checkpoint_step) return tag;
    // Show: "trained_ckpt156000 (step 156000)"
    return `${tag} (step ${versionData.checkpoint_step})`;
}

/**
 * Update checkpoint info display
 */
function updateCheckpointInfo(versionData) {
    const infoEl = document.getElementById('skillCheckpointInfo');
    if (!infoEl) return;

    if (versionData?.checkpoint_step) {
        infoEl.textContent = `checkpoint-${versionData.checkpoint_step}`;
        infoEl.style.display = 'inline';
    } else if (versionData?.tag?.startsWith(BASE_VERSION_PREFIX)) {
        infoEl.textContent = 'base model';
        infoEl.style.display = 'inline';
    } else {
        infoEl.style.display = 'none';
    }
}

/**
 * Handle version selector change
 */
document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('skillVersionSelect');
    if (select) {
        select.addEventListener('change', (e) => {
            selectedSkillVersion = e.target.value;
            const versionData = skillVersionsData[selectedSkillVersion];

            // Update checkpoint info
            updateCheckpointInfo(versionData);

            // Update both cards with new version vs base
            ['syllable', 'binary'].forEach(skill => {
                updateSkillCardUnified(skill, versionData, baseVersionData);
            });
        });
    }
});

/**
 * Unified skill card update - DRY implementation
 * Always shows current version with improvement delta vs base
 */
function updateSkillCardUnified(skill, versionData, baseData) {
    const indicatorId = `${skill}Indicator`;

    // Check if we have valid data
    if (!versionData?.skills?.[skill]) {
        setCardOffline(indicatorId);
        return;
    }

    setCardOnline(indicatorId);

    const skillData = versionData.skills[skill];
    const baseSkillData = baseData?.skills?.[skill];

    // Update overall accuracy
    const displayAcc = skillData.overall_accuracy || 0;
    setText(`${skill}Overall`, `${displayAcc.toFixed(1)}%`);

    // Calculate and display improvement vs base
    const baseAcc = baseSkillData?.overall_accuracy || 0;
    const improvement = displayAcc - baseAcc;
    const isBaseVersion = versionData.tag?.startsWith(BASE_VERSION_PREFIX);

    updateDeltaDisplay(`${skill}Delta`, improvement, isBaseVersion);

    // Update difficulty bars
    updateDifficultyBars(skill, skillData.by_difficulty || {});

    // Update timestamp
    updateTimestamp(`${skill}LastEval`, versionData.timestamp || skillData.timestamp);
}

/**
 * Update delta/improvement display element
 */
function updateDeltaDisplay(elementId, improvement, isBaseVersion) {
    const deltaEl = document.getElementById(elementId);
    if (!deltaEl) return;

    const arrowEl = deltaEl.querySelector('.delta-arrow');
    const valueEl = deltaEl.querySelector('.delta-value');
    if (!arrowEl || !valueEl) return;

    // Reset classes
    deltaEl.classList.remove('positive', 'negative', 'neutral');

    if (isBaseVersion) {
        // Viewing base model - no comparison
        deltaEl.classList.add('neutral');
        arrowEl.textContent = '○';
        valueEl.textContent = 'base';
    } else if (improvement > IMPROVEMENT_THRESHOLD) {
        // Positive improvement
        deltaEl.classList.add('positive');
        arrowEl.textContent = '↑';
        valueEl.textContent = `+${improvement.toFixed(1)}%`;
    } else if (improvement < -IMPROVEMENT_THRESHOLD) {
        // Negative (regression)
        deltaEl.classList.add('negative');
        arrowEl.textContent = '↓';
        valueEl.textContent = `${improvement.toFixed(1)}%`;
    } else {
        // Neutral (minimal change)
        deltaEl.classList.add('neutral');
        arrowEl.textContent = '→';
        valueEl.textContent = '~0%';
    }
}

/**
 * Update difficulty bars for a skill - DRY helper
 */
function updateDifficultyBars(skill, byDifficulty) {
    ['easy', 'medium', 'hard'].forEach(diff => {
        const capitalDiff = diff.charAt(0).toUpperCase() + diff.slice(1);
        const barId = `${skill}${capitalDiff}Bar`;
        const valueId = `${skill}${capitalDiff}`;

        const diffData = byDifficulty[diff] || {};
        const acc = diffData.accuracy || 0;

        // Update bar width
        const barEl = document.getElementById(barId);
        if (barEl) {
            barEl.style.width = `${Math.min(100, acc)}%`;
        }

        // Update percentage text
        setText(valueId, `${acc.toFixed(0)}%`);
    });
}

/**
 * Update Transfer Learning summary card
 */
function updateTransferLearning(data) {
    const source = data.sources?.skill_metrics;
    if (!source || source.status !== 'ok') {
        setCardOffline('transferIndicator');
        return;
    }

    setCardOnline('transferIndicator');

    const summary = source.data?.summary || {};
    const skills = source.data?.skills || {};

    // Update primitives summary
    const primData = summary.primitives || {};
    setText('primitivesAvg', primData.trained_avg ? `${primData.trained_avg.toFixed(0)}%` : '--');
    updateTransferDelta('primitivesDelta', primData.delta, primData.count);

    // Update bAbI/benchmark summary - aggregate babi + bigbench
    const benchData = summary.benchmarks || {};
    setText('babiAvg', benchData.trained_avg ? `${benchData.trained_avg.toFixed(0)}%` : '--');
    updateTransferDelta('babiDelta', benchData.delta, benchData.count);

    // Count and show Big-Bench separately (could also aggregate)
    let bbCount = 0, bbBase = 0, bbTrained = 0;
    Object.entries(skills).forEach(([name, data]) => {
        if (name.startsWith('bb_') || name.startsWith('bbh_')) {
            const base = data.base?.overall_accuracy || 0;
            const trained = data.trained?.overall_accuracy || 0;
            if (base > 0 || trained > 0) {
                bbCount++;
                bbBase += base;
                bbTrained += trained;
            }
        }
    });
    if (bbCount > 0) {
        setText('bigbenchAvg', `${(bbTrained / bbCount).toFixed(0)}%`);
        updateTransferDelta('bigbenchDelta', (bbTrained - bbBase) / bbCount, bbCount);
    }

    // Show best/worst performers
    updateTransferDetails(skills);
}

/**
 * Update transfer delta display
 */
function updateTransferDelta(elementId, delta, count) {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (delta === undefined || delta === null || count === 0) {
        el.textContent = 'no data';
        el.style.color = '#888';
        return;
    }

    const sign = delta >= 0 ? '+' : '';
    el.textContent = `${sign}${delta.toFixed(1)}% (n=${count || 0})`;

    if (delta > 2) {
        el.style.color = 'var(--color-success)';
    } else if (delta < -2) {
        el.style.color = 'var(--color-danger)';
    } else {
        el.style.color = '#888';
    }
}

/**
 * Show best/worst performers in transfer learning
 */
function updateTransferDetails(skills) {
    const detailsEl = document.getElementById('transferDetails');
    if (!detailsEl) return;

    // Collect skills with both base and trained data
    const comparisons = [];
    Object.entries(skills).forEach(([name, data]) => {
        if (data.category === 'trained') return; // Skip trained skills
        const baseAcc = data.base?.overall_accuracy || 0;
        const trainedAcc = data.trained?.overall_accuracy || 0;
        if (baseAcc > 0 || trainedAcc > 0) {
            comparisons.push({
                name: name.replace('babi_', '').replace('bb_', '').replace('bbh_', ''),
                delta: trainedAcc - baseAcc,
                base: baseAcc,
                trained: trainedAcc
            });
        }
    });

    if (comparisons.length === 0) {
        detailsEl.innerHTML = '<span style="color: #666;">Run baseline tests to see transfer effects</span>';
        return;
    }

    // Sort by delta
    comparisons.sort((a, b) => b.delta - a.delta);

    // Show top 3 gains and top 3 losses
    const best = comparisons.filter(c => c.delta > 0).slice(0, 3);
    const worst = comparisons.filter(c => c.delta < 0).slice(-3).reverse();

    let html = '';
    if (best.length > 0) {
        html += '<span style="color: var(--color-success);">↑ Best: ';
        html += best.map(c => `${c.name} +${c.delta.toFixed(0)}%`).join(', ');
        html += '</span><br>';
    }
    if (worst.length > 0) {
        html += '<span style="color: var(--color-danger);">↓ Worst: ';
        html += worst.map(c => `${c.name} ${c.delta.toFixed(0)}%`).join(', ');
        html += '</span>';
    }

    detailsEl.innerHTML = html || '<span style="color: #666;">No significant transfer effects</span>';
}

/**
 * Update Layer Drift Analysis card
 * Shows how much each transformer layer has changed from base model
 */
function updateLayerDrift(data) {
    const source = data.sources?.training_analytics;
    if (!source || source.status !== 'ok') {
        setCardOffline('layerDriftIndicator');
        return;
    }

    const drift = source.data?.layer_drift;
    if (!drift || !drift.available) {
        setCardOffline('layerDriftIndicator');
        return;
    }

    setCardOnline('layerDriftIndicator');

    // Update summary metrics
    const totalChange = drift.total_relative_change;
    setText('driftTotalChange', totalChange ? `${(totalChange * 100).toFixed(1)}%` : '--');

    const pattern = drift.summary?.pattern || '--';
    const patternEl = document.getElementById('driftPattern');
    if (patternEl) {
        patternEl.textContent = pattern;
        patternEl.className = 'drift-value ' + pattern;
    }

    const maxLayer = drift.summary?.max_drift_layer;
    const maxValue = drift.summary?.max_drift_value;
    setText('driftMaxLayer', maxLayer !== undefined ?
        `L${maxLayer} (${(maxValue * 100).toFixed(1)}%)` : '--');

    // Update comparison reference
    setText('driftRef', `${drift.current || '--'} vs ${drift.reference || '--'}`);

    // Build layer drift chart
    const chartContainer = document.getElementById('layerDriftChart');
    if (chartContainer && drift.layers && drift.layers.length > 0) {
        chartContainer.innerHTML = '';

        // Find max change for scaling
        const maxChange = Math.max(...drift.layers.map(l => l.relative_change || 0));

        drift.layers.forEach(layer => {
            const change = layer.relative_change || 0;
            const percent = change * 100;
            const height = maxChange > 0 ? (change / maxChange * 40) : 0;

            // Classify by change level
            let barClass = 'low';
            if (percent > 35) barClass = 'high';
            else if (percent > 20) barClass = 'medium';

            const wrapper = document.createElement('div');
            wrapper.className = 'layer-bar-wrapper';
            wrapper.title = `Layer ${layer.layer}: ${percent.toFixed(1)}% change`;

            const bar = document.createElement('div');
            bar.className = `layer-bar ${barClass}`;
            bar.style.height = `${height}px`;

            const label = document.createElement('span');
            label.className = 'layer-bar-label';
            label.textContent = layer.layer;

            wrapper.appendChild(bar);
            wrapper.appendChild(label);
            chartContainer.appendChild(wrapper);
        });
    }
}

/**
 * Update Parameter Stability card
 * Shows weight norm health and alerts for each layer
 */
function updateParameterStability(data) {
    const source = data.sources?.training_analytics;
    if (!source || source.status !== 'ok') {
        setCardOffline('paramStabilityIndicator');
        return;
    }

    const stability = source.data?.parameter_stability;
    if (!stability || !stability.available) {
        setCardOffline('paramStabilityIndicator');
        return;
    }

    setCardOnline('paramStabilityIndicator');

    // Update summary metrics
    const health = stability.health_status || 'unknown';
    const healthEl = document.getElementById('stabilityHealth');
    if (healthEl) {
        healthEl.textContent = health.toUpperCase();
        healthEl.className = 'stability-value ' + health;
    }

    const avgNorm = stability.summary?.avg_weight_norm;
    setText('stabilityAvgNorm', avgNorm ? avgNorm.toFixed(1) : '--');

    const totalAlerts = stability.alerts?.length || 0;
    const alertsEl = document.getElementById('stabilityAlerts');
    if (alertsEl) {
        alertsEl.textContent = totalAlerts;
        alertsEl.style.color = totalAlerts > 0 ? 'var(--color-warning)' : 'var(--color-success)';
    }

    // Display alerts
    const alertsContainer = document.getElementById('stabilityAlertsContainer');
    if (alertsContainer) {
        alertsContainer.innerHTML = '';

        if (stability.alerts && stability.alerts.length > 0) {
            stability.alerts.slice(0, 3).forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `stability-alert ${alert.severity || 'warning'}`;
                alertDiv.innerHTML = `
                    <span class="alert-icon">${alert.severity === 'critical' ? '⚠' : '⚡'}</span>
                    <span>L${alert.layer}: ${alert.type} (${alert.value.toFixed(1)})</span>
                `;
                alertsContainer.appendChild(alertDiv);
            });
        }
    }

    // Build layer norms chart
    const chartContainer = document.getElementById('layerNormsChart');
    if (chartContainer && stability.layer_norms && stability.layer_norms.length > 0) {
        chartContainer.innerHTML = '';

        // Find max norm for scaling
        const maxNorm = Math.max(...stability.layer_norms.map(l => l.weight_norm || 0));
        const threshold = 100; // Alert threshold for max_abs_weight

        stability.layer_norms.forEach(layer => {
            const norm = layer.weight_norm || 0;
            const maxAbs = layer.max_abs_weight || 0;
            const height = maxNorm > 0 ? (norm / maxNorm * 40) : 0;

            // Classify by health
            let barClass = 'norm-healthy';
            if (maxAbs > threshold) barClass = 'norm-warning';
            if (maxAbs > threshold * 1.5) barClass = 'norm-critical';

            const wrapper = document.createElement('div');
            wrapper.className = 'layer-bar-wrapper';
            wrapper.title = `Layer ${layer.layer}: norm=${norm.toFixed(1)}, max=${maxAbs.toFixed(1)}`;

            const bar = document.createElement('div');
            bar.className = `layer-bar ${barClass}`;
            bar.style.height = `${height}px`;

            const label = document.createElement('span');
            label.className = 'layer-bar-label';
            label.textContent = layer.layer;

            wrapper.appendChild(bar);
            wrapper.appendChild(label);
            chartContainer.appendChild(wrapper);
        });
    }
}

/**
 * Update timestamp display - DRY helper
 */
function updateTimestamp(elementId, timestamp) {
    if (!timestamp) {
        setText(elementId, '--');
        return;
    }

    const date = new Date(timestamp);
    const formatted = date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
    setText(elementId, formatted);
}

/**
 * Helper: Format numeric value safely (handles 0 correctly)
 * Fixes falsy-value bug where 0 was displayed as '--'
 */
function formatNum(value, digits = 3) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
        return '--';
    }
    return value.toFixed(digits);
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

// ===== Training Queue Functions =====
let allQueueFilesMain = [];
let currentQueueFilter = 'all';

async function fetchQueueData() {
    try {
        const apiBase = CONFIG.apiUrl.replace('/api/unified', '');
        const response = await fetch(`${apiBase}/api/queue`);
        if (!response.ok) return;

        const queueData = await response.json();
        updateQueueDisplay(queueData);
    } catch (e) {
        console.log('Queue fetch error:', e);
    }
}

function updateQueueDisplay(queueData) {
    if (!queueData) return;

    allQueueFilesMain = queueData.files || [];
    const totalFiles = queueData.total_files || 0;
    const totalExamples = queueData.total_examples || 0;
    const inboxCount = queueData.inbox_count || 0;

    // Update summary
    const filesEl = document.getElementById('queueTotalFiles');
    const examplesEl = document.getElementById('queueTotalExamples');
    const inboxEl = document.getElementById('queueInboxCount');

    if (filesEl) filesEl.textContent = totalFiles;
    if (examplesEl) examplesEl.textContent = formatQueueNumber(totalExamples);
    if (inboxEl) inboxEl.textContent = inboxCount;

    // Count by priority
    const counts = { high: 0, normal: 0, low: 0 };
    allQueueFilesMain.forEach(f => {
        const p = f.priority || 'normal';
        if (counts[p] !== undefined) counts[p]++;
    });

    const highEl = document.getElementById('qCountHigh');
    const normalEl = document.getElementById('qCountNormal');
    const lowEl = document.getElementById('qCountLow');

    if (highEl) highEl.textContent = counts.high;
    if (normalEl) normalEl.textContent = counts.normal;
    if (lowEl) lowEl.textContent = counts.low;

    // Render files
    renderQueueFilesMain(currentQueueFilter);
}

function renderQueueFilesMain(filter) {
    const files = filter === 'all'
        ? allQueueFilesMain
        : allQueueFilesMain.filter(f => f.priority === filter);

    const listEl = document.getElementById('queueFileList');
    const emptyEl = document.getElementById('queueEmptyState');

    if (!listEl) return;

    if (files.length === 0) {
        listEl.innerHTML = '';
        if (emptyEl) emptyEl.style.display = 'block';
        return;
    }

    if (emptyEl) emptyEl.style.display = 'none';

    listEl.innerHTML = files.map(f => `
        <div class="q-file" title="${f.name}">
            <div class="q-dot ${f.priority || 'normal'}"></div>
            <div class="q-info">
                <div class="q-name">${f.name}</div>
                <div class="q-meta">${(f.examples || 0).toLocaleString()} ex · ${f.size_mb || '?'} MB</div>
            </div>
            <div class="q-size">${formatQueueNumber(f.examples)}</div>
        </div>
    `).join('');
}

function filterQueueMain(priority) {
    currentQueueFilter = priority;
    // Update tab styles
    document.querySelectorAll('.q-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.priority === priority);
    });
    renderQueueFilesMain(priority);
}

async function refreshQueue() {
    await fetchQueueData();
}

function formatQueueNumber(n) {
    if (!n) return '--';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toString();
}

// ===== Error Card Management =====

/**
 * Process system health data and show/hide error cards
 */
function updateErrorCards(data) {
    const systemHealth = data.sources?.system_health?.data;
    if (!systemHealth) return;

    const currentIssues = detectIssues(systemHealth);

    // Track which issues are still active
    const activeIssueIds = new Set(currentIssues.map(i => i.id));

    // Remove resolved issues (with animation)
    for (const [id, cardInfo] of activeErrorCards) {
        if (!activeIssueIds.has(id)) {
            resolveErrorCard(id);
        }
    }

    // Add new issues
    for (const issue of currentIssues) {
        if (!activeErrorCards.has(issue.id) && !dismissedErrors.has(issue.id)) {
            createErrorCard(issue);
        } else if (activeErrorCards.has(issue.id)) {
            // Update existing card (e.g., downtime counter)
            updateErrorCardDowntime(issue.id);
        }
    }

    previousHealthState = systemHealth;
}

/**
 * Detect issues from system health data
 */
function detectIssues(healthData) {
    const issues = [];

    // Check for unreachable machines
    for (const [machineName, machineData] of Object.entries(healthData.machines || {})) {
        if (!machineData.reachable) {
            issues.push({
                id: `machine_unreachable_${machineName}`,
                type: 'machine_unreachable',
                severity: 'critical',
                name: `${machineName} Unreachable`,
                machine: machineName,
                message: machineData.error || 'Cannot connect to machine',
                firstSeen: Date.now()
            });
        }

        // Check for missing processes
        for (const missingProcess of (machineData.processes_missing || [])) {
            issues.push({
                id: `process_down_${machineName}_${missingProcess}`,
                type: 'process_down',
                severity: isCriticalProcess(missingProcess) ? 'critical' : 'warning',
                name: formatProcessName(missingProcess),
                machine: machineName,
                message: 'Process not running',
                processName: missingProcess,
                firstSeen: Date.now()
            });
        }

        // Check GPU availability
        if (machineData.reachable && !machineData.gpu_available) {
            issues.push({
                id: `gpu_unavailable_${machineName}`,
                type: 'gpu_unavailable',
                severity: 'warning',
                name: `GPU Unavailable`,
                machine: machineName,
                message: 'GPU not detected or not accessible',
                firstSeen: Date.now()
            });
        }
    }

    // Check individual process health from processes array
    for (const process of (healthData.processes || [])) {
        if (!process.running && process.error) {
            const id = `process_error_${process.machine}_${process.name}`;
            // Don't duplicate if we already have this from processes_missing
            if (!issues.some(i => i.id.includes(process.name) && i.machine === process.machine)) {
                issues.push({
                    id: id,
                    type: 'process_error',
                    severity: isCriticalProcess(process.name) ? 'critical' : 'warning',
                    name: formatProcessName(process.name),
                    machine: process.machine,
                    message: process.error || 'Process error',
                    processName: process.name,
                    firstSeen: Date.now()
                });
            }
        }
    }

    return issues;
}

/**
 * Check if a process is critical (affects training flow)
 */
function isCriticalProcess(processName) {
    const criticalProcesses = [
        'training_daemon',
        'inference_server'
    ];
    return criticalProcesses.some(cp => processName.toLowerCase().includes(cp.toLowerCase()));
}

/**
 * Format process name for display
 */
function formatProcessName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Create and display an error card
 */
function createErrorCard(issue) {
    const container = document.getElementById('errorCardsContainer');
    if (!container) return;

    // Preserve first seen time if re-showing (e.g., after page reload)
    const firstSeen = activeErrorCards.get(issue.id)?.firstSeen || issue.firstSeen || Date.now();

    const card = document.createElement('div');
    card.className = `error-card ${issue.severity}`;
    card.id = `error-card-${issue.id}`;
    card.dataset.firstSeen = firstSeen;

    const icon = getIssueIcon(issue.type, issue.severity);

    card.innerHTML = `
        <div class="error-card-header">
            <div class="error-card-title">
                <span class="error-card-icon">${icon}</span>
                <span class="error-card-name">${issue.name}</span>
            </div>
            <button class="error-card-dismiss" onclick="dismissErrorCard('${issue.id}')" title="Dismiss">×</button>
        </div>
        <div class="error-card-body">
            <div class="error-card-detail">
                <span class="error-card-label">Machine</span>
                <span class="error-card-value machine-${issue.machine}">${issue.machine}</span>
            </div>
            <div class="error-card-detail">
                <span class="error-card-label">Issue</span>
                <span class="error-card-value">${issue.message}</span>
            </div>
            <div class="error-card-detail">
                <span class="error-card-label">Down for</span>
                <span class="error-card-value error-card-downtime" id="downtime-${issue.id}">
                    ${formatDowntime(firstSeen)}
                </span>
            </div>
        </div>
    `;

    container.appendChild(card);

    activeErrorCards.set(issue.id, {
        issue: issue,
        element: card,
        firstSeen: firstSeen
    });

    console.log(`Error card created: ${issue.name} on ${issue.machine}`);
}

/**
 * Get appropriate icon for issue type
 */
function getIssueIcon(type, severity) {
    const icons = {
        'machine_unreachable': '🔌',
        'process_down': '⚠',
        'process_error': '❌',
        'gpu_unavailable': '🎮'
    };

    if (severity === 'critical') {
        return '🚨';
    }

    return icons[type] || '⚠';
}

/**
 * Format downtime duration
 */
function formatDowntime(firstSeen) {
    const elapsed = Date.now() - firstSeen;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
        return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

/**
 * Update downtime counter for an existing error card
 */
function updateErrorCardDowntime(issueId) {
    const cardInfo = activeErrorCards.get(issueId);
    if (!cardInfo) return;

    const downtimeEl = document.getElementById(`downtime-${issueId}`);
    if (downtimeEl) {
        downtimeEl.textContent = formatDowntime(cardInfo.firstSeen);
    }
}

/**
 * Resolve (auto-dismiss) an error card when issue is fixed
 */
function resolveErrorCard(issueId) {
    const cardInfo = activeErrorCards.get(issueId);
    if (!cardInfo) return;

    const card = cardInfo.element;
    card.classList.add('resolved');

    // Remove after animation
    setTimeout(() => {
        card.remove();
        activeErrorCards.delete(issueId);
        // Don't add to dismissedErrors - resolved issues should reappear if they recur
    }, 500);

    console.log(`Error card resolved: ${cardInfo.issue.name}`);
}

/**
 * Manually dismiss an error card
 */
function dismissErrorCard(issueId) {
    const cardInfo = activeErrorCards.get(issueId);
    if (!cardInfo) return;

    const card = cardInfo.element;
    card.classList.add('dismissing');

    // Add to dismissed set so it doesn't reappear until page reload
    dismissedErrors.add(issueId);

    // Remove after animation
    setTimeout(() => {
        card.remove();
        activeErrorCards.delete(issueId);
    }, 300);

    console.log(`Error card dismissed: ${cardInfo.issue.name}`);
}

/**
 * Dismiss all error cards
 */
function dismissAllErrorCards() {
    for (const [id] of activeErrorCards) {
        dismissErrorCard(id);
    }
}

/**
 * Update all downtime counters (called periodically)
 */
function updateAllDowntimes() {
    for (const [id] of activeErrorCards) {
        updateErrorCardDowntime(id);
    }
}

// Update downtime counters every second
setInterval(updateAllDowntimes, 1000);

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
