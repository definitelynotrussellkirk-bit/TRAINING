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
        updateDataHealth(data);
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
        updateSyllo10LevelBars(data);  // Override SYLLO with 10-level data
        updateTransferLearning(data);
        updateLayerDrift(data);
        updateParameterStability(data);

        // NEW: Update scheduler, retention, data impact cards
        updateScheduler(data);
        updateRetention(data);
        updateDataImpact(data);

        // Fetch and update data lineage (separate API call)
        fetchDataLineage();

        // Update error cards based on system health
        updateErrorCards(data);

        // Update daemon status badge in header
        updateDaemonStatusBadge(data);

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
        setActionHint('trainingAction', 'Training status unavailable', 'warning');
        return;
    }

    setCardOnline('trainingStatusIndicator');
    const training = source.data;

    // Status
    const status = training.status || '--';
    setText('trainingStatus', status);

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

    // Color gap based on threshold
    const gapEl = document.getElementById('trainingGap');
    if (gapEl && gap !== null && gap !== undefined) {
        if (gap > 0.5) {
            gapEl.style.color = 'var(--color-danger)';
        } else if (gap > 0.3) {
            gapEl.style.color = 'var(--color-warning)';
        } else {
            gapEl.style.color = 'var(--color-success)';
        }
    }

    // Accuracy
    const acc = training.accuracy_percent;
    setText('trainingAccuracy', acc !== null && acc !== undefined ? `${formatNum(acc, 1)}%` : '--');

    // Tokens/sec
    const tokens = training.tokens_per_sec;
    setText('trainingTokens', formatNum(tokens, 0));

    // Compute action hint based on training state
    updateTrainingActionHint(status, gap, loss, acc);
}

/**
 * Compute and set training action hint
 */
function updateTrainingActionHint(status, gap, loss, acc) {
    let actionText = '';
    let actionLevel = 'good';

    const statusLower = (status || '').toLowerCase();

    // Check training state first
    if (statusLower === 'stopped' || statusLower === 'idle') {
        actionText = 'Training not running. Start daemon or add data to queue.';
        actionLevel = 'warning';
    } else if (statusLower === 'paused') {
        actionText = 'Training paused. Use controller to resume when ready.';
        actionLevel = 'warning';
    } else if (statusLower === 'training') {
        // Training is running - check metrics
        if (gap !== null && gap !== undefined && gap > 0.5) {
            actionText = `Overfitting risk: val-train gap is ${gap.toFixed(2)}. Consider early stopping or more data.`;
            actionLevel = 'critical';
        } else if (gap !== null && gap !== undefined && gap > 0.3) {
            actionText = `Monitor: val-train gap is ${gap.toFixed(2)}. Watch for overfitting.`;
            actionLevel = 'warning';
        } else if (loss !== null && loss !== undefined && loss > 2.0) {
            actionText = `High loss (${loss.toFixed(2)}). Model still learning fundamentals.`;
            actionLevel = 'warning';
        } else if (acc !== null && acc !== undefined && acc < 30) {
            actionText = `Low accuracy (${acc.toFixed(0)}%). Model needs more training.`;
            actionLevel = 'warning';
        } else {
            actionText = 'Training progressing normally.';
            actionLevel = 'good';
        }
    } else {
        actionText = `Status: ${status}`;
        actionLevel = 'neutral';
    }

    setActionHint('trainingAction', actionText, actionLevel);
}

/**
 * Update Data Health card (Protocol Conformance)
 * Displays emoji/direct mode mix and protocol violations
 */
function updateDataHealth(data) {
    const source = data.sources?.training_status;
    if (!source || source.status !== 'ok') {
        setCardOffline('dataHealthIndicator');
        return;
    }

    const training = source.data;
    const stats = training.protocol_stats;

    // If no protocol stats yet, show defaults
    if (!stats) {
        setText('protocolEmojiPct', '--');
        setText('protocolDirectPct', '--');
        setText('protocolMalformed', '--');
        setText('protocolChecked', '0');
        setCardNeutral('dataHealthIndicator');
        return;
    }

    // Update text values
    const emojiPct = stats.emoji_mode_pct || 0;
    const directPct = stats.direct_mode_pct || 0;
    const malformed = stats.malformed || 0;
    const checked = stats.protocol_checked || 0;

    setText('protocolEmojiPct', `${emojiPct.toFixed(1)}%`);
    setText('protocolDirectPct', `${directPct.toFixed(1)}%`);
    setText('protocolMalformed', malformed.toString());
    setText('protocolChecked', checked.toString());

    // Update progress bars
    const emojiBar = document.getElementById('protocolEmojiBar');
    const directBar = document.getElementById('protocolDirectBar');
    if (emojiBar) emojiBar.style.width = `${emojiPct}%`;
    if (directBar) directBar.style.width = `${directPct}%`;

    // Update status indicator
    const indicator = document.getElementById('dataHealthIndicator');
    const malformedEl = document.getElementById('protocolMalformed');

    if (malformed > 0) {
        // Has errors - critical
        if (indicator) {
            indicator.className = 'status-indicator critical';
        }
        if (malformedEl) {
            malformedEl.classList.add('has-errors');
        }
    } else if (checked > 0 && (emojiPct < 30 || emojiPct > 70)) {
        // Mix is skewed - warning
        if (indicator) {
            indicator.className = 'status-indicator warning';
        }
        if (malformedEl) {
            malformedEl.classList.remove('has-errors');
        }
    } else if (checked > 0) {
        // All good
        if (indicator) {
            indicator.className = 'status-indicator healthy';
        }
        if (malformedEl) {
            malformedEl.classList.remove('has-errors');
        }
    } else {
        // No data yet
        setCardNeutral('dataHealthIndicator');
    }
}

/**
 * Set card to neutral (not yet populated)
 */
function setCardNeutral(indicatorId) {
    const el = document.getElementById(indicatorId);
    if (el) {
        el.className = 'status-indicator';
        el.style.color = 'var(--text-muted)';
    }
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
 * Update curriculum optimization card + data flow
 */
function updateCurriculum(data) {
    const source = data.sources?.curriculum_optimization;
    if (!source || source.status !== 'ok') {
        setCardOffline('curriculumIndicator');
        setActionHint('curriculumAction', 'Curriculum data unavailable', 'warning');
        return;
    }

    setCardOnline('curriculumIndicator');
    const curriculum = source.data;
    const latest = curriculum.latest_summary;
    const dataFlow = curriculum.data_flow;

    // Update 10-level accuracy display from skill_metrics.syllo_10level
    const tenLevel = data.sources?.skill_metrics?.data?.syllo_10level;
    if (tenLevel?.available) {
        const byLevel = tenLevel.by_level || {};
        setText('curriculumL1', byLevel.L1 ? `${byLevel.L1.accuracy.toFixed(0)}%` : '--');
        setText('curriculumL2', byLevel.L2 ? `${byLevel.L2.accuracy.toFixed(0)}%` : '--');
        // L3-4 average
        const l3 = byLevel.L3?.accuracy || 0;
        const l4 = byLevel.L4?.accuracy || 0;
        const l34Avg = (l3 + l4) / 2;
        setText('curriculumL34', `${l34Avg.toFixed(0)}%`);
        setText('curriculumStep', latest?.step || '--');
    } else if (latest) {
        // Fallback to legacy if 10-level not available
        setText('curriculumL1', '--');
        setText('curriculumL2', '--');
        setText('curriculumL34', '--');
        setText('curriculumStep', latest.step || '--');
    } else {
        setText('curriculumL1', '--');
        setText('curriculumL2', '--');
        setText('curriculumL34', '--');
        setText('curriculumStep', '--');
    }

    // Update data flow display
    if (dataFlow && dataFlow.available) {
        updateSkillFlowCard('syllo', dataFlow.skills?.syllo);
        updateSkillFlowCard('binary', dataFlow.skills?.binary);
        updateCurriculumActionHint(dataFlow);
    } else {
        setActionHint('curriculumAction', 'Data flow not initialized', 'warning');
    }
}

/**
 * Update a skill flow card with data
 */
function updateSkillFlowCard(skillName, skillData) {
    const prefix = skillName === 'syllo' ? 'syllo' : 'binary';
    const cardEl = document.getElementById(`${prefix}FlowCard`);

    if (!skillData) {
        if (cardEl) cardEl.classList.add('inactive');
        return;
    }

    // Update level
    setText(`${prefix}FlowLevel`, `L${skillData.current_level}/${skillData.max_level}`);

    // Update progress bar
    const progressEl = document.getElementById(`${prefix}FlowProgress`);
    if (progressEl) {
        progressEl.style.width = `${skillData.progress_to_next || 0}%`;
    }

    // Update accuracy
    setText(`${prefix}FlowAcc`, `${skillData.recent_accuracy || 0}%`);

    // Update trend
    const trendEl = document.getElementById(`${prefix}FlowTrend`);
    if (trendEl) {
        const trend = skillData.trend || 'stable';
        trendEl.className = `skill-flow-trend ${trend}`;
        if (trend === 'improving') {
            trendEl.textContent = '↑';
        } else if (trend === 'declining') {
            trendEl.textContent = '↓';
        } else {
            trendEl.textContent = '→';
        }
    }

    // Update eval count
    setText(`${prefix}FlowEvals`, `${skillData.eval_count || 0} evals`);

    // Mark active if has evals
    if (cardEl) {
        if (skillData.eval_count > 0) {
            cardEl.classList.remove('inactive');
            cardEl.classList.add('active');
        } else {
            cardEl.classList.add('inactive');
            cardEl.classList.remove('active');
        }
    }
}

/**
 * Compute and set curriculum action hint
 */
function updateCurriculumActionHint(dataFlow) {
    let actionText = '';
    let actionLevel = 'good';

    const syllo = dataFlow.skills?.syllo;
    const binary = dataFlow.skills?.binary;

    // Check SYLLO status (primary skill)
    if (syllo) {
        const acc = syllo.recent_accuracy || 0;
        const progress = syllo.progress_to_next || 0;
        const level = syllo.current_level || 1;
        const evalCount = syllo.eval_count || 0;

        if (evalCount === 0) {
            actionText = 'SYLLO curriculum starting. Awaiting first evaluation.';
            actionLevel = 'warning';
        } else if (acc === 0 && evalCount > 10) {
            actionText = `SYLLO stuck at 0% after ${evalCount} evals. Check model or data format.`;
            actionLevel = 'critical';
        } else if (acc < 20 && evalCount > 5) {
            actionText = `SYLLO accuracy low (${acc}%). Model struggling with L${level}.`;
            actionLevel = 'warning';
        } else if (progress >= 80) {
            actionText = `SYLLO ready to advance! ${progress}% progress to L${level + 1}.`;
            actionLevel = 'good';
        } else if (syllo.trend === 'improving') {
            actionText = `SYLLO improving. ${progress}% to L${level + 1}.`;
            actionLevel = 'good';
        } else if (syllo.trend === 'declining') {
            actionText = `SYLLO declining. Check recent training data.`;
            actionLevel = 'warning';
        } else {
            actionText = `SYLLO L${level}: ${acc}% accuracy, ${progress}% to advance.`;
            actionLevel = 'good';
        }
    } else {
        actionText = 'Curriculum not initialized.';
        actionLevel = 'warning';
    }

    setActionHint('curriculumAction', actionText, actionLevel);
}

/**
 * Update regression monitoring card
 */
function updateRegression(data) {
    const source = data.sources?.regression_monitoring;
    if (!source || source.status !== 'ok') {
        setCardOffline('regressionIndicator');
        setActionHint('regressionAction', 'Regression monitoring unavailable', 'warning');
        return;
    }

    setCardOnline('regressionIndicator');
    const regression = source.data;
    const latest = regression.latest_summary;
    const totalRegressions = regression.total_regressions || 0;

    let regressionDetected = false;
    let lossIncrease = null;
    let accDrop = null;

    if (latest) {
        regressionDetected = latest.regression_detected;
        lossIncrease = latest.loss_increase;
        accDrop = latest.accuracy_drop;

        const detected = regressionDetected ? 'YES' : 'NO';
        setText('regressionDetected', detected);
        document.getElementById('regressionDetected').style.color =
            regressionDetected ? 'var(--color-danger)' : 'var(--color-success)';

        setText('regressionLossInc', lossIncrease ?
            `${lossIncrease.toFixed(1)}%` : '--');
        setText('regressionAccDrop', accDrop ?
            `${accDrop.toFixed(1)}%` : '--');

        // Color loss increase based on severity
        const lossIncEl = document.getElementById('regressionLossInc');
        if (lossIncEl && lossIncrease !== null) {
            if (lossIncrease > 15) {
                lossIncEl.style.color = 'var(--color-danger)';
            } else if (lossIncrease > 5) {
                lossIncEl.style.color = 'var(--color-warning)';
            } else {
                lossIncEl.style.color = 'var(--text-primary)';
            }
        }

        // Color accuracy drop based on severity
        const accDropEl = document.getElementById('regressionAccDrop');
        if (accDropEl && accDrop !== null) {
            if (accDrop > 10) {
                accDropEl.style.color = 'var(--color-danger)';
            } else if (accDrop > 3) {
                accDropEl.style.color = 'var(--color-warning)';
            } else {
                accDropEl.style.color = 'var(--text-primary)';
            }
        }
    } else {
        setText('regressionDetected', '--');
        setText('regressionLossInc', '--');
        setText('regressionAccDrop', '--');
    }

    setText('regressionTotal', totalRegressions);

    // Compute action hint
    updateRegressionActionHint(regressionDetected, lossIncrease, accDrop, totalRegressions);
}

/**
 * Compute and set regression action hint
 */
function updateRegressionActionHint(detected, lossIncrease, accDrop, totalRegressions) {
    let actionText = '';
    let actionLevel = 'good';

    if (detected) {
        // Regression detected - severity based on magnitude
        if (lossIncrease !== null && lossIncrease > 15) {
            actionText = `Severe regression: ${lossIncrease.toFixed(1)}% loss increase. Consider rolling back checkpoint.`;
            actionLevel = 'critical';
        } else if (accDrop !== null && accDrop > 10) {
            actionText = `Significant accuracy drop: ${accDrop.toFixed(1)}%. Investigate recent training data.`;
            actionLevel = 'critical';
        } else {
            actionText = 'Minor regression detected. Monitor next evaluation.';
            actionLevel = 'warning';
        }
    } else if (totalRegressions > 5) {
        actionText = `${totalRegressions} total regressions this run. Training may be unstable.`;
        actionLevel = 'warning';
    } else if (totalRegressions > 0) {
        actionText = `No current regression. ${totalRegressions} historical regression(s).`;
        actionLevel = 'good';
    } else {
        actionText = 'No regressions detected. Model improving steadily.';
        actionLevel = 'good';
    }

    setActionHint('regressionAction', actionText, actionLevel);
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
 * Update SYLLO 10-level bars from syllo_10level data
 */
function updateSyllo10LevelBars(data) {
    const tenLevel = data.sources?.skill_metrics?.data?.syllo_10level;
    if (!tenLevel?.available) return;

    const byLevel = tenLevel.by_level || {};

    // Update L1-L4 directly
    ['L1', 'L2', 'L3', 'L4'].forEach(lvl => {
        const levelData = byLevel[lvl] || {};
        const acc = levelData.accuracy || 0;

        const barEl = document.getElementById(`syllable${lvl}Bar`);
        if (barEl) {
            barEl.style.width = `${Math.min(100, acc)}%`;
        }
        setText(`syllable${lvl}`, `${acc.toFixed(0)}%`);
    });

    // L5+ is average of L5-L10
    let l5PlusTotal = 0, l5PlusCount = 0;
    ['L5', 'L6', 'L7', 'L8', 'L9', 'L10'].forEach(lvl => {
        const levelData = byLevel[lvl];
        if (levelData && levelData.total > 0) {
            l5PlusTotal += levelData.accuracy || 0;
            l5PlusCount++;
        }
    });
    const l5PlusAcc = l5PlusCount > 0 ? l5PlusTotal / l5PlusCount : 0;

    const l5Bar = document.getElementById('syllableL5Bar');
    if (l5Bar) {
        l5Bar.style.width = `${Math.min(100, l5PlusAcc)}%`;
    }
    setText('syllableL5', `${l5PlusAcc.toFixed(0)}%`);

    // Update overall accuracy from 10-level data
    setText('syllableOverall', `${tenLevel.overall_accuracy?.toFixed(1) || 0}%`);

    // Update indicator
    setCardOnline('syllableIndicator');
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
        <div class="q-file queue-file" title="Click to preview: ${f.name}" onclick="openPreview('${f.name}')">
            <div class="q-dot ${f.priority || 'normal'}"></div>
            <div class="q-info">
                <div class="q-name queue-file-name">${f.name}</div>
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

// ===== Daemon Status Badge & Dropdown =====

let daemonDropdownOpen = false;
let allDaemonProcesses = []; // Store all processes for dropdown

/**
 * Toggle the daemon status dropdown
 */
function toggleDaemonDropdown() {
    const dropdown = document.getElementById('daemonDropdown');
    if (!dropdown) return;

    daemonDropdownOpen = !daemonDropdownOpen;
    dropdown.classList.toggle('open', daemonDropdownOpen);

    // Close on outside click
    if (daemonDropdownOpen) {
        setTimeout(() => {
            document.addEventListener('click', closeDaemonDropdownOnOutsideClick);
        }, 10);
    } else {
        document.removeEventListener('click', closeDaemonDropdownOnOutsideClick);
    }
}

/**
 * Close dropdown when clicking outside
 */
function closeDaemonDropdownOnOutsideClick(event) {
    const wrapper = document.getElementById('daemonStatusWrapper');
    if (wrapper && !wrapper.contains(event.target)) {
        const dropdown = document.getElementById('daemonDropdown');
        if (dropdown) {
            dropdown.classList.remove('open');
            daemonDropdownOpen = false;
        }
        document.removeEventListener('click', closeDaemonDropdownOnOutsideClick);
    }
}

/**
 * Update daemon status badge and dropdown content
 */
function updateDaemonStatusBadge(data) {
    const systemHealth = data.sources?.system_health?.data;
    if (!systemHealth) return;

    const badge = document.getElementById('daemonStatusBadge');
    const iconEl = document.getElementById('daemonBadgeIcon');
    const textEl = document.getElementById('daemonBadgeText');
    const contentEl = document.getElementById('daemonDropdownContent');

    if (!badge || !iconEl || !textEl || !contentEl) return;

    // Get all processes and count status
    const processes = systemHealth.processes || [];
    allDaemonProcesses = processes;

    const running = processes.filter(p => p.running).length;
    const down = processes.filter(p => !p.running).length;
    const total = processes.length;

    // Update badge appearance
    badge.classList.remove('healthy', 'warning', 'critical');

    if (down === 0) {
        badge.classList.add('healthy');
        iconEl.textContent = '✓';
        textEl.textContent = `${running}/${total} OK`;
    } else if (down <= 2) {
        badge.classList.add('warning');
        iconEl.textContent = '⚠';
        textEl.textContent = `${down} down`;
    } else {
        badge.classList.add('critical');
        iconEl.textContent = '🚨';
        textEl.textContent = `${down} down`;
    }

    // Build dropdown content grouped by machine
    const byMachine = {};
    for (const proc of processes) {
        const machine = proc.machine || 'unknown';
        if (!byMachine[machine]) byMachine[machine] = [];
        byMachine[machine].push(proc);
    }

    let html = '';

    // Sort machines: 4090 first, then 3090
    const machines = Object.keys(byMachine).sort((a, b) => {
        if (a === '4090') return -1;
        if (b === '4090') return 1;
        return a.localeCompare(b);
    });

    for (const machine of machines) {
        const procs = byMachine[machine];
        const machineDown = procs.filter(p => !p.running).length;
        const machineTotal = procs.length;

        html += `<div class="daemon-machine-section">`;
        html += `<div class="daemon-machine-header m-${machine}">`;
        html += `<span>RTX ${machine}</span>`;
        html += `<span style="margin-left:auto; font-weight:400;">${machineTotal - machineDown}/${machineTotal}</span>`;
        html += `</div>`;

        // Sort: down processes first
        procs.sort((a, b) => {
            if (!a.running && b.running) return -1;
            if (a.running && !b.running) return 1;
            return a.name.localeCompare(b.name);
        });

        for (const proc of procs) {
            const statusClass = proc.running ? 'running' : 'down';
            const icon = proc.running ? '✓' : '✗';
            const statusIcon = proc.running ? '●' : '○';

            html += `<div class="daemon-item ${statusClass}" onclick="scrollToErrorCard('${proc.machine}', '${proc.name}')">`;
            html += `<span class="daemon-item-icon">${proc.running ? '✓' : '✗'}</span>`;
            html += `<div class="daemon-item-info">`;
            html += `<div class="daemon-item-name">${formatProcessName(proc.name)}</div>`;
            html += `<div class="daemon-item-meta">`;
            html += `<span class="daemon-item-status ${statusClass}">${statusIcon} ${proc.running ? 'Running' : 'Down'}</span>`;
            if (proc.running && proc.uptime) {
                html += `<span class="daemon-item-uptime">${proc.uptime}</span>`;
            }
            if (!proc.running && proc.error) {
                html += `<span style="color: var(--color-danger); font-size: 0.7rem;">${proc.error}</span>`;
            }
            html += `</div></div></div>`;
        }

        html += `</div>`;
    }

    if (processes.length === 0) {
        html = `<div class="daemon-dropdown-empty">No daemon data available</div>`;
    }

    contentEl.innerHTML = html;
}

/**
 * Scroll to and highlight an error card for a specific daemon
 */
function scrollToErrorCard(machine, processName) {
    // Find the error card for this daemon
    const cardId = `error-card-process_down_${machine}_${processName}`;
    let card = document.getElementById(cardId);

    // Try alternate ID format
    if (!card) {
        const altCardId = `error-card-process_error_${machine}_${processName}`;
        card = document.getElementById(altCardId);
    }

    if (card) {
        // Close dropdown
        toggleDaemonDropdown();

        // Scroll into view and highlight
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        card.style.transform = 'scale(1.02)';
        card.style.boxShadow = '0 0 20px rgba(239, 68, 68, 0.5)';

        setTimeout(() => {
            card.style.transform = '';
            card.style.boxShadow = '';
        }, 1500);
    } else {
        // No error card (daemon might be running) - just close dropdown
        toggleDaemonDropdown();
    }
}

// ================================
// NEW: Scheduler Card Update
// ================================
function updateScheduler(data) {
    const scheduler = data.sources?.scheduler?.data || {};
    const indicator = document.getElementById('schedulerIndicator');

    if (!scheduler.available) {
        indicator.style.color = 'var(--text-muted)';
        setActionHint('schedulerAction', 'Scheduler not responding', 'critical');
        return;
    }

    // Update metrics
    document.getElementById('schedulerQueue').textContent = scheduler.queue_length || 0;
    document.getElementById('schedulerActiveTask').textContent = scheduler.active_task || 'idle';
    document.getElementById('schedulerGpuUtil').textContent = `${Math.round(scheduler.gpu_utilization || 0)}%`;
    document.getElementById('schedulerCompleted').textContent = scheduler.tasks_completed || 0;

    // Update utilization band status
    const bandEl = document.getElementById('schedulerBandStatus');
    const util = scheduler.gpu_utilization || 0;
    const queueLen = scheduler.queue_length || 0;

    bandEl.className = 'scheduler-band';
    let bandText = '';
    let actionLevel = 'good';
    let actionText = 'Operating normally (20-80% GPU utilization)';

    if (util >= 20 && util <= 80) {
        bandEl.classList.add('band-good');
        bandText = `In target band (${Math.round(util)}%)`;
        indicator.style.color = 'var(--color-success)';
    } else if (util < 20 && queueLen > 0) {
        bandEl.classList.add('band-warning');
        bandText = `Underutilized (${Math.round(util)}%)`;
        actionLevel = 'warning';
        actionText = 'GPU underutilized with tasks queued. Check scheduler logs.';
        indicator.style.color = 'var(--color-warning)';
    } else if (util > 80) {
        bandEl.classList.add('band-warning');
        bandText = `High utilization (${Math.round(util)}%)`;
        actionLevel = 'warning';
        actionText = 'GPU near saturation. Queue processing may slow.';
        indicator.style.color = 'var(--color-warning)';
    } else {
        bandEl.classList.add('band-good');
        bandText = `Idle (${Math.round(util)}%)`;
        indicator.style.color = 'var(--color-success)';
    }

    // Check for failures
    if (scheduler.tasks_failed > 0) {
        actionLevel = 'warning';
        actionText = `${scheduler.tasks_failed} failed task(s). Check logs.`;
    }

    bandEl.querySelector('.band-text').textContent = bandText;
    setActionHint('schedulerAction', actionText, actionLevel);
}

// ================================
// NEW: Retention Card Update
// ================================
function updateRetention(data) {
    const retention = data.sources?.retention?.data || {};
    const indicator = document.getElementById('retentionIndicator');

    if (!retention.total_size_gb && retention.total_size_gb !== 0) {
        indicator.style.color = 'var(--text-muted)';
        setActionHint('retentionAction', 'Retention data unavailable', 'warning');
        return;
    }

    const usagePct = retention.usage_pct || 0;
    const totalGb = retention.total_size_gb || 0;
    const limitGb = retention.limit_gb || 150;
    const checkpoints = retention.checkpoints?.count || 0;
    const snapshots = retention.snapshots?.count || 0;

    // Update metrics
    document.getElementById('retentionUsage').textContent = `${Math.round(usagePct)}%`;
    document.getElementById('retentionCheckpoints').textContent = checkpoints;
    document.getElementById('retentionSnapshots').textContent = snapshots;

    // Update bar
    const bar = document.getElementById('retentionBar');
    bar.style.width = `${Math.min(usagePct, 100)}%`;

    // Color bar based on usage
    if (usagePct >= 90) {
        bar.style.background = 'var(--color-danger)';
    } else if (usagePct >= 80) {
        bar.style.background = 'var(--color-warning)';
    } else {
        bar.style.background = 'linear-gradient(90deg, var(--color-success), var(--color-warning))';
    }

    document.getElementById('retentionBarLabel').textContent = `${totalGb.toFixed(1)} / ${limitGb} GB`;

    // Update warnings
    const warningsEl = document.getElementById('retentionWarnings');
    const warnings = retention.warnings || [];
    warningsEl.innerHTML = warnings.map(w => `<div class="warning-item">${w}</div>`).join('');

    // Set indicator and action hint
    let actionText = 'Storage healthy';
    let actionLevel = 'good';

    if (usagePct >= 95) {
        indicator.style.color = 'var(--color-danger)';
        actionLevel = 'critical';
        actionText = 'Storage critical! Cleanup needed immediately.';
    } else if (usagePct >= 80) {
        indicator.style.color = 'var(--color-warning)';
        actionLevel = 'warning';
        actionText = 'Storage getting full. Consider cleanup or NAS sync.';
    } else {
        indicator.style.color = 'var(--color-success)';
    }

    setActionHint('retentionAction', actionText, actionLevel);
}

// ================================
// NEW: Data Impact Card Update
// ================================
function updateDataImpact(data) {
    const analytics = data.sources?.training_analytics?.data || {};
    const indicator = document.getElementById('dataImpactIndicator');
    const fileImpact = analytics.data_file_impact || {};

    if (!fileImpact.available) {
        indicator.style.color = 'var(--text-muted)';
        document.getElementById('positiveImpactFiles').innerHTML = '<li class="impact-loading">No data available</li>';
        document.getElementById('negativeImpactFiles').innerHTML = '<li class="impact-loading">No data available</li>';
        setActionHint('dataImpactAction', 'Data impact analysis not running', 'warning');
        return;
    }

    indicator.style.color = 'var(--color-success)';

    const recentImpacts = fileImpact.recent_impacts || [];

    // Sort by impact score
    const sorted = [...recentImpacts].sort((a, b) => (b.impact_score || 0) - (a.impact_score || 0));

    // Top 5 positive (highest positive impact)
    const positive = sorted.filter(f => (f.impact_score || 0) > 0).slice(0, 5);
    // Top 5 negative (most negative impact)
    const negative = sorted.filter(f => (f.impact_score || 0) < 0).slice(-5).reverse();

    const positiveEl = document.getElementById('positiveImpactFiles');
    const negativeEl = document.getElementById('negativeImpactFiles');

    if (positive.length > 0) {
        positiveEl.innerHTML = positive.map(f => {
            const name = f.filename || 'unknown';
            const shortName = name.length > 25 ? name.slice(0, 22) + '...' : name;
            const score = (f.impact_score || 0).toFixed(3);
            return `<li title="${name}">${shortName} (+${score})</li>`;
        }).join('');
    } else {
        positiveEl.innerHTML = '<li class="impact-loading">No positive files</li>';
    }

    if (negative.length > 0) {
        negativeEl.innerHTML = negative.map(f => {
            const name = f.filename || 'unknown';
            const shortName = name.length > 25 ? name.slice(0, 22) + '...' : name;
            const score = (f.impact_score || 0).toFixed(3);
            return `<li title="${name}">${shortName} (${score})</li>`;
        }).join('');
    } else {
        negativeEl.innerHTML = '<li class="impact-loading">No negative files</li>';
    }

    // Action hint
    if (negative.length > 0) {
        setActionHint('dataImpactAction', `${negative.length} file(s) have negative impact. Consider removing or re-labeling.`, 'warning');
    } else {
        setActionHint('dataImpactAction', 'All analyzed files have neutral or positive impact.', 'good');
    }
}

// ================================
// Helper: Set action hint with level
// ================================
function setActionHint(elementId, text, level = 'neutral') {
    const el = document.getElementById(elementId);
    if (!el) return;

    el.textContent = text;
    el.className = 'action-hint';

    if (level === 'good') el.classList.add('action-good');
    else if (level === 'warning') el.classList.add('action-warning');
    else if (level === 'critical') el.classList.add('action-critical');
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

// ===== File Preview Modal =====

let previewData = null;
let previewIndex = 0;
let previewRotationInterval = null;

/**
 * Open the preview modal for a queue file
 */
async function openPreview(filename) {
    const modal = document.getElementById('previewModal');
    if (!modal) return;

    // Show modal with loading state
    modal.style.display = 'flex';
    document.getElementById('previewFileName').textContent = filename;
    document.getElementById('previewCounter').textContent = 'Loading...';
    document.getElementById('previewPrompt').textContent = 'Loading examples...';
    document.getElementById('previewResponse').textContent = '';
    document.getElementById('previewTotalExamples').textContent = '-- examples';
    document.getElementById('previewFormat').textContent = '--';

    try {
        const apiBase = CONFIG.apiUrl.replace('/api/unified', '');
        const response = await fetch(`${apiBase}/api/queue/preview/${encodeURIComponent(filename)}?count=10`);
        const data = await response.json();

        if (data.error) {
            document.getElementById('previewPrompt').textContent = `Error: ${data.error}`;
            document.getElementById('previewResponse').textContent = '';
            return;
        }

        previewData = data;
        previewIndex = 0;

        document.getElementById('previewTotalExamples').textContent = `${data.total_examples.toLocaleString()} examples`;
        displayPreviewItem();

        // Start auto-rotation every 5 seconds
        startPreviewRotation();

    } catch (error) {
        document.getElementById('previewPrompt').textContent = `Fetch error: ${error.message}`;
        document.getElementById('previewResponse').textContent = '';
    }
}

/**
 * Display the current preview item
 */
function displayPreviewItem() {
    if (!previewData || !previewData.previews || previewData.previews.length === 0) return;

    const item = previewData.previews[previewIndex];
    document.getElementById('previewCounter').textContent = `${previewIndex + 1}/${previewData.previews.length}`;
    document.getElementById('previewPrompt').textContent = item.prompt || '(no prompt)';
    document.getElementById('previewResponse').textContent = item.response || '(no response)';
    document.getElementById('previewFormat').textContent = item.format || 'unknown';
}

/**
 * Go to next preview item
 */
function nextPreview() {
    if (!previewData || !previewData.previews) return;
    previewIndex = (previewIndex + 1) % previewData.previews.length;
    displayPreviewItem();
    restartPreviewRotation();
}

/**
 * Go to previous preview item
 */
function prevPreview() {
    if (!previewData || !previewData.previews) return;
    previewIndex = (previewIndex - 1 + previewData.previews.length) % previewData.previews.length;
    displayPreviewItem();
    restartPreviewRotation();
}

/**
 * Shuffle and reload previews
 */
async function shufflePreview() {
    if (!previewData) return;
    const filename = previewData.filename;
    await openPreview(filename);
}

/**
 * Start auto-rotation of preview items
 */
function startPreviewRotation() {
    stopPreviewRotation();
    previewRotationInterval = setInterval(() => {
        if (previewData && previewData.previews && previewData.previews.length > 1) {
            previewIndex = (previewIndex + 1) % previewData.previews.length;
            displayPreviewItem();
        }
    }, 5000); // Rotate every 5 seconds
}

/**
 * Stop auto-rotation
 */
function stopPreviewRotation() {
    if (previewRotationInterval) {
        clearInterval(previewRotationInterval);
        previewRotationInterval = null;
    }
}

/**
 * Restart rotation (when user manually navigates)
 */
function restartPreviewRotation() {
    startPreviewRotation();
}

/**
 * Close the preview modal
 */
function closePreview() {
    const modal = document.getElementById('previewModal');
    if (modal) {
        modal.style.display = 'none';
    }
    stopPreviewRotation();
    previewData = null;
    previewIndex = 0;
}

// Close modal on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closePreview();
    }
    // Arrow key navigation when modal is open
    const modal = document.getElementById('previewModal');
    if (modal && modal.style.display !== 'none') {
        if (e.key === 'ArrowRight') nextPreview();
        if (e.key === 'ArrowLeft') prevPreview();
    }
});

// Close modal when clicking outside content
document.addEventListener('click', (e) => {
    const modal = document.getElementById('previewModal');
    if (e.target === modal) {
        closePreview();
    }
});

// ================================
// Data Lineage Card
// ================================

const LINEAGE_API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8081/api/lineage'
    : `http://${window.location.hostname}:8081/api/lineage`;

/**
 * Fetch data lineage stats from dedicated API endpoint
 */
async function fetchDataLineage() {
    try {
        const response = await fetch(LINEAGE_API_URL);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        updateDataLineage(data);
    } catch (error) {
        console.error('Error fetching lineage data:', error);
        setCardOffline('lineageIndicator');
    }
}

/**
 * Update the Data Lineage card with generator/validator stats
 */
function updateDataLineage(data) {
    const indicator = document.getElementById('lineageIndicator');
    if (!indicator) return;

    // Check if we have data
    const totalValidations = data.total_validations || 0;
    const generators = data.generators || {};
    const validators = data.validators || {};
    const summary = data.summary || {};

    if (totalValidations === 0) {
        indicator.className = 'status-indicator offline';
        document.getElementById('lineageTotalValidations').textContent = '0';
        document.getElementById('lineageOverallFailRate').textContent = '--%';
        return;
    }

    // Determine card status based on overall fail rate
    const overallFailRate = summary.overall_fail_rate || 0;
    if (overallFailRate > 10) {
        indicator.className = 'status-indicator critical';
    } else if (overallFailRate > 5) {
        indicator.className = 'status-indicator warning';
    } else {
        indicator.className = 'status-indicator healthy';
    }

    // Update summary stats
    document.getElementById('lineageTotalValidations').textContent = totalValidations.toLocaleString();
    document.getElementById('lineageOverallFailRate').textContent = `${overallFailRate.toFixed(1)}%`;

    // Update generator table
    const genBody = document.getElementById('generatorTableBody');
    if (genBody) {
        if (Object.keys(generators).length === 0) {
            genBody.innerHTML = '<tr><td colspan="3" class="empty-row">No data yet</td></tr>';
        } else {
            // Sort by total descending
            const sorted = Object.entries(generators)
                .sort((a, b) => b[1].total - a[1].total)
                .slice(0, 5); // Top 5

            genBody.innerHTML = sorted.map(([key, stats]) => {
                const shortKey = key.split('@')[0]; // Remove version for brevity
                const failRate = stats.fail_rate || 0;
                const failClass = getFailRateClass(failRate);
                return `<tr>
                    <td title="${key}">${shortKey}</td>
                    <td>${stats.total}</td>
                    <td class="${failClass}">${failRate.toFixed(1)}%</td>
                </tr>`;
            }).join('');
        }
    }

    // Update validator table
    const valBody = document.getElementById('validatorTableBody');
    if (valBody) {
        if (Object.keys(validators).length === 0) {
            valBody.innerHTML = '<tr><td colspan="3" class="empty-row">No data yet</td></tr>';
        } else {
            // Sort by total descending
            const sorted = Object.entries(validators)
                .sort((a, b) => b[1].total - a[1].total)
                .slice(0, 5); // Top 5

            valBody.innerHTML = sorted.map(([key, stats]) => {
                const shortKey = key.split('@')[0]; // Remove version for brevity
                const failRate = stats.fail_rate || 0;
                const failClass = getFailRateClass(failRate);
                return `<tr>
                    <td title="${key}">${shortKey}</td>
                    <td>${stats.total}</td>
                    <td class="${failClass}">${failRate.toFixed(1)}%</td>
                </tr>`;
            }).join('');
        }
    }

    // Update worst generator/validator warning
    const worstEl = document.getElementById('lineageWorst');
    if (worstEl) {
        const worstGen = summary.worst_generator;
        const worstVal = summary.worst_validator;

        if (worstGen && worstGen.fail_rate > 5) {
            worstEl.innerHTML = `⚠️ <strong>${worstGen.id}</strong> has ${worstGen.fail_rate.toFixed(1)}% rejection rate (${worstGen.total} validations)`;
            worstEl.classList.add('visible');
        } else if (worstVal && worstVal.fail_rate > 5) {
            worstEl.innerHTML = `⚠️ <strong>${worstVal.id}</strong> validator rejecting ${worstVal.fail_rate.toFixed(1)}% (${worstVal.total} validations)`;
            worstEl.classList.add('visible');
        } else {
            worstEl.classList.remove('visible');
        }
    }
}

/**
 * Get CSS class for fail rate coloring
 */
function getFailRateClass(rate) {
    if (rate > 10) return 'fail-rate-bad';
    if (rate > 5) return 'fail-rate-warn';
    return 'fail-rate-ok';
}
