/**
 * REALM OF TRAINING - Game UI JavaScript
 *
 * Handles:
 * - Live data fetching from the unified API
 * - Idle game mechanics (XP ticks, resource accumulation)
 * - Animations and visual feedback
 * - Battle status updates
 */

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
    API_BASE: '/api',
    UPDATE_INTERVAL: 1000,      // Main update every 1 second
};

// ============================================
// GAME STATE
// ============================================

const GameState = {
    // Hero
    totalLevel: 0,      // Sum of all MASTERED skill levels
    currentStep: 0,
    previousStep: 0,
    totalEvals: 0,      // Real: total skill evaluations

    // Current skill being trained
    currentSkill: 'BINARY',
    currentSkillMastered: 0,   // Highest level mastered
    currentSkillTraining: 1,   // Level being trained (mastered + 1)
    currentSkillAcc: 0,

    // Training
    isTraining: false,
    currentQuest: null,
    questProgress: 0,
    totalSteps: 0,
    loss: 0,            // Strain
    valLoss: 0,         // Val Strain
    perplexity: 0,      // For Clarity calculation
    lossHistory: [],    // Loss history for graph
    stepsPerSecond: 0,
    etaSeconds: 0,
    learningRate: 0,

    // Skills (real curriculum data) - MASTERED levels
    sylloMastered: 0,
    sylloTraining: 1,
    sylloAcc: 0,
    sylloEvals: 0,
    binaryMastered: 0,
    binaryTraining: 1,
    binaryAcc: 0,
    binaryEvals: 0,

    // GPU (real hardware stats)
    vramUsed: 0,
    vramTotal: 24,
    gpuTemp: 0,

    // Vault (real checkpoint data)
    checkpointCount: 0,
    totalSize: 0,
    bestCheckpoint: null,
};

// ============================================
// SCHEMA VALIDATION (POLICY 1)
// ============================================

/**
 * Validate game data (client-side)
 *
 * POLICY 1: Schema Validation at All Boundaries
 * This catches schema mismatches before they cause UI bugs.
 *
 * Supports both formats:
 * - New RealmStore: {state: {training: ...}, events: [], timestamp: "..."}
 * - Old /api/game: {training: ..., gpu: ..., curriculum: ...}
 *
 * @param {object} data - Data from /api/game or /api/realm-state
 * @returns {{valid: boolean, errors: string[]}}
 */
function validateRealmStore(data) {
    const errors = [];

    // Detect format
    const isRealmStore = 'state' in data;
    const isLegacyGame = 'training' in data;

    if (!isRealmStore && !isLegacyGame) {
        errors.push("Data missing both 'state' (RealmStore) and 'training' (/api/game) fields");
        return {valid: false, errors};
    }

    // Get training data from either format
    const training = isRealmStore ? data.state?.training : data.training;

    if (!training) {
        errors.push("Missing training data");
        return {valid: false, errors};
    }

    // Validate training fields (lenient - only check critical ones)
    // status and step are required, updated_at is optional for legacy format
    if (training.status !== undefined && typeof training.status !== 'string') {
        errors.push("training.status must be a string");
    }
    if (training.step !== undefined && typeof training.step !== 'number' && typeof training.current_step !== 'number') {
        errors.push("training.step or current_step must be a number");
    }

    return {
        valid: errors.length === 0,
        errors: errors
    };
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

// Formatting functions - delegate to shared Format module (formatters.js)
// These wrappers maintain backward compatibility with existing code
function formatNumber(num) {
    return Format.compact(num);
}

function formatStrain(loss) {
    return Format.loss(loss);
}

function formatClarity(perplexity) {
    return Format.clarity(perplexity);
}

function formatETA(seconds) {
    return Format.eta(seconds);
}

function formatTime(date) {
    return Format.timestamp(date);
}

// ============================================
// ACTION HINTS - Contextual Advice
// ============================================

// Additional state for hints
const HintState = {
    queueTotal: 0,
    queueHigh: 0,
    queueNormal: 0,
    queueLow: 0,
    inboxCount: 0,
    gpu3090Idle: false,
    gpu3090Available: false,
    taskMasterRunning: false,
    lastTaskSuccess: null,
    valTrainGap: 0,
};

// ============================================
// REALM STATE INTEGRATION
// Sync from unified RealmState to GameState
// ============================================

/**
 * Subscribe to RealmState and sync to GameState.
 * This replaces the old fetchGameData polling for core training status.
 */
function initRealmStateSync() {
    // Check if RealmState is available
    if (typeof RealmState === 'undefined') {
        console.warn('[Game] RealmState not available, falling back to legacy polling');
        return false;
    }

    console.log('[Game] Syncing with RealmState (single source of truth)');

    // Subscribe to training state changes
    RealmState.subscribe('training', (training) => {
        // Sync training data to GameState
        GameState.isTraining = training.status === 'training';
        GameState.currentStep = training.step || 0;
        GameState.totalSteps = training.totalSteps || 0;
        GameState.loss = training.loss || 0;
        GameState.learningRate = training.learningRate || 0;
        GameState.currentQuest = training.file || null;
        GameState.stepsPerSecond = training.speed || 0;
        GameState.etaSeconds = training.etaSeconds || 0;

        // Calculate quest progress
        if (GameState.totalSteps > 0) {
            GameState.questProgress = Math.round((GameState.currentStep / GameState.totalSteps) * 100);
        }

        // Update UI
        updateBattleStatus();
        updateTrainingUI();
    });

    // Subscribe to queue state changes
    RealmState.subscribe('queue', (queue) => {
        HintState.queueTotal = queue.depth || 0;
        HintState.queueHigh = queue.highPriority || 0;
        HintState.queueNormal = queue.normalPriority || 0;
        HintState.queueLow = queue.lowPriority || 0;

        // Update queue warning
        updateQueueWarning(queue.status);
    });

    // Subscribe to events (battle log)
    RealmState.subscribe('events', (events) => {
        // Update battle log display
        renderBattleLogFromRealm(events);
    });

    return true;
}

/**
 * Update queue warning banner based on queue status
 */
function updateQueueWarning(status) {
    const questLabel = document.querySelector('.quest-label');
    if (!questLabel) return;

    if (status === 'empty' || status === 'low') {
        questLabel.innerHTML = '<span class="warning">📦 Queue running low. Prepare more quests.</span>';
    }
}

/**
 * Render battle log from RealmState events
 */
function renderBattleLogFromRealm(events) {
    const container = document.getElementById('logEntries');
    if (!container || !events || events.length === 0) return;

    // Clear and rebuild (events come newest first)
    container.innerHTML = '';

    for (const event of events.slice(0, 30)) {  // Show last 30
        const entry = document.createElement('div');
        entry.className = `log-entry ${event.severity || 'info'}`;

        const time = event.timestamp
            ? new Date(event.timestamp).toLocaleTimeString('en-US', { hour12: false })
            : '--:--:--';

        entry.innerHTML = `${event.icon || '📢'} <span class="log-time">[${time}]</span> <span class="log-channel">${event.channel || 'system'}</span> ${escapeHtmlSafe(event.message || '')}`;
        container.appendChild(entry);
    }
}

/**
 * Safe HTML escaping
 */
function escapeHtmlSafe(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Fetch additional data for hints (called less frequently)
async function fetchHintData() {
    try {
        // Fetch quests data
        const questsRes = await fetch('/api/quests');
        if (questsRes.ok) {
            const quests = await questsRes.json();
            HintState.queueTotal = (quests.high?.length || 0) + (quests.normal?.length || 0) + (quests.low?.length || 0);
            HintState.queueHigh = quests.high?.length || 0;
            HintState.queueNormal = quests.normal?.length || 0;
            HintState.queueLow = quests.low?.length || 0;
            HintState.inboxCount = quests.inbox_count || 0;
        }

        // Fetch task master data
        const taskRes = await fetch('/api/task-master');
        if (taskRes.ok) {
            const task = await taskRes.json();
            HintState.gpu3090Available = task.gpus?.['3090']?.available || false;
            HintState.gpu3090Idle = task.gpus?.['3090']?.is_idle || false;
            HintState.taskMasterRunning = task.daemon_running || false;
            HintState.lastTaskSuccess = task.last_task?.success;
        }
    } catch (err) {
        console.debug('Failed to fetch hint data:', err);
    }
}

// Compute the best hint based on current state
function computeActionHint() {
    const hints = [];

    // Priority 1: Training-specific hints
    if (GameState.isTraining) {
        // Check for overfitting (val loss significantly higher than train loss)
        HintState.valTrainGap = GameState.valLoss - GameState.loss;
        if (GameState.valLoss > 0 && GameState.loss > 0 && HintState.valTrainGap > 0.5) {
            hints.push({ priority: 10, text: '⚠️ High val-train gap. Consider more diverse data.' });
        }

        // High loss warning
        if (GameState.loss > 2.0) {
            hints.push({ priority: 8, text: `💪 Strain is high. ${GameState.heroName || 'Hero'} is struggling with this material.` });
        }

        // Low queue warning during training
        if (HintState.queueTotal < 3) {
            hints.push({ priority: 7, text: '📦 Queue running low. Prepare more quests.' });
        }

        return hints.length > 0 ? hints.sort((a, b) => b.priority - a.priority)[0].text : '';
    }

    // Priority 2: Idle-specific hints
    if (!GameState.isTraining) {
        // Empty queue
        if (HintState.queueTotal === 0 && HintState.inboxCount === 0) {
            hints.push({ priority: 10, text: '📜 No quests queued. Drop training files in inbox/' });
        } else if (HintState.queueTotal === 0 && HintState.inboxCount > 0) {
            hints.push({ priority: 9, text: `📥 ${HintState.inboxCount} files in inbox awaiting processing.` });
        } else if (HintState.queueTotal > 0) {
            hints.push({ priority: 5, text: `📋 ${HintState.queueTotal} quests ready. Start training to continue.` });
        }

        // 3090 status hints
        if (HintState.gpu3090Available && HintState.gpu3090Idle && !HintState.taskMasterRunning) {
            hints.push({ priority: 6, text: '🤖 3090 is idle. Task Master could run sparring.' });
        }

        // Skill accuracy hints
        if (GameState.currentSkillAcc < 50 && GameState.currentSkillAcc > 0) {
            hints.push({ priority: 4, text: `📈 ${GameState.currentSkill} accuracy is ${GameState.currentSkillAcc.toFixed(0)}%. More training needed.` });
        }
    }

    // Return highest priority hint
    if (hints.length > 0) {
        hints.sort((a, b) => b.priority - a.priority);
        return hints[0].text;
    }

    return '';
}

// Update the action hint display
function updateActionHint() {
    const hint = computeActionHint();
    const el = document.getElementById('actionHint');
    if (el) {
        el.textContent = hint;
        el.style.display = hint ? 'block' : 'none';
    }
}

// ============================================
// LOSS CHART RENDERING
// ============================================

function renderLossChart(lossHistory) {
    const canvas = document.getElementById('lossChart');
    if (!canvas || !lossHistory || lossHistory.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 4;

    // Clear canvas
    ctx.fillStyle = '#0a0c10';
    ctx.fillRect(0, 0, width, height);

    // Extract effort values (cumulative strain)
    // If effort field doesn't exist (old data), fall back to loss
    const efforts = lossHistory.map(h => h.effort !== undefined ? h.effort : (h.loss || 0));
    const minEffort = Math.min(...efforts);
    const maxEffort = Math.max(...efforts);
    const range = maxEffort - minEffort || 1;

    // Update footer stats
    const minEl = document.getElementById('lossMin');
    const maxEl = document.getElementById('lossMax');
    const trendEl = document.getElementById('lossTrend');
    const rangeEl = document.getElementById('lossGraphRange');

    if (minEl) minEl.textContent = `Min: ${minEffort.toFixed(1)}`;
    if (maxEl) maxEl.textContent = `Max: ${maxEffort.toFixed(1)}`;
    if (rangeEl) rangeEl.textContent = `Last ${efforts.length} steps`;

    // Calculate trend (effort rate: compare first half vs second half slope)
    // Effort should be increasing (it's cumulative), but we want to know if the RATE is increasing or decreasing
    if (efforts.length >= 4) {
        const halfIdx = Math.floor(efforts.length / 2);
        const firstHalfRate = (efforts[halfIdx] - efforts[0]) / halfIdx;  // Effort per step in first half
        const secondHalfRate = (efforts[efforts.length - 1] - efforts[halfIdx]) / (efforts.length - halfIdx);  // Effort per step in second half
        const rateDiff = secondHalfRate - firstHalfRate;

        if (trendEl) {
            trendEl.classList.remove('improving', 'degrading', 'stable');
            // Lower effort rate = learning easier = improving
            if (rateDiff < -0.05) {
                trendEl.textContent = '↓ Learning Easier';
                trendEl.classList.add('improving');
            } else if (rateDiff > 0.05) {
                trendEl.textContent = '↑ Struggling More';
                trendEl.classList.add('degrading');
            } else {
                trendEl.textContent = '→ Steady';
                trendEl.classList.add('stable');
            }
        }
    }

    // No threshold line for effort (it's cumulative, always increasing)

    // Draw effort line (cumulative, always increasing)
    ctx.strokeStyle = '#3b82f6';  // Blue for effort
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const stepX = (width - 2 * padding) / (efforts.length - 1 || 1);
    efforts.forEach((effort, i) => {
        const x = padding + i * stepX;
        const y = height - padding - ((effort - minEffort) / range) * (height - 2 * padding);

        // Color based on effort rate (slope)
        if (i > 0) {
            const prevEffort = efforts[i - 1];
            const effortRate = effort - prevEffort;  // Effort added this step

            // Color: green = low effort rate, yellow = medium, red = high
            if (effortRate < 0.3) {
                ctx.strokeStyle = '#22c55e';  // Green - learning easily
            } else if (effortRate < 0.6) {
                ctx.strokeStyle = '#f59e0b';  // Yellow - moderate effort
            } else {
                ctx.strokeStyle = '#ef4444';  // Red - high effort (struggling)
            }
        }

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // Draw current value marker (last point)
    if (efforts.length > 0) {
        const lastEffort = efforts[efforts.length - 1];
        const lastX = width - padding;
        const lastY = height - padding - ((lastEffort - minEffort) / range) * (height - 2 * padding);

        // Color marker based on recent effort rate
        const recentEffortRate = efforts.length > 1 ? (lastEffort - efforts[efforts.length - 2]) : 0;
        ctx.fillStyle = recentEffortRate < 0.3 ? '#22c55e' : recentEffortRate < 0.6 ? '#f59e0b' : '#ef4444';
        ctx.beginPath();
        ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
        ctx.fill();
    }
}

// ============================================
// DOM HELPERS
// ============================================

function $(selector) {
    return document.querySelector(selector);
}

function $$(selector) {
    return document.querySelectorAll(selector);
}

function setText(selector, text) {
    const el = $(selector);
    if (el) el.textContent = text;
}

function setWidth(selector, percent) {
    const el = $(selector);
    if (el) el.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

// ============================================
// UI UPDATE FUNCTIONS
// ============================================

function updateHeader() {
    setText('#totalSteps', formatNumber(GameState.currentStep));
    setText('#totalEvals', formatNumber(GameState.totalEvals));
}

function updateHeroStats() {
    // Total level (sum of all MASTERED skills)
    setText('#totalLevel', GameState.totalLevel);

    // Current skill being trained - show training level (what they're working on)
    // Include skill icon if available from skill_context
    const skillDisplay = GameState.skillIcon
        ? `${GameState.skillIcon} ${GameState.currentSkill}`
        : GameState.currentSkill;
    setText('#currentSkillName', skillDisplay);
    setText('#currentSkillLevel', GameState.currentSkillTraining);

    // Stats are for current skill - show target if available
    const accDisplay = GameState.currentSkillAcc
        ? `${GameState.currentSkillAcc.toFixed(1)}%`
        : '0%';
    setText('#statAcc', accDisplay);

    // Add tooltip showing target accuracy if available
    const statAccEl = document.querySelector('#statAcc')?.parentElement;
    if (statAccEl && GameState.skillTargetAcc) {
        statAccEl.title = `Target: ${GameState.skillTargetAcc.toFixed(0)}% for level-up`;
    }

    setText('#statClarity', formatClarity(GameState.perplexity));
    setText('#statStrain', formatStrain(GameState.loss));
}

function updateBattleStatus() {
    const battleStatus = $('#battleStatus');
    const idleIndicator = $('#idleIndicator');

    if (GameState.isTraining) {
        // Training mode
        battleStatus.classList.add('fighting');
        idleIndicator.classList.add('active');
        setText('.idle-icon', '⚔️');
        setText('.idle-text', 'TRAINING');

        setText('#battleIcon', '⚔️');
        setText('#battleTitle', 'Training in Progress');

        const questName = GameState.currentQuest || 'Unknown file';
        setText('#questName', questName.length > 50 ? questName.slice(0, 47) + '...' : questName);

        setWidth('#questProgressBar', GameState.questProgress);
        setText('#questProgressText', `${GameState.questProgress.toFixed(1)}%`);

        setText('#battleStep', `${GameState.currentStep.toLocaleString()}/${GameState.totalSteps.toLocaleString()}`);
        setText('#battleSpeed', GameState.stepsPerSecond > 0 ? `${GameState.stepsPerSecond.toFixed(2)}/s` : '--');
        setText('#battleStrain', formatStrain(GameState.loss));
        setText('#battleETA', formatETA(GameState.etaSeconds));

    } else {
        // Idle mode
        battleStatus.classList.remove('fighting');
        idleIndicator.classList.remove('active');
        setText('.idle-icon', '💤');
        setText('.idle-text', 'IDLE');

        setText('#battleIcon', '💤');
        setText('#battleTitle', 'Awaiting Orders');
        setText('#questName', 'Drop training files in inbox/');

        setWidth('#questProgressBar', 0);
        setText('#questProgressText', '0%');

        setText('#battleStep', '--');
        setText('#battleSpeed', '--');
        setText('#battleStrain', '--');
        setText('#battleETA', '--');
    }

    // Update control buttons
    updateControlButtons();
}

// ============================================
// SKILLS (Dynamic from YAML configs)
// ============================================

let skillsData = [];
let skillsLoaded = false;

async function fetchSkills() {
    try {
        const response = await fetch(`/skills?_t=${Date.now()}`, {
            cache: 'no-store'
        });
        if (!response.ok) return;

        const data = await response.json();
        if (data.skills && data.skills.length > 0) {
            skillsData = data.skills;
            GameState.totalLevel = data.total_mastered || 0;

            // Update individual skill state for backward compat
            for (const skill of data.skills) {
                if (skill.id === 'sy' || skill.id === 'syllo') {
                    GameState.sylloMastered = skill.mastered_level;
                    GameState.sylloTraining = skill.training_level;
                    GameState.sylloAcc = skill.accuracy;
                    GameState.sylloEvals = skill.eval_count;
                } else if (skill.id === 'bin' || skill.id === 'binary') {
                    GameState.binaryMastered = skill.mastered_level;
                    GameState.binaryTraining = skill.training_level;
                    GameState.binaryAcc = skill.accuracy;
                    GameState.binaryEvals = skill.eval_count;
                }
            }

            renderSkills();
            skillsLoaded = true;
        }
    } catch (error) {
        console.error('Failed to fetch skills:', error);
    }
}

function renderSkills() {
    const container = $('#skillsContainer');
    if (!container || skillsData.length === 0) return;

    container.innerHTML = skillsData.map(skill => {
        const progressPct = skill.max_level > 0
            ? (skill.mastered_level / skill.max_level) * 100
            : 0;

        const isActive = skill.id === GameState.currentSkill?.toLowerCase() ||
                        skill.short_name === GameState.currentSkill;

        return `
            <div class="skill-card clickable ${isActive ? 'active' : ''}" data-skill="${skill.id}" style="--skill-color: ${skill.color}">
                <div class="skill-header">
                    <span class="skill-icon">${skill.icon}</span>
                    <span class="skill-name">${skill.short_name}</span>
                    <span class="skill-level">L${skill.mastered_level}/${skill.max_level}</span>
                </div>
                <div class="skill-bar-container">
                    <div class="skill-bar" style="width: ${progressPct}%; background: ${skill.color}"></div>
                </div>
                <div class="skill-meta">
                    <span class="skill-acc">${skill.accuracy.toFixed(1)}%</span>
                    <span class="skill-desc">${skill.rpg_name}</span>
                </div>
                <div class="skill-training">
                    Training L${skill.training_level}
                </div>
            </div>
        `;
    }).join('');

    // Add click handlers for skill detail pages
    container.querySelectorAll('.skill-card').forEach(card => {
        card.addEventListener('click', () => {
            const skillId = card.dataset.skill;
            if (skillId) {
                window.location.href = `/skill/${skillId}`;
            }
        });
    });
}

function updateSkills() {
    // If skills loaded from API, re-render with latest state
    if (skillsLoaded && skillsData.length > 0) {
        // Update skill data from GameState
        for (const skill of skillsData) {
            if (skill.id === 'sy' || skill.id === 'syllo') {
                skill.mastered_level = GameState.sylloMastered;
                skill.training_level = GameState.sylloTraining;
                skill.accuracy = GameState.sylloAcc;
            } else if (skill.id === 'bin' || skill.id === 'binary') {
                skill.mastered_level = GameState.binaryMastered;
                skill.training_level = GameState.binaryTraining;
                skill.accuracy = GameState.binaryAcc;
            }
        }
        renderSkills();
    }
}

function updateVault() {
    setText('#vaultCheckpoints', GameState.checkpointCount || '--');
    setText('#vaultSize', GameState.totalSize ? `${GameState.totalSize.toFixed(1)} GB` : '-- GB');
    setText('#vaultBest', GameState.bestCheckpoint || '--');
}

// Fetch vault data directly (more reliable than unified API)
async function fetchVaultData() {
    try {
        const response = await fetch(`/vault/assets?_t=${Date.now()}`, {
            cache: 'no-store'
        });
        if (!response.ok) return;

        const data = await response.json();
        GameState.checkpointCount = data.checkpoint_count || 0;
        GameState.totalSize = data.total_size_gb || 0;

        // Find best checkpoint
        if (data.checkpoints?.length > 0) {
            const champion = data.checkpoints.find(cp => cp.is_champion);
            if (champion) {
                GameState.bestCheckpoint = champion.name.replace('checkpoint-', '');
            } else {
                GameState.bestCheckpoint = data.checkpoints[0].name.replace('checkpoint-', '');
            }
        }

        updateVault();
    } catch (error) {
        console.error('Failed to fetch vault data:', error);
    }
}

function updateForge() {
    // VRAM
    const vramPercent = GameState.vramTotal > 0 ? (GameState.vramUsed / GameState.vramTotal) * 100 : 0;
    setWidth('#forgeVramBar', vramPercent);
    setText('#forgeVram', GameState.vramUsed ? `${GameState.vramUsed.toFixed(1)} GB` : '-- GB');

    // Temperature
    const tempPercent = Math.min(100, (GameState.gpuTemp / 90) * 100);
    setWidth('#forgeTempBar', tempPercent);
    setText('#forgeTemp', GameState.gpuTemp ? `${Math.round(GameState.gpuTemp)}°C` : '--°C');
}

function updateTime() {
    setText('#timeDisplay', formatTime(new Date()));
}

// ============================================
// BATTLE LOG (from core.battle_log)
// ============================================

// Track displayed events to avoid duplicates
let lastBattleLogTimestamp = null;
let battleLogInitialized = false;
let displayedEventIds = new Set();

async function fetchSaga() {
    // Now uses the battle_log API instead of saga
    try {
        const response = await fetch(`/api/battle_log?limit=30&_t=${Date.now()}`, {
            cache: 'no-store'
        });
        if (!response.ok) return;

        const data = await response.json();
        if (data.events && data.events.length > 0) {
            renderBattleLog(data.events);
        }
    } catch (error) {
        console.error('Failed to fetch battle log:', error);
    }
}

function renderBattleLog(events) {
    const container = $('#logEntries');
    if (!container) return;

    // First load: replace everything
    if (!battleLogInitialized) {
        container.innerHTML = '';
        battleLogInitialized = true;
    }

    // Events come newest-first, we display newest at top
    // Only add events we haven't seen (based on id)
    const newEvents = [];
    for (const event of events) {
        if (displayedEventIds.has(event.id)) {
            continue;  // Already displayed
        }
        newEvents.push(event);
        displayedEventIds.add(event.id);
    }

    // Update timestamp tracker
    if (events.length > 0) {
        lastBattleLogTimestamp = events[0].timestamp;
    }

    // Add new events at the top (reverse to maintain order)
    for (const event of newEvents.reverse()) {
        const entry = document.createElement('div');
        entry.className = `log-entry ${getSeverityClass(event.severity)}`;

        // Format timestamp to just time
        const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString('en-US', { hour12: false }) : '--:--:--';

        entry.innerHTML = `${event.icon || '📢'} <span class="log-time">[${time}]</span> <span class="log-channel">${event.channel}</span> ${escapeHtml(event.message)}`;
        container.insertBefore(entry, container.firstChild);
    }

    // Keep only 30 entries max
    while (container.children.length > 30) {
        const removed = container.lastChild;
        container.removeChild(removed);
    }
}

function getSeverityClass(severity) {
    const classMap = {
        'info': 'info',
        'success': 'success',
        'warning': 'warning',
        'error': 'error',
    };
    return classMap[severity] || 'info';
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Legacy alias for compatibility
function getCategoryClass(category) {
    return getSeverityClass(category);
}

// Local log for immediate feedback (before chronicle catches up)
function addLocalLog(message, type = 'info') {
    const container = $('#logEntries');
    if (!container) return;

    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    const icons = { info: '📝', success: '✅', warning: '⚠️', error: '❌' };
    const icon = icons[type] || '📝';

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `${icon} [${time}] ${message}`;

    container.insertBefore(entry, container.firstChild);

    // Keep only 20 entries
    while (container.children.length > 20) {
        container.removeChild(container.lastChild);
    }
}

// ============================================
// POLICY 3: STALENESS DETECTION
// ============================================

/**
 * Check if training data is stale (>60s old during active training)
 *
 * @param {object} training - Training state from RealmStore
 * @returns {{is_stale: boolean, message: string, age_seconds: number}}
 */
function checkDataFreshness(training) {
    if (!training || !training.updated_at) {
        return {
            is_stale: false,
            message: "No timestamp available",
            age_seconds: 0
        };
    }

    const updatedAt = new Date(training.updated_at);
    const ageSeconds = (Date.now() - updatedAt) / 1000;

    // Only consider stale if training is active AND data is old
    const isTraining = training.status === 'training';
    const isStale = isTraining && ageSeconds > 60;

    if (isStale) {
        return {
            is_stale: true,
            message: `Training data is ${ageSeconds.toFixed(0)}s old - daemon may be stuck`,
            age_seconds: ageSeconds
        };
    }

    return {is_stale: false, message: "", age_seconds: ageSeconds};
}

/**
 * Update staleness warning banner
 */
function updateFreshnessWarning(training) {
    const banner = document.getElementById('dataFreshnessWarning');
    const messageEl = document.getElementById('freshnessMessage');

    if (!banner || !messageEl) return;

    const check = checkDataFreshness(training);

    if (check.is_stale) {
        messageEl.textContent = check.message;
        banner.style.display = 'block';
    } else {
        banner.style.display = 'none';
    }
}

// ============================================
// POLICY 5: FAIL-FAST ERROR STATES
// ============================================

/**
 * Show schema validation error in UI
 * POLICY 5: Fail-fast - make errors LOUD and OBVIOUS
 */
function showSchemaError(errors) {
    const battleStatus = document.getElementById('battleStatus');
    if (!battleStatus) return;

    battleStatus.classList.add('error-state');
    setText('#battleIcon', '⚠️');
    setText('#battleTitle', 'DATA PIPELINE BROKEN');
    setText('#questName', 'Schema validation failed - check console for details');
    setText('#battleStep', '--');
    setText('#battleSpeed', '--');
    setText('#battleStrain', '--');
    setText('#battleETA', '--');

    // Log to battle log
    addLocalLog(`Schema validation failed: ${errors[0]}`, 'error');

    console.error("Schema errors:", errors);
}

// ============================================
// NOTIFICATIONS
// ============================================

function showNotification(title, text, type = 'info') {
    const container = $('#notifications');
    if (!container) return;

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-title">${title}</div>
        <div class="notification-text">${text}</div>
    `;

    container.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// ============================================
// API FETCHING
// ============================================

async function fetchGameData() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/game?_t=${Date.now()}`, {
            cache: 'no-store'
        });
        if (!response.ok) throw new Error('API error');

        const data = await response.json();
        processGameData(data);

    } catch (error) {
        console.error('Failed to fetch game data:', error);
    }
}

function processGameData(data) {
    /**
     * Process fresh game data from /api/game endpoint.
     * Data structure:
     *   - training: training status
     *   - gpu: GPU stats from nvidia-smi
     *   - curriculum: skill progression state
     *   - vault: checkpoint data
     *   - comparison: model comparison results
     */

    // POLICY 1: Validate schema before processing
    const validation = validateRealmStore(data);
    if (!validation.valid) {
        console.error("❌ SCHEMA VALIDATION FAILED!");
        console.error("Errors:", validation.errors);
        // POLICY 5: Show error state in UI instead of gracefully degrading
        showSchemaError(validation.errors);
        return;  // Stop processing - don't render invalid data
    }

    const prevTraining = GameState.isTraining;
    const prevStep = GameState.currentStep;
    const prevTotalLevel = GameState.totalLevel;

    // Training status - handle both RealmStore and legacy /api/game format
    const training = data.state?.training || data.training;
    if (training) {
        GameState.isTraining = training.status === 'training';
        GameState.currentStep = training.step || training.current_step || 0;
        GameState.totalSteps = training.total_steps || 0;
        GameState.loss = training.loss || 0;
        GameState.valLoss = training.validation_loss || 0;
        GameState.currentQuest = training.current_file || null;
        GameState.questProgress = training.progress_percent || 0;
        GameState.stepsPerSecond = training.steps_per_second || 0;
        GameState.etaSeconds = training.eta_seconds || 0;
        GameState.learningRate = training.learning_rate || 0;
        // Loss history for graph - DO NOT USE train_loss_history (always empty)
        // Instead, we build it from real-time events (training.step)
        // This prevents duplicate injection and jumping graphs

        // POLICY 3: Check staleness and update warning
        updateFreshnessWarning(training);
    }

    // GPU stats
    const gpu = data.gpu;
    if (gpu) {
        GameState.vramUsed = gpu.vram_used_gb || 0;
        GameState.vramTotal = gpu.vram_total_gb || 24;
        GameState.gpuTemp = gpu.temperature_c || 0;
    }

    // Curriculum (skills)
    // current_level = level being TRAINED on (not mastered)
    // mastered = current_level - 1 (minimum 0)
    const curriculum = data.curriculum;
    if (curriculum?.skills) {
        const skills = curriculum.skills;

        if (skills.syllo) {
            const trainingLevel = skills.syllo.current_level || 1;
            GameState.sylloMastered = Math.max(0, trainingLevel - 1);
            GameState.sylloTraining = trainingLevel;
            GameState.sylloAcc = skills.syllo.recent_accuracy || 0;
            GameState.sylloEvals = skills.syllo.eval_count || 0;
        }
        if (skills.binary) {
            const trainingLevel = skills.binary.current_level || 1;
            GameState.binaryMastered = Math.max(0, trainingLevel - 1);
            GameState.binaryTraining = trainingLevel;
            GameState.binaryAcc = skills.binary.recent_accuracy || 0;
            GameState.binaryEvals = skills.binary.eval_count || 0;
        }

        // Total level = sum of all MASTERED skill levels
        GameState.totalLevel = GameState.sylloMastered + GameState.binaryMastered;

        // Total evals = SYLLO + BINARY evaluations
        GameState.totalEvals = GameState.sylloEvals + GameState.binaryEvals;

        // Determine current skill - prefer skill_context from training status
        if (training?.skill_context) {
            // Use authoritative skill context from training daemon
            const ctx = training.skill_context;
            GameState.currentSkill = ctx.skill_name || ctx.skill_id?.toUpperCase() || 'UNKNOWN';
            GameState.currentSkillTraining = ctx.skill_level || 1;
            GameState.currentSkillMastered = Math.max(0, GameState.currentSkillTraining - 1);
            GameState.currentSkillAcc = ctx.skill_last_accuracy != null
                ? ctx.skill_last_accuracy * 100
                : 0;
            GameState.skillIcon = ctx.skill_icon || '⚔️';
            GameState.skillTargetAcc = ctx.skill_target_accuracy
                ? ctx.skill_target_accuracy * 100
                : 80;
        } else {
            // Fallback: infer from training file name
            const questName = (GameState.currentQuest || '').toLowerCase();
            if (questName.includes('syllo') || questName.includes('sy_')) {
                GameState.currentSkill = 'SYLLO';
                GameState.currentSkillMastered = GameState.sylloMastered;
                GameState.currentSkillTraining = GameState.sylloTraining;
                GameState.currentSkillAcc = GameState.sylloAcc;
            } else {
                GameState.currentSkill = 'BINARY';
                GameState.currentSkillMastered = GameState.binaryMastered;
                GameState.currentSkillTraining = GameState.binaryTraining;
                GameState.currentSkillAcc = GameState.binaryAcc;
            }
            GameState.skillIcon = null;
            GameState.skillTargetAcc = 80;
        }
    }

    // Vault data (checkpoints)
    const vault = data.vault;
    if (vault) {
        GameState.checkpointCount = vault.checkpoint_count || 0;
        GameState.totalSize = vault.total_size_gb || 0;

        // Find best checkpoint
        if (vault.checkpoints?.length > 0) {
            const champion = vault.checkpoints.find(cp => cp.is_champion);
            if (champion) {
                GameState.bestCheckpoint = champion.name.replace('checkpoint-', '');
            } else {
                GameState.bestCheckpoint = vault.checkpoints[0].name.replace('checkpoint-', '');
            }
        }
    }

    // Model comparison
    const comparison = data.comparison;
    if (comparison) {
        if (comparison.best_checkpoint) {
            // Extract step number from checkpoint name
            const match = comparison.best_checkpoint.match(/checkpoint-(\d+)/);
            if (match) {
                GameState.bestCheckpoint = match[1];
            }
        }
    }

    // Check for events
    if (!prevTraining && GameState.isTraining) {
        showNotification('Training Started!', 'The Skeptic begins questioning', 'success');
    } else if (prevTraining && !GameState.isTraining) {
        showNotification('Training Complete!', 'The Skeptic rests', 'success');
    }

    // Skill level up detection
    if (GameState.totalLevel > prevTotalLevel && prevTotalLevel > 0) {
        showNotification('Skill Level Up!', `Total level is now ${GameState.totalLevel}!`, 'success');
    }

    // Update all UI
    updateAll();
}

function updateAll() {
    updateHeader();
    updateHeroStats();
    updateBattleStatus();
    updateActionHint();
    updateSkills();
    updateVault();
    updateForge();
    updateTime();
}

// ============================================
// CAMPAIGN / HERO DATA
// ============================================

async function fetchCampaignData() {
    try {
        // Fetch active hero info and titles in parallel
        const [heroResp, titlesResp] = await Promise.all([
            fetch('/api/hero'),
            fetch('/api/titles')
        ]);

        const hero = heroResp.ok ? await heroResp.json() : null;
        const titles = titlesResp.ok ? await titlesResp.json() : null;

        if (!hero || !hero.name) return;

        // Update hero display
        const nameEl = document.getElementById('heroName');
        const titleEl = document.getElementById('heroTitle');
        const classEl = document.getElementById('heroClass');
        const iconEl = document.querySelector('.dio-icon');

        if (nameEl) nameEl.textContent = hero.name;

        // Use dynamic title from titles API, fallback to rpg_name
        if (titleEl) {
            if (titles && titles.primary && titles.primary.name) {
                titleEl.textContent = titles.primary.name;
                titleEl.title = titles.primary.description || '';
            } else {
                titleEl.textContent = hero.rpg_name;
            }
        }

        if (classEl) classEl.textContent = hero.model_name || hero.hero_id;
        if (iconEl) iconEl.textContent = hero.icon || '🦸';

        // Store hero info for other uses
        GameState.heroName = hero.name;
        GameState.heroIcon = hero.icon;
        GameState.modelName = hero.model_name;
        GameState.titles = titles;

        // Log warnings if any
        if (titles && titles.warnings && titles.warnings.length > 0) {
            titles.warnings.forEach(w => {
                console.warn(`Title warning: ${w.icon || '⚠️'} ${w.name} - ${w.description}`);
            });
        }

        console.log(`Hero loaded: ${hero.name} (${titles?.primary?.name || hero.rpg_name})`);
    } catch (err) {
        console.error('Error fetching hero:', err);
    }
}

// ============================================
// INITIALIZATION
// ============================================

// Global poller group for managing all polling loops
const pollers = createPollerGroup();

// ============================================
// NEXT ACTION - Main Call to Action
// ============================================

/**
 * Refresh the Next Action widget
 * Shows either:
 * - A blocker message (if momentum is blocked)
 * - A recommendation (what to do next)
 */
async function refreshNextAction() {
    try {
        const [campaignRes, momentumRes] = await Promise.all([
            fetch('/api/campaigns/active'),
            fetch('/api/momentum?check=false'), // Don't re-run checks, just get state
        ]);

        const campaign = campaignRes.ok ? await campaignRes.json() : null;
        const momentum = momentumRes.ok ? await momentumRes.json() : { status: 'go' };

        updateNextActionUI(campaign, momentum);
    } catch (err) {
        console.error('Failed to refresh next action:', err);
    }
}

/**
 * Update the Next Action UI based on campaign and momentum state
 */
function updateNextActionUI(campaign, momentum) {
    const section = document.getElementById('nextActionSection');
    const titleEl = document.getElementById('nextActionTitle');
    const bodyEl = document.getElementById('nextActionBody');
    const btnEl = document.getElementById('nextActionButton');

    if (!section) return;

    // Reset classes
    section.className = 'next-action-card';
    btnEl.disabled = false;

    // Check if daemon is running - CRITICAL for honest state
    const daemon = momentum.daemon || {};
    if (!daemon.running) {
        section.classList.add('blocked');
        titleEl.textContent = 'Training Daemon Not Running';
        bodyEl.innerHTML = `
            The training daemon needs to be started before you can train.
        `;
        btnEl.textContent = 'Start Daemon';
        btnEl.onclick = () => startDaemon(btnEl);
        return;
    }

    // If blocked, show blocker info
    if (momentum.status === 'blocked' && momentum.primary_blocker) {
        const b = momentum.primary_blocker;
        section.classList.add('blocked');
        titleEl.textContent = "Can't move forward yet";
        bodyEl.textContent = `${b.why_i_failed} ${b.how_to_fix}`;

        // Set button based on suggested action
        const action = b.suggested_action;
        if (action === 'open_campaign') {
            btnEl.textContent = 'Open Campaign';
            btnEl.onclick = () => { window.location.href = '/campaign'; };
        } else if (action === 'open_guild') {
            btnEl.textContent = 'Open Guild';
            btnEl.onclick = () => { window.location.href = '/guild'; };
        } else if (action === 'open_quests') {
            btnEl.textContent = 'Open Quests';
            btnEl.onclick = () => { window.location.href = '/quests'; };
        } else {
            btnEl.textContent = 'Fix This';
            btnEl.onclick = () => { window.location.href = '/settings'; };
        }
        return;
    }

    // No campaign at all - check if first run
    if (!campaign || !campaign.id) {
        section.classList.add('no-campaign');

        // Check setup status to determine if truly first run
        checkSetupStatus().then(setup => {
            if (setup && setup.is_first_run) {
                // True first run - show welcome with Quick Start
                titleEl.textContent = 'Welcome to the Realm!';
                bodyEl.innerHTML = `
                    Begin your hero's training journey with one click.<br>
                    <small style="opacity:0.7">Creates a campaign and generates training data automatically.</small>
                `;
                btnEl.textContent = 'Quick Start';
                btnEl.onclick = () => quickStart(btnEl);
            } else {
                // Has campaigns but none active
                titleEl.textContent = 'No Active Campaign';
                bodyEl.textContent = 'Select a campaign to continue your hero\'s journey.';
                btnEl.textContent = 'Open Campaign';
                btnEl.onclick = () => { window.location.href = '/campaign'; };
            }
        }).catch(() => {
            // Fallback if setup check fails
            titleEl.textContent = 'No Active Campaign';
            bodyEl.textContent = 'Create or select a campaign to begin your hero\'s journey.';
            btnEl.textContent = 'Open Campaign';
            btnEl.onclick = () => { window.location.href = '/campaign'; };
        });
        return;
    }

    // Check if queue is empty (warning blocker)
    const queueEmpty = momentum.blockers && momentum.blockers['QUEUE_EMPTY'];
    if (queueEmpty) {
        section.classList.add('blocked');
        titleEl.textContent = 'No Training Data';
        bodyEl.innerHTML = `
            <div style="margin-bottom: 0.75rem;">The training queue is empty.</div>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                <button class="action-btn secondary" onclick="toggleAutoQueue(this)" style="font-size: 0.85rem; padding: 0.4rem 0.8rem;">
                    Enable Auto-Queue
                </button>
            </div>
        `;
        btnEl.textContent = 'Generate 5,000 Examples';
        btnEl.onclick = () => generateTrainingData(btnEl, 5000);

        // Check auto-queue status and update button
        checkAutoQueueStatus();
        return;
    }

    // Show recommendation
    const rec = campaign.recommendation;
    if (!rec) {
        titleEl.textContent = 'Ready to Train';
        bodyEl.innerHTML = `
            <div style="margin-bottom: 0.75rem;">Run a training session to push your hero further.</div>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                <button class="action-btn secondary" onclick="toggleAutoRun(this)" style="font-size: 0.85rem; padding: 0.4rem 0.8rem;">
                    Enable Auto-Run
                </button>
            </div>
        `;
        btnEl.textContent = 'Run 2,000 Steps';
        btnEl.onclick = () => startTraining(2000);

        // Check auto-run status and update button
        checkAutoRunStatus();
        return;
    }

    titleEl.textContent = rec.title;
    bodyEl.innerHTML = `
        <div style="margin-bottom: 0.75rem;">${rec.description}${rec.reason ? ` (${rec.reason})` : ''}</div>
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
            <button class="action-btn secondary" onclick="toggleAutoRun(this)" style="font-size: 0.85rem; padding: 0.4rem 0.8rem;">
                Enable Auto-Run
            </button>
        </div>
    `;

    const steps = rec.suggested_steps || 2000;
    btnEl.textContent = `Run ${steps.toLocaleString()} Steps`;
    btnEl.onclick = () => startTraining(steps);

    // Check auto-run status and update button
    checkAutoRunStatus();
}

/**
 * Start a training session
 */
async function startTraining(steps) {
    const btnEl = document.getElementById('nextActionButton');
    if (!btnEl) return;

    // Disable button while submitting
    btnEl.disabled = true;
    btnEl.textContent = 'Starting...';

    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ steps }),
        });

        const result = await res.json();

        if (result.ok) {
            btnEl.textContent = 'Request Submitted!';
            // Show notification
            if (typeof showNotification === 'function') {
                showNotification(`Training request: ${steps.toLocaleString()} steps`, 'success');
            }
            // Refresh after a moment
            setTimeout(() => {
                refreshNextAction();
                fetchMomentum();
            }, 2000);
        } else {
            btnEl.textContent = 'Error - Retry';
            btnEl.disabled = false;
            if (typeof showNotification === 'function') {
                showNotification(result.message || result.error, 'error');
            }
            // Refresh momentum to show any blockers
            fetchMomentum();
        }
    } catch (err) {
        console.error('Failed to start training:', err);
        btnEl.textContent = 'Error - Retry';
        btnEl.disabled = false;
    }
}

/**
 * Start the training daemon
 */
async function startDaemon(btnEl) {
    if (!btnEl) btnEl = document.getElementById('nextActionButton');
    if (!btnEl) return;

    // Disable button while starting
    btnEl.disabled = true;
    btnEl.textContent = 'Starting...';

    try {
        const res = await fetch('/api/daemon/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'start' }),
        });

        const result = await res.json();

        if (result.success) {
            btnEl.textContent = 'Daemon Starting...';
            if (typeof showNotification === 'function') {
                showNotification('Training daemon is starting...', 'success');
            }
            // Poll for daemon to be ready
            let attempts = 0;
            const pollInterval = setInterval(async () => {
                attempts++;
                try {
                    const mRes = await fetch('/api/momentum');
                    const m = await mRes.json();
                    if (m.daemon && m.daemon.running) {
                        clearInterval(pollInterval);
                        btnEl.textContent = 'Daemon Started!';
                        if (typeof showNotification === 'function') {
                            showNotification('Training daemon is ready!', 'success');
                        }
                        // Refresh UI
                        setTimeout(() => {
                            refreshNextAction();
                            fetchMomentum();
                        }, 1000);
                    } else if (attempts >= 15) {
                        // 15 seconds timeout
                        clearInterval(pollInterval);
                        btnEl.textContent = 'Start Daemon';
                        btnEl.disabled = false;
                        if (typeof showNotification === 'function') {
                            showNotification('Daemon taking too long to start. Check logs.', 'warning');
                        }
                    }
                } catch (e) {
                    // Keep polling
                }
            }, 1000);
        } else {
            btnEl.textContent = 'Start Daemon';
            btnEl.disabled = false;
            if (typeof showNotification === 'function') {
                showNotification(result.error || 'Failed to start daemon', 'error');
            }
        }
    } catch (err) {
        console.error('Failed to start daemon:', err);
        btnEl.textContent = 'Start Daemon';
        btnEl.disabled = false;
        if (typeof showNotification === 'function') {
            showNotification('Failed to start daemon', 'error');
        }
    }
}

/**
 * Check setup/first-run status
 */
async function checkSetupStatus() {
    try {
        const res = await fetch('/api/setup/status');
        if (!res.ok) return null;
        return await res.json();
    } catch (err) {
        console.error('Failed to check setup status:', err);
        return null;
    }
}

/**
 * Quick Start - One-click setup for new users
 */
async function quickStart(btnEl) {
    if (!btnEl) btnEl = document.getElementById('nextActionButton');
    if (!btnEl) return;

    // Disable button while running
    btnEl.disabled = true;
    btnEl.textContent = 'Setting up...';

    try {
        const res = await fetch('/api/setup/quick-start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ examples_count: 1000 }),
        });

        const result = await res.json();

        if (result.ok) {
            btnEl.textContent = 'Ready!';
            if (typeof showNotification === 'function') {
                let msg;
                if (result.skipped) {
                    msg = 'Campaign already exists!';
                } else {
                    const count = result.generated?.[0]?.count || 0;
                    const daemon = result.daemon_started ? ' Daemon starting...' : '';
                    msg = `Campaign created! Generated ${count} training examples.${daemon}`;
                }
                showNotification(msg, 'success');
            }
            // Refresh everything (give daemon time to start)
            const delay = result.daemon_started ? 3000 : 1500;
            setTimeout(() => {
                fetchCampaignData();
                refreshNextAction();
                fetchMomentum();
            }, delay);
        } else {
            btnEl.textContent = 'Quick Start';
            btnEl.disabled = false;
            if (typeof showNotification === 'function') {
                showNotification(result.error || 'Quick start failed', 'error');
            }
        }
    } catch (err) {
        console.error('Quick start failed:', err);
        btnEl.textContent = 'Quick Start';
        btnEl.disabled = false;
        if (typeof showNotification === 'function') {
            showNotification('Quick start failed', 'error');
        }
    }
}

/**
 * Check auto-queue status and update button text
 */
async function checkAutoQueueStatus() {
    try {
        const res = await fetch('/config');
        if (!res.ok) return;
        const config = await res.json();
        const enabled = config.auto_generate?.enabled || false;

        // Find and update the auto-queue button
        const btn = document.querySelector('button[onclick*="toggleAutoQueue"]');
        if (btn) {
            btn.textContent = enabled ? 'Disable Auto-Queue' : 'Enable Auto-Queue';
            btn.classList.toggle('active', enabled);
        }
    } catch (err) {
        console.error('Failed to check auto-queue status:', err);
    }
}

/**
 * Toggle auto-queue (auto-generate) setting
 */
async function toggleAutoQueue(btnEl) {
    if (!btnEl) return;

    btnEl.disabled = true;
    const wasEnabled = btnEl.textContent.includes('Disable');
    btnEl.textContent = 'Updating...';

    try {
        // Get current config
        const configRes = await fetch('/config');
        if (!configRes.ok) throw new Error('Failed to load config');
        const config = await configRes.json();

        // Toggle auto_generate.enabled
        if (!config.auto_generate) config.auto_generate = {};
        config.auto_generate.enabled = !wasEnabled;

        // Save config
        const saveRes = await fetch('/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });

        const result = await saveRes.json();

        if (result.success || result.ok) {
            const newState = config.auto_generate.enabled;
            btnEl.textContent = newState ? 'Disable Auto-Queue' : 'Enable Auto-Queue';
            btnEl.classList.toggle('active', newState);
            if (typeof showNotification === 'function') {
                showNotification(
                    newState ? 'Auto-queue enabled! Data will generate automatically.' : 'Auto-queue disabled.',
                    'success'
                );
            }
        } else {
            btnEl.textContent = wasEnabled ? 'Disable Auto-Queue' : 'Enable Auto-Queue';
            if (typeof showNotification === 'function') {
                showNotification(result.error || 'Failed to update setting', 'error');
            }
        }
    } catch (err) {
        console.error('Failed to toggle auto-queue:', err);
        btnEl.textContent = wasEnabled ? 'Disable Auto-Queue' : 'Enable Auto-Queue';
        if (typeof showNotification === 'function') {
            showNotification('Failed to toggle auto-queue', 'error');
        }
    } finally {
        btnEl.disabled = false;
    }
}

/**
 * Check auto-run status and update button text
 */
async function checkAutoRunStatus() {
    try {
        const res = await fetch('/config');
        if (!res.ok) return;
        const config = await res.json();
        const enabled = config.auto_run?.enabled || false;

        // Find and update the auto-run button
        const btn = document.querySelector('button[onclick*="toggleAutoRun"]');
        if (btn) {
            btn.textContent = enabled ? 'Disable Auto-Run' : 'Enable Auto-Run';
            btn.classList.toggle('active', enabled);
        }
    } catch (err) {
        console.error('Failed to check auto-run status:', err);
    }
}

/**
 * Toggle auto-run (automatic training) setting
 */
async function toggleAutoRun(btnEl) {
    if (!btnEl) return;

    btnEl.disabled = true;
    const wasEnabled = btnEl.textContent.includes('Disable');
    btnEl.textContent = 'Updating...';

    try {
        // Get current config
        const configRes = await fetch('/config');
        if (!configRes.ok) throw new Error('Failed to load config');
        const config = await configRes.json();

        // Toggle auto_run.enabled
        if (!config.auto_run) config.auto_run = {};
        config.auto_run.enabled = !wasEnabled;

        // Save config
        const saveRes = await fetch('/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });

        const result = await saveRes.json();

        if (result.success || result.ok) {
            const newState = config.auto_run.enabled;
            btnEl.textContent = newState ? 'Disable Auto-Run' : 'Enable Auto-Run';
            btnEl.classList.toggle('active', newState);
            if (typeof showNotification === 'function') {
                showNotification(
                    newState ? 'Auto-run enabled! Training will start automatically.' : 'Auto-run disabled. Use Run button to start.',
                    'success'
                );
            }
        } else {
            btnEl.textContent = wasEnabled ? 'Disable Auto-Run' : 'Enable Auto-Run';
            if (typeof showNotification === 'function') {
                showNotification(result.error || 'Failed to update setting', 'error');
            }
        }
    } catch (err) {
        console.error('Failed to toggle auto-run:', err);
        btnEl.textContent = wasEnabled ? 'Disable Auto-Run' : 'Enable Auto-Run';
        if (typeof showNotification === 'function') {
            showNotification('Failed to toggle auto-run', 'error');
        }
    } finally {
        btnEl.disabled = false;
    }
}

/**
 * Generate training data and add to queue
 */
async function generateTrainingData(btnEl, count = 1000) {
    if (!btnEl) btnEl = document.getElementById('nextActionButton');
    if (!btnEl) return;

    // Disable button while generating
    btnEl.disabled = true;
    btnEl.textContent = 'Generating...';

    try {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ count }),
        });

        const result = await res.json();

        if (result.ok) {
            btnEl.textContent = `Generated ${result.examples_generated}!`;
            if (typeof showNotification === 'function') {
                showNotification(`Generated ${result.examples_generated} training examples for ${result.skill} L${result.level}`, 'success');
            }
            // Refresh UI to show updated state
            setTimeout(() => {
                refreshNextAction();
                fetchMomentum();
            }, 1500);
        } else {
            btnEl.textContent = 'Generate 1,000 Examples';
            btnEl.disabled = false;
            if (typeof showNotification === 'function') {
                showNotification(result.error || 'Failed to generate data', 'error');
            }
        }
    } catch (err) {
        console.error('Failed to generate training data:', err);
        btnEl.textContent = 'Generate 1,000 Examples';
        btnEl.disabled = false;
        if (typeof showNotification === 'function') {
            showNotification('Failed to generate training data', 'error');
        }
    }
}

// ============================================
// MOMENTUM ENGINE - Forward Progress Tracking
// ============================================

/**
 * Fetch and display momentum state (blockers, forward progress)
 */
async function fetchMomentum() {
    try {
        const res = await fetch('/api/momentum');
        if (!res.ok) return;

        const m = await res.json();
        updateMomentumUI(m);
        // Also update next action since momentum affects it
        refreshNextAction();
    } catch (err) {
        console.error('Failed to fetch momentum:', err);
    }
}

/**
 * Update the momentum card UI based on state
 */
function updateMomentumUI(m) {
    const card = document.getElementById('momentumCard');
    const badge = document.getElementById('momentumBadge');
    const statusEl = document.getElementById('momentumStatus');
    const blockerDiv = document.getElementById('momentumBlocker');

    if (!card) return;

    const status = m.status || 'idle';
    const pb = m.primary_blocker;
    const blockerCount = m.blocker_count || 0;

    // Update badge
    badge.className = 'momentum-badge ' + status;
    statusEl.textContent = status.toUpperCase();

    // Update card class
    card.className = 'momentum-card';
    if (status === 'blocked') card.classList.add('blocked');
    if (pb && pb.severity === 'error') card.classList.add('error');

    // Show card only if there are blockers
    if (blockerCount > 0) {
        card.style.display = 'block';

        // Show primary blocker details
        if (pb) {
            blockerDiv.style.display = 'block';
            document.getElementById('blockerWhat').textContent =
                `Trying to: ${pb.what_i_was_trying}`;
            document.getElementById('blockerWhy').textContent = pb.why_i_failed;
            document.getElementById('blockerHow').textContent = pb.how_to_fix;

            // Set up action button
            const btn = document.getElementById('blockerActionBtn');
            const action = pb.suggested_action;

            if (action === 'open_campaign') {
                btn.textContent = 'Open Campaign';
                btn.onclick = () => (window.location.href = '/campaign');
            } else if (action === 'open_guild') {
                btn.textContent = 'Open Guild';
                btn.onclick = () => (window.location.href = '/guild');
            } else if (action === 'open_settings') {
                btn.textContent = 'Open Settings';
                btn.onclick = () => (window.location.href = '/settings');
            } else if (action === 'open_quests') {
                btn.textContent = 'Open Quests';
                btn.onclick = () => (window.location.href = '/quests');
            } else if (action === 'open_analysis') {
                btn.textContent = 'Open Analysis';
                btn.onclick = () => (window.location.href = '/analysis');
            } else {
                btn.textContent = 'Show Details';
                btn.onclick = () => console.log('Blocker context:', pb.context);
            }
        } else {
            blockerDiv.style.display = 'none';
        }
    } else {
        // No blockers - hide the card entirely
        card.style.display = 'none';
    }
}

function init() {
    console.log('Realm of Training initializing...');

    // Load active campaign/hero first
    fetchCampaignData();

    // Initial UI update
    updateAll();

    // Fetch skills (from YAML configs)
    fetchSkills();

    // Try to use unified RealmState (single source of truth)
    const realmSyncActive = initRealmStateSync();

    if (realmSyncActive) {
        console.log('[Game] Using RealmState for training/queue/events data');
        // RealmState handles: training status, queue, battle log
        // We still need: GPU, vault, curriculum from /api/game
        fetchGameData();
        pollers.add('gameData', fetchGameData, CONFIG.UPDATE_INTERVAL, { immediate: false });
    } else {
        console.log('[Game] Falling back to legacy polling');
        // Legacy: Fetch saga (battle log) and game data separately
        fetchSaga();
        fetchGameData();
        pollers.add('gameData', fetchGameData, CONFIG.UPDATE_INTERVAL, { immediate: false });
        pollers.add('saga', fetchSaga, 5000, { immediate: false });
    }

    // Fetch hint data (quests, task master) - still needed
    fetchHintData();

    // Fetch momentum state (blockers, forward progress)
    fetchMomentum();

    // Fetch next action recommendation (main CTA)
    refreshNextAction();

    // Set up polling for secondary data
    pollers.add('hints', fetchHintData, 10000, { immediate: false });
    pollers.add('skills', fetchSkills, 30000, { immediate: false });
    pollers.add('time', updateTime, 1000, { immediate: false });
    pollers.add('momentum', fetchMomentum, 15000, { immediate: false });

    // Start all pollers
    pollers.startAll();

    // Register for real-time SSE training updates (in addition to RealmState)
    registerTrainingEventHandlers();

    console.log('[Game] Pollers initialized:', Object.keys(pollers.getAllStats()));
}

/**
 * Register handlers for SSE training events
 * Provides instant UI updates without polling
 */
function registerTrainingEventHandlers() {
    // Wait for eventStream to be initialized
    const waitForEventStream = setInterval(() => {
        if (window.eventStream) {
            clearInterval(waitForEventStream);

            // Handle training.step events - real-time progress updates
            window.eventStream.on('training.step', (event) => {
                const data = event.data || {};

                // Update GameState with real-time data
                GameState.isTraining = true;
                GameState.currentStep = data.step || GameState.currentStep;
                GameState.totalSteps = data.max_steps || GameState.totalSteps;
                GameState.loss = data.loss || GameState.loss;
                GameState.questProgress = data.progress_percent || GameState.questProgress;
                GameState.stepsPerSecond = data.steps_per_second || GameState.stepsPerSecond;
                GameState.etaSeconds = data.eta_seconds || GameState.etaSeconds;
                GameState.learningRate = data.learning_rate || GameState.learningRate;

                // Update VRAM if provided
                if (data.vram_gb) {
                    GameState.vramUsed = data.vram_gb;
                }

                // Add to effort history for graph
                // Effort = cumulative strain (sum of (loss - floor) over time)
                if (data.loss && data.step) {
                    const floor = 0.5;  // Comfort zone loss (could come from config)
                    const strain = Math.max(0, data.loss - floor);  // Strain = loss - floor, min 0

                    // Calculate cumulative effort
                    let effort = strain;
                    if (GameState.lossHistory.length > 0) {
                        const lastEntry = GameState.lossHistory[GameState.lossHistory.length - 1];
                        effort = (lastEntry.effort || lastEntry.loss) + strain;  // Cumulative
                    }

                    GameState.lossHistory.push({
                        step: data.step,
                        loss: data.loss,  // Keep raw loss for reference
                        strain: strain,   // Instantaneous strain
                        effort: effort    // Cumulative effort
                    });

                    // Keep last 200 entries
                    if (GameState.lossHistory.length > 200) {
                        GameState.lossHistory = GameState.lossHistory.slice(-200);
                    }
                    renderLossChart(GameState.lossHistory);
                }

                // Update UI immediately
                updateBattleStatus();
                updateForge();
            });

            // Handle training.started events
            window.eventStream.on('training.started', (event) => {
                const data = event.data || {};
                GameState.isTraining = true;
                GameState.currentQuest = data.file || 'Training';
                GameState.lossHistory = [];  // Reset loss history for new training
                updateBattleStatus();
                console.log('[Game] Training started:', data.file);
            });

            // Handle training.completed events
            window.eventStream.on('training.completed', (event) => {
                const data = event.data || {};
                GameState.isTraining = false;
                updateBattleStatus();
                console.log('[Game] Training completed:', data.steps, 'steps');
            });

            console.log('[Game] Registered SSE training event handlers');
        }
    }, 100);
}

// ============================================
// TRAINING CONTROL
// ============================================

/**
 * Send control command to training daemon
 * Uses TrainingClient for API abstraction
 * @param {string} action - 'pause', 'resume', or 'stop'
 */
async function controlTraining(action) {
    const btnPause = document.getElementById('btnPause');
    const btnResume = document.getElementById('btnResume');
    const btnStop = document.getElementById('btnStop');

    // Disable buttons during request
    if (btnPause) btnPause.disabled = true;
    if (btnResume) btnResume.disabled = true;
    if (btnStop) btnStop.disabled = true;

    try {
        // Use TrainingClient for API call
        const result = await TrainingClient.controlTraining(action);

        // Update button states based on action
        if (action === 'pause') {
            if (btnPause) btnPause.style.display = 'none';
            if (btnResume) btnResume.style.display = 'inline-flex';
            addLocalLog(`Training paused - ${result.message}`, 'warning');
        } else if (action === 'resume') {
            if (btnPause) btnPause.style.display = 'inline-flex';
            if (btnResume) btnResume.style.display = 'none';
            addLocalLog(`Training resumed - ${result.message}`, 'success');
        } else if (action === 'stop') {
            addLocalLog(`Training stopped - ${result.message}`, 'warning');
        }
    } catch (error) {
        addLocalLog(`Control error: ${error.message}`, 'error');
    } finally {
        // Re-enable buttons
        if (btnPause) btnPause.disabled = false;
        if (btnResume) btnResume.disabled = false;
        if (btnStop) btnStop.disabled = false;
    }
}

/**
 * Update control button states based on training status
 */
function updateControlButtons() {
    const btnPause = document.getElementById('btnPause');
    const btnResume = document.getElementById('btnResume');
    const controls = document.getElementById('trainingControls');

    // Show controls only when training
    if (controls) {
        controls.style.display = GameState.isTraining ? 'flex' : 'none';
    }

    // TODO: Check actual pause state from API and show correct button
}

// ============================================
// 🎮 MUD CONSOLE - COMMAND THE REALM
// ============================================

/**
 * Realm Console - zMUD-style command interface
 */
const RealmConsole = {
    isOpen: false,
    isLoggedIn: false,
    password: 'ADMIN123',  // Pseudo-security, like the old days
    history: [],
    historyIndex: -1,

    // MUD-style room descriptions
    locations: {
        tavern: "You are in the Tavern. Torches flicker on the walls. The air hums with potential energy.",
        forge: "The Forge blazes before you. Heat radiates from GPU cores. This is where heroes are tempered.",
        guild: "The Guild Hall stretches before you. Skill crystals line the walls, each pulsing with learned knowledge.",
        vault: "The Vault's iron doors stand open. Rows of checkpoints gleam like treasure.",
        oracle: "The Oracle's chamber is quiet. Ancient inference patterns swirl in the darkness."
    },
    currentLocation: 'tavern',

    init() {
        const input = document.getElementById('consoleInput');
        const output = document.getElementById('consoleOutput');
        if (!input || !output) return;

        // Listen for ` (backtick) to toggle console globally
        document.addEventListener('keydown', (e) => {
            if (e.key === '`' && !e.ctrlKey && !e.altKey) {
                // Don't trigger if typing in an input
                if (document.activeElement.tagName === 'INPUT' && document.activeElement.id !== 'consoleInput') {
                    return;
                }
                e.preventDefault();
                this.toggle();
            }
        });

        // Command input handling
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const cmd = input.value;
                input.value = '';
                if (cmd.trim()) {
                    this.history.unshift(cmd);
                    this.historyIndex = -1;
                    this.execute(cmd);
                }
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (this.history.length > 0) {
                    this.historyIndex = Math.min(this.historyIndex + 1, this.history.length - 1);
                    input.value = this.history[this.historyIndex];
                }
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (this.historyIndex > 0) {
                    this.historyIndex--;
                    input.value = this.history[this.historyIndex];
                } else {
                    this.historyIndex = -1;
                    input.value = '';
                }
            } else if (e.key === 'Escape') {
                this.toggle();
            }
        });

        // Set initial prompt state
        this.updatePrompt();

        console.log('🎮 Realm Console initialized');
    },

    toggle() {
        const console = document.getElementById('realmConsole');
        const input = document.getElementById('consoleInput');
        if (!console) return;

        this.isOpen = !this.isOpen;
        console.classList.toggle('collapsed', !this.isOpen);

        if (this.isOpen) {
            setTimeout(() => input?.focus(), 100);
        }
    },

    print(text, type = '') {
        const output = document.getElementById('consoleOutput');
        if (!output) return;

        const line = document.createElement('div');
        line.className = 'console-line ' + (type ? `console-${type}` : '');
        line.innerHTML = text;
        output.appendChild(line);
        output.scrollTop = output.scrollHeight;
    },

    updatePrompt() {
        const prompt = document.getElementById('consolePrompt');
        if (!prompt) return;

        if (!this.isLoggedIn) {
            prompt.textContent = 'Password:';
            prompt.className = 'console-prompt';
        } else if (GameState.isTraining) {
            prompt.textContent = '>';
            prompt.className = 'console-prompt training';
        } else {
            prompt.textContent = '>';
            prompt.className = 'console-prompt';
        }
    },

    async execute(cmd) {
        const trimmed = cmd.trim();

        // Not logged in - check password
        if (!this.isLoggedIn) {
            if (trimmed.toUpperCase() === this.password) {
                this.isLoggedIn = true;
                this.print('*** ACCESS GRANTED ***', 'success');
                this.print('Welcome to the Realm, Dungeon Master.', 'system');
                this.print(`Type <span class="cmd">help</span> for available commands.`);
                this.updatePrompt();
                return;
            } else {
                this.print('*** ACCESS DENIED ***', 'error');
                return;
            }
        }

        // Logged in - parse command
        this.print(`> ${trimmed}`, 'cmd');

        const [verb, ...args] = trimmed.toLowerCase().split(/\s+/);
        const arg = args.join(' ');

        try {
            switch (verb) {
                case 'help':
                case '?':
                    this.cmdHelp();
                    break;
                case 'look':
                case 'l':
                    this.cmdLook();
                    break;
                case 'status':
                case 'stat':
                case 's':
                    await this.cmdStatus();
                    break;
                case 'who':
                case 'w':
                    await this.cmdWho();
                    break;
                case 'skills':
                    await this.cmdSkills();
                    break;
                case 'checkpoints':
                case 'ck':
                    await this.cmdCheckpoints();
                    break;
                case 'evals':
                case 'eval':
                    await this.cmdEvals(arg);
                    break;
                case 'weaver':
                    await this.cmdWeaver(arg);
                    break;
                case 'train':
                    await this.cmdTrain(arg);
                    break;
                case 'go':
                case 'goto':
                    this.cmdGo(arg);
                    break;
                case 'clear':
                case 'cls':
                    this.cmdClear();
                    break;
                case 'say':
                    this.print(`You say: "${arg}"`, 'system');
                    break;
                case 'quit':
                case 'logout':
                    this.isLoggedIn = false;
                    this.print('You fade from the Realm...', 'system');
                    this.print('Enter password to reconnect.', 'info');
                    this.updatePrompt();
                    break;
                default:
                    this.print(`Unknown command: ${verb}. Type <span class="cmd">help</span> for commands.`, 'error');
            }
        } catch (err) {
            console.error('Console error:', err);
            this.print(`Error: ${err.message}`, 'error');
        }
    },

    cmdHelp() {
        this.print('=== REALM CONSOLE COMMANDS ===', 'info');
        this.print('<span class="cmd">look</span> (l)         - Describe your surroundings');
        this.print('<span class="cmd">status</span> (s)       - Show realm status');
        this.print('<span class="cmd">who</span> (w)          - Show connected services');
        this.print('<span class="cmd">skills</span>          - Show DIO\'s skills');
        this.print('<span class="cmd">checkpoints</span> (ck) - List recent checkpoints');
        this.print('<span class="cmd">evals</span> [skill]    - Show recent evaluations');
        this.print('<span class="cmd">weaver</span> [start|stop|status] - Control the Weaver');
        this.print('<span class="cmd">train</span> [start|stop] - Control training');
        this.print('<span class="cmd">go</span> [location]    - Travel (tavern/forge/guild/vault/oracle)');
        this.print('<span class="cmd">clear</span>           - Clear console');
        this.print('<span class="cmd">logout</span>          - Disconnect from realm');
    },

    cmdLook() {
        const desc = this.locations[this.currentLocation] || 'You are somewhere in the Realm.';
        this.print(desc, 'system');

        // Add exits
        const exits = Object.keys(this.locations).filter(l => l !== this.currentLocation);
        this.print(`Exits: ${exits.join(', ')}`, 'info');
    },

    async cmdStatus() {
        try {
            const res = await fetch('/api/state');
            const data = await res.json();

            const mode = data.mode || 'unknown';
            const step = data.step?.toLocaleString() || '--';
            const loss = data.loss?.toFixed(4) || '--';
            const skill = data.active_skill || '--';
            const level = data.training_level || '--';

            this.print(`=== REALM STATUS ===`, 'info');
            this.print(`Mode: <span class="val">${mode.toUpperCase()}</span>`);
            this.print(`Step: <span class="val">${step}</span> | Loss: <span class="val">${loss}</span>`);
            this.print(`Active Skill: <span class="highlight">${skill}</span> L${level}`);

            if (data.queue_size !== undefined) {
                this.print(`Queue: <span class="val">${data.queue_size}</span> jobs waiting`);
            }
        } catch (err) {
            this.print('Failed to fetch realm status', 'error');
        }
    },

    async cmdWho() {
        try {
            const res = await fetch('/api/cluster/summary');
            const data = await res.json();

            this.print('=== CONNECTED SERVICES ===', 'info');

            if (data.hosts) {
                for (const [name, info] of Object.entries(data.hosts)) {
                    const status = info.online ? '<span class="val">ONLINE</span>' : '<span class="console-error">OFFLINE</span>';
                    this.print(`${name}: ${status}`);
                }
            } else {
                this.print('No host registry found', 'info');
            }
        } catch (err) {
            this.print('Failed to fetch cluster info', 'error');
        }
    },

    async cmdSkills() {
        try {
            const res = await fetch('/api/skills');
            const data = await res.json();

            this.print('=== DIO\'S SKILLS ===', 'info');

            if (data.skills) {
                for (const skill of data.skills) {
                    const bar = '█'.repeat(skill.mastered_level || 0) + '░'.repeat(Math.max(0, 10 - (skill.mastered_level || 0)));
                    this.print(`<span class="highlight">${skill.id.toUpperCase()}</span> [${bar}] L${skill.mastered_level || 0}`);
                }
            }
        } catch (err) {
            this.print('Failed to fetch skills', 'error');
        }
    },

    async cmdCheckpoints() {
        try {
            const res = await fetch('/api/ledger?limit=5');
            const data = await res.json();

            this.print('=== RECENT CHECKPOINTS ===', 'info');

            const checkpoints = data.checkpoints || data || [];
            for (const cp of checkpoints.slice(0, 5)) {
                const step = cp.step?.toLocaleString() || cp.name || '??';
                const loss = cp.train_loss?.toFixed(4) || '--';
                this.print(`Step <span class="val">${step}</span> | Loss: ${loss}`);
            }
        } catch (err) {
            this.print('Failed to fetch checkpoints', 'error');
        }
    },

    async cmdEvals(skill) {
        try {
            const url = skill ? `/api/evals/skill/${skill}` : '/api/evals?limit=10';
            const res = await fetch(url);
            const data = await res.json();

            this.print(`=== RECENT EVALS ${skill ? `(${skill.toUpperCase()})` : ''} ===`, 'info');

            const evals = data.evaluations || data || [];
            for (const ev of evals.slice(0, 5)) {
                const acc = (ev.accuracy * 100).toFixed(1);
                const level = ev.level || '?';
                const skillId = ev.skill_id || ev.skill || '?';
                this.print(`<span class="highlight">${skillId.toUpperCase()}</span> L${level}: <span class="val">${acc}%</span>`);
            }
        } catch (err) {
            this.print('Failed to fetch evals', 'error');
        }
    },

    async cmdWeaver(action) {
        if (!action) {
            this.print('Usage: weaver [start|stop|status]', 'info');
            return;
        }

        switch (action) {
            case 'status':
                try {
                    const res = await fetch('/api/weaver/status');
                    const data = await res.json();
                    this.print(`Weaver: <span class="val">${data.running ? 'RUNNING' : 'STOPPED'}</span>`);
                } catch {
                    this.print('Weaver status unknown (API not available)', 'error');
                }
                break;
            case 'start':
                this.print('Awakening the Weaver...', 'system');
                try {
                    await fetch('/api/weaver/start', { method: 'POST' });
                    this.print('Weaver awakened!', 'success');
                } catch {
                    this.print('Failed to start Weaver', 'error');
                }
                break;
            case 'stop':
                this.print('Putting the Weaver to sleep...', 'system');
                try {
                    await fetch('/api/weaver/stop', { method: 'POST' });
                    this.print('Weaver sleeps.', 'success');
                } catch {
                    this.print('Failed to stop Weaver', 'error');
                }
                break;
            default:
                this.print('Unknown weaver action. Use: start, stop, status', 'error');
        }
    },

    async cmdTrain(action) {
        if (!action) {
            this.print('Usage: train [start|stop]', 'info');
            return;
        }

        switch (action) {
            case 'start':
                this.print('Initiating training sequence...', 'system');
                try {
                    await fetch('/api/realm/mode', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ mode: 'training' })
                    });
                    this.print('Training mode activated!', 'success');
                    this.updatePrompt();
                } catch {
                    this.print('Failed to start training', 'error');
                }
                break;
            case 'stop':
                this.print('Halting training...', 'system');
                try {
                    await fetch('/api/realm/mode', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ mode: 'idle' })
                    });
                    this.print('Training halted.', 'success');
                    this.updatePrompt();
                } catch {
                    this.print('Failed to stop training', 'error');
                }
                break;
            default:
                this.print('Unknown train action. Use: start, stop', 'error');
        }
    },

    cmdGo(location) {
        if (!location) {
            this.print('Go where? Options: ' + Object.keys(this.locations).join(', '), 'info');
            return;
        }

        if (this.locations[location]) {
            this.currentLocation = location;
            this.print(`You travel to the ${location.charAt(0).toUpperCase() + location.slice(1)}...`, 'system');
            this.cmdLook();
        } else {
            this.print(`Unknown location: ${location}`, 'error');
        }
    },

    cmdClear() {
        const output = document.getElementById('consoleOutput');
        if (output) output.innerHTML = '';
    }
};

// ============================================
// 🔥 RPG FLAIR - THE TAVERN IS ALIVE
// ============================================

/**
 * RPG Flair Module - Brings the Tavern to life!
 */
const RPGFlair = {
    // Hero sprite - keep consistent, just add occasional sparkle
    heroBase: '🧔🏽',
    heroWalking: false,

    // Mini-map state
    minimapHeroPos: { x: 2, y: 1 },
    minimapCols: 5,
    minimapRows: 3,
    minimapTerrain: [],
    minimapStructures: [
        { x: 0, y: 0, icon: '🏰', name: 'Tavern' },
        { x: 4, y: 0, icon: '🏛️', name: 'Guild' },
        { x: 4, y: 2, icon: '🗃️', name: 'Vault' },
        { x: 0, y: 2, icon: '🔮', name: 'Oracle' },
    ],

    // Room descriptions
    roomDescriptions: [
        "You stand in the Training Grounds. The air crackles with gradients. Distant echoes of loss curves rumble like thunder across the Realm.",
        "The Tavern hums with potential energy. Torches flicker as DIO contemplates the next challenge.",
        "You feel the weight of accumulated steps. The hero's journey continues ever forward.",
        "Gradients swirl in the air like motes of dust. Each one carries the memory of a training run.",
        "The smell of burning GPU fills the air. In the distance, a checkpoint crystallizes into being.",
    ],

    // Achievement queue
    achievementQueue: [],
    achievementShowing: false,

    // Previous state for detecting changes
    prevStep: 0,
    prevLevel: 0,
    prevAccuracy: 0,

    /**
     * Initialize all RPG flair systems
     */
    init() {
        console.log('🔥 RPG Flair initializing...');

        // Set random room description
        this.setRandomRoomDescription();

        // Initialize mini-map
        this.initMinimap();

        // Start hero animation loop
        this.startHeroAnimation();

        // Set up spell cast effects on buttons
        this.setupSpellcastButtons();

        // Start mini-map wandering
        this.startMinimapWander();

        // Initialize HP/MP bars
        this.updateResourceBars();

        console.log('🔥 RPG Flair ready!');
    },

    /**
     * Set a random room description
     */
    setRandomRoomDescription() {
        const descEl = document.getElementById('roomDescription');
        if (!descEl) return;

        const desc = this.roomDescriptions[Math.floor(Math.random() * this.roomDescriptions.length)];
        // Keep the exits, just update the main text
        const exitsHTML = descEl.querySelector('.room-exits')?.outerHTML || '';
        descEl.innerHTML = desc + exitsHTML;
    },

    /**
     * Initialize the mini-map with terrain and structures
     */
    initMinimap() {
        const mapEl = document.getElementById('realmMinimap');
        if (!mapEl) return;

        mapEl.innerHTML = '';

        // Add random terrain
        const terrainTypes = ['🌲', '🪨', '🌿', ''];
        for (let y = 0; y < this.minimapRows; y++) {
            for (let x = 0; x < this.minimapCols; x++) {
                // Skip structure positions
                if (this.minimapStructures.some(s => s.x === x && s.y === y)) continue;

                if (Math.random() < 0.3) {
                    const terrain = terrainTypes[Math.floor(Math.random() * terrainTypes.length)];
                    if (terrain) {
                        const el = document.createElement('div');
                        el.className = 'minimap-cell minimap-terrain';
                        el.textContent = terrain;
                        el.style.left = `${10 + x * 25}px`;
                        el.style.top = `${10 + y * 25}px`;
                        mapEl.appendChild(el);
                    }
                }
            }
        }

        // Add structures
        for (const struct of this.minimapStructures) {
            const el = document.createElement('div');
            el.className = 'minimap-cell minimap-structure';
            el.textContent = struct.icon;
            el.title = struct.name;
            el.style.left = `${10 + struct.x * 25}px`;
            el.style.top = `${10 + struct.y * 25}px`;
            mapEl.appendChild(el);
        }

        // Add hero
        const heroEl = document.createElement('div');
        heroEl.className = 'minimap-cell minimap-hero';
        heroEl.id = 'minimapHero';
        heroEl.textContent = '🧙‍♂️';
        heroEl.style.left = `${10 + this.minimapHeroPos.x * 25}px`;
        heroEl.style.top = `${10 + this.minimapHeroPos.y * 25}px`;
        mapEl.appendChild(heroEl);
    },

    /**
     * Start the hero animation - subtle sparkle only, no frame swapping
     */
    startHeroAnimation() {
        const heroEl = document.getElementById('heroSprite');
        if (!heroEl) return;

        // Set base hero and let CSS handle the breathing animation
        heroEl.textContent = this.heroBase;

        // Occasional sparkle (every ~5 seconds when training)
        setInterval(() => {
            if (GameState.isTraining) {
                heroEl.parentElement?.parentElement?.classList.add('hero-walking');
            } else {
                heroEl.parentElement?.parentElement?.classList.remove('hero-walking');
            }
        }, 2000);
    },

    /**
     * Start the mini-map hero wandering
     */
    startMinimapWander() {
        setInterval(() => {
            const heroEl = document.getElementById('minimapHero');
            if (!heroEl) return;

            // Random movement
            const directions = [
                { dx: 0, dy: -1 },
                { dx: 0, dy: 1 },
                { dx: -1, dy: 0 },
                { dx: 1, dy: 0 },
            ];

            const dir = directions[Math.floor(Math.random() * directions.length)];
            const newX = this.minimapHeroPos.x + dir.dx;
            const newY = this.minimapHeroPos.y + dir.dy;

            // Bounds check
            if (newX >= 0 && newX < this.minimapCols && newY >= 0 && newY < this.minimapRows) {
                this.minimapHeroPos.x = newX;
                this.minimapHeroPos.y = newY;
                heroEl.style.left = `${10 + newX * 25}px`;
                heroEl.style.top = `${10 + newY * 25}px`;
            }
        }, 1200);
    },

    /**
     * Set up spell cast effects on buttons
     */
    setupSpellcastButtons() {
        document.querySelectorAll('.btn-spellcast').forEach(btn => {
            btn.addEventListener('click', () => {
                btn.classList.remove('casting');
                void btn.offsetWidth; // Force reflow
                btn.classList.add('casting');
                setTimeout(() => btn.classList.remove('casting'), 600);
            });
        });

        // Also add to any future buttons
        document.addEventListener('click', (e) => {
            const btn = e.target.closest('.btn-spellcast, .next-action-btn, .control-btn');
            if (btn && !btn.classList.contains('casting')) {
                btn.classList.add('casting');
                setTimeout(() => btn.classList.remove('casting'), 600);
            }
        });
    },

    /**
     * Show a floating number animation
     */
    showFloatingNumber(text, type = 'xp', targetEl = null) {
        const container = targetEl || document.querySelector('.hero-frame');
        if (!container) return;

        const rect = container.getBoundingClientRect();
        const floater = document.createElement('div');
        floater.className = `floating-number ${type}`;
        floater.textContent = text;
        floater.style.left = `${rect.left + rect.width / 2}px`;
        floater.style.top = `${rect.top + rect.height / 3}px`;
        floater.style.position = 'fixed';

        document.body.appendChild(floater);

        // Remove after animation
        setTimeout(() => floater.remove(), 1500);
    },

    /**
     * Show critical hit effect
     */
    showCriticalHit(text = 'CRITICAL!') {
        const overlay = document.getElementById('critOverlay');
        if (!overlay) return;

        const textEl = overlay.querySelector('.critical-hit-text');
        if (textEl) textEl.textContent = text;

        overlay.classList.remove('active');
        void overlay.offsetWidth;
        overlay.classList.add('active');

        // Screen shake
        document.body.classList.add('screen-shake');
        setTimeout(() => {
            document.body.classList.remove('screen-shake');
            overlay.classList.remove('active');
        }, 800);
    },

    /**
     * Show achievement toast
     */
    showAchievement(name, icon = '🏆') {
        this.achievementQueue.push({ name, icon });
        this.processAchievementQueue();
    },

    processAchievementQueue() {
        if (this.achievementShowing || this.achievementQueue.length === 0) return;

        const { name, icon } = this.achievementQueue.shift();
        const toast = document.getElementById('achievementToast');
        const iconEl = document.getElementById('achievementIcon');
        const nameEl = document.getElementById('achievementName');

        if (!toast) return;

        if (iconEl) iconEl.textContent = icon;
        if (nameEl) nameEl.textContent = name;

        this.achievementShowing = true;
        toast.classList.remove('show');
        void toast.offsetWidth;
        toast.classList.add('show');

        setTimeout(() => {
            toast.classList.remove('show');
            this.achievementShowing = false;
            this.processAchievementQueue();
        }, 3500);
    },

    /**
     * Show zone discovery animation
     */
    showZoneDiscovery(zoneName, subtitle = 'Zone Discovered') {
        const el = document.getElementById('zoneDiscovery');
        const nameEl = document.getElementById('zoneName');
        const subEl = el?.querySelector('.zone-discovery-subtitle');

        if (!el) return;

        if (nameEl) nameEl.textContent = zoneName;
        if (subEl) subEl.textContent = subtitle;

        el.classList.remove('show');
        void el.offsetWidth;
        el.classList.add('show');

        setTimeout(() => el.classList.remove('show'), 3000);
    },

    /**
     * Set weather effect
     */
    setWeather(type = null) {
        const overlay = document.getElementById('weatherOverlay');
        if (!overlay) return;

        overlay.className = 'weather-overlay';
        if (type) {
            overlay.classList.add(type);
        }
    },

    /**
     * Flash lightning (for high strain moments)
     */
    flashLightning() {
        const overlay = document.getElementById('weatherOverlay');
        if (!overlay) return;

        overlay.classList.add('lightning');
        setTimeout(() => overlay.classList.remove('lightning'), 200);
    },

    /**
     * Update HP/MP resource bars
     */
    updateResourceBars() {
        const hpFill = document.getElementById('hpBarFill');
        const hpText = document.getElementById('hpBarText');
        const mpFill = document.getElementById('mpBarFill');
        const mpText = document.getElementById('mpBarText');

        // VRAM as HP
        if (hpFill && GameState.vramTotal > 0) {
            const pct = (GameState.vramUsed / GameState.vramTotal) * 100;
            hpFill.style.width = `${pct}%`;
        }
        if (hpText) {
            hpText.textContent = `${GameState.vramUsed?.toFixed(1) || '--'} / ${GameState.vramTotal || 24} GB`;
        }

        // RAM as MP (estimate or placeholder)
        // Could fetch from /api/system in the future
        if (mpFill) {
            mpFill.style.width = '45%'; // Placeholder
        }
        if (mpText) {
            mpText.textContent = '-- / -- GB';
        }
    },

    /**
     * Add shimmer effect to new battle log entries
     */
    shimmerLogEntry(entry) {
        entry.classList.add('new-event');
        setTimeout(() => entry.classList.remove('new-event'), 800);
    },

    /**
     * Check for state changes and trigger effects
     */
    checkForEvents(prevState) {
        // XP gained (steps increased)
        const stepDiff = GameState.currentStep - this.prevStep;
        if (stepDiff > 0 && stepDiff < 1000 && GameState.isTraining) {
            // Show floating XP every ~100 steps
            if (Math.random() < 0.05) {
                this.showFloatingNumber(`+${stepDiff} XP`, 'xp');
            }
        }

        // Level up detection
        if (GameState.totalLevel > this.prevLevel && this.prevLevel > 0) {
            this.showAchievement(`Level ${GameState.totalLevel}!`, '⬆️');
            this.showZoneDiscovery(`Level ${GameState.totalLevel}`, 'Level Up!');
        }

        // High accuracy = critical hit
        if (GameState.currentSkillAcc > 90 && this.prevAccuracy <= 90 && this.prevAccuracy > 0) {
            this.showCriticalHit('90%+ ACCURACY!');
        }

        // High strain = storm weather
        if (GameState.loss > 2.0) {
            this.setWeather('storm');
            if (Math.random() < 0.1) {
                this.flashLightning();
            }
        } else {
            this.setWeather(null);
        }

        // Update previous state
        this.prevStep = GameState.currentStep;
        this.prevLevel = GameState.totalLevel;
        this.prevAccuracy = GameState.currentSkillAcc;
    },

    /**
     * Roll dice animation for eval results
     */
    rollDice(callback) {
        const diceFrames = ['🎲', '🎲', '🎲'];
        let rolls = 0;
        const maxRolls = 8;

        const roll = setInterval(() => {
            rolls++;
            // Could update a dice display element here
            if (rolls >= maxRolls) {
                clearInterval(roll);
                if (callback) callback();
            }
        }, 100);
    },
};

// Hook into the main update loop
const originalUpdateAll = updateAll;
updateAll = function() {
    originalUpdateAll();
    RPGFlair.checkForEvents();
    RPGFlair.updateResourceBars();
};

// Hook into battle log rendering
const originalRenderBattleLog = renderBattleLog;
renderBattleLog = function(events) {
    originalRenderBattleLog(events);
    // Add shimmer to newest entries
    const container = document.getElementById('logEntries');
    if (container && container.firstChild) {
        RPGFlair.shimmerLogEntry(container.firstChild);
    }
};

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        init();
        RealmConsole.init();
        RPGFlair.init();
    });
} else {
    init();
    RealmConsole.init();
    RPGFlair.init();
}
