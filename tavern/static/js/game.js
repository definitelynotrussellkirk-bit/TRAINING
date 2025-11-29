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

    // Extract loss values
    const losses = lossHistory.map(h => h.loss || 0);
    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const range = maxLoss - minLoss || 1;

    // Update footer stats
    const minEl = document.getElementById('lossMin');
    const maxEl = document.getElementById('lossMax');
    const trendEl = document.getElementById('lossTrend');
    const rangeEl = document.getElementById('lossGraphRange');

    if (minEl) minEl.textContent = `Min: ${minLoss.toFixed(3)}`;
    if (maxEl) maxEl.textContent = `Max: ${maxLoss.toFixed(3)}`;
    if (rangeEl) rangeEl.textContent = `Last ${losses.length} updates`;

    // Calculate trend (compare first half vs second half average)
    if (losses.length >= 4) {
        const halfIdx = Math.floor(losses.length / 2);
        const firstHalfAvg = losses.slice(0, halfIdx).reduce((a, b) => a + b, 0) / halfIdx;
        const secondHalfAvg = losses.slice(halfIdx).reduce((a, b) => a + b, 0) / (losses.length - halfIdx);
        const diff = secondHalfAvg - firstHalfAvg;

        if (trendEl) {
            trendEl.classList.remove('improving', 'degrading', 'stable');
            if (diff < -0.05) {
                trendEl.textContent = '↓ Improving';
                trendEl.classList.add('improving');
            } else if (diff > 0.05) {
                trendEl.textContent = '↑ Degrading';
                trendEl.classList.add('degrading');
            } else {
                trendEl.textContent = '→ Stable';
                trendEl.classList.add('stable');
            }
        }
    }

    // Draw threshold line at loss=1.0 (good target)
    const thresholdY = height - padding - ((1.0 - minLoss) / range) * (height - 2 * padding);
    if (thresholdY > padding && thresholdY < height - padding) {
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padding, thresholdY);
        ctx.lineTo(width - padding, thresholdY);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw loss line
    ctx.strokeStyle = '#ef4444';  // Red for high loss
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const stepX = (width - 2 * padding) / (losses.length - 1 || 1);
    losses.forEach((loss, i) => {
        const x = padding + i * stepX;
        const y = height - padding - ((loss - minLoss) / range) * (height - 2 * padding);

        // Color based on loss value
        if (loss < 1.0) {
            ctx.strokeStyle = '#22c55e';  // Green for low loss
        } else if (loss < 2.0) {
            ctx.strokeStyle = '#f59e0b';  // Yellow for medium
        } else {
            ctx.strokeStyle = '#ef4444';  // Red for high
        }

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // Draw current value marker (last point)
    if (losses.length > 0) {
        const lastLoss = losses[losses.length - 1];
        const lastX = width - padding;
        const lastY = height - padding - ((lastLoss - minLoss) / range) * (height - 2 * padding);

        ctx.fillStyle = lastLoss < 1.0 ? '#22c55e' : lastLoss < 2.0 ? '#f59e0b' : '#ef4444';
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
// SAGA (Persistent Battle Log)
// ============================================

// Track what we've already displayed to avoid duplicates
let lastSagaTimestamp = null;
let sagaInitialized = false;

async function fetchSaga() {
    try {
        const response = await fetch(`/saga?limit=30&_t=${Date.now()}`, {
            cache: 'no-store'
        });
        if (!response.ok) return;

        const data = await response.json();
        if (data.tales && data.tales.length > 0) {
            renderSaga(data.tales);
        }
    } catch (error) {
        console.error('Failed to fetch saga:', error);
    }
}

function renderSaga(tales) {
    const container = $('#logEntries');
    if (!container) return;

    // First load: replace everything
    if (!sagaInitialized) {
        container.innerHTML = '';
        sagaInitialized = true;
    }

    // Tales come newest-first, we display newest at top
    // Only add tales we haven't seen (based on timestamp)
    const newTales = [];
    for (const tale of tales) {
        const taleTs = `${tale.date}T${tale.time}`;
        if (lastSagaTimestamp && taleTs <= lastSagaTimestamp) {
            break;  // Already have this and older ones
        }
        newTales.push(tale);
    }

    // Update timestamp tracker
    if (tales.length > 0) {
        lastSagaTimestamp = `${tales[0].date}T${tales[0].time}`;
    }

    // Add new tales at the top (they're already newest-first)
    for (const tale of newTales.reverse()) {
        const entry = document.createElement('div');
        entry.className = `log-entry ${getCategoryClass(tale.category)}`;
        entry.innerHTML = `${tale.icon} [${tale.time}] ${tale.message}`;
        container.insertBefore(entry, container.firstChild);
    }

    // Keep only 30 entries max
    while (container.children.length > 30) {
        container.removeChild(container.lastChild);
    }
}

function getCategoryClass(category) {
    const classMap = {
        'quest': 'info',
        'combat': 'info',
        'hero': 'success',
        'champion': 'success',
        'training': 'info',
        'system': 'warning',
    };
    return classMap[category] || 'info';
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
    const prevTraining = GameState.isTraining;
    const prevStep = GameState.currentStep;
    const prevTotalLevel = GameState.totalLevel;

    // Training status
    const training = data.training;
    if (training) {
        GameState.isTraining = training.status === 'training';
        GameState.currentStep = training.current_step || 0;
        GameState.totalSteps = training.total_steps || 0;
        GameState.loss = training.loss || 0;
        GameState.valLoss = training.validation_loss || 0;
        GameState.currentQuest = training.current_file || null;
        GameState.questProgress = training.progress_percent || 0;
        GameState.stepsPerSecond = training.steps_per_second || 0;
        GameState.etaSeconds = training.eta_seconds || 0;
        GameState.learningRate = training.learning_rate || 0;
        // Loss history for graph
        if (training.train_loss_history) {
            GameState.lossHistory = training.train_loss_history;
            renderLossChart(GameState.lossHistory);
        }
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

    // No campaign at all
    if (!campaign || !campaign.id) {
        section.classList.add('no-campaign');
        titleEl.textContent = 'No Active Campaign';
        bodyEl.textContent = 'Create or select a campaign to begin your hero\'s journey.';
        btnEl.textContent = 'Open Campaign';
        btnEl.onclick = () => { window.location.href = '/campaign'; };
        return;
    }

    // Show recommendation
    const rec = campaign.recommendation;
    if (!rec) {
        titleEl.textContent = 'Ready to Train';
        bodyEl.textContent = 'Run a training session to push your hero further.';
        btnEl.textContent = 'Run 2,000 Steps';
        btnEl.onclick = () => startTraining(2000);
        return;
    }

    titleEl.textContent = rec.title;
    bodyEl.textContent = rec.description + (rec.reason ? ` (${rec.reason})` : '');

    const steps = rec.suggested_steps || 2000;
    btnEl.textContent = `Run ${steps.toLocaleString()} Steps`;
    btnEl.onclick = () => startTraining(steps);
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

    // Fetch saga (persistent battle log)
    fetchSaga();

    // Start data fetching (includes vault data)
    fetchGameData();

    // Fetch hint data (quests, task master)
    fetchHintData();

    // Fetch momentum state (blockers, forward progress)
    fetchMomentum();

    // Fetch next action recommendation (main CTA)
    refreshNextAction();

    // Set up polling using createPollerGroup for centralized management
    pollers.add('gameData', fetchGameData, CONFIG.UPDATE_INTERVAL, { immediate: false });
    pollers.add('saga', fetchSaga, 5000, { immediate: false });
    pollers.add('hints', fetchHintData, 10000, { immediate: false });
    pollers.add('skills', fetchSkills, 30000, { immediate: false });
    pollers.add('time', updateTime, 1000, { immediate: false });
    pollers.add('momentum', fetchMomentum, 15000, { immediate: false }); // Check momentum every 15s

    // Start all pollers
    pollers.startAll();

    // Register for real-time SSE training updates
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

                // Add to loss history for graph
                if (data.loss && data.step) {
                    GameState.lossHistory.push({ step: data.step, loss: data.loss });
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

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
