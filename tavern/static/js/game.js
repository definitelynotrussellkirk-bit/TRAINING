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
// SOUND MANAGER (Web Audio API)
// ============================================

const SoundManager = {
    ctx: null,
    muted: false,
    volume: 0.3,

    init() {
        // Load mute preference
        this.muted = localStorage.getItem('realm_sound_muted') === 'true';
        // AudioContext created on first user interaction (browser requirement)
    },

    _ensureContext() {
        if (!this.ctx) {
            try {
                this.ctx = new (window.AudioContext || window.webkitAudioContext)();
            } catch (e) {
                console.warn('[Sound] Web Audio API not supported');
                return false;
            }
        }
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
        return true;
    },

    toggle() {
        this.muted = !this.muted;
        localStorage.setItem('realm_sound_muted', this.muted);
        return this.muted;
    },

    // Play a tone with given frequency and duration
    _playTone(freq, duration = 0.15, type = 'sine', volume = this.volume) {
        if (this.muted || !this._ensureContext()) return;

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = type;
        osc.frequency.setValueAtTime(freq, this.ctx.currentTime);

        gain.gain.setValueAtTime(volume, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start();
        osc.stop(this.ctx.currentTime + duration);
    },

    // Play a sequence of tones (for melodies)
    _playSequence(notes, baseVolume = this.volume) {
        if (this.muted || !this._ensureContext()) return;

        let time = this.ctx.currentTime;
        notes.forEach(([freq, duration, type = 'sine']) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();

            osc.type = type;
            osc.frequency.setValueAtTime(freq, time);

            gain.gain.setValueAtTime(baseVolume, time);
            gain.gain.exponentialRampToValueAtTime(0.001, time + duration * 0.9);

            osc.connect(gain);
            gain.connect(this.ctx.destination);

            osc.start(time);
            osc.stop(time + duration);
            time += duration;
        });
    },

    // === SOUND EFFECTS ===

    // Level up - rising triumphant melody
    levelUp() {
        this._playSequence([
            [523, 0.1],   // C5
            [659, 0.1],   // E5
            [784, 0.1],   // G5
            [1047, 0.25], // C6 (hold)
        ], this.volume * 0.8);
    },

    // Achievement unlocked - fanfare
    achievement() {
        this._playSequence([
            [784, 0.08],  // G5
            [880, 0.08],  // A5
            [988, 0.08],  // B5
            [1047, 0.15], // C6
            [1175, 0.2],  // D6
        ], this.volume * 0.7);
    },

    // Quest complete - satisfying ding
    questComplete() {
        this._playSequence([
            [880, 0.1],   // A5
            [1109, 0.2],  // C#6
        ], this.volume * 0.6);
    },

    // Checkpoint saved - soft confirmation
    checkpoint() {
        this._playTone(659, 0.1, 'sine', this.volume * 0.4);  // E5
        setTimeout(() => this._playTone(784, 0.15, 'sine', this.volume * 0.3), 100);  // G5
    },

    // Training step - subtle tick (very quiet, for ambient feel)
    step() {
        this._playTone(440 + Math.random() * 100, 0.03, 'sine', this.volume * 0.05);
    },

    // Error/warning - dissonant
    error() {
        this._playTone(220, 0.15, 'sawtooth', this.volume * 0.3);
        setTimeout(() => this._playTone(185, 0.2, 'sawtooth', this.volume * 0.25), 100);
    },

    // Button click - subtle UI feedback
    click() {
        this._playTone(600, 0.05, 'sine', this.volume * 0.15);
    },

    // Zone discovery - mystical
    zoneDiscovery() {
        this._playSequence([
            [392, 0.15, 'triangle'],  // G4
            [494, 0.15, 'triangle'],  // B4
            [587, 0.15, 'triangle'],  // D5
            [784, 0.3, 'triangle'],   // G5
        ], this.volume * 0.5);
    },

    // Critical hit - impact
    critical() {
        this._playTone(150, 0.1, 'square', this.volume * 0.4);
        setTimeout(() => this._playTone(200, 0.15, 'square', this.volume * 0.3), 50);
    },
};

// Initialize sound manager
SoundManager.init();

// ============================================
// MUD-STYLE TEXT UTILITIES
// ============================================

const MudStyle = {
    /**
     * Generate a text-based progress bar using Unicode blocks
     * @param {number} percent - Value from 0-100
     * @param {number} width - Number of characters for the bar
     * @returns {object} - {filled: "████", empty: "░░░░", text: "[████░░░░]"}
     */
    progressBar(percent, width = 10) {
        const pct = Math.max(0, Math.min(100, percent));
        const filled = Math.round((pct / 100) * width);
        const empty = width - filled;

        const filledStr = '█'.repeat(filled);
        const emptyStr = '░'.repeat(empty);

        return {
            filled: filledStr,
            empty: emptyStr,
            text: `[${filledStr}${emptyStr}]`,
            html: `<span class="mud-bar-fill">${filledStr}</span><span class="mud-bar-empty">${emptyStr}</span>`
        };
    },

    /**
     * Format a number with color class based on type
     */
    colorValue(value, type) {
        const classes = {
            damage: 'val-damage',
            crit: 'val-crit',
            heal: 'val-heal',
            xp: 'val-xp',
            gold: 'val-gold',
            level: 'val-level',
            step: 'val-step',
            loss: 'val-loss',
            acc: 'val-acc'
        };
        return `<span class="${classes[type] || ''}">${value}</span>`;
    },

    /**
     * Format combat message with colored values
     */
    formatCombat(message) {
        // Replace common patterns with colored versions
        return message
            // Loss values
            .replace(/loss[:\s]+(\d+\.?\d*)/gi, (m, v) => `loss: ${this.colorValue(v, 'loss')}`)
            // Accuracy
            .replace(/(\d+\.?\d*%)\s*(accuracy|acc)/gi, (m, v) => `${this.colorValue(v, 'acc')} accuracy`)
            // XP/Steps
            .replace(/\+?(\d+)\s*(xp|experience|steps?)/gi, (m, v, t) => `+${this.colorValue(v, 'xp')} ${t}`)
            // Gold
            .replace(/\+?(\d+)\s*gold/gi, (m, v) => `+${this.colorValue(v, 'gold')} gold`)
            // Level references
            .replace(/level\s*(\d+)/gi, (m, v) => `Level ${this.colorValue(v, 'level')}`)
            // Checkpoint steps
            .replace(/step\s*(\d+)/gi, (m, v) => `step ${this.colorValue(v, 'step')}`);
    },

    /**
     * Add decorative separator
     */
    separator(type = 'single') {
        const chars = {
            single: '────────────────────────',
            double: '════════════════════════',
            ornate: '╔═══╦═══╦═══╦═══╗'
        };
        return chars[type] || chars.single;
    },

    /**
     * Apply training glow to hero frame
     */
    setHeroTraining(isTraining) {
        const portrait = document.getElementById('heroPortrait');
        if (portrait) {
            portrait.classList.toggle('training', isTraining);
        }
    },

    /**
     * Add text animation class temporarily
     */
    animateText(element, animation = 'text-glow', duration = 2000) {
        if (!element) return;
        element.classList.add(animation);
        setTimeout(() => element.classList.remove(animation), duration);
    }
};

// Export for global use
window.MudStyle = MudStyle;

// Global toggle function for HTML onclick
function toggleSound() {
    const muted = SoundManager.toggle();
    const btn = document.getElementById('soundToggle');
    const icon = document.getElementById('soundIcon');

    if (btn) {
        btn.classList.toggle('muted', muted);
    }
    if (icon) {
        icon.textContent = muted ? '🔇' : '🔊';
    }

    // Play a click sound to confirm (if not muted)
    if (!muted) {
        SoundManager.click();
    }

    console.log(`[Sound] ${muted ? 'Muted' : 'Unmuted'}`);
}

// Initialize sound toggle button state on load
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('soundToggle');
    const icon = document.getElementById('soundIcon');

    if (btn && SoundManager.muted) {
        btn.classList.add('muted');
    }
    if (icon && SoundManager.muted) {
        icon.textContent = '🔇';
    }
});

// ============================================
// GAME STATE
// ============================================

const GameState = {
    // Hero
    totalLevel: 0,      // Sum of all MASTERED skill levels
    currentStep: 0,
    previousStep: 0,
    totalEvals: 0,      // Real: total skill evaluations (curriculum + passive)
    passiveEvals: 0,    // Passive evals from eval_runner

    // Current skill being trained
    currentSkill: 'BINARY',
    currentSkillMastered: 0,   // Highest level mastered
    currentSkillTraining: 1,   // Level being trained (mastered + 1)
    currentSkillAcc: 0,

    // Training
    isTraining: false,
    isPaused: false,
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
    sylloEffort: 0,       // Cumulative effort for SY skill
    binaryMastered: 0,
    binaryTraining: 1,
    binaryAcc: 0,
    binaryEvals: 0,
    binaryEffort: 0,      // Cumulative effort for BIN skill

    // GPU (real hardware stats)
    vramUsed: 0,
    vramTotal: 24,
    ramUsed: 0,
    ramTotal: 0,
    gpuTemp: 0,
    gpuUtil: 0,

    // Vault (real checkpoint data)
    checkpointCount: 0,
    totalSize: 0,
    bestCheckpoint: null,

    // State flags
    realmSyncActive: false,
    noCampaign: true,  // True until we confirm a campaign is active
    heroId: null,
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
        // Use nullish coalescing (??) to preserve existing values when data is null/undefined
        // This prevents flickering when different data sources have different update frequencies
        GameState.isTraining = training.status === 'training';
        GameState.isPaused = training.status === 'paused';
        GameState.currentStep = training.step ?? GameState.currentStep ?? 0;
        GameState.totalSteps = training.totalSteps ?? GameState.totalSteps ?? 0;
        GameState.loss = training.loss ?? GameState.loss ?? 0;
        GameState.learningRate = training.learningRate ?? GameState.learningRate ?? 0;
        GameState.currentQuest = training.file ?? GameState.currentQuest ?? null;
        GameState.stepsPerSecond = training.speed ?? GameState.stepsPerSecond ?? 0;
        GameState.etaSeconds = training.etaSeconds ?? GameState.etaSeconds ?? 0;

        // Quest progress comes from progress_percent (batch progress)
        // NOT from currentStep/totalSteps (campaign steps - meaningless for continuous training)

        // Update UI
        updateBattleStatus();
        updateHeroStats();
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

    // Subscribe to skills state changes
    RealmState.subscribe('skills', (skillsData) => {
        if (!skillsData?.skills) return;

        const skills = skillsData.skills;

        // Update SY skill
        const syData = skills.sy || skills.syllo;
        if (syData) {
            GameState.sylloMastered = syData.mastered_level ?? GameState.sylloMastered ?? 0;
            GameState.sylloTraining = syData.training_level ?? GameState.sylloTraining ?? 1;
            GameState.sylloAcc = (syData.accuracy ?? syData.last_accuracy ?? 0) * 100;
            if (syData.eval_count != null) GameState.sylloEvals = syData.eval_count;
        }

        // Update BIN skill
        const binData = skills.bin || skills.binary;
        if (binData) {
            GameState.binaryMastered = binData.mastered_level ?? GameState.binaryMastered ?? 0;
            GameState.binaryTraining = binData.training_level ?? GameState.binaryTraining ?? 1;
            GameState.binaryAcc = (binData.accuracy ?? binData.last_accuracy ?? 0) * 100;
            if (binData.eval_count != null) GameState.binaryEvals = binData.eval_count;
        }

        // Only update totals if we have actual skill data
        if (syData || binData) {
            GameState.totalLevel = GameState.sylloMastered + GameState.binaryMastered;
        }

        // Get passive evals from API response
        if (data.total_passive_evals !== undefined) {
            GameState.passiveEvals = data.total_passive_evals;
            // Include passive evals in total count
            GameState.totalEvals = GameState.sylloEvals + GameState.binaryEvals + GameState.passiveEvals;
        }

        // Update UI
        renderSkills();
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

        // Strain zone hints (curriculum guidance)
        const floor = 0.02;
        const strain = Math.max(0, GameState.loss - floor);
        if (strain >= 0.5) {
            // Overload zone
            hints.push({ priority: 9, text: '⚠️ Overload zone - material too hard. Consider easier curriculum.' });
        } else if (strain < 0.1 && GameState.loss > 0) {
            // Recovery zone
            hints.push({ priority: 5, text: '↓ Recovery zone - hero is coasting. Consider leveling up.' });
        } else if (strain >= 0.1 && strain < 0.3) {
            // Productive zone - optional positive feedback
            hints.push({ priority: 2, text: '✓ Productive zone - optimal learning pace.' });
        }

        // High loss warning (legacy - keep for backwards compatibility)
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

function setText(selector, text) {
    const el = document.querySelector(selector);
    if (el) el.textContent = text;
}

function setWidth(selector, percent) {
    const el = document.querySelector(selector);
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
    updateStrainZone();
}

function updateBattleStatus() {
    const battleStatus = document.querySelector('#battleStatus');
    const idleIndicator = document.querySelector('#idleIndicator');

    // Priority 1: No campaign active - show error/setup state
    if (GameState.noCampaign) {
        battleStatus.classList.remove('fighting');
        battleStatus.classList.add('no-campaign');
        idleIndicator.classList.remove('active');
        idleIndicator.classList.add('error');
        setText('.idle-icon', '⚠️');
        setText('.idle-text', 'NO CAMPAIGN');

        setText('#battleIcon', '🚫');
        setText('#battleTitle', 'No Campaign Active');
        setText('#questName', 'Start a campaign to begin training');

        setWidth('#questProgressBar', 0);
        setText('#questProgressText', '--');

        setText('#battleStep', '--');
        setText('#battleSpeed', '--');
        setText('#battleStrain', '--');
        setText('#battleETA', '--');

    } else if (GameState.isTraining) {
        // Priority 2: Training mode
        battleStatus.classList.remove('no-campaign');
        battleStatus.classList.add('fighting');
        idleIndicator.classList.remove('error');
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
        // Priority 3: Idle mode (campaign active but not training)
        battleStatus.classList.remove('fighting');
        battleStatus.classList.remove('no-campaign');
        idleIndicator.classList.remove('active');
        idleIndicator.classList.remove('error');
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
        const response = await fetch(`/api/skills?_t=${Date.now()}`, {
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
    const container = document.querySelector('#skillsContainer');
    if (!container || skillsData.length === 0) return;

    container.innerHTML = skillsData.map(skill => {
        const progressPct = skill.max_level > 0
            ? (skill.mastered_level / skill.max_level) * 100
            : 0;

        const isActive = skill.id === GameState.currentSkill?.toLowerCase() ||
                        skill.short_name === GameState.currentSkill;

        // Format time ago for last eval
        let timeAgo = '';
        if (skill.last_eval_time) {
            const evalDate = new Date(skill.last_eval_time);
            const now = new Date();
            const diffMs = now - evalDate;
            const diffMins = Math.floor(diffMs / 60000);
            const diffHours = Math.floor(diffMs / 3600000);
            const diffDays = Math.floor(diffMs / 86400000);

            if (diffMins < 1) timeAgo = 'just now';
            else if (diffMins < 60) timeAgo = `${diffMins}m ago`;
            else if (diffHours < 24) timeAgo = `${diffHours}h ago`;
            else timeAgo = `${diffDays}d ago`;
        }

        // Last accuracy with time indicator
        const lastAccDisplay = skill.last_accuracy !== undefined
            ? `${skill.last_accuracy.toFixed(0)}%${timeAgo ? ` <span class="eval-time">(${timeAgo})</span>` : ''}`
            : '--';

        // Get effort for this skill
        let skillEffort = 0;
        if (skill.id === 'sy' || skill.id === 'syllo') {
            skillEffort = GameState.sylloEffort || 0;
        } else if (skill.id === 'bin' || skill.id === 'binary') {
            skillEffort = GameState.binaryEffort || 0;
        }

        // Effort bar - scaled to a reasonable max (100 = full bar)
        const effortPct = Math.min(100, (skillEffort / 100) * 100);
        const effortDisplay = skillEffort > 0 ? skillEffort.toFixed(1) : '0';

        // Generate MUD-style text bar for progress
        const mudProgress = MudStyle.progressBar(progressPct, 12);

        return `
            <div class="skill-card clickable ${isActive ? 'active' : ''}" data-skill="${skill.id}" style="--skill-color: ${skill.color}">
                <div class="skill-header">
                    <span class="skill-icon">${skill.icon}</span>
                    <span class="skill-name">${skill.short_name}</span>
                    <span class="skill-level">${progressPct.toFixed(0)}%</span>
                </div>
                <div class="skill-bar-container">
                    <div class="skill-bar" style="width: ${progressPct}%; background: ${skill.color}"></div>
                </div>
                <div class="mud-bar ${skill.id}" title="Progress: ${progressPct.toFixed(0)}%">[${mudProgress.html}]</div>
                <div class="skill-effort" title="Cumulative effort invested in this skill">
                    <span class="effort-label">Effort</span>
                    <div class="effort-bar-container">
                        <div class="effort-bar" style="width: ${effortPct}%"></div>
                    </div>
                    <span class="effort-value">${effortDisplay}</span>
                </div>
                <div class="skill-meta">
                    <span class="skill-acc">${lastAccDisplay}</span>
                    <span class="skill-desc">L${skill.mastered_level}/${skill.max_level}</span>
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

// ============================================
// EFFICIENCY DASHBOARD
// ============================================

async function updateEfficiency() {
    const container = document.getElementById('efficiencyBars');
    if (!container) return;

    try {
        const response = await fetch(`/api/strain/efficiency?_t=${Date.now()}`, { cache: 'no-store' });
        if (!response.ok) throw new Error('Failed to fetch efficiency');

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        const skills = data.skills || {};
        const ranking = data.ranking || [];

        // Find max efficiency for scaling bars
        let maxEfficiency = 0;
        for (const skill of Object.values(skills)) {
            if (skill.efficiency > maxEfficiency) maxEfficiency = skill.efficiency;
        }
        // Use level as fallback if no efficiency data yet
        if (maxEfficiency === 0) {
            for (const skill of Object.values(skills)) {
                if (skill.current_level > maxEfficiency) maxEfficiency = skill.current_level;
            }
        }
        maxEfficiency = maxEfficiency || 1;  // Avoid division by zero

        // Build HTML for each skill
        let html = '';
        ranking.forEach((skillId, index) => {
            const skill = skills[skillId];
            if (!skill) return;

            // Use effort from frontend if backend has none
            const frontendEffort = skillId === 'sy' ? GameState.sylloEffort : GameState.binaryEffort;
            const effort = skill.effort > 0 ? skill.effort : (frontendEffort || 0);

            // Calculate efficiency using frontend effort
            const efficiency = effort > 0 ? (skill.plastic_gain / effort) : 0;
            const displayValue = effort > 0 ? efficiency.toFixed(3) : skill.current_level;

            // Scale bar to max
            const barWidth = maxEfficiency > 0 ? (displayValue / maxEfficiency * 100) : 0;

            const rankClass = index === 0 ? 'first' : 'second';
            const rankIcon = index === 0 ? '1st' : '2nd';

            html += `
                <div class="efficiency-row">
                    <span class="efficiency-rank ${rankClass}">${rankIcon}</span>
                    <span class="efficiency-label">${skill.skill_name}</span>
                    <div class="efficiency-bar-container">
                        <div class="efficiency-bar ${skillId}" style="width: ${Math.min(100, barWidth)}%">
                            ${barWidth > 20 ? `<span class="efficiency-value">${displayValue}</span>` : ''}
                        </div>
                        ${barWidth <= 20 ? `<span class="efficiency-value">${displayValue}</span>` : ''}
                    </div>
                </div>
            `;
        });

        container.innerHTML = html || '<div class="efficiency-loading">No efficiency data</div>';
    } catch (error) {
        console.error('[Efficiency] Update failed:', error);
        container.innerHTML = '<div class="efficiency-loading">--</div>';
    }
}

// ============================================
// LEVEL TRANSITIONS LOG
// ============================================

async function updateTransitions() {
    const container = document.getElementById('transitionsList');
    if (!container) return;

    try {
        const response = await fetch(`/api/strain/transitions?_t=${Date.now()}`, { cache: 'no-store' });
        if (!response.ok) throw new Error('Failed to fetch transitions');

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        const transitions = data.transitions || [];

        if (transitions.length === 0) {
            container.innerHTML = '<div class="transitions-loading">No level-ups yet</div>';
            return;
        }

        // Build HTML for each transition (most recent first, limited to 5)
        let html = '';
        transitions.slice(0, 5).forEach((t) => {
            const skillIcon = t.skill_id === 'sy' ? '🧩' : '🔢';
            const timeAgo = formatTimeAgo(t.timestamp);

            html += `
                <div class="transition-item">
                    <span class="transition-icon">${skillIcon}</span>
                    <span class="transition-skill ${t.skill_id}">${t.skill_name}</span>
                    <span class="transition-level">
                        L${t.from_level}<span class="arrow">→</span>L${t.to_level}
                    </span>
                    <span class="transition-step">Step ${t.step}</span>
                </div>
            `;
        });

        container.innerHTML = html;
    } catch (error) {
        console.error('[Transitions] Update failed:', error);
        container.innerHTML = '<div class="transitions-loading">--</div>';
    }
}

function formatTimeAgo(timestamp) {
    if (!timestamp) return '';
    try {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffDays > 0) return `${diffDays}d ago`;
        if (diffHours > 0) return `${diffHours}h ago`;
        if (diffMins > 0) return `${diffMins}m ago`;
        return 'just now';
    } catch {
        return '';
    }
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
    const container = document.querySelector('#logEntries');
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

        // Determine log entry class based on channel/content
        let entryClass = getSeverityClass(event.severity);
        const channel = (event.channel || '').toLowerCase();
        const msg = (event.message || '').toLowerCase();

        // Add channel-specific classes for MUD-style coloring
        if (channel.includes('checkpoint') || msg.includes('checkpoint')) {
            entryClass += ' checkpoint';
        } else if (channel.includes('level') || msg.includes('level up')) {
            entryClass += ' levelup';
        } else if (channel.includes('quest') || msg.includes('quest complete')) {
            entryClass += ' quest-complete';
        } else if (msg.includes('loss') || msg.includes('strain')) {
            entryClass += ' combat-damage';
        }

        entry.className = `log-entry ${entryClass}`;

        // Format timestamp to just time
        const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString('en-US', { hour12: false }) : '--:--:--';

        // Apply MUD-style color formatting to message
        const formattedMsg = MudStyle.formatCombat(escapeHtml(event.message));

        entry.innerHTML = `${event.icon || '📢'} <span class="log-time">[${time}]</span> <span class="log-channel">${event.channel}</span> ${formattedMsg}`;
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
    const container = document.querySelector('#logEntries');
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
    const container = document.querySelector('#notifications');
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

/**
 * Fetch effort history from training logs for the Effort History chart.
 * This provides initial data on page load (before SSE events arrive).
 */
async function fetchEffortHistory() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/effort-history?limit=200&_t=${Date.now()}`, {
            cache: 'no-store'
        });
        if (!response.ok) return;

        const data = await response.json();
        if (data.history && data.history.length > 0) {
            // Populate lossHistory from API data
            GameState.lossHistory = data.history.map(entry => ({
                step: entry.step,
                loss: entry.loss,
                strain: entry.strain,
                effort: entry.effort,
            }));
            console.log(`[Game] Loaded ${data.history.length} effort history entries`);
            renderLossChart(GameState.lossHistory);
        }
    } catch (error) {
        console.error('Failed to fetch effort history:', error);
    }
}

/**
 * Update strain zone indicator based on current loss.
 * Strain = loss - floor (floor ~0.01-0.02 for most skills)
 * Zones: recovery (<0.1), productive (0.1-0.3), stretch (0.3-0.5), overload (>0.5)
 */
function updateStrainZone() {
    const el = document.getElementById('strainZone');
    if (!el) return;

    const loss = GameState.loss;
    if (!loss || !GameState.isTraining) {
        el.textContent = '';
        el.className = 'strain-zone';
        el.title = 'Training Zone';
        return;
    }

    // Approximate floor for current skill (default 0.02)
    const floor = 0.02;
    const strain = Math.max(0, loss - floor);

    let zone, icon, hint;
    if (strain < 0.1) {
        zone = 'recovery';
        icon = '\u2193';  // ↓
        hint = 'Under-challenged - consider leveling up';
    } else if (strain < 0.3) {
        zone = 'productive';
        icon = '\u2713';  // ✓
        hint = 'Optimal learning zone';
    } else if (strain < 0.5) {
        zone = 'stretch';
        icon = '\u2191';  // ↑
        hint = 'Challenging but sustainable';
    } else {
        zone = 'overload';
        icon = '\u26A0';  // ⚠
        hint = 'Too hard - consider reducing difficulty';
    }

    el.textContent = `${icon} ${zone.toUpperCase()}`;
    el.className = `strain-zone ${zone}`;
    el.title = hint;
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

    // Training status - SKIP when RealmState is active (it's the source of truth)
    // This prevents rubber-banding between /api/game and RealmState
    const training = data.state?.training || data.training;
    if (training && !GameState.realmSyncActive) {
        // Only process training data from /api/game when RealmState is NOT active
        GameState.isTraining = training.status === 'training';
        GameState.isPaused = training.status === 'paused';
        GameState.currentStep = training.step ?? training.current_step ?? GameState.currentStep ?? 0;
        GameState.totalSteps = training.total_steps ?? GameState.totalSteps ?? 0;
        GameState.loss = training.loss ?? GameState.loss ?? 0;
        GameState.valLoss = training.validation_loss ?? GameState.valLoss ?? 0;
        GameState.currentQuest = training.current_file ?? GameState.currentQuest ?? null;
        GameState.questProgress = training.progress_percent ?? GameState.questProgress ?? 0;
        GameState.stepsPerSecond = training.steps_per_second ?? training.speed ?? GameState.stepsPerSecond ?? 0;
        GameState.etaSeconds = training.eta_seconds ?? GameState.etaSeconds ?? 0;
        GameState.learningRate = training.learning_rate ?? GameState.learningRate ?? 0;

        // POLICY 3: Check staleness and update warning
        updateFreshnessWarning(training);
    } else if (training && GameState.realmSyncActive) {
        // When RealmState is active, only update quest progress (batch progress)
        // which isn't available from RealmState
        GameState.questProgress = training.progress_percent ?? GameState.questProgress ?? 0;
    }

    // GPU stats
    const gpu = data.gpu;
    if (gpu) {
        GameState.vramUsed = gpu.vram_used_gb || 0;
        GameState.vramTotal = gpu.vram_total_gb || 24;
        GameState.gpuTemp = gpu.temperature_c || 0;
        GameState.gpuUtil = gpu.utilization_pct || 0;
    }

    // System RAM stats
    const system = data.system;
    if (system) {
        GameState.ramUsed = system.ram_used_gb || 0;
        GameState.ramTotal = system.ram_total_gb || 0;
    }

    // Curriculum (skills) - SKIP when RealmState is active (it handles skills)
    // This prevents rubber-banding between /api/game and RealmState
    const curriculum = data.curriculum;
    if (curriculum?.skills && !GameState.realmSyncActive) {
        const skills = curriculum.skills;

        // Handle both old (syllo/binary) and new (sy/bin) skill IDs
        const syData = skills.sy || skills.syllo;
        if (syData) {
            GameState.sylloMastered = syData.mastered_level ?? Math.max(0, (syData.current_level || 1) - 1);
            GameState.sylloTraining = syData.training_level ?? syData.current_level ?? 1;
            GameState.sylloAcc = syData.recent_accuracy || 0;
            GameState.sylloEvals = syData.eval_count || 0;
        }
        const binData = skills.bin || skills.binary;
        if (binData) {
            GameState.binaryMastered = binData.mastered_level ?? Math.max(0, (binData.current_level || 1) - 1);
            GameState.binaryTraining = binData.training_level ?? binData.current_level ?? 1;
            GameState.binaryAcc = binData.recent_accuracy || 0;
            GameState.binaryEvals = binData.eval_count || 0;
        }

        // Total level = sum of all MASTERED skill levels
        GameState.totalLevel = GameState.sylloMastered + GameState.binaryMastered;

        // Total evals = SYLLO + BINARY + passive evaluations
        GameState.totalEvals = GameState.sylloEvals + GameState.binaryEvals + GameState.passiveEvals;
    }

    // Always update eval counts from /api/game (RealmState doesn't track these)
    if (curriculum?.skills) {
        const skills = curriculum.skills;
        const syData = skills.sy || skills.syllo;
        const binData = skills.bin || skills.binary;
        if (syData) GameState.sylloEvals = syData.eval_count || 0;
        if (binData) GameState.binaryEvals = binData.eval_count || 0;
        // Include passive evals in total
        GameState.totalEvals = GameState.sylloEvals + GameState.binaryEvals + GameState.passiveEvals;
    }

    // Determine current skill - always process skill_context from training (useful for both modes)
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
    } else if (!GameState.realmSyncActive) {
        // Fallback: infer from training file name (only when RealmState not active)
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

    // Move "Run Steps" card to bottom when training is active
    const nextActionCard = document.getElementById('nextActionSection');
    if (nextActionCard) {
        nextActionCard.classList.toggle('training-active', GameState.isTraining);
    }

    // Skill level up detection
    if (GameState.totalLevel > prevTotalLevel && prevTotalLevel > 0) {
        showNotification('Skill Level Up!', `Total level is now ${GameState.totalLevel}!`, 'success');
        SoundManager.levelUp();
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

    // MUD-style hero training glow
    MudStyle.setHeroTraining(GameState.isTraining);

    // Check for new achievements
    Achievements.checkAll();
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

        // Check if we have an active campaign
        if (!hero || !hero.hero_id) {
            GameState.noCampaign = true;
            GameState.heroId = null;
            updateBattleStatus();  // Refresh UI to show NO CAMPAIGN state
            return;
        }

        GameState.noCampaign = false;
        GameState.heroId = hero.hero_id;

        if (!hero.name) return;

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
async function updateNextActionUI(campaign, momentum) {
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

    // Show recommendation based on kind
    const rec = campaign.recommendation;
    const queueFiles = campaign.queue_files || 0;

    // Check auto-queue status for smarter guidance
    let autoQueueEnabled = false;
    try {
        const configRes = await fetch('/config');
        if (configRes.ok) {
            const config = await configRes.json();
            autoQueueEnabled = config.auto_generate?.enabled || false;
        }
    } catch (e) { /* ignore */ }

    // Handle different recommendation kinds
    if (!rec || rec.kind === 'train_steps') {
        // Training recommendation (default)
        if (queueFiles === 0) {
            // Override: No queue data
            section.classList.add('needs-data');

            if (autoQueueEnabled) {
                // Auto-queue is on - it should be generating
                titleEl.textContent = 'Generating Training Data...';
                bodyEl.innerHTML = `
                    <div style="margin-bottom: 0.75rem;">Auto-queue is enabled. Data should generate automatically.</div>
                    <div style="font-size: 0.85rem; opacity: 0.8;">If this persists, check the daemon status or generate manually.</div>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                        <button class="action-btn secondary" onclick="toggleAutoQueue(this)" style="font-size: 0.85rem; padding: 0.4rem 0.8rem;">
                            Disable Auto-Queue
                        </button>
                    </div>
                `;
                btnEl.textContent = 'Generate Now';
                btnEl.onclick = () => generateTrainingData(btnEl, 5000);
            } else {
                // Auto-queue is off - prompt to generate or enable
                titleEl.textContent = 'Generate Training Data';
                bodyEl.innerHTML = `
                    <div style="margin-bottom: 0.75rem;">The quest board is empty. Generate training quests to begin.</div>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                        <button class="action-btn secondary" onclick="toggleAutoQueue(this)" style="font-size: 0.85rem; padding: 0.4rem 0.8rem;">
                            Enable Auto-Queue
                        </button>
                    </div>
                `;
                btnEl.textContent = 'Generate 5,000 Examples';
                btnEl.onclick = () => generateTrainingData(btnEl, 5000);
            }
            checkAutoQueueStatus();
        } else {
            // Has data, show training option
            titleEl.textContent = rec?.title || 'Ready to Train';
            const desc = rec?.description || 'Run a training session to push your hero further.';
            const reason = rec?.reason ? ` (${rec.reason})` : '';
            bodyEl.innerHTML = `
                <div style="margin-bottom: 0.75rem;">${desc}${reason}</div>
                <div style="margin-bottom: 0.5rem; font-size: 0.85rem; opacity: 0.8;">📋 ${queueFiles} quest${queueFiles !== 1 ? 's' : ''} in queue</div>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                    <button class="action-btn secondary" onclick="toggleAutoRun(this)" style="font-size: 0.85rem; padding: 0.4rem 0.8rem;">
                        Enable Auto-Run
                    </button>
                </div>
            `;
            const steps = rec?.suggested_steps || 2000;
            btnEl.textContent = `Run ${steps.toLocaleString()} Steps`;
            btnEl.onclick = () => startTraining(steps);
            checkAutoRunStatus();
        }
    } else if (rec.kind === 'create_quest') {
        // Need to generate data
        section.classList.add('needs-data');

        if (autoQueueEnabled) {
            // Auto-queue is on - it should be generating
            titleEl.textContent = 'Generating Training Data...';
            bodyEl.innerHTML = `
                <div style="margin-bottom: 0.75rem;">Auto-queue is enabled. Data should generate automatically.</div>
                <div style="font-size: 0.85rem; opacity: 0.8;">If this persists, check the daemon status or generate manually.</div>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                    <button class="action-btn secondary" onclick="toggleAutoQueue(this)" style="font-size: 0.85rem; padding: 0.4rem 0.8rem;">
                        Disable Auto-Queue
                    </button>
                </div>
            `;
            btnEl.textContent = 'Generate Now';
            btnEl.onclick = () => generateTrainingData(btnEl, 5000);
        } else {
            titleEl.textContent = rec.title;
            bodyEl.innerHTML = `
                <div style="margin-bottom: 0.75rem;">${rec.description}</div>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                    <button class="action-btn secondary" onclick="toggleAutoQueue(this)" style="font-size: 0.85rem; padding: 0.4rem 0.8rem;">
                        Enable Auto-Queue
                    </button>
                </div>
            `;
            btnEl.textContent = 'Generate 5,000 Examples';
            btnEl.onclick = () => generateTrainingData(btnEl, 5000);
        }
        checkAutoQueueStatus();
    } else if (rec.kind === 'wait') {
        // Training in progress
        section.classList.add('training-active');
        titleEl.textContent = rec.title;
        bodyEl.innerHTML = `<div style="margin-bottom: 0.75rem;">${rec.description}</div>`;
        btnEl.textContent = 'Training...';
        btnEl.disabled = true;
    } else {
        // Unknown kind, fallback
        titleEl.textContent = rec.title || 'Next Step';
        bodyEl.innerHTML = `<div style="margin-bottom: 0.75rem;">${rec.description || 'Continue your journey.'}</div>`;
        btnEl.textContent = 'Continue';
        btnEl.onclick = () => { window.location.href = '/campaign'; };
    }
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

    // Initialize achievement system
    Achievements.init();

    // Load active campaign/hero first
    fetchCampaignData();

    // Initial UI update
    updateAll();

    // Fetch skills (from YAML configs)
    fetchSkills();

    // Fetch effort history for the chart (from training logs)
    fetchEffortHistory();

    // Try to use unified RealmState (single source of truth)
    GameState.realmSyncActive = initRealmStateSync();

    if (GameState.realmSyncActive) {
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
    pollers.add('vault', fetchVaultData, 60000, { immediate: true }); // Vault data from ledger (accurate)
    pollers.add('efficiency', updateEfficiency, 30000, { immediate: true }); // Efficiency dashboard
    pollers.add('transitions', updateTransitions, 60000, { immediate: true }); // Level transitions log

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
                    const floor = 0.02;  // Target floor loss (sy=0.01, bin=0.02)
                    const strain = Math.max(0, data.loss - floor);  // Strain = loss - floor, min 0

                    // Calculate cumulative effort
                    let effort = strain;
                    if (GameState.lossHistory.length > 0) {
                        const lastEntry = GameState.lossHistory[GameState.lossHistory.length - 1];
                        effort = (lastEntry.effort || lastEntry.loss) + strain;  // Cumulative
                    }

                    // Track per-skill effort
                    const activeSkill = (GameState.currentSkill || '').toLowerCase();
                    if (activeSkill.includes('sy') || activeSkill.includes('syllo')) {
                        GameState.sylloEffort += strain;
                    } else if (activeSkill.includes('bin') || activeSkill.includes('binary')) {
                        GameState.binaryEffort += strain;
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

                // Show floating XP number for quest completion
                const steps = data.steps || 0;
                if (steps > 0) {
                    RPGFlair.showFloatingNumber(`+${steps} XP`, 'xp');
                    showNotification('Quest Complete!', `Earned ${steps} experience points`, 'success');
                    SoundManager.questComplete();
                }
            });

            // Handle training.checkpoint events (new champion/save)
            window.eventStream.on('training.checkpoint', (event) => {
                const data = event.data || {};
                const step = data.step || '?';
                console.log('[Game] Checkpoint saved:', step);

                // Show gold floating number (saving = treasure)
                RPGFlair.showFloatingNumber('+100 Gold', 'gold');
                showNotification('Checkpoint Saved!', `Progress secured at step ${step}`, 'success');
                SoundManager.checkpoint();

                // Increment checkpoint count
                GameState.checkpointCount = (GameState.checkpointCount || 0) + 1;
                updateVault();
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

    // Show controls when training or paused (not when idle/completed)
    const showControls = GameState.isTraining || GameState.isPaused;
    if (controls) {
        controls.style.display = showControls ? 'flex' : 'none';
    }

    // Show correct button based on pause state
    if (btnPause && btnResume) {
        if (GameState.isPaused) {
            // Training is paused - show Resume, hide Pause
            btnPause.style.display = 'none';
            btnResume.style.display = 'inline-flex';
        } else if (GameState.isTraining) {
            // Training is active - show Pause, hide Resume
            btnPause.style.display = 'inline-flex';
            btnResume.style.display = 'none';
        } else {
            // Idle - hide both
            btnPause.style.display = 'none';
            btnResume.style.display = 'none';
        }
    }
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
// 🏆 ACHIEVEMENT SYSTEM
// ============================================

/**
 * Achievement definitions and tracking
 */
const Achievements = {
    // Achievement definitions
    definitions: {
        // Training milestones
        'first_steps': { name: 'First Steps', icon: '👣', desc: 'Complete your first training step', check: () => GameState.currentStep >= 1 },
        'century': { name: 'Century', icon: '💯', desc: 'Reach 100 training steps', check: () => GameState.currentStep >= 100 },
        'thousand': { name: 'Thousand Steps', icon: '🚶', desc: 'Reach 1,000 training steps', check: () => GameState.currentStep >= 1000 },
        'ten_thousand': { name: 'Marathon', icon: '🏃', desc: 'Reach 10,000 training steps', check: () => GameState.currentStep >= 10000 },
        'hundred_thousand': { name: 'Ultramarathon', icon: '🏅', desc: 'Reach 100,000 training steps', check: () => GameState.currentStep >= 100000 },

        // Level milestones
        'level_1': { name: 'Awakening', icon: '✨', desc: 'Reach total level 1', check: () => GameState.totalLevel >= 1 },
        'level_5': { name: 'Apprentice', icon: '📚', desc: 'Reach total level 5', check: () => GameState.totalLevel >= 5 },
        'level_10': { name: 'Journeyman', icon: '🎓', desc: 'Reach total level 10', check: () => GameState.totalLevel >= 10 },
        'level_20': { name: 'Expert', icon: '🏆', desc: 'Reach total level 20', check: () => GameState.totalLevel >= 20 },
        'level_50': { name: 'Master', icon: '👑', desc: 'Reach total level 50', check: () => GameState.totalLevel >= 50 },

        // Accuracy achievements
        'sharp_eye': { name: 'Sharp Eye', icon: '🎯', desc: 'Achieve 80% accuracy on any skill', check: () => GameState.currentSkillAcc >= 80 },
        'precision': { name: 'Precision', icon: '💎', desc: 'Achieve 90% accuracy on any skill', check: () => GameState.currentSkillAcc >= 90 },
        'perfectionist': { name: 'Perfectionist', icon: '🌟', desc: 'Achieve 100% accuracy on any skill', check: () => GameState.currentSkillAcc >= 100 },

        // Checkpoint achievements
        'first_save': { name: 'First Save', icon: '💾', desc: 'Save your first checkpoint', check: () => GameState.checkpointCount >= 1 },
        'hoarder': { name: 'Hoarder', icon: '📦', desc: 'Accumulate 10 checkpoints', check: () => GameState.checkpointCount >= 10 },
        'archivist': { name: 'Archivist', icon: '🏛️', desc: 'Accumulate 50 checkpoints', check: () => GameState.checkpointCount >= 50 },

        // Skill achievements
        'dual_wielder': { name: 'Dual Wielder', icon: '⚔️', desc: 'Train in 2 different skills', check: () => (GameState.sylloMastered > 0 ? 1 : 0) + (GameState.binaryMastered > 0 ? 1 : 0) >= 2 },

        // Strain zone achievements
        'in_the_zone': { name: 'In The Zone', icon: '🟢', desc: 'Train in the productive zone', check: () => {
            const strain = Math.max(0, GameState.loss - 0.02);
            return strain >= 0.1 && strain < 0.3 && GameState.isTraining;
        }},
        'pushing_limits': { name: 'Pushing Limits', icon: '🟠', desc: 'Train in the stretch zone', check: () => {
            const strain = Math.max(0, GameState.loss - 0.02);
            return strain >= 0.3 && strain < 0.5 && GameState.isTraining;
        }},
    },

    // Unlocked achievements (loaded from localStorage)
    unlocked: {},

    // Initialize from localStorage
    init() {
        try {
            const saved = localStorage.getItem('realm_achievements');
            if (saved) {
                this.unlocked = JSON.parse(saved);
                console.log(`[Achievements] Loaded ${Object.keys(this.unlocked).length} achievements`);
            }
        } catch (e) {
            console.error('[Achievements] Failed to load:', e);
            this.unlocked = {};
        }
    },

    // Save to localStorage
    save() {
        try {
            localStorage.setItem('realm_achievements', JSON.stringify(this.unlocked));
        } catch (e) {
            console.error('[Achievements] Failed to save:', e);
        }
    },

    // Check for new achievements
    checkAll() {
        for (const [id, def] of Object.entries(this.definitions)) {
            if (!this.unlocked[id] && def.check()) {
                this.unlock(id);
            }
        }
    },

    // Unlock an achievement
    unlock(id) {
        if (this.unlocked[id]) return; // Already unlocked

        const def = this.definitions[id];
        if (!def) return;

        this.unlocked[id] = {
            unlockedAt: new Date().toISOString(),
            name: def.name
        };
        this.save();

        console.log(`[Achievements] Unlocked: ${def.name}`);

        // Show achievement toast via RPGFlair
        if (typeof RPGFlair !== 'undefined' && RPGFlair.showAchievement) {
            RPGFlair.showAchievement(def.name, def.icon);
        }

        // Also show notification
        if (typeof showNotification === 'function') {
            showNotification('Achievement Unlocked!', `${def.icon} ${def.name} - ${def.desc}`, 'success');
        }

        // Play achievement sound
        SoundManager.achievement();
    },

    // Get count of unlocked achievements
    getCount() {
        return Object.keys(this.unlocked).length;
    },

    // Get total achievements
    getTotal() {
        return Object.keys(this.definitions).length;
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
    // All possible locations (randomly selected for display)
    allLocations: [
        { icon: '🏰', name: 'Tavern', url: '/' },
        { icon: '🏛️', name: 'Guild', url: '/guild' },
        { icon: '🗝️', name: 'Vault', url: '/vault' },
        { icon: '🔮', name: 'Oracle', url: '/oracle' },
        { icon: '🛕', name: 'Temple', url: '/temple' },
        { icon: '📜', name: 'Quests', url: '/quests' },
        { icon: '🛡️', name: 'Garrison', url: '/garrison' },
        { icon: '🔥', name: 'Forge', url: '/forge' },
        { icon: '📋', name: 'Jobs', url: '/jobs' },
    ],
    // Corner positions for structures
    cornerPositions: [
        { x: 0, y: 0 },  // top-left
        { x: 4, y: 0 },  // top-right
        { x: 0, y: 2 },  // bottom-left
        { x: 4, y: 2 },  // bottom-right
    ],
    minimapStructures: [],  // Populated randomly on init

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

        // Randomly select 4 locations from the pool
        const shuffled = [...this.allLocations].sort(() => Math.random() - 0.5);
        this.minimapStructures = shuffled.slice(0, 4).map((loc, i) => ({
            ...loc,
            x: this.cornerPositions[i].x,
            y: this.cornerPositions[i].y,
        }));

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

        // Add structures (clickable)
        for (const struct of this.minimapStructures) {
            const el = document.createElement('div');
            el.className = 'minimap-cell minimap-structure';
            el.textContent = struct.icon;
            el.title = struct.name;
            el.style.left = `${10 + struct.x * 25}px`;
            el.style.top = `${10 + struct.y * 25}px`;
            el.style.cursor = 'pointer';
            el.onclick = () => window.location.href = struct.url;
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
     * Update HP/MP/Stamina resource bars (WoW + MUD style)
     */
    updateResourceBars() {
        const hpFill = document.getElementById('hpBarFill');
        const hpText = document.getElementById('hpBarText');
        const mpFill = document.getElementById('mpBarFill');
        const mpText = document.getElementById('mpBarText');
        const staminaFill = document.getElementById('staminaBarFill');
        const staminaText = document.getElementById('staminaBarText');

        // MUD bar elements
        const mudHp = document.getElementById('mudBarHp');
        const mudMp = document.getElementById('mudBarMp');
        const mudStamina = document.getElementById('mudBarStamina');

        // VRAM as HP (round total to nearest integer for cleaner display)
        const vramTotalRounded = Math.round(GameState.vramTotal || 24);
        let hpPct = 0;
        if (hpFill && GameState.vramTotal > 0) {
            hpPct = (GameState.vramUsed / GameState.vramTotal) * 100;
            hpFill.style.width = `${hpPct}%`;
        }
        if (hpText) {
            hpText.textContent = `${GameState.vramUsed?.toFixed(1) || '--'} / ${vramTotalRounded} GB`;
        }
        // Update MUD HP bar
        if (mudHp) {
            const bar = MudStyle.progressBar(hpPct, 12);
            mudHp.innerHTML = `[${bar.html}]`;
        }

        // RAM as MP
        const ramTotalRounded = Math.round(GameState.ramTotal || 0);
        let mpPct = 0;
        if (mpFill && GameState.ramTotal > 0) {
            mpPct = (GameState.ramUsed / GameState.ramTotal) * 100;
            mpFill.style.width = `${mpPct}%`;
        }
        if (mpText) {
            if (GameState.ramTotal > 0) {
                mpText.textContent = `${GameState.ramUsed?.toFixed(1) || '--'} / ${ramTotalRounded} GB`;
            } else {
                mpText.textContent = '-- / -- GB';
            }
        }
        // Update MUD MP bar
        if (mudMp) {
            const bar = MudStyle.progressBar(mpPct, 12);
            mudMp.innerHTML = `[${bar.html}]`;
        }

        // GPU Utilization as Stamina
        if (staminaFill) {
            staminaFill.style.width = `${GameState.gpuUtil}%`;
        }
        if (staminaText) {
            staminaText.textContent = `${Math.round(GameState.gpuUtil)}%`;
        }
        // Update MUD Stamina bar
        if (mudStamina) {
            const bar = MudStyle.progressBar(GameState.gpuUtil || 0, 12);
            mudStamina.innerHTML = `[${bar.html}]`;
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
            SoundManager.zoneDiscovery();
        }

        // High accuracy = critical hit
        if (GameState.currentSkillAcc > 90 && this.prevAccuracy <= 90 && this.prevAccuracy > 0) {
            this.showCriticalHit('90%+ ACCURACY!');
            SoundManager.critical();
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
