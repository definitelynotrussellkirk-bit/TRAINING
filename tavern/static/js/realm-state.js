/**
 * Realm State Manager - Single Source of Truth for all UI data
 *
 * This module fetches from /api/realm ONCE and distributes the data
 * to all UI components. No more multiple fetch calls to different endpoints.
 *
 * Architecture:
 *   /api/realm -> RealmState (cache) -> Subscribers (UI components)
 *
 * Usage:
 *   // Subscribe to state changes
 *   RealmState.subscribe('training', (training) => {
 *       updateTrainingUI(training);
 *   });
 *
 *   // Read current state directly
 *   const step = RealmState.training.step;
 *
 *   // Manually trigger refresh
 *   await RealmState.refresh();
 */

const RealmState = {
    // =================================================================
    // STATE SECTIONS (mirrors backend realm_store.py)
    // =================================================================

    training: {
        status: 'idle',      // idle, training, paused, stopped
        step: 0,
        totalSteps: 0,
        loss: null,
        learningRate: null,
        file: null,
        speed: null,         // steps/sec
        etaSeconds: null,
        strain: null,        // loss - floor
        updatedAt: null,
    },

    queue: {
        depth: 0,
        highPriority: 0,
        normalPriority: 0,
        lowPriority: 0,
        status: 'ok',        // ok, low, empty, stale
        updatedAt: null,
    },

    workers: {},  // worker_id -> { status, role, device, currentJob, lastHeartbeat }

    hero: {
        name: 'DIO',
        title: '',
        level: 0,
        xp: 0,
        campaignId: null,
        currentSkill: null,
        currentSkillLevel: 0,
        updatedAt: null,
    },

    mode: 'idle',            // training, idle, eval_only, maintenance
    health: 'unknown',       // healthy, warning, error, unknown
    warnings: [],

    events: [],              // Recent battle log events

    // =================================================================
    // METADATA
    // =================================================================

    lastFetch: null,
    lastError: null,
    fetchCount: 0,
    isPolling: false,

    // =================================================================
    // SUBSCRIBERS
    // =================================================================

    _subscribers: {
        training: [],
        queue: [],
        workers: [],
        hero: [],
        mode: [],
        events: [],
        all: [],  // Called on any change
    },

    /**
     * Subscribe to state changes for a section
     * @param {string} section - 'training', 'queue', 'workers', 'hero', 'mode', 'events', or 'all'
     * @param {function} callback - Called with the section data when it changes
     */
    subscribe(section, callback) {
        if (this._subscribers[section]) {
            this._subscribers[section].push(callback);
        } else {
            console.warn(`Unknown section: ${section}`);
        }
    },

    /**
     * Unsubscribe from state changes
     */
    unsubscribe(section, callback) {
        if (this._subscribers[section]) {
            const idx = this._subscribers[section].indexOf(callback);
            if (idx > -1) {
                this._subscribers[section].splice(idx, 1);
            }
        }
    },

    /**
     * Notify subscribers of a section change
     */
    _notify(section, data) {
        // Notify section-specific subscribers
        if (this._subscribers[section]) {
            for (const cb of this._subscribers[section]) {
                try {
                    cb(data);
                } catch (e) {
                    console.error(`Subscriber error (${section}):`, e);
                }
            }
        }
        // Notify 'all' subscribers
        for (const cb of this._subscribers.all) {
            try {
                cb(section, data);
            } catch (e) {
                console.error('Subscriber error (all):', e);
            }
        }
    },

    // =================================================================
    // FETCHING
    // =================================================================

    /**
     * Fetch latest state from /api/realm
     */
    async refresh() {
        try {
            const res = await fetch(`/api/realm?_t=${Date.now()}`, {
                cache: 'no-store',
            });

            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }

            const data = await res.json();
            this.fetchCount++;
            this.lastFetch = new Date();
            this.lastError = null;

            // Process the response
            this._processState(data);

            return data;

        } catch (err) {
            console.error('RealmState fetch error:', err);
            this.lastError = err.message;
            this.health = 'error';
            this._notify('mode', { mode: this.mode, health: 'error' });
            return null;
        }
    },

    /**
     * Process state response and update local cache
     */
    _processState(data) {
        const state = data.state || {};
        const events = data.events || [];

        // Track what changed for notifications
        const changes = [];

        // Training state
        if (state.training) {
            const t = state.training;
            const changed = (
                this.training.step !== t.step ||
                this.training.status !== t.status ||
                this.training.loss !== t.loss
            );

            this.training = {
                status: t.status || 'idle',
                step: t.step || 0,
                totalSteps: t.total_steps || 0,
                loss: t.loss,
                learningRate: t.learning_rate,
                file: t.file,
                speed: t.speed,
                etaSeconds: t.eta_seconds,
                strain: t.strain,
                updatedAt: t.updated_at,
            };

            if (changed) changes.push('training');
        }

        // Queue state
        if (state.queue) {
            const q = state.queue;
            const changed = this.queue.depth !== q.depth;

            this.queue = {
                depth: q.depth || 0,
                highPriority: q.high_priority || 0,
                normalPriority: q.normal_priority || 0,
                lowPriority: q.low_priority || 0,
                status: q.status || 'ok',
                updatedAt: q.updated_at,
            };

            if (changed) changes.push('queue');
        }

        // Workers state
        if (state.workers) {
            const workersChanged = JSON.stringify(this.workers) !== JSON.stringify(state.workers);
            this.workers = {};
            for (const [id, w] of Object.entries(state.workers)) {
                this.workers[id] = {
                    workerId: w.worker_id,
                    role: w.role,
                    status: w.status,
                    device: w.device,
                    currentJob: w.current_job,
                    lastHeartbeat: w.last_heartbeat,
                };
            }
            if (workersChanged) changes.push('workers');
        }

        // Hero state
        if (state.hero) {
            const h = state.hero;
            const changed = this.hero.level !== h.level || this.hero.xp !== h.xp;

            this.hero = {
                name: h.name || 'DIO',
                title: h.title || '',
                level: h.level || 0,
                xp: h.xp || 0,
                campaignId: h.campaign_id,
                currentSkill: h.current_skill,
                currentSkillLevel: h.current_skill_level || 0,
                updatedAt: h.updated_at,
            };

            if (changed) changes.push('hero');
        }

        // Mode state
        const newMode = state.mode || 'idle';
        if (this.mode !== newMode) {
            this.mode = newMode;
            changes.push('mode');
        }

        // Events
        if (events.length > 0) {
            const hadEvents = this.events.length;
            this.events = events;
            if (events.length !== hadEvents) {
                changes.push('events');
            }
        }

        // Compute health from workers
        this._computeHealth();

        // Notify all changed sections
        for (const section of changes) {
            this._notify(section, this[section]);
        }
    },

    /**
     * Compute health status from workers
     */
    _computeHealth() {
        const workerList = Object.values(this.workers);
        if (workerList.length === 0) {
            this.health = 'unknown';
            this.warnings = ['No workers registered'];
            return;
        }

        const stale = workerList.filter(w => w.status === 'stale');
        const errors = workerList.filter(w => w.status === 'error');

        if (errors.length > 0) {
            this.health = 'error';
            this.warnings = errors.map(w => `${w.workerId}: error`);
        } else if (stale.length > 0) {
            this.health = 'warning';
            this.warnings = stale.map(w => `${w.workerId}: stale`);
        } else {
            this.health = 'healthy';
            this.warnings = [];
        }
    },

    // =================================================================
    // POLLING
    // =================================================================

    _pollInterval: null,

    /**
     * Start polling for state updates
     * @param {number} intervalMs - Poll interval in milliseconds (default 2000)
     */
    startPolling(intervalMs = 2000) {
        if (this.isPolling) return;

        this.isPolling = true;
        this.refresh();  // Initial fetch

        this._pollInterval = setInterval(() => {
            this.refresh();
        }, intervalMs);

        console.log(`[RealmState] Polling started (${intervalMs}ms interval)`);
    },

    /**
     * Stop polling
     */
    stopPolling() {
        if (this._pollInterval) {
            clearInterval(this._pollInterval);
            this._pollInterval = null;
        }
        this.isPolling = false;
        console.log('[RealmState] Polling stopped');
    },

    // =================================================================
    // MODE CONTROL
    // =================================================================

    /**
     * Set realm mode via API
     */
    async setMode(mode, reason = 'user action') {
        try {
            const res = await fetch('/api/realm-mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode, reason }),
            });

            if (!res.ok) {
                const err = await res.json();
                console.error('Failed to set mode:', err);
                return false;
            }

            // Refresh state
            await this.refresh();
            return true;

        } catch (err) {
            console.error('Failed to set realm mode:', err);
            return false;
        }
    },

    // =================================================================
    // CONVENIENCE GETTERS
    // =================================================================

    get isTraining() {
        return this.training.status === 'training';
    },

    get isPaused() {
        return this.training.status === 'paused';
    },

    get isIdle() {
        return this.training.status === 'idle';
    },

    get queueIsLow() {
        return this.queue.depth < 5;
    },

    get queueIsEmpty() {
        return this.queue.depth === 0;
    },

    get workerCount() {
        return Object.keys(this.workers).length;
    },

    get healthyWorkerCount() {
        return Object.values(this.workers).filter(w => w.status === 'running').length;
    },
};

// =================================================================
// HEALTH ICONS (for UI)
// =================================================================

const HEALTH_ICONS = {
    healthy: '🟢',
    warning: '🟡',
    error: '🔴',
    unknown: '⚪',
};

const MODE_ICONS = {
    training: '⚔️',
    idle: '💤',
    eval_only: '📊',
    maintenance: '🔧',
    offline: '⛔',
};

const MODE_LABELS = {
    training: 'TRAINING',
    idle: 'IDLE',
    eval_only: 'EVAL ONLY',
    maintenance: 'MAINT',
    offline: 'OFFLINE',
};

// =================================================================
// UI UPDATE FUNCTIONS (moved from world-state.js)
// =================================================================

/**
 * Update header UI (mode toggle, health indicator)
 */
function updateHeaderUI() {
    // Mode toggle buttons
    const btnTraining = document.getElementById('btnTraining');
    const btnIdle = document.getElementById('btnIdle');

    if (btnTraining && btnIdle) {
        btnTraining.classList.toggle('active', RealmState.mode === 'training');
        btnIdle.classList.toggle('active', RealmState.mode === 'idle');

        const specialMode = ['eval_only', 'maintenance', 'offline'].includes(RealmState.mode);
        btnTraining.disabled = specialMode;
        btnIdle.disabled = specialMode;
    }

    // Health indicator
    const healthIcon = document.getElementById('healthIcon');
    const healthIndicator = document.getElementById('healthIndicator');
    if (healthIcon) {
        healthIcon.textContent = HEALTH_ICONS[RealmState.health] || HEALTH_ICONS.unknown;
    }
    if (healthIndicator) {
        healthIndicator.title = RealmState.warnings.length > 0
            ? `Warnings:\n${RealmState.warnings.join('\n')}`
            : 'Realm healthy';
        healthIndicator.classList.toggle('has-warnings', RealmState.warnings.length > 0);
    }

    // Mode indicator
    const modeIcon = document.getElementById('modeIcon');
    const modeText = document.getElementById('modeText');
    const idleIndicator = document.getElementById('idleIndicator');

    if (modeIcon) {
        modeIcon.textContent = MODE_ICONS[RealmState.mode] || MODE_ICONS.idle;
    }
    if (modeText) {
        modeText.textContent = MODE_LABELS[RealmState.mode] || RealmState.mode.toUpperCase();
    }
    if (idleIndicator) {
        idleIndicator.className = 'idle-indicator';
        if (RealmState.mode === 'training') {
            idleIndicator.classList.add('mode-training');
        } else if (RealmState.mode === 'idle') {
            idleIndicator.classList.add('mode-idle');
        } else {
            idleIndicator.classList.add('mode-other');
        }

        if (RealmState.isTraining) {
            idleIndicator.classList.add('is-running');
        }
    }

    // Update total steps/evals in header
    const totalSteps = document.getElementById('totalSteps');
    const totalEvals = document.getElementById('totalEvals');
    if (totalSteps) {
        totalSteps.textContent = RealmState.training.step.toLocaleString();
    }
    if (totalEvals) {
        // TODO: Get from hero state when available
        totalEvals.textContent = RealmState.hero.level || 0;
    }
}

/**
 * Initialize header button handlers
 */
function initHeaderHandlers() {
    const btnTraining = document.getElementById('btnTraining');
    const btnIdle = document.getElementById('btnIdle');

    if (btnTraining) {
        btnTraining.addEventListener('click', async () => {
            if (RealmState.mode !== 'training') {
                btnTraining.disabled = true;
                await RealmState.setMode('training', 'clicked TRAIN button');
                btnTraining.disabled = false;
            }
        });
    }

    if (btnIdle) {
        btnIdle.addEventListener('click', async () => {
            if (RealmState.mode !== 'idle') {
                btnIdle.disabled = true;
                await RealmState.setMode('idle', 'clicked IDLE button');
                btnIdle.disabled = false;
            }
        });
    }
}

// =================================================================
// INITIALIZATION
// =================================================================

function initRealmState() {
    // Subscribe to mode changes to update header
    RealmState.subscribe('mode', updateHeaderUI);
    RealmState.subscribe('training', updateHeaderUI);

    // Initialize header handlers
    initHeaderHandlers();

    // Start polling (2 second interval)
    RealmState.startPolling(2000);

    console.log('[RealmState] Initialized');
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initRealmState);
} else {
    initRealmState();
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    RealmState.stopPolling();
});

// Export for other modules
window.RealmState = RealmState;
