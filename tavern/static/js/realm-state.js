/**
 * Realm State Manager - Single Source of Truth for all UI data
 *
 * Features:
 * - Server-Sent Events (SSE) for real-time push updates
 * - Automatic fallback to polling if SSE unavailable
 * - Subscriber pattern for component updates
 * - Skills/curriculum state support
 * - Connection health monitoring
 *
 * Architecture:
 *   SSE /api/stream -> RealmState (cache) -> Subscribers (UI components)
 *   OR
 *   Polling /api/realm -> RealmState (cache) -> Subscribers (UI components)
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
 *   // Manually trigger refresh (polling mode only)
 *   await RealmState.refresh();
 */

const RealmState = {
    // =================================================================
    // STATE SECTIONS (mirrors backend realm_store.py)
    // =================================================================

    // Campaign tracking (for reset on campaign change)
    campaignId: null,
    heroId: null,
    campaignMaxStep: 0,  // Max step from ledger for current campaign

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

    skills: {
        skills: {},          // skill_id -> { mastered_level, training_level, accuracy, ... }
        activeSkill: null,
        updatedAt: null,
    },

    mode: 'idle',            // training, idle, eval_only, maintenance
    health: 'unknown',       // healthy, warning, error, unknown
    warnings: [],

    events: [],              // Recent battle log events

    // =================================================================
    // METADATA & CONNECTION
    // =================================================================

    lastFetch: null,
    lastError: null,
    fetchCount: 0,
    schemaVersion: null,

    // Connection state
    _connectionMode: 'none',  // 'sse', 'polling', 'none'
    _eventSource: null,
    _pollInterval: null,
    _reconnectAttempts: 0,
    _maxReconnectAttempts: 5,
    _reconnectDelay: 1000,

    // =================================================================
    // SUBSCRIBERS
    // =================================================================

    _subscribers: {
        training: [],
        queue: [],
        workers: [],
        hero: [],
        skills: [],
        mode: [],
        events: [],
        all: [],  // Called on any change
    },

    /**
     * Subscribe to state changes for a section
     * @param {string} section - 'training', 'queue', 'workers', 'hero', 'skills', 'mode', 'events', or 'all'
     * @param {function} callback - Called with the section data when it changes
     */
    subscribe(section, callback) {
        if (this._subscribers[section]) {
            this._subscribers[section].push(callback);
        } else {
            console.warn(`[RealmState] Unknown section: ${section}`);
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
                    console.error(`[RealmState] Subscriber error (${section}):`, e);
                }
            }
        }
        // Notify 'all' subscribers
        for (const cb of this._subscribers.all) {
            try {
                cb(section, data);
            } catch (e) {
                console.error('[RealmState] Subscriber error (all):', e);
            }
        }
    },

    // =================================================================
    // SSE CONNECTION
    // =================================================================

    /**
     * Connect via Server-Sent Events for real-time updates
     */
    connectSSE() {
        if (this._eventSource) {
            this._eventSource.close();
        }

        const url = `/api/realm/stream`;
        console.log(`[RealmState] Connecting to SSE: ${url}`);

        try {
            this._eventSource = new EventSource(url);
            this._connectionMode = 'sse';

            this._eventSource.onopen = () => {
                console.log('[RealmState] SSE connected');
                this._reconnectAttempts = 0;
                this.health = 'healthy';
                this._notify('mode', { mode: this.mode, health: 'healthy' });
            };

            // Handle initial state
            this._eventSource.addEventListener('init', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    console.log('[RealmState] Received initial state via SSE');
                    this._processState(data);
                } catch (err) {
                    console.error('[RealmState] Error parsing init event:', err);
                }
            });

            // Handle training updates
            this._eventSource.addEventListener('training', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    this._updateTraining(data);
                } catch (err) {
                    console.error('[RealmState] Error parsing training event:', err);
                }
            });

            // Handle queue updates
            this._eventSource.addEventListener('queue', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    this._updateQueue(data);
                } catch (err) {
                    console.error('[RealmState] Error parsing queue event:', err);
                }
            });

            // Handle worker updates
            this._eventSource.addEventListener('worker_update', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    this._updateWorker(data);
                } catch (err) {
                    console.error('[RealmState] Error parsing worker event:', err);
                }
            });

            this._eventSource.addEventListener('workers', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    this._updateWorkers(data);
                } catch (err) {
                    console.error('[RealmState] Error parsing workers event:', err);
                }
            });

            // Handle hero updates
            this._eventSource.addEventListener('hero', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    this._updateHero(data);
                } catch (err) {
                    console.error('[RealmState] Error parsing hero event:', err);
                }
            });

            // Handle skills updates
            this._eventSource.addEventListener('skills', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    this._updateSkills(data);
                } catch (err) {
                    console.error('[RealmState] Error parsing skills event:', err);
                }
            });

            // Handle battle log events
            this._eventSource.addEventListener('event', (e) => {
                try {
                    const event = JSON.parse(e.data);
                    this._addEvent(event);
                } catch (err) {
                    console.error('[RealmState] Error parsing event:', err);
                }
            });

            // Handle errors
            this._eventSource.onerror = (e) => {
                console.warn('[RealmState] SSE error, will reconnect...');
                this._eventSource.close();
                this._eventSource = null;
                this._connectionMode = 'none';
                this.health = 'warning';

                // Attempt reconnection with backoff
                if (this._reconnectAttempts < this._maxReconnectAttempts) {
                    this._reconnectAttempts++;
                    const delay = this._reconnectDelay * Math.pow(2, this._reconnectAttempts - 1);
                    console.log(`[RealmState] Reconnecting in ${delay}ms (attempt ${this._reconnectAttempts})`);
                    setTimeout(() => this.connectSSE(), delay);
                } else {
                    console.log('[RealmState] Max reconnect attempts reached, falling back to polling');
                    this.startPolling(2000);
                }
            };

        } catch (err) {
            console.error('[RealmState] Failed to create EventSource:', err);
            this.startPolling(2000);
        }
    },

    /**
     * Disconnect SSE
     */
    disconnectSSE() {
        if (this._eventSource) {
            this._eventSource.close();
            this._eventSource = null;
        }
        this._connectionMode = 'none';
    },

    // =================================================================
    // POLLING (fallback)
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
            this.schemaVersion = data.schema_version;

            // Process the response
            this._processState(data);

            return data;

        } catch (err) {
            console.error('[RealmState] Fetch error:', err);
            this.lastError = err.message;
            this.health = 'error';
            this._notify('mode', { mode: this.mode, health: 'error' });
            return null;
        }
    },

    /**
     * Start polling for state updates (fallback when SSE unavailable)
     * @param {number} intervalMs - Poll interval in milliseconds (default 2000)
     */
    startPolling(intervalMs = 2000) {
        if (this._connectionMode === 'polling') return;

        this.stopPolling();
        this._connectionMode = 'polling';
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
        if (this._connectionMode === 'polling') {
            this._connectionMode = 'none';
        }
    },

    // =================================================================
    // STATE PROCESSING
    // =================================================================

    /**
     * Process complete state response
     */
    _processState(data) {
        const state = data.state || {};
        const events = data.events || [];

        // Track what changed for notifications
        const changes = [];

        // Training state
        if (state.training) {
            if (this._updateTrainingInternal(state.training)) {
                changes.push('training');
            }
        }

        // Queue state
        if (state.queue) {
            if (this._updateQueueInternal(state.queue)) {
                changes.push('queue');
            }
        }

        // Workers state
        if (state.workers) {
            if (this._updateWorkersInternal(state.workers)) {
                changes.push('workers');
            }
        }

        // Hero state
        if (state.hero) {
            if (this._updateHeroInternal(state.hero)) {
                changes.push('hero');
            }
        }

        // Skills state
        if (state.skills) {
            if (this._updateSkillsInternal(state.skills)) {
                changes.push('skills');
            }
        }

        // Campaign state (campaign-scoped step count)
        const campaign = data.campaign;
        if (campaign) {
            // Detect campaign change and reset cached values
            if (campaign.campaign_id !== this.campaignId || campaign.hero_id !== this.heroId) {
                console.log(`[RealmState] Campaign changed: ${this.heroId}/${this.campaignId} -> ${campaign.hero_id}/${campaign.campaign_id}`);
                // Reset campaign-specific cached values
                this.campaignMaxStep = 0;
            }
            // Update campaign tracking
            this.heroId = campaign.hero_id;
            this.campaignId = campaign.campaign_id;
            this.campaignMaxStep = campaign.campaign_max_step || 0;
        }

        // Mode state
        const modeInfo = state.mode_info || {};
        const newMode = modeInfo.mode || state.mode || 'idle';
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

    // Individual update methods (for SSE events)
    _updateTraining(data) {
        if (this._updateTrainingInternal(data)) {
            this._notify('training', this.training);
        }
    },

    _updateTrainingInternal(t) {
        const changed = (
            this.training.step !== t.step ||
            this.training.status !== t.status ||
            this.training.loss !== t.loss ||
            this.training.speed !== t.speed
        );

        // Preserve existing values when incoming data is null/undefined
        // This prevents flickering when different endpoints have different data
        this.training = {
            status: t.status ?? this.training.status ?? 'idle',
            step: t.step ?? this.training.step ?? 0,
            totalSteps: t.total_steps ?? this.training.totalSteps ?? 0,
            loss: t.loss ?? this.training.loss,
            learningRate: t.learning_rate ?? this.training.learningRate,
            file: t.file ?? this.training.file,
            speed: t.speed ?? this.training.speed,
            etaSeconds: t.eta_seconds ?? this.training.etaSeconds,
            strain: t.strain ?? this.training.strain,
            updatedAt: t.updated_at ?? this.training.updatedAt,
        };

        return changed;
    },

    _updateQueue(data) {
        if (this._updateQueueInternal(data)) {
            this._notify('queue', this.queue);
        }
    },

    _updateQueueInternal(q) {
        const changed = this.queue.depth !== q.depth;

        this.queue = {
            depth: q.depth || 0,
            highPriority: q.high_priority || 0,
            normalPriority: q.normal_priority || 0,
            lowPriority: q.low_priority || 0,
            status: q.status || 'ok',
            updatedAt: q.updated_at,
        };

        return changed;
    },

    _updateWorker(data) {
        const workerId = data.worker_id;
        if (!workerId) return;

        this.workers[workerId] = {
            workerId: data.worker_id,
            role: data.role,
            status: data.status,
            device: data.device,
            currentJob: data.current_job,
            lastHeartbeat: data.last_heartbeat,
        };

        this._computeHealth();
        this._notify('workers', this.workers);
    },

    _updateWorkers(data) {
        if (this._updateWorkersInternal(data)) {
            this._notify('workers', this.workers);
        }
    },

    _updateWorkersInternal(workersData) {
        const workersChanged = JSON.stringify(this.workers) !== JSON.stringify(workersData);
        this.workers = {};
        for (const [id, w] of Object.entries(workersData)) {
            this.workers[id] = {
                workerId: w.worker_id,
                role: w.role,
                status: w.status,
                device: w.device,
                currentJob: w.current_job,
                lastHeartbeat: w.last_heartbeat,
            };
        }
        return workersChanged;
    },

    _updateHero(data) {
        if (this._updateHeroInternal(data)) {
            this._notify('hero', this.hero);
        }
    },

    _updateHeroInternal(h) {
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

        return changed;
    },

    _updateSkills(data) {
        if (this._updateSkillsInternal(data)) {
            this._notify('skills', this.skills);
        }
    },

    _updateSkillsInternal(s) {
        const changed = JSON.stringify(this.skills.skills) !== JSON.stringify(s.skills);

        this.skills = {
            skills: s.skills || {},
            activeSkill: s.active_skill || null,
            updatedAt: s.updated_at,
        };

        return changed;
    },

    _addEvent(event) {
        // Add to front (newest first)
        this.events.unshift(event);
        // Keep only last 100
        if (this.events.length > 100) {
            this.events = this.events.slice(0, 100);
        }
        this._notify('events', this.events);
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
                console.error('[RealmState] Failed to set mode:', err);
                return false;
            }

            // In polling mode, refresh state
            if (this._connectionMode === 'polling') {
                await this.refresh();
            }
            return true;

        } catch (err) {
            console.error('[RealmState] Failed to set realm mode:', err);
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

    get connectionMode() {
        return this._connectionMode;
    },

    get isConnected() {
        return this._connectionMode !== 'none';
    },

    // =================================================================
    // SKILL HELPERS
    // =================================================================

    getSkill(skillId) {
        return this.skills.skills?.[skillId] || null;
    },

    getSkillLevel(skillId) {
        const skill = this.getSkill(skillId);
        return skill ? skill.training_level : 0;
    },

    getSkillMasteredLevel(skillId) {
        const skill = this.getSkill(skillId);
        return skill ? skill.mastered_level : 0;
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

const CONNECTION_ICONS = {
    sse: '📡',
    polling: '🔄',
    none: '❌',
};

// =================================================================
// UI UPDATE FUNCTIONS
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
    // campaignMaxStep = authoritative historical max from ledger (campaign-scoped)
    // training.step = live step during active training
    // Only use training.step when actively training to avoid stale data
    const totalSteps = document.getElementById('totalSteps');
    const totalEvals = document.getElementById('totalEvals');
    if (totalSteps) {
        const liveStep = RealmState.isTraining ? (RealmState.training.step || 0) : 0;
        const stepCount = Math.max(RealmState.campaignMaxStep || 0, liveStep);
        totalSteps.textContent = stepCount.toLocaleString();
    }
    if (totalEvals) {
        totalEvals.textContent = RealmState.hero.level || 0;
    }

    // Connection indicator (if present)
    const connIcon = document.getElementById('connectionIcon');
    if (connIcon) {
        connIcon.textContent = CONNECTION_ICONS[RealmState.connectionMode] || CONNECTION_ICONS.none;
        connIcon.title = `Connection: ${RealmState.connectionMode}`;
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

    // Try SSE first, fall back to polling
    // Check if SSE is supported and if the endpoint exists
    if (typeof EventSource !== 'undefined') {
        // Try SSE connection
        RealmState.connectSSE();
    } else {
        // Fall back to polling
        console.log('[RealmState] EventSource not supported, using polling');
        RealmState.startPolling(2000);
    }

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
    RealmState.disconnectSSE();
    RealmState.stopPolling();
});

// Export for other modules
window.RealmState = RealmState;
