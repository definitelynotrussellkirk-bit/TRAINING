/**
 * World State - Realm mode toggle and health indicator
 *
 * Polls /api/world-state and provides:
 * - Mode toggle (TRAINING <-> IDLE)
 * - Health indicator (healthy/warning/error)
 * - Training status from heartbeats
 */

const WorldState = {
    mode: 'idle',
    health: 'unknown',
    warnings: [],
    training: {
        status: 'idle',
        currentJob: null,
        step: 0,
        totalSteps: 0,
        itPerSec: null,
    },
    workers: [],
    gpus: [],
    lastUpdate: null,
};

// Health icons mapping
const HEALTH_ICONS = {
    healthy: '🟢',
    warning: '🟡',
    error: '🔴',
    unknown: '⚪',
};

// Mode icons mapping
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

/**
 * Fetch world state from API
 */
async function fetchWorldState() {
    try {
        const res = await fetch('/api/world-state');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Update state
        WorldState.mode = data.realm_mode || 'unknown';
        WorldState.health = data.health || 'unknown';
        WorldState.warnings = data.warnings || [];
        WorldState.workers = data.workers || [];
        WorldState.gpus = data.gpus || [];
        WorldState.lastUpdate = new Date();

        // Training status
        if (data.state?.training) {
            const training = data.state.training;
            WorldState.training = {
                status: training.status || 'idle',
                currentJob: training.file || training.current_job_name || null,
                step: training.step || training.progress?.step || 0,
                totalSteps: training.total_steps || training.progress?.total || 0,
                itPerSec: training.speed || training.progress?.it_per_sec || null,
            };
        }

        // Update UI
        updateWorldStateUI();

        return data;
    } catch (err) {
        console.error('Failed to fetch world state:', err);
        WorldState.health = 'error';
        updateWorldStateUI();
        return null;
    }
}

/**
 * Set realm mode via API
 */
async function setRealmMode(mode, reason = 'user action') {
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

        const data = await res.json();
        console.log('Mode set:', data);

        // Refresh world state
        await fetchWorldState();
        return true;
    } catch (err) {
        console.error('Failed to set realm mode:', err);
        return false;
    }
}

/**
 * Update UI elements based on current state
 */
function updateWorldStateUI() {
    // Mode toggle buttons
    const btnTraining = document.getElementById('btnTraining');
    const btnIdle = document.getElementById('btnIdle');

    if (btnTraining && btnIdle) {
        // Highlight active mode
        btnTraining.classList.toggle('active', WorldState.mode === 'training');
        btnIdle.classList.toggle('active', WorldState.mode === 'idle');

        // Disable if in special modes
        const specialMode = ['eval_only', 'maintenance', 'offline'].includes(WorldState.mode);
        btnTraining.disabled = specialMode;
        btnIdle.disabled = specialMode;
    }

    // Health indicator
    const healthIcon = document.getElementById('healthIcon');
    const healthIndicator = document.getElementById('healthIndicator');
    if (healthIcon) {
        healthIcon.textContent = HEALTH_ICONS[WorldState.health] || HEALTH_ICONS.unknown;
    }
    if (healthIndicator) {
        healthIndicator.title = WorldState.warnings.length > 0
            ? `Warnings:\n${WorldState.warnings.join('\n')}`
            : 'Realm healthy';
        healthIndicator.classList.toggle('has-warnings', WorldState.warnings.length > 0);
    }

    // Mode indicator (idle indicator)
    const modeIcon = document.getElementById('modeIcon');
    const modeText = document.getElementById('modeText');
    const idleIndicator = document.getElementById('idleIndicator');

    if (modeIcon) {
        modeIcon.textContent = MODE_ICONS[WorldState.mode] || MODE_ICONS.idle;
    }
    if (modeText) {
        modeText.textContent = MODE_LABELS[WorldState.mode] || WorldState.mode.toUpperCase();
    }
    if (idleIndicator) {
        // Update class based on mode
        idleIndicator.className = 'idle-indicator';
        if (WorldState.mode === 'training') {
            idleIndicator.classList.add('mode-training');
        } else if (WorldState.mode === 'idle') {
            idleIndicator.classList.add('mode-idle');
        } else {
            idleIndicator.classList.add('mode-other');
        }

        // Also show if training is actually happening
        if (WorldState.training.status === 'running') {
            idleIndicator.classList.add('is-running');
        }
    }
}

/**
 * Initialize world state polling and event handlers
 */
function initWorldState() {
    // Initial fetch
    fetchWorldState();

    // Poll every 5 seconds
    setInterval(fetchWorldState, 5000);

    // Mode toggle button handlers
    const btnTraining = document.getElementById('btnTraining');
    const btnIdle = document.getElementById('btnIdle');

    if (btnTraining) {
        btnTraining.addEventListener('click', async () => {
            if (WorldState.mode !== 'training') {
                btnTraining.disabled = true;
                await setRealmMode('training', 'clicked TRAIN button');
                btnTraining.disabled = false;
            }
        });
    }

    if (btnIdle) {
        btnIdle.addEventListener('click', async () => {
            if (WorldState.mode !== 'idle') {
                btnIdle.disabled = true;
                await setRealmMode('idle', 'clicked IDLE button');
                btnIdle.disabled = false;
            }
        });
    }
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWorldState);
} else {
    initWorldState();
}
