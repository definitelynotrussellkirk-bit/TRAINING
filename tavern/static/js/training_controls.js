/**
 * Training Controls - Shared pause/resume/stop button logic
 *
 * Provides consistent control UX across all UIs that have training controls.
 * Handles button state, disabling during requests, and status toggling.
 */

/**
 * Attach training control handlers to buttons
 *
 * @param {object} config - Configuration object
 * @param {HTMLElement} config.pauseBtn - Pause button element
 * @param {HTMLElement} config.resumeBtn - Resume button element
 * @param {HTMLElement} config.stopBtn - Stop button (optional)
 * @param {Function} config.onLog - Callback for logging messages (message, type)
 * @param {Function} config.onStatusChange - Callback when status changes (action, result)
 * @param {HTMLElement} config.container - Container to show/hide based on training state
 */
function attachTrainingControls(config) {
    const {
        pauseBtn,
        resumeBtn,
        stopBtn,
        onLog = () => {},
        onStatusChange = () => {},
        container = null
    } = config;

    const buttons = [pauseBtn, resumeBtn, stopBtn].filter(Boolean);

    /**
     * Disable all control buttons
     */
    function disableAll() {
        buttons.forEach(btn => {
            if (btn) btn.disabled = true;
        });
    }

    /**
     * Enable all control buttons
     */
    function enableAll() {
        buttons.forEach(btn => {
            if (btn) btn.disabled = false;
        });
    }

    /**
     * Handle a control action
     * @param {string} action - 'pause', 'resume', or 'stop'
     */
    async function handleAction(action) {
        disableAll();

        try {
            // Use TrainingClient if available, otherwise fall back to direct fetch
            let result;
            if (typeof TrainingClient !== 'undefined') {
                result = await TrainingClient.controlTraining(action);
            } else {
                const response = await fetch('/api/daemon/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action })
                });
                result = await response.json();
                if (!result.success) {
                    throw new Error(result.error || 'Control failed');
                }
            }

            // Update button visibility based on action
            if (action === 'pause') {
                if (pauseBtn) pauseBtn.style.display = 'none';
                if (resumeBtn) resumeBtn.style.display = 'inline-flex';
                onLog(`Training paused - ${result.message}`, 'warning');
            } else if (action === 'resume') {
                if (pauseBtn) pauseBtn.style.display = 'inline-flex';
                if (resumeBtn) resumeBtn.style.display = 'none';
                onLog(`Training resumed - ${result.message}`, 'success');
            } else if (action === 'stop') {
                onLog(`Training stopped - ${result.message}`, 'warning');
            }

            onStatusChange(action, result);

        } catch (error) {
            onLog(`Control error: ${error.message}`, 'error');
        } finally {
            enableAll();
        }
    }

    // Attach click handlers
    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => handleAction('pause'));
    }
    if (resumeBtn) {
        resumeBtn.addEventListener('click', () => handleAction('resume'));
    }
    if (stopBtn) {
        stopBtn.addEventListener('click', () => handleAction('stop'));
    }

    // Return control object for external use
    return {
        /**
         * Update button states based on training status
         * @param {boolean} isTraining - Whether training is active
         * @param {boolean} isPaused - Whether training is paused
         */
        updateState(isTraining, isPaused = false) {
            if (container) {
                container.style.display = isTraining ? 'flex' : 'none';
            }

            if (isTraining) {
                if (isPaused) {
                    if (pauseBtn) pauseBtn.style.display = 'none';
                    if (resumeBtn) resumeBtn.style.display = 'inline-flex';
                } else {
                    if (pauseBtn) pauseBtn.style.display = 'inline-flex';
                    if (resumeBtn) resumeBtn.style.display = 'none';
                }
            }
        },

        /**
         * Programmatically trigger an action
         * @param {string} action - 'pause', 'resume', or 'stop'
         */
        trigger(action) {
            return handleAction(action);
        },

        /**
         * Disable all buttons (e.g., during external operations)
         */
        disable: disableAll,

        /**
         * Enable all buttons
         */
        enable: enableAll
    };
}

/**
 * Create a standalone control bar element
 *
 * @param {object} options - Configuration
 * @param {Function} options.onLog - Logging callback
 * @param {Function} options.onStatusChange - Status change callback
 * @returns {object} { element: HTMLElement, controls: ControlObject }
 */
function createTrainingControlBar(options = {}) {
    const {
        onLog = () => {},
        onStatusChange = () => {}
    } = options;

    // Create container
    const container = document.createElement('div');
    container.className = 'training-controls';
    container.style.display = 'none';

    // Create buttons
    const pauseBtn = document.createElement('button');
    pauseBtn.className = 'control-btn pause';
    pauseBtn.innerHTML = '<span class="btn-icon">⏸</span> Pause';
    pauseBtn.title = 'Pause training after current step';

    const resumeBtn = document.createElement('button');
    resumeBtn.className = 'control-btn resume';
    resumeBtn.innerHTML = '<span class="btn-icon">▶</span> Resume';
    resumeBtn.title = 'Resume training';
    resumeBtn.style.display = 'none';

    const stopBtn = document.createElement('button');
    stopBtn.className = 'control-btn stop';
    stopBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop';
    stopBtn.title = 'Stop training completely';

    container.appendChild(pauseBtn);
    container.appendChild(resumeBtn);
    container.appendChild(stopBtn);

    // Attach handlers
    const controls = attachTrainingControls({
        pauseBtn,
        resumeBtn,
        stopBtn,
        container,
        onLog,
        onStatusChange
    });

    return { element: container, controls };
}

// Export for module systems if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { attachTrainingControls, createTrainingControlBar };
}
