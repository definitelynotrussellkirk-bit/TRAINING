/**
 * Training Client - Unified API layer for Realm of Training
 *
 * Centralizes all API calls. Changes to endpoints, error handling,
 * auth headers, or request options happen here only.
 */

const TrainingClient = (() => {
    const API_BASE = '/api';

    // Cache-busting helper
    function cacheBust(url) {
        const sep = url.includes('?') ? '&' : '?';
        return `${url}${sep}_t=${Date.now()}`;
    }

    /**
     * Generic fetch wrapper with error handling
     * @param {string} url - Endpoint URL
     * @param {object} options - Fetch options
     * @returns {Promise<any>} Response JSON
     */
    async function request(url, options = {}) {
        const defaultOptions = {
            cache: 'no-store',
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const response = await fetch(
            cacheBust(url),
            { ...defaultOptions, ...options }
        );

        if (!response.ok) {
            const error = new Error(`API error: ${response.status}`);
            error.status = response.status;
            throw error;
        }

        return response.json();
    }

    /**
     * GET request helper
     */
    async function get(endpoint) {
        return request(endpoint);
    }

    /**
     * POST request helper
     */
    async function post(endpoint, data) {
        return request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    // ========================================
    // GAME STATE APIs
    // ========================================

    /**
     * Get unified game state (training, GPU, curriculum, vault)
     * @returns {Promise<object>} Game state data
     */
    async function getGameState() {
        return get(`${API_BASE}/game`);
    }

    /**
     * Get hero info (name, icon, class)
     * @returns {Promise<object>} Hero data
     */
    async function getHero() {
        return get(`${API_BASE}/hero`);
    }

    /**
     * Get dynamic hero titles
     * @returns {Promise<object>} Titles data
     */
    async function getTitles() {
        return get(`${API_BASE}/titles`);
    }

    // ========================================
    // TRAINING CONTROL APIs
    // ========================================

    /**
     * Send control command to training daemon
     * @param {string} action - 'pause', 'resume', or 'stop'
     * @returns {Promise<object>} Control result with success/error
     */
    async function controlTraining(action) {
        const result = await post(`${API_BASE}/daemon/control`, { action });
        if (!result.success) {
            const error = new Error(result.error || 'Control failed');
            error.result = result;
            throw error;
        }
        return result;
    }

    /**
     * Pause training
     */
    async function pause() {
        return controlTraining('pause');
    }

    /**
     * Resume training
     */
    async function resume() {
        return controlTraining('resume');
    }

    /**
     * Stop training
     */
    async function stop() {
        return controlTraining('stop');
    }

    // ========================================
    // QUEUE & TASKS APIs
    // ========================================

    /**
     * Get quest queue status
     * @returns {Promise<object>} Queue data (high, normal, low priorities)
     */
    async function getQuests() {
        return get(`${API_BASE}/quests`);
    }

    /**
     * Get task master status (GPU scheduler)
     * @returns {Promise<object>} Task master state
     */
    async function getTaskMaster() {
        return get(`${API_BASE}/task-master`);
    }

    // ========================================
    // SYSTEM STATUS APIs
    // ========================================

    /**
     * Get Weaver daemon status
     * @returns {Promise<object>} Weaver health data
     */
    async function getWeaverStatus() {
        return get(`${API_BASE}/weaver/status`);
    }

    /**
     * Get training status directly
     * @returns {Promise<object>} Training status
     */
    async function getTrainingStatus() {
        return get(`${API_BASE}/status/training`);
    }

    // ========================================
    // SKILLS & CURRICULUM APIs
    // ========================================

    /**
     * Get skill configurations (from YAML)
     * @returns {Promise<object>} Skills data
     */
    async function getSkills() {
        return get('/skills');
    }

    // ========================================
    // VAULT & CHECKPOINTS APIs
    // ========================================

    /**
     * Get vault assets (checkpoints)
     * @returns {Promise<object>} Vault data
     */
    async function getVaultAssets() {
        return get('/vault/assets');
    }

    // ========================================
    // BATTLE LOG APIs
    // ========================================

    /**
     * Get battle saga entries
     * @param {number} limit - Max entries to return
     * @returns {Promise<object>} Saga data with tales array
     */
    async function getSaga(limit = 30) {
        return get(`/saga?limit=${limit}`);
    }

    // ========================================
    // EXPORT
    // ========================================

    return {
        // Core
        request,
        get,
        post,

        // Game state
        getGameState,
        getHero,
        getTitles,

        // Training control
        controlTraining,
        pause,
        resume,
        stop,

        // Queue & tasks
        getQuests,
        getTaskMaster,

        // System status
        getWeaverStatus,
        getTrainingStatus,

        // Skills
        getSkills,

        // Vault
        getVaultAssets,

        // Battle log
        getSaga
    };
})();

// Export for module systems if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TrainingClient;
}
