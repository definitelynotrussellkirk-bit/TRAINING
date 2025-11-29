/**
 * Settings Client - Centralized settings API layer
 *
 * All settings load/save operations go through this module.
 * Changes to endpoints, error handling, or request options happen here only.
 */

const SettingsClient = (() => {

    /**
     * Load the current config from server
     * @returns {Promise<object>} The full config object
     */
    async function load() {
        const res = await fetch('/config');
        if (!res.ok) throw new Error(`Config load failed (${res.status})`);
        return res.json();
    }

    /**
     * Save config to server
     * @param {object} config - The config object to save
     * @returns {Promise<object>} Result with success/error
     */
    async function save(config) {
        const res = await fetch('/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Config save failed');
        return data;
    }

    /**
     * Save current config as the default
     * @returns {Promise<object>} Result with success/error
     */
    async function saveAsDefault() {
        const res = await fetch('/config/save-default', { method: 'POST' });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Failed to save as default');
        return data;
    }

    /**
     * Restore config from saved default
     * @returns {Promise<object>} Result with success/error and restored config
     */
    async function restoreDefault() {
        const res = await fetch('/config/restore-default', { method: 'POST' });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Failed to restore default');
        return data;
    }

    /**
     * Load inference hosts for remote eval
     * @returns {Promise<object>} Hosts data
     */
    async function loadInferenceHosts() {
        const res = await fetch('/oracle/hosts');
        if (!res.ok) throw new Error('Failed to load inference hosts');
        return res.json();
    }

    /**
     * Load scheduler status
     * @returns {Promise<object>} Scheduler status
     */
    async function loadScheduler() {
        const res = await fetch('/api/scheduler');
        if (!res.ok) throw new Error('Failed to load scheduler');
        return res.json();
    }

    /**
     * Load scheduler presets
     * @returns {Promise<object>} Presets data
     */
    async function loadSchedulerPresets() {
        const res = await fetch('/api/scheduler/presets');
        if (!res.ok) throw new Error('Failed to load scheduler presets');
        return res.json();
    }

    /**
     * Apply a scheduler preset
     * @param {string} presetId - The preset ID to apply
     * @returns {Promise<object>} Result with new status
     */
    async function applySchedulerPreset(presetId) {
        const res = await fetch('/api/scheduler/preset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preset: presetId })
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Failed to apply preset');
        return data;
    }

    /**
     * Save scheduler config
     * @param {object} config - Scheduler config (strategy, skills)
     * @returns {Promise<object>} Result
     */
    async function saveScheduler(config) {
        const res = await fetch('/api/scheduler/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Failed to save scheduler');
        return data;
    }

    return {
        load,
        save,
        saveAsDefault,
        restoreDefault,
        loadInferenceHosts,
        loadScheduler,
        loadSchedulerPresets,
        applySchedulerPreset,
        saveScheduler
    };
})();

// Export for module systems if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SettingsClient;
}
