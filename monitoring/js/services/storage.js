/**
 * Storage Service
 * Type-safe localStorage wrapper with defaults
 */

import CONFIG from '../core/config.js';

export class StorageService {
    /**
     * Get a value from localStorage
     * @param {string} key - Storage key
     * @param {any} defaultValue - Default value if not found
     * @returns {any} Stored value or default
     */
    get(key, defaultValue = null) {
        try {
            const value = localStorage.getItem(key);
            if (value === null) return defaultValue;

            // Try to parse as JSON
            try {
                return JSON.parse(value);
            } catch {
                // Return as string if not JSON
                return value;
            }
        } catch (error) {
            console.error(`Error reading from localStorage (${key}):`, error);
            return defaultValue;
        }
    }

    /**
     * Set a value in localStorage
     * @param {string} key - Storage key
     * @param {any} value - Value to store
     */
    set(key, value) {
        try {
            const serialized = typeof value === 'string'
                ? value
                : JSON.stringify(value);
            localStorage.setItem(key, serialized);
        } catch (error) {
            console.error(`Error writing to localStorage (${key}):`, error);
        }
    }

    /**
     * Remove a value from localStorage
     * @param {string} key - Storage key
     */
    remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (error) {
            console.error(`Error removing from localStorage (${key}):`, error);
        }
    }

    /**
     * Clear all localStorage
     */
    clear() {
        try {
            localStorage.clear();
        } catch (error) {
            console.error('Error clearing localStorage:', error);
        }
    }

    /**
     * Get boolean value
     * @param {string} key - Storage key
     * @param {boolean} defaultValue - Default value
     * @returns {boolean} Stored boolean value
     */
    getBoolean(key, defaultValue = false) {
        const value = this.get(key, defaultValue);
        return value === true || value === 'true';
    }

    /**
     * Get number value
     * @param {string} key - Storage key
     * @param {number} defaultValue - Default value
     * @returns {number} Stored number value
     */
    getNumber(key, defaultValue = 0) {
        const value = this.get(key, defaultValue);
        const num = Number(value);
        return Number.isFinite(num) ? num : defaultValue;
    }

    /**
     * Get string value
     * @param {string} key - Storage key
     * @param {string} defaultValue - Default value
     * @returns {string} Stored string value
     */
    getString(key, defaultValue = '') {
        const value = this.get(key, defaultValue);
        return String(value);
    }

    /**
     * Get array value
     * @param {string} key - Storage key
     * @param {Array} defaultValue - Default value
     * @returns {Array} Stored array value
     */
    getArray(key, defaultValue = []) {
        const value = this.get(key, defaultValue);
        return Array.isArray(value) ? value : defaultValue;
    }

    /**
     * Get object value
     * @param {string} key - Storage key
     * @param {Object} defaultValue - Default value
     * @returns {Object} Stored object value
     */
    getObject(key, defaultValue = {}) {
        const value = this.get(key, defaultValue);
        return typeof value === 'object' && value !== null ? value : defaultValue;
    }

    // Convenience methods for app-specific settings

    getCompactMode() {
        return this.getBoolean(CONFIG.STORAGE_KEYS.COMPACT_MODE, false);
    }

    setCompactMode(enabled) {
        this.set(CONFIG.STORAGE_KEYS.COMPACT_MODE, enabled);
    }

    getDarkTheme() {
        return this.getBoolean(CONFIG.STORAGE_KEYS.DARK_THEME, true);
    }

    setDarkTheme(enabled) {
        this.set(CONFIG.STORAGE_KEYS.DARK_THEME, enabled);
    }

    getSoundEnabled() {
        return this.getBoolean(CONFIG.STORAGE_KEYS.SOUND_ENABLED, false);
    }

    setSoundEnabled(enabled) {
        this.set(CONFIG.STORAGE_KEYS.SOUND_ENABLED, enabled);
    }

    getNotificationsEnabled() {
        return this.getBoolean(CONFIG.STORAGE_KEYS.NOTIFICATIONS_ENABLED, false);
    }

    setNotificationsEnabled(enabled) {
        this.set(CONFIG.STORAGE_KEYS.NOTIFICATIONS_ENABLED, enabled);
    }
}

// Create singleton instance
export const storage = new StorageService();

export default storage;
