/**
 * Event Bus Module
 * Pub/sub event system for decoupled component communication
 */

export class EventBus {
    constructor() {
        this._events = new Map();
    }

    /**
     * Subscribe to an event
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     * @returns {Function} Unsubscribe function
     */
    on(event, callback) {
        if (!this._events.has(event)) {
            this._events.set(event, []);
        }
        this._events.get(event).push(callback);

        // Return unsubscribe function
        return () => this.off(event, callback);
    }

    /**
     * Subscribe to an event (one-time only)
     */
    once(event, callback) {
        const wrapped = (...args) => {
            callback(...args);
            this.off(event, wrapped);
        };
        return this.on(event, wrapped);
    }

    /**
     * Unsubscribe from an event
     */
    off(event, callback) {
        if (this._events.has(event)) {
            const callbacks = this._events.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
            // Clean up empty event arrays
            if (callbacks.length === 0) {
                this._events.delete(event);
            }
        }
    }

    /**
     * Emit an event
     * @param {string} event - Event name
     * @param {...any} args - Arguments to pass to callbacks
     */
    emit(event, ...args) {
        if (this._events.has(event)) {
            this._events.get(event).forEach(callback => {
                try {
                    callback(...args);
                } catch (error) {
                    console.error(`Error in event handler for "${event}":`, error);
                }
            });
        }
    }

    /**
     * Remove all listeners for an event, or all events
     */
    removeAllListeners(event) {
        if (event) {
            this._events.delete(event);
        } else {
            this._events.clear();
        }
    }

    /**
     * Get listener count for an event
     */
    listenerCount(event) {
        return this._events.has(event) ? this._events.get(event).length : 0;
    }

    /**
     * Get all registered events
     */
    eventNames() {
        return Array.from(this._events.keys());
    }
}

// Create singleton instance
export const events = new EventBus();

// Export default for convenience
export default events;

// Common event names (for documentation and IDE autocomplete)
export const EVENTS = {
    // Data events
    DATA_LOADED: 'data:loaded',
    DATA_ERROR: 'data:error',
    STATUS_UPDATED: 'status:updated',
    GPU_UPDATED: 'gpu:updated',
    MEMORY_UPDATED: 'memory:updated',

    // Training events
    TRAINING_STARTED: 'training:started',
    TRAINING_STOPPED: 'training:stopped',
    TRAINING_COMPLETED: 'training:completed',
    STEP_COMPLETED: 'step:completed',
    FILE_COMPLETED: 'file:completed',

    // UI events
    THEME_CHANGED: 'ui:theme_changed',
    COMPACT_TOGGLED: 'ui:compact_toggled',
    MODAL_OPENED: 'ui:modal_opened',
    MODAL_CLOSED: 'ui:modal_closed',

    // Alert events
    ALERT_WARNING: 'alert:warning',
    ALERT_ERROR: 'alert:error',
    ALERT_SUCCESS: 'alert:success'
};
