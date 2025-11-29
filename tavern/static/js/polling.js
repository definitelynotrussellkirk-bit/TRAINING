/**
 * Polling Helper - Generic polling utilities for Realm of Training
 *
 * Replaces scattered setInterval() calls with a structured poller that:
 * - Handles errors gracefully
 * - Supports dynamic interval changes
 * - Can pause/resume polling
 * - Tracks polling state
 */

/**
 * Create a poller instance
 * @param {Function} fn - Async function to poll
 * @param {number} intervalMs - Polling interval in milliseconds
 * @param {object} options - Optional configuration
 * @returns {object} Poller controls
 */
function createPoller(fn, intervalMs, options = {}) {
    const {
        immediate = true,           // Run immediately on start
        errorHandler = console.error, // Error handler
        name = 'poller'             // Name for logging
    } = options;

    let timer = null;
    let running = false;
    let interval = intervalMs;
    let lastError = null;
    let runCount = 0;
    let errorCount = 0;

    async function tick() {
        if (!running) return;

        try {
            await fn();
            runCount++;
            lastError = null;
        } catch (err) {
            errorCount++;
            lastError = err;
            errorHandler(`[${name}] Polling error:`, err);
        }

        // Schedule next tick if still running
        if (running) {
            timer = setTimeout(tick, interval);
        }
    }

    return {
        /**
         * Start polling
         */
        start() {
            if (running) return;
            running = true;

            if (immediate) {
                tick();
            } else {
                timer = setTimeout(tick, interval);
            }
        },

        /**
         * Stop polling
         */
        stop() {
            running = false;
            if (timer) {
                clearTimeout(timer);
                timer = null;
            }
        },

        /**
         * Restart polling (stop + start)
         */
        restart() {
            this.stop();
            this.start();
        },

        /**
         * Update polling interval
         * @param {number} newInterval - New interval in milliseconds
         */
        setInterval(newInterval) {
            interval = newInterval;
            // If running, restart with new interval
            if (running) {
                this.restart();
            }
        },

        /**
         * Check if poller is running
         * @returns {boolean}
         */
        isRunning() {
            return running;
        },

        /**
         * Get poller statistics
         * @returns {object}
         */
        getStats() {
            return {
                running,
                interval,
                runCount,
                errorCount,
                lastError
            };
        },

        /**
         * Force an immediate poll (outside normal schedule)
         */
        async pollNow() {
            if (!running) return;

            // Clear pending timer
            if (timer) {
                clearTimeout(timer);
            }

            // Run immediately
            await tick();
        }
    };
}

/**
 * Create a group of pollers that can be controlled together
 * @returns {object} PollerGroup controls
 */
function createPollerGroup() {
    const pollers = new Map();

    return {
        /**
         * Add a poller to the group
         * @param {string} name - Unique identifier
         * @param {Function} fn - Async function to poll
         * @param {number} intervalMs - Polling interval
         * @param {object} options - Optional configuration
         * @returns {object} The created poller
         */
        add(name, fn, intervalMs, options = {}) {
            if (pollers.has(name)) {
                pollers.get(name).stop();
            }
            const poller = createPoller(fn, intervalMs, { ...options, name });
            pollers.set(name, poller);
            return poller;
        },

        /**
         * Get a poller by name
         * @param {string} name
         * @returns {object|undefined}
         */
        get(name) {
            return pollers.get(name);
        },

        /**
         * Start all pollers
         */
        startAll() {
            for (const poller of pollers.values()) {
                poller.start();
            }
        },

        /**
         * Stop all pollers
         */
        stopAll() {
            for (const poller of pollers.values()) {
                poller.stop();
            }
        },

        /**
         * Get stats for all pollers
         * @returns {object} Map of name -> stats
         */
        getAllStats() {
            const stats = {};
            for (const [name, poller] of pollers) {
                stats[name] = poller.getStats();
            }
            return stats;
        },

        /**
         * Remove a poller
         * @param {string} name
         */
        remove(name) {
            const poller = pollers.get(name);
            if (poller) {
                poller.stop();
                pollers.delete(name);
            }
        },

        /**
         * Remove all pollers
         */
        clear() {
            this.stopAll();
            pollers.clear();
        }
    };
}

// Export for module systems if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createPoller, createPollerGroup };
}
