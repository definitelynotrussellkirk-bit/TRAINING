/**
 * API Service
 * Handles all HTTP requests with error handling, retry logic, and caching
 */

import CONFIG from '../core/config.js';

export class APIService {
    constructor() {
        this.errorCount = 0;
        this.currentBackoff = 0;
        this.cache = new Map();
        this.inFlight = new Map();
    }

    /**
     * Fetch with retry and exponential backoff
     */
    async _fetchWithRetry(url, options = {}, retries = CONFIG.LIMITS.MAX_ERROR_RETRIES) {
        for (let i = 0; i <= retries; i++) {
            try {
                const response = await fetch(url, options);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                this.errorCount = 0;
                this.currentBackoff = 0;
                return response;
            } catch (error) {
                if (i === retries) {
                    this.errorCount++;
                    this.currentBackoff = Math.min(
                        CONFIG.TIMING.MAX_BACKOFF_MS,
                        Math.pow(2, this.errorCount) * 1000
                    );
                    throw error;
                }

                // Wait before retry
                const delay = Math.pow(2, i) * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    /**
     * Fetch with in-flight request deduplication
     */
    async _fetch(url, options = {}) {
        // Check if request is already in flight
        if (this.inFlight.has(url)) {
            return this.inFlight.get(url);
        }

        // Create promise and store it
        const promise = this._fetchWithRetry(url, options)
            .finally(() => {
                // Remove from in-flight when done
                this.inFlight.delete(url);
            });

        this.inFlight.set(url, promise);
        return promise;
    }

    /**
     * Fetch training status
     * @returns {Promise<Object>} Training status data
     */
    async fetchStatus() {
        try {
            const response = await this._fetch(CONFIG.API.STATUS);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching status:', error);
            throw error;
        }
    }

    /**
     * Fetch GPU statistics
     * @returns {Promise<Object>} GPU stats
     */
    async fetchGPU() {
        try {
            const response = await this._fetch(CONFIG.API.GPU_STATS);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching GPU stats:', error);
            return null; // GPU stats optional
        }
    }

    /**
     * Fetch memory statistics
     * @returns {Promise<Object>} Memory stats
     */
    async fetchMemory() {
        try {
            const response = await this._fetch(CONFIG.API.MEMORY_STATS);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching memory stats:', error);
            return null; // Memory stats optional
        }
    }

    /**
     * Fetch inbox files
     * @returns {Promise<Array>} Inbox files list
     */
    async fetchInbox() {
        try {
            const response = await this._fetch(CONFIG.API.INBOX_FILES);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching inbox files:', error);
            return [];
        }
    }

    /**
     * Fetch config
     * @returns {Promise<Object>} Config data
     */
    async fetchConfig() {
        try {
            const response = await this._fetch(CONFIG.API.CONFIG);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching config:', error);
            return null;
        }
    }

    /**
     * Fetch queue samples (file snippets from queue)
     * @returns {Promise<Array>} Queue sample files
     */
    async fetchQueueSamples() {
        try {
            const response = await fetch(CONFIG.API.QUEUE_SAMPLES);
            const text = await response.text();

            // Parse directory listing (simple HTML parsing)
            const parser = new DOMParser();
            const doc = parser.parseFromString(text, 'text/html');
            const links = Array.from(doc.querySelectorAll('a'))
                .map(a => a.getAttribute('href'))
                .filter(href => href && href.endsWith('.jsonl'));

            return links;
        } catch (error) {
            console.error('Error fetching queue samples:', error);
            return [];
        }
    }

    /**
     * Fetch queue sample file content
     * @param {string} filename - File name
     * @returns {Promise<Array>} Sample data lines
     */
    async fetchQueueFile(filename) {
        try {
            const url = CONFIG.API.QUEUE_SAMPLES + filename;
            const response = await fetch(url);
            const text = await response.text();

            // Parse JSONL (up to 5 lines)
            const lines = text.trim().split('\n').slice(0, 5);
            const samples = lines
                .map(line => {
                    try {
                        return JSON.parse(line);
                    } catch {
                        return null;
                    }
                })
                .filter(s => s !== null);

            return samples;
        } catch (error) {
            console.error(`Error fetching queue file ${filename}:`, error);
            return [];
        }
    }

    /**
     * Fetch all data in parallel
     * @returns {Promise<Object>} All data combined
     */
    async fetchAll() {
        const [status, gpu, memory, inbox, config] = await Promise.allSettled([
            this.fetchStatus(),
            this.fetchGPU(),
            this.fetchMemory(),
            this.fetchInbox(),
            this.fetchConfig()
        ]);

        return {
            status: status.status === 'fulfilled' ? status.value : null,
            gpu: gpu.status === 'fulfilled' ? gpu.value : null,
            memory: memory.status === 'fulfilled' ? memory.value : null,
            inbox: inbox.status === 'fulfilled' ? inbox.value : [],
            config: config.status === 'fulfilled' ? config.value : null
        };
    }

    /**
     * Get current error count
     */
    getErrorCount() {
        return this.errorCount;
    }

    /**
     * Get current backoff delay
     */
    getBackoffDelay() {
        return this.currentBackoff;
    }

    /**
     * Reset error state
     */
    resetErrors() {
        this.errorCount = 0;
        this.currentBackoff = 0;
    }
}

// Create singleton instance
export const api = new APIService();

export default api;
