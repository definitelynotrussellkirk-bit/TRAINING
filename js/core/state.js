/**
 * State Management Module
 * Centralized reactive state management for the training monitor
 */

export class State {
    constructor() {
        this._data = {
            // Training state
            status: 'idle',
            currentStep: 0,
            totalSteps: 0,
            epoch: 0,
            loss: null,
            validationLoss: null,
            learningRate: null,

            // File progress
            currentFile: null,
            batchStep: null,
            batchTotalSteps: null,
            queueSize: 0,
            queueSizeMB: 0,

            // Performance metrics
            tokensPerSec: null,
            tokensPerSecAvg: null,
            tokensPerStep: null,
            stepsPerSec: null,
            examplesPerSec: null,
            throughput: null,

            // Model config
            modelName: null,
            maxOutputTokens: null,
            contextWindow: null,
            loraR: null,
            loraAlpha: null,
            loraDropout: null,
            batchSize: null,
            gradAccum: null,
            maxLength: null,

            // Advanced metrics
            streamingCE: null,
            tokenEntropy: null,
            timeElapsed: null,
            etaRemaining: null,
            etaTime: null,

            // Analytics
            queueVelocity: null,
            logitPenaltyStats: null,
            patternLossTrend: null,
            patternLayerCorrelation: null,
            layerStabilitySummary: null,
            layerActivitySummary: null,
            lengthCoverage: null,

            // Accuracy
            accuracyPercent: 0,
            totalCorrect: 0,
            totalEvals: 0,
            accuracyHistory: [],

            // Think tags
            thinkTagPercent: 0,
            thinkTagCount: 0,

            // GPU/Memory
            gpuTemp: null,
            gpuUtil: null,
            gpuMemUsed: null,
            gpuMemTotal: null,
            gpuMemPercent: null,
            gpuName: null,
            gpuPowerDraw: null,
            gpuPowerLimit: null,
            ramUsed: null,
            ramTotal: null,
            ramPercent: null,
            ramAvailable: null,
            trainingProcessGB: null,
            swapUsed: null,
            swapTotal: null,
            swapPercent: null,

            // Timers
            trainingStartTime: null,
            lastUpdateTime: null,
            startStep: null,
            startTime: null,

            // Historical data
            lossHistory: [],
            trainLossHistory: [],
            recentExamples: [],

            // Current example
            currentPrompt: null,
            currentSystemPrompt: null,
            goldenAnswer: null,
            modelAnswer: null,
            answerMatches: null,

            // UI state
            compactMode: false,
            darkTheme: true,
            soundEnabled: false,
            isPaused: false,

            // Error state
            errorCount: 0,
            lastError: null
        };

        // Change listeners
        this._listeners = new Map();
    }

    /**
     * Subscribe to state changes
     * @param {string} key - State key to watch (or '*' for all changes)
     * @param {Function} callback - Function to call on change
     */
    on(key, callback) {
        if (!this._listeners.has(key)) {
            this._listeners.set(key, []);
        }
        this._listeners.get(key).push(callback);
    }

    /**
     * Unsubscribe from state changes
     */
    off(key, callback) {
        if (this._listeners.has(key)) {
            const callbacks = this._listeners.get(key);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    /**
     * Notify listeners of state change
     */
    _notify(key, value, oldValue) {
        // Notify specific key listeners
        if (this._listeners.has(key)) {
            this._listeners.get(key).forEach(cb => cb(value, oldValue));
        }

        // Notify wildcard listeners
        if (this._listeners.has('*')) {
            this._listeners.get('*').forEach(cb => cb(key, value, oldValue));
        }
    }

    /**
     * Get state value
     */
    get(key) {
        return this._data[key];
    }

    /**
     * Set state value (triggers listeners)
     */
    set(key, value) {
        const oldValue = this._data[key];
        if (oldValue !== value) {
            this._data[key] = value;
            this._notify(key, value, oldValue);
        }
    }

    /**
     * Update multiple state values at once
     */
    update(updates) {
        Object.entries(updates).forEach(([key, value]) => {
            this.set(key, value);
        });
    }

    /**
     * Get all state data
     */
    getAll() {
        return { ...this._data };
    }

    /**
     * Reset state to defaults
     */
    reset() {
        const keys = Object.keys(this._data);
        keys.forEach(key => {
            if (Array.isArray(this._data[key])) {
                this.set(key, []);
            } else if (typeof this._data[key] === 'boolean') {
                // Preserve UI preferences
                if (!['compactMode', 'darkTheme', 'soundEnabled'].includes(key)) {
                    this.set(key, false);
                }
            } else if (typeof this._data[key] === 'number') {
                this.set(key, 0);
            } else {
                this.set(key, null);
            }
        });
    }

    /**
     * Add to history array (with max length)
     */
    pushHistory(key, value, maxLength = 100) {
        const history = this.get(key) || [];
        history.push(value);
        if (history.length > maxLength) {
            history.shift();
        }
        this.set(key, history);
    }
}

// Create singleton instance
export const state = new State();

// Export default for convenience
export default state;
