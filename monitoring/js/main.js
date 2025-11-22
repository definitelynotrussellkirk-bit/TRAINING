/**
 * Main Application Controller
 * Orchestrates all components and manages the application lifecycle
 */

import CONFIG from './core/config.js';
import { state } from './core/state.js';
import { events, EVENTS } from './core/events.js';
import { api } from './services/api.js';
import { storage } from './services/storage.js';
import StatusBar from './ui/status-bar.js';
import GPUPanel from './ui/gpu-panel.js';
import RAMPanel from './ui/ram-panel.js';
import TrainingPanel from './ui/training-panel.js';
import LossPanel from './ui/loss-panel.js';
import TimersPanel from './ui/timers-panel.js';
import AnalyticsPanel from './ui/analytics-panel.js';
import ExamplesPanel from './ui/examples-panel.js';
import { initAudio, setSoundEnabled, playCompletionSound } from './utils/audio.js';
import { showToast } from './utils/animations.js';

class TrainingMonitor {
    constructor() {
        // Components
        this.statusBar = new StatusBar();
        this.gpuPanel = new GPUPanel();
        this.ramPanel = new RAMPanel();
        this.trainingPanel = new TrainingPanel();
        this.lossPanel = new LossPanel();
        this.timersPanel = new TimersPanel();
        this.analyticsPanel = new AnalyticsPanel();
        this.examplesPanel = new ExamplesPanel();

        // State
        this.pollTimer = null;
        this.isPaused = false;
        this.lastStatusForNotification = null;
        this.hasCompletedBefore = false;

        // Stats tracking
        this.stepsPerSecEMA = null;
        this.lastStep = 0;
        this.lastTimestamp = null;

        // File tracking for throughput
        this.currentFileSizeMB = 0;
        this.currentFileStartTime = null;
        this.throughputHistory = [];

        // Initialize
        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        console.log('🚀 Initializing Training Monitor...');

        // Load preferences from storage
        this.loadPreferences();

        // Initialize audio
        initAudio(storage.getSoundEnabled());

        // Set up event listeners
        this.setupEventListeners();

        // Start polling
        this.startPolling();

        // Initial data fetch
        await this.poll();

        console.log('✅ Training Monitor initialized');
    }

    /**
     * Load user preferences from storage
     */
    loadPreferences() {
        const soundEnabled = storage.getSoundEnabled();

        // Set sound
        setSoundEnabled(soundEnabled);

        // Update state
        state.set('soundEnabled', soundEnabled);
    }

    /**
     * Set up event listeners for UI controls
     */
    setupEventListeners() {
        // Pause toggle
        const pauseToggle = document.getElementById('pauseToggle');
        if (pauseToggle) {
            pauseToggle.addEventListener('click', () => this.togglePause());
        }

        // Visibility change (pause when tab hidden)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('Tab hidden - continuing background updates');
            } else {
                console.log('Tab visible - resuming');
                this.poll(); // Immediate update when tab becomes visible
            }
        });
    }

    /**
     * Start polling for data
     */
    startPolling() {
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
        }

        // Initial poll
        this.poll();

        // Set up interval
        const interval = CONFIG.TIMING.BASE_REFRESH_MS + api.getBackoffDelay();
        this.pollTimer = setInterval(() => {
            if (!this.isPaused) {
                this.poll();
            }
        }, interval);
    }

    /**
     * Main polling function - fetches and updates all data
     */
    async poll() {
        try {
            // Fetch all data in parallel
            const { status, gpu, memory, config } = await api.fetchAll();

            if (!status) {
                console.warn('No status data received');
                return;
            }

            // Process and update state
            this.processStatusData(status);
            this.processGPUData(gpu);
            this.processMemoryData(memory);
            this.processConfigData(config);

            // Update UI components
            this.updateComponents();

            // Check for critical events
            this.checkCriticalEvents(status);

            // Update last update time
            state.set('lastUpdateTime', Date.now());

            // Emit event
            events.emit(EVENTS.DATA_LOADED, { status, gpu, memory });

        } catch (error) {
            console.error('Poll error:', error);
            events.emit(EVENTS.DATA_ERROR, error);
        }
    }

    /**
     * Process status data and update state
     */
    processStatusData(data) {
        // Update state with new data
        state.update({
            status: data.status,
            currentStep: data.current_step,
            totalSteps: data.total_steps,
            epoch: data.epoch,
            loss: data.loss,
            validationLoss: data.validation_loss,
            learningRate: data.learning_rate,
            currentFile: data.current_file,
            batchStep: data.batch_step,
            batchTotalSteps: data.batch_total_steps,
            queueSize: data.batch_queue_size,
            tokensPerSec: data.tokens_per_sec,
            tokensPerSecAvg: data.tokens_per_sec_avg,
            tokensPerStep: data.tokens_per_step,
            stepsPerSec: data.steps_per_sec,
            examplesPerSec: data.examples_per_sec,
            accuracyPercent: data.accuracy_percent,
            thinkTagPercent: data.think_tag_percent,
            thinkTagCount: data.think_tag_count,
            totalCorrect: data.total_correct,
            totalEvals: data.total_evals,
            modelName: data.model_name,
            maxOutputTokens: data.max_output_tokens,
            contextWindow: data.context_window,
            streamingCE: data.streaming_ce,
            tokenEntropy: data.token_entropy,
            timeElapsed: data.time_elapsed,
            etaRemaining: data.eta_remaining,
            etaTime: data.eta_time,
            loraR: data.lora_r,
            loraAlpha: data.lora_alpha,
            batchSize: data.batch_size,
            gradAccum: data.gradient_accumulation_steps,

            // Analytics
            queueVelocity: data.queue_velocity,
            logitPenaltyStats: data.logit_penalty_stats,
            patternLossTrend: data.pattern_loss_trend,
            patternLayerCorrelation: data.pattern_layer_correlation,
            layerStabilitySummary: data.layer_stability_summary,
            layerActivitySummary: data.layer_activity_summary,
            lengthCoverage: data.length_coverage,

            // Current example
            currentPrompt: data.current_prompt,
            currentSystemPrompt: data.current_system_prompt,
            goldenAnswer: data.golden_answer,
            modelAnswer: data.model_answer,
            answerMatches: data.answer_matches,
            recentExamples: data.recent_examples
        });

        // Calculate throughput
        this.calculateThroughput(data);

        // Track completion events
        this.trackCompletionEvents(data);
    }

    /**
     * Process GPU data
     */
    processGPUData(gpu) {
        if (!gpu) return;

        state.update({
            gpuTemp: gpu.temperature,
            gpuUtil: gpu.gpu_utilization,
            gpuMemUsed: gpu.memory_used_mb,
            gpuMemTotal: gpu.memory_total_mb,
            gpuMemPercent: (gpu.memory_used_mb / gpu.memory_total_mb) * 100,
            gpuName: gpu.gpu_name,
            gpuPowerDraw: gpu.power_draw_w,
            gpuPowerLimit: gpu.power_limit_w
        });
    }

    /**
     * Process memory data
     */
    processMemoryData(memory) {
        if (!memory) return;

        state.update({
            ramUsed: memory.used_gb,
            ramTotal: memory.total_gb,
            ramPercent: memory.percent,
            ramAvailable: memory.available_gb,
            trainingProcessGB: memory.training_process_gb,
            swapUsed: memory.swap_used_gb,
            swapTotal: memory.swap_total_gb,
            swapPercent: memory.swap_percent
        });
    }

    /**
     * Process config data
     */
    processConfigData(config) {
        if (!config) return;

        // Only update if not already set (config doesn't change often)
        if (!state.get('loraR')) {
            state.update({
                loraR: config.lora_r,
                loraAlpha: config.lora_alpha,
                loraDropout: config.lora_dropout,
                batchSize: config.batch_size,
                gradAccum: config.gradient_accumulation_steps,
                maxLength: config.max_length
            });
        }
    }

    /**
     * Calculate throughput
     */
    calculateThroughput(data) {
        // Start tracking new file
        if (data.status === 'training' && !this.currentFileStartTime) {
            this.currentFileStartTime = Date.now();
            this.currentFileSizeMB = data.current_file_size_mb || 40;
        }

        // File completed - calculate throughput
        if (data.status === 'idle' && this.hasCompletedBefore) {
            if (this.currentFileStartTime && this.currentFileSizeMB > 0) {
                const timeElapsedHours = (Date.now() - this.currentFileStartTime) / (1000 * 3600);
                const throughput = this.currentFileSizeMB / timeElapsedHours;

                this.throughputHistory.push({
                    mb: this.currentFileSizeMB,
                    hours: timeElapsedHours,
                    throughput
                });

                // Keep last 10 measurements
                if (this.throughputHistory.length > 10) {
                    this.throughputHistory.shift();
                }
            }

            // Reset for next file
            this.currentFileStartTime = null;
            this.currentFileSizeMB = 0;
        }
    }

    /**
     * Track completion events
     */
    trackCompletionEvents(data) {
        if (data.status === 'idle' && this.lastStatusForNotification === 'training') {
            this.hasCompletedBefore = true;

            // Play completion sound
            if (state.get('soundEnabled')) {
                playCompletionSound();
            }

            // Show notification
            showToast('File training completed!', 'success');

            // Emit event
            events.emit(EVENTS.FILE_COMPLETED, data);
        }

        this.lastStatusForNotification = data.status;
    }

    /**
     * Update all UI components
     */
    updateComponents() {
        // Prepare data for status bar
        const statusBarData = {
            status: state.get('status'),
            current_step: state.get('currentStep'),
            total_steps: state.get('totalSteps'),
            loss: state.get('loss'),
            batch_queue_size: state.get('queueSize'),
            gpuTemp: state.get('gpuTemp'),
            gpuMemUsed: state.get('gpuMemUsed'),
            gpuMemTotal: state.get('gpuMemTotal'),
            gpuMemPercent: state.get('gpuMemPercent'),
            ramUsed: state.get('ramUsed'),
            ramPercent: state.get('ramPercent'),
            throughputValue: this.getAverageThroughput()
        };

        // Prepare data for GPU panel
        const gpuData = {
            gpuName: state.get('gpuName'),
            gpuTemp: state.get('gpuTemp'),
            gpuUtil: state.get('gpuUtil'),
            gpuMemUsed: state.get('gpuMemUsed'),
            gpuMemTotal: state.get('gpuMemTotal'),
            gpuPowerDraw: state.get('gpuPowerDraw'),
            gpuPowerLimit: state.get('gpuPowerLimit')
        };

        // Prepare data for RAM panel
        const ramData = {
            ramUsed: state.get('ramUsed'),
            ramTotal: state.get('ramTotal'),
            ramPercent: state.get('ramPercent'),
            ramAvailable: state.get('ramAvailable'),
            trainingProcessGB: state.get('trainingProcessGB'),
            swapUsed: state.get('swapUsed'),
            swapTotal: state.get('swapTotal'),
            swapPercent: state.get('swapPercent')
        };

        // Prepare data for training panel
        const trainingData = {
            status: state.get('status'),
            currentStep: state.get('currentStep'),
            totalSteps: state.get('totalSteps'),
            epoch: state.get('epoch'),
            batchStep: state.get('batchStep'),
            batchTotalSteps: state.get('batchTotalSteps'),
            maxOutputTokens: state.get('maxOutputTokens'),
            contextWindow: state.get('contextWindow'),
            modelName: state.get('modelName'),
            layerActivitySummary: state.get('layerActivitySummary'),
            batchSize: state.get('batchSize'),
            gradAccum: state.get('gradAccum')
        };

        // Prepare data for loss panel
        const lossData = {
            loss: state.get('loss'),
            validationLoss: state.get('validationLoss'),
            streamingCE: state.get('streamingCE'),
            tokenEntropy: state.get('tokenEntropy'),
            learningRate: state.get('learningRate'),
            tokensPerSec: state.get('tokensPerSec'),
            tokensPerStep: state.get('tokensPerStep'),
            stepsPerSec: state.get('stepsPerSec'),
            examplesPerSec: state.get('examplesPerSec'),
            timeElapsed: state.get('timeElapsed'),
            etaRemaining: state.get('etaRemaining'),
            etaTime: state.get('etaTime')
        };

        // Prepare data for timers panel
        const timersData = {
            status: state.get('status'),
            currentStep: state.get('currentStep'),
            totalSteps: state.get('totalSteps'),
            stepsPerSec: state.get('stepsPerSec'),
            tokensPerSec: state.get('tokensPerSec'),
            etaRemaining: state.get('etaRemaining'),
            throughputValue: this.getAverageThroughput(),
            queueSizeMB: state.get('queueSizeMB')
        };

        // Prepare data for analytics panel
        const analyticsData = {
            queueVelocity: state.get('queueVelocity'),
            logitPenaltyStats: state.get('logitPenaltyStats'),
            patternLossTrend: state.get('patternLossTrend'),
            patternLayerCorrelation: state.get('patternLayerCorrelation'),
            layerStabilitySummary: state.get('layerStabilitySummary'),
            layerActivitySummary: state.get('layerActivitySummary'),
            lengthCoverage: state.get('lengthCoverage')
        };

        // Prepare data for examples panel
        const examplesData = {
            currentPrompt: state.get('currentPrompt'),
            currentSystemPrompt: state.get('currentSystemPrompt'),
            goldenAnswer: state.get('goldenAnswer'),
            modelAnswer: state.get('modelAnswer'),
            answerMatches: state.get('answerMatches'),
            recentExamples: state.get('recentExamples')
        };

        // Update all components
        this.statusBar.update(statusBarData);
        this.gpuPanel.update(gpuData);
        this.ramPanel.update(ramData);
        this.trainingPanel.update(trainingData);
        this.lossPanel.update(lossData);
        this.timersPanel.update(timersData);
        this.analyticsPanel.update(analyticsData);
        this.examplesPanel.update(examplesData);
    }

    /**
     * Get average throughput from history
     */
    getAverageThroughput() {
        if (this.throughputHistory.length === 0) return null;

        const sum = this.throughputHistory.reduce((acc, item) => acc + item.throughput, 0);
        return sum / this.throughputHistory.length;
    }

    /**
     * Check for critical events (overfitting, errors, etc.)
     */
    checkCriticalEvents(data) {
        // Check for overfitting
        const gap = data.val_train_gap;
        if (gap > CONFIG.LOSS.DANGER_GAP) {
            events.emit(EVENTS.ALERT_WARNING, {
                type: 'overfitting',
                message: `Train-val gap is ${gap.toFixed(3)} (>0.5) - possible overfitting`
            });
        }

        // Check for high think tag percentage
        if (data.think_tag_percent > CONFIG.THINK_TAG.WARNING) {
            events.emit(EVENTS.ALERT_WARNING, {
                type: 'think_tags',
                message: `Think tag percentage is ${data.think_tag_percent.toFixed(1)}% (>60%)`
            });
        }

        // Check for empty queue
        if (data.batch_queue_size === 0 && data.status === 'idle') {
            events.emit(EVENTS.ALERT_WARNING, {
                type: 'empty_queue',
                message: 'Training queue is empty'
            });
        }
    }

    /**
     * Toggle pause
     */
    togglePause() {
        this.isPaused = !this.isPaused;
        state.set('isPaused', this.isPaused);

        const button = document.getElementById('pauseToggle');
        if (button) {
            button.textContent = this.isPaused ? '▶️ Resume' : '⏸️ Pause';
            button.setAttribute('aria-pressed', this.isPaused);
        }

        if (!this.isPaused) {
            this.poll(); // Immediate update when resuming
        }
    }
}

// Initialize on DOM loaded
document.addEventListener('DOMContentLoaded', () => {
    window.monitor = new TrainingMonitor();
});

export default TrainingMonitor;
