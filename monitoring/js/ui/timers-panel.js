/**
 * Timers Panel Component
 * Displays training duration, ETA, and throughput timers
 */

import { formatNumber } from '../utils/formatters.js';

export class TimersPanel {
    constructor() {
        this.elements = {
            trainingTimer: document.getElementById('trainingTimer'),
            etaCountdown: document.getElementById('etaCountdown'),
            throughput: document.getElementById('throughput'),
            throughputDetails: document.getElementById('throughputDetails')
        };

        this.trainingStartTime = null;
    }

    /**
     * Update all timer displays
     */
    update(data) {
        if (!data) return;

        this.updateTrainingDuration(data);
        this.updateETA(data);
        this.updateThroughput(data);
    }

    /**
     * Update training duration timer
     */
    updateTrainingDuration(data) {
        if (!this.elements.trainingTimer) return;

        // Start timer when training begins
        if (data.status === 'training' && !this.trainingStartTime) {
            this.trainingStartTime = Date.now();
        }

        // Reset timer when idle
        if (data.status === 'idle') {
            this.trainingStartTime = null;
            this.elements.trainingTimer.textContent = '00:00:00';
            return;
        }

        // Calculate elapsed time
        if (this.trainingStartTime) {
            const elapsed = Math.floor((Date.now() - this.trainingStartTime) / 1000);
            this.elements.trainingTimer.textContent = this.formatDuration(elapsed);
        }
    }

    /**
     * Update ETA countdown
     */
    updateETA(data) {
        if (!this.elements.etaCountdown) return;

        // Use etaRemaining from status if available
        if (data.etaRemaining !== null && data.etaRemaining !== undefined) {
            this.elements.etaCountdown.textContent = this.formatDuration(data.etaRemaining);
            return;
        }

        // Calculate ETA from current progress
        if (data.currentStep && data.totalSteps && data.stepsPerSec) {
            const stepsRemaining = data.totalSteps - data.currentStep;
            const secondsRemaining = stepsRemaining / data.stepsPerSec;
            this.elements.etaCountdown.textContent = this.formatDuration(secondsRemaining);
            return;
        }

        // No ETA available
        this.elements.etaCountdown.textContent = '--:--:--';
    }

    /**
     * Update throughput display
     */
    updateThroughput(data) {
        if (!this.elements.throughput) return;

        // Check if we have throughput data
        if (data.throughputValue !== null && data.throughputValue !== undefined && data.throughputValue > 0) {
            this.elements.throughput.textContent = formatNumber(data.throughputValue, 1) + ' MB/hr';

            if (this.elements.throughputDetails) {
                // Calculate completion estimate if we have queue data
                if (data.queueSizeMB && data.queueSizeMB > 0) {
                    const hoursRemaining = data.queueSizeMB / data.throughputValue;
                    const completionTime = new Date(Date.now() + hoursRemaining * 3600 * 1000);
                    this.elements.throughputDetails.textContent =
                        'Queue: ' + formatNumber(data.queueSizeMB, 1) + ' MB (~' + hoursRemaining.toFixed(1) + 'h remaining)';
                } else {
                    this.elements.throughputDetails.textContent = 'No files queued';
                }
            }
        } else {
            // Use tokens/sec as fallback
            if (data.tokensPerSec && data.tokensPerSec > 0) {
                const tokPerSecK = (data.tokensPerSec / 1000).toFixed(1);
                this.elements.throughput.textContent = tokPerSecK + 'K tok/s';

                if (this.elements.throughputDetails) {
                    this.elements.throughputDetails.textContent = 'Token throughput';
                }
            } else {
                this.elements.throughput.textContent = '-- MB/hr';

                if (this.elements.throughputDetails) {
                    this.elements.throughputDetails.textContent = 'No data yet';
                }
            }
        }
    }

    /**
     * Format seconds into HH:MM:SS
     */
    formatDuration(seconds) {
        if (!seconds || seconds < 0 || !Number.isFinite(seconds)) {
            return '00:00:00';
        }

        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        return [hours, minutes, secs]
            .map(v => v.toString().padStart(2, '0'))
            .join(':');
    }
}

export default TimersPanel;
