/**
 * Status Bar Component
 * Displays training status in the persistent top bar
 */

import CONFIG, { COLORS } from '../core/config.js';
import { formatLoss, formatLargeNumber, formatNumber } from '../utils/formatters.js';
import { updateDelta, animateValueChange } from '../utils/animations.js';

export class StatusBar {
    constructor() {
        this.previousValues = {
            loss: null,
            gpuTemp: null,
            vramUsed: null,
            ramUsed: null
        };
    }

    /**
     * Update all status bar metrics
     * @param {Object} data - Combined data from all sources
     */
    update(data) {
        if (!data) return;

        this.updateHealth(data.status);
        this.updateStatus(data.status);
        this.updateLoss(data.loss);
        this.updateProgress(data.current_step, data.total_steps);
        this.updateGPU(data.gpuTemp);
        this.updateVRAM(data.gpuMemUsed, data.gpuMemTotal, data.gpuMemPercent);
        this.updateRAM(data.ramUsed, data.ramPercent);
        this.updateQueue(data.batch_queue_size);
    }

    /**
     * Update health indicator dot
     */
    updateHealth(status) {
        const dot = document.getElementById('healthDot');
        const label = document.getElementById('healthLabel');
        if (!dot || !label) return;

        const healthStates = {
            'training': {
                color: COLORS.ACCENT_GREEN,
                label: 'TRAINING'
            },
            'idle': {
                color: COLORS.ACCENT_YELLOW,
                label: 'IDLE'
            },
            'paused': {
                color: COLORS.ACCENT_YELLOW,
                label: 'PAUSED'
            },
            'error': {
                color: COLORS.ACCENT_RED,
                label: 'ERROR'
            },
            'crashed': {
                color: COLORS.ACCENT_RED,
                label: 'CRASHED'
            }
        };

        const state = healthStates[status] || {
            color: COLORS.TEXT_SECONDARY,
            label: status ? status.toUpperCase() : 'UNKNOWN'
        };

        dot.style.background = state.color;
        label.textContent = state.label;
        label.style.color = state.color;
    }

    /**
     * Update status text
     */
    updateStatus(status) {
        const el = document.getElementById('summaryStatus');
        if (!el) return;

        el.textContent = status ? status.toUpperCase() : 'UNKNOWN';

        // Color code
        const colors = {
            'training': COLORS.ACCENT_GREEN,
            'idle': COLORS.ACCENT_YELLOW,
            'error': COLORS.ACCENT_RED,
            'crashed': COLORS.ACCENT_RED
        };
        el.style.color = colors[status] || COLORS.TEXT_PRIMARY;
    }

    /**
     * Update loss value and delta
     */
    updateLoss(loss) {
        const valueEl = document.getElementById('lossValue');
        const deltaEl = document.getElementById('lossDelta');
        if (!valueEl) return;

        if (loss !== null && loss !== undefined && Number.isFinite(loss)) {
            valueEl.textContent = formatLoss(loss);

            // Update delta
            if (deltaEl) {
                updateDelta('lossDelta', loss, this.previousValues.loss, true);
            }

            // Animate if changed significantly
            if (this.previousValues.loss !== null &&
                Math.abs(loss - this.previousValues.loss) > 0.1) {
                animateValueChange('lossValue');
            }

            this.previousValues.loss = loss;
        } else {
            valueEl.textContent = '-';
            if (deltaEl) deltaEl.textContent = '';
        }
    }

    /**
     * Update progress percentage
     */
    updateProgress(currentStep, totalSteps) {
        const el = document.getElementById('progressValue');
        if (!el) return;

        if (currentStep && totalSteps && totalSteps > 0) {
            const percent = (currentStep / totalSteps) * 100;
            el.textContent = percent.toFixed(1) + '%';
        } else {
            el.textContent = '0%';
        }
    }

    /**
     * Update GPU temperature
     */
    updateGPU(temp) {
        const valueEl = document.getElementById('gpuValue');
        const deltaEl = document.getElementById('gpuDelta');
        if (!valueEl) return;

        if (temp !== null && temp !== undefined && Number.isFinite(temp)) {
            valueEl.textContent = Math.round(temp) + '°C';

            // Update delta
            if (deltaEl) {
                updateDelta('gpuDelta', temp, this.previousValues.gpuTemp, true);
            }

            // Color code based on temperature
            if (temp > CONFIG.LIMITS.GPU_TEMP_DANGER) {
                valueEl.style.color = COLORS.ACCENT_RED;
            } else if (temp > CONFIG.LIMITS.GPU_TEMP_HOT) {
                valueEl.style.color = COLORS.ACCENT_YELLOW;
            } else {
                valueEl.style.color = COLORS.ACCENT_GREEN;
            }

            // Animate if changed significantly
            if (this.previousValues.gpuTemp !== null &&
                Math.abs(temp - this.previousValues.gpuTemp) > 5) {
                animateValueChange('gpuValue');
            }

            this.previousValues.gpuTemp = temp;
        } else {
            valueEl.textContent = '-°C';
            if (deltaEl) deltaEl.textContent = '';
        }
    }

    /**
     * Update VRAM usage
     */
    updateVRAM(vramUsed, vramTotal, vramPercent) {
        const valueEl = document.getElementById('vramValue');
        const deltaEl = document.getElementById('vramDelta');
        if (!valueEl) return;

        if (vramUsed !== null && vramUsed !== undefined && Number.isFinite(vramUsed)) {
            const vramGB = vramUsed / 1024; // Convert MB to GB
            valueEl.textContent = formatNumber(vramGB, 1) + ' GB';

            // Update delta
            if (deltaEl) {
                updateDelta('vramDelta', vramGB, this.previousValues.vramUsed, true);
            }

            // Color code based on percentage
            if (vramPercent > 90) {
                valueEl.style.color = COLORS.ACCENT_RED;
            } else if (vramPercent > 75) {
                valueEl.style.color = COLORS.ACCENT_YELLOW;
            } else {
                valueEl.style.color = COLORS.ACCENT_GREEN;
            }

            // Animate if changed significantly
            if (this.previousValues.vramUsed !== null &&
                Math.abs(vramGB - this.previousValues.vramUsed) > 0.5) {
                animateValueChange('vramValue');
            }

            this.previousValues.vramUsed = vramGB;
        } else {
            valueEl.textContent = '- GB';
            if (deltaEl) deltaEl.textContent = '';
        }
    }

    /**
     * Update RAM usage
     */
    updateRAM(ramUsed, ramPercent) {
        const valueEl = document.getElementById('ramValue');
        const deltaEl = document.getElementById('ramDelta');
        if (!valueEl) return;

        if (ramUsed !== null && ramUsed !== undefined && Number.isFinite(ramUsed)) {
            valueEl.textContent = formatNumber(ramUsed, 1) + ' GB';

            // Update delta
            if (deltaEl) {
                updateDelta('ramDelta', ramUsed, this.previousValues.ramUsed, true);
            }

            // Color code based on percentage
            if (ramPercent > CONFIG.LIMITS.RAM_DANGER) {
                valueEl.style.color = COLORS.ACCENT_RED;
            } else if (ramPercent > CONFIG.LIMITS.RAM_WARN) {
                valueEl.style.color = COLORS.ACCENT_YELLOW;
            } else {
                valueEl.style.color = COLORS.ACCENT_GREEN;
            }

            // Animate if changed significantly
            if (this.previousValues.ramUsed !== null &&
                Math.abs(ramUsed - this.previousValues.ramUsed) > 0.5) {
                animateValueChange('ramValue');
            }

            this.previousValues.ramUsed = ramUsed;
        } else {
            valueEl.textContent = '- GB';
            if (deltaEl) deltaEl.textContent = '';
        }
    }

    /**
     * Update queue size
     */
    updateQueue(queueSize) {
        const el = document.getElementById('queueValue');
        if (!el) return;

        if (queueSize !== null && queueSize !== undefined) {
            el.textContent = queueSize + ' files';

            // Color code based on queue size
            if (queueSize === CONFIG.QUEUE.EMPTY) {
                el.style.color = COLORS.ACCENT_RED;
            } else if (queueSize < CONFIG.QUEUE.LOW) {
                el.style.color = COLORS.ACCENT_YELLOW;
            } else {
                el.style.color = COLORS.ACCENT_GREEN;
            }
        } else {
            el.textContent = '-';
            el.style.color = COLORS.TEXT_SECONDARY;
        }
    }
}

export default StatusBar;
