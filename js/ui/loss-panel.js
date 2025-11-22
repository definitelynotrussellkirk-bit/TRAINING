/**
 * Loss Panel Component
 * Displays loss metrics, learning rate, and trends
 */

import { formatLoss, formatScientific, formatNumber, formatLargeNumber } from '../utils/formatters.js';
import { COLORS } from '../core/config.js';

export class LossPanel {
    constructor() {
        this.elements = {
            currentLoss: document.getElementById('currentLoss'),
            validationLoss: document.getElementById('validationLoss'),
            streamingCE: document.getElementById('streamingCE'),
            tokenEntropy: document.getElementById('tokenEntropy'),
            learningRate: document.getElementById('learningRate'),
            lossTrend: document.getElementById('lossTrend'),
            // Speed metrics
            tokensPerSec: document.getElementById('tokensPerSec'),
            tokensPerStep: document.getElementById('tokensPerStep'),
            stepsPerSec: document.getElementById('stepsPerSec'),
            examplesPerSec: document.getElementById('examplesPerSec'),
            timeElapsed: document.getElementById('timeElapsed'),
            etaRemaining: document.getElementById('etaRemaining'),
            etaTime: document.getElementById('etaTime')
        };

        this.lossHistory = [];
    }

    /**
     * Update loss panel with new data
     */
    update(data) {
        if (!data) return;

        // Current loss (big display)
        if (this.elements.currentLoss && data.loss !== null) {
            this.elements.currentLoss.textContent = formatLoss(data.loss);

            // Track loss history
            this.lossHistory.push(data.loss);
            if (this.lossHistory.length > 20) {
                this.lossHistory.shift();
            }

            // Color code based on trend
            if (this.lossHistory.length >= 2) {
                const recent = this.lossHistory[this.lossHistory.length - 1];
                const previous = this.lossHistory[this.lossHistory.length - 2];
                
                if (recent < previous) {
                    this.elements.currentLoss.style.color = COLORS.ACCENT_GREEN;
                } else if (recent > previous) {
                    this.elements.currentLoss.style.color = COLORS.ACCENT_RED;
                } else {
                    this.elements.currentLoss.style.color = COLORS.ACCENT_YELLOW;
                }
            }
        }

        // Validation Loss with overfitting indicator
        if (this.elements.validationLoss) {
            if (data.validationLoss !== null && data.validationLoss !== undefined) {
                const valLoss = data.validationLoss;
                this.elements.validationLoss.textContent = formatLoss(valLoss);

                // Color code based on train/val gap (overfitting detection)
                if (data.loss !== null && data.loss !== undefined) {
                    const gap = valLoss - data.loss;
                    if (gap > 0.5) {
                        this.elements.validationLoss.style.color = COLORS.ACCENT_RED; // Overfitting!
                    } else if (gap > 0.2) {
                        this.elements.validationLoss.style.color = COLORS.ACCENT_YELLOW; // Caution
                    } else {
                        this.elements.validationLoss.style.color = COLORS.ACCENT_GREEN; // Good
                    }
                }
            } else {
                this.elements.validationLoss.textContent = '--';
                this.elements.validationLoss.style.color = COLORS.TEXT_SECONDARY;
            }
        }

        // Streaming CE (EMA)
        if (this.elements.streamingCE && data.streamingCE !== null) {
            this.elements.streamingCE.textContent = formatLoss(data.streamingCE);
        }

        // Token Entropy
        if (this.elements.tokenEntropy && data.tokenEntropy !== null) {
            this.elements.tokenEntropy.textContent = formatNumber(data.tokenEntropy, 3);
        }

        // Learning Rate
        if (this.elements.learningRate && data.learningRate !== null) {
            this.elements.learningRate.textContent = formatScientific(data.learningRate);
        }

        // Loss Trend
        if (this.elements.lossTrend && this.lossHistory.length >= 10) {
            const recent5 = this.lossHistory.slice(-5).reduce((a, b) => a + b, 0) / 5;
            const previous5 = this.lossHistory.slice(-10, -5).reduce((a, b) => a + b, 0) / 5;
            
            let trend = '→ Stable';
            let color = COLORS.ACCENT_YELLOW;
            
            if (recent5 < previous5 - 0.01) {
                trend = '↓ Decreasing';
                color = COLORS.ACCENT_GREEN;
            } else if (recent5 > previous5 + 0.01) {
                trend = '↑ Increasing';
                color = COLORS.ACCENT_RED;
            }
            
            this.elements.lossTrend.textContent = trend;
            this.elements.lossTrend.style.color = color;
        }

        // Speed metrics
        if (this.elements.tokensPerSec && data.tokensPerSec !== null) {
            this.elements.tokensPerSec.textContent = formatLargeNumber(data.tokensPerSec);
        }

        if (this.elements.tokensPerStep && data.tokensPerStep !== null) {
            this.elements.tokensPerStep.textContent = formatLargeNumber(data.tokensPerStep);
        }

        if (this.elements.stepsPerSec && data.stepsPerSec !== null) {
            this.elements.stepsPerSec.textContent = formatNumber(data.stepsPerSec, 2);
        }

        if (this.elements.examplesPerSec && data.examplesPerSec !== null) {
            this.elements.examplesPerSec.textContent = formatNumber(data.examplesPerSec, 2);
        }

        // Time metrics
        if (this.elements.timeElapsed && data.timeElapsed) {
            this.elements.timeElapsed.textContent = this.formatDuration(data.timeElapsed);
        }

        if (this.elements.etaRemaining && data.etaRemaining) {
            this.elements.etaRemaining.textContent = this.formatDuration(data.etaRemaining);
        }

        if (this.elements.etaTime && data.etaTime) {
            this.elements.etaTime.textContent = data.etaTime;
        }
    }

    /**
     * Format duration in seconds to HH:MM:SS
     */
    formatDuration(seconds) {
        if (!seconds || seconds < 0) return '--:--:--';

        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        return [hours, minutes, secs]
            .map(v => v.toString().padStart(2, '0'))
            .join(':');
    }
}

export default LossPanel;
