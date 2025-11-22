/**
 * Training Panel Component
 * Displays training status, progress, and metrics
 */

import { formatNumber, formatLargeNumber } from '../utils/formatters.js';
import { COLORS } from '../core/config.js';

export class TrainingPanel {
    constructor() {
        this.elements = {
            status: document.getElementById('status'),
            currentStep: document.getElementById('currentStep'),
            totalSteps: document.getElementById('totalSteps'),
            epoch: document.getElementById('epoch'),
            progressPercent: document.getElementById('progressPercent'),
            maxOutputTokens: document.getElementById('maxOutputTokens'),
            contextWindow: document.getElementById('contextWindow'),
            progressBar: document.getElementById('progressBar'),
            overallProgressBar: document.getElementById('overallProgressBar'),
            overallProgressText: document.getElementById('overallProgressText'),
            overallProgressLabel: document.getElementById('overallProgressLabel'),
            // Model config
            loraRank: document.getElementById('loraRank'),
            loraAlpha: document.getElementById('loraAlpha'),
            batchSize: document.getElementById('batchSize'),
            gradAccum: document.getElementById('gradAccum'),
            effectiveBatch: document.getElementById('effectiveBatch'),
            modelName: document.getElementById('modelName')
        };
    }

    /**
     * Update training panel with new data
     */
    update(data) {
        if (!data) return;

        // Status
        if (this.elements.status && data.status) {
            this.elements.status.textContent = data.status.toUpperCase();
            
            // Color code
            const statusColors = {
                'training': COLORS.ACCENT_GREEN,
                'idle': COLORS.ACCENT_YELLOW,
                'error': COLORS.ACCENT_RED,
                'crashed': COLORS.ACCENT_RED
            };
            this.elements.status.style.color = statusColors[data.status] || COLORS.TEXT_PRIMARY;
        }

        // Steps
        if (this.elements.currentStep && data.currentStep !== null) {
            this.elements.currentStep.textContent = formatLargeNumber(data.currentStep);
        }

        if (this.elements.totalSteps && data.totalSteps !== null) {
            this.elements.totalSteps.textContent = formatLargeNumber(data.totalSteps);
        }

        // Epoch
        if (this.elements.epoch && data.epoch !== null) {
            this.elements.epoch.textContent = data.epoch;
        }

        // Progress
        if (data.currentStep !== null && data.totalSteps !== null && data.totalSteps > 0) {
            const percent = (data.currentStep / data.totalSteps) * 100;
            
            if (this.elements.progressPercent) {
                this.elements.progressPercent.textContent = percent.toFixed(1) + '%';
            }

            if (this.elements.progressBar) {
                this.elements.progressBar.style.width = percent + '%';
            }
        }

        // Overall progress (batch progress)
        if (data.batchStep !== null && data.batchTotalSteps !== null && data.batchTotalSteps > 0) {
            const batchPercent = (data.batchStep / data.batchTotalSteps) * 100;

            if (this.elements.overallProgressBar) {
                this.elements.overallProgressBar.style.width = batchPercent + '%';
            }

            if (this.elements.overallProgressText) {
                this.elements.overallProgressText.textContent = batchPercent.toFixed(1) + '%';
            }

            if (this.elements.overallProgressLabel) {
                this.elements.overallProgressLabel.textContent = batchPercent.toFixed(1) + '%';
            }
        }

        // Max output tokens
        if (this.elements.maxOutputTokens && data.maxOutputTokens !== null) {
            this.elements.maxOutputTokens.textContent = formatLargeNumber(data.maxOutputTokens);
        }

        // Context window
        if (this.elements.contextWindow && data.contextWindow !== null) {
            this.elements.contextWindow.textContent = formatLargeNumber(data.contextWindow);
        }

        // Model config
        if (this.elements.modelName && data.modelName) {
            this.elements.modelName.textContent = data.modelName;
        }

        // Layer monitoring stats (element IDs are misleading - they're for layer stats, not LoRA)
        if (this.elements.loraRank && data.layerActivitySummary) {
            // Show total layers tracked
            const totalLayers = data.layerActivitySummary.overall?.total_layers || 0;
            this.elements.loraRank.textContent = totalLayers;
        }

        if (this.elements.loraAlpha && data.layerActivitySummary) {
            // Show average weight delta
            const avgDelta = data.layerActivitySummary.overall?.avg_delta || 0;
            this.elements.loraAlpha.textContent = formatNumber(avgDelta, 6);
        }

        // Batch config
        if (this.elements.batchSize && data.batchSize !== null) {
            this.elements.batchSize.textContent = data.batchSize;
        }

        if (this.elements.gradAccum && data.gradAccum !== null) {
            this.elements.gradAccum.textContent = data.gradAccum;
        }

        if (this.elements.effectiveBatch && data.batchSize !== null && data.gradAccum !== null) {
            this.elements.effectiveBatch.textContent = data.batchSize * data.gradAccum;
        }
    }
}

export default TrainingPanel;
