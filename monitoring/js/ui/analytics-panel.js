/**
 * Analytics Panel Component
 * Displays advanced training analytics
 */

import { formatNumber, formatLargeNumber } from '../utils/formatters.js';

export class AnalyticsPanel {
    constructor() {
        this.elements = {
            penaltyStats: document.getElementById('penaltyStats'),
            queueVelocityStats: document.getElementById('queueVelocityStats'),
            lengthCoverageList: document.getElementById('lengthCoverageList'),
            patternLossList: document.getElementById('patternLossList'),
            patternLayerList: document.getElementById('patternLayerList'),
            layerStabilityList: document.getElementById('layerStabilityList'),
            vramScatterStatus: document.getElementById('vramScatterStatus'),
            vramScatter: document.getElementById('vramScatter')
        };
    }

    /**
     * Update all analytics displays
     */
    update(data) {
        if (!data) return;

        this.updatePenaltyStats(data.logitPenaltyStats);
        this.updateQueueVelocity(data.queueVelocity);
        this.updateLengthCoverage(data.lengthCoverage);
        this.updatePatternLoss(data.patternLossTrend);
        this.updatePatternLayer(data.patternLayerCorrelation);
        this.updateLayerStability(data.layerStabilitySummary);
    }

    /**
     * Update penalty statistics
     */
    updatePenaltyStats(stats) {
        if (!this.elements.penaltyStats) return;

        if (!stats || !stats.total_hits) {
            this.elements.penaltyStats.textContent = 'No penalties';
            return;
        }

        const html = `
            <div style="font-size: 1.2em; font-weight: bold; color: #ff4444;">${formatLargeNumber(stats.total_hits)}</div>
            <div style="font-size: 0.8em; color: #888; margin-top: 4px;">
                Avg: ${formatNumber(stats.avg_per_sample, 2)}/sample<br>
                Rate: ${formatNumber(stats.hit_rate * 100, 1)}%
            </div>
        `;
        this.elements.penaltyStats.innerHTML = html;
    }

    /**
     * Update queue velocity statistics
     */
    updateQueueVelocity(velocity) {
        if (!this.elements.queueVelocityStats) return;

        if (!velocity) {
            this.elements.queueVelocityStats.textContent = '--';
            return;
        }

        const html = `
            <div style="font-size: 1.2em; font-weight: bold; color: #00ff88;">
                ${formatNumber(velocity.samples_per_sec, 1)} samples/s
            </div>
            <div style="font-size: 0.8em; color: #888; margin-top: 4px;">
                ${formatLargeNumber(velocity.samples_per_hour)}/hour<br>
                Batch: ${velocity.effective_batch}
            </div>
        `;
        this.elements.queueVelocityStats.innerHTML = html;
    }

    /**
     * Update length coverage statistics
     */
    updateLengthCoverage(coverage) {
        if (!this.elements.lengthCoverageList) return;

        if (!coverage || !Array.isArray(coverage)) {
            this.elements.lengthCoverageList.textContent = 'No data';
            return;
        }

        const html = coverage.slice(0, 5).map(item => `
            <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                <span style="color: #888;">${item.range}</span>
                <span style="color: #00d9ff;">${formatNumber(item.percent, 1)}%</span>
            </div>
        `).join('');

        this.elements.lengthCoverageList.innerHTML = html || 'No data';
    }

    /**
     * Update pattern loss trends
     */
    updatePatternLoss(trends) {
        if (!this.elements.patternLossList) return;

        if (!trends || typeof trends !== 'object') {
            this.elements.patternLossList.textContent = 'No patterns';
            return;
        }

        // Convert object to array and sort by recent_loss
        const trendsArray = Object.entries(trends)
            .map(([pattern, data]) => ({
                pattern,
                loss: data.recent_loss || data.avg_loss || 0,
                samples: data.samples || 0
            }))
            .sort((a, b) => b.loss - a.loss);

        const html = trendsArray.slice(0, 5).map(item => {
            const color = item.loss < 0.5 ? '#00ff88' : item.loss < 1.0 ? '#ffaa00' : '#ff4444';
            return `
                <div style="display: flex; justify-content: space-between; margin: 2px 0;">
                    <span style="color: #888; font-size: 0.85em;">${item.pattern}</span>
                    <span style="color: ${color};">${formatNumber(item.loss, 3)}</span>
                </div>
            `;
        }).join('');

        this.elements.patternLossList.innerHTML = html || 'No data';
    }

    /**
     * Update pattern-layer correlation
     */
    updatePatternLayer(correlation) {
        if (!this.elements.patternLayerList) return;

        if (!correlation || !Array.isArray(correlation)) {
            this.elements.patternLayerList.textContent = 'No correlations';
            return;
        }

        const html = correlation.slice(0, 5).map(item => `
            <div style="margin: 3px 0; font-size: 0.85em;">
                <div style="color: #888;">${item.pattern || 'Unknown'}</div>
                <div style="color: #00d9ff;">Layer ${item.layer}: ${formatNumber(item.correlation, 2)}</div>
            </div>
        `).join('');

        this.elements.patternLayerList.innerHTML = html || 'No data';
    }

    /**
     * Update layer stability information
     */
    updateLayerStability(summary) {
        if (!this.elements.layerStabilityList) return;

        if (!summary) {
            this.elements.layerStabilityList.textContent = 'No data';
            return;
        }

        const html = `
            <div style="font-size: 1.2em; font-weight: bold; color: #00ff88;">
                ${formatNumber(summary.avg_stability || 0, 3)}
            </div>
            <div style="font-size: 0.8em; color: #888; margin-top: 4px;">
                Avg stability score<br>
                ${summary.stable_layers || 0} stable layers
            </div>
        `;
        this.elements.layerStabilityList.innerHTML = html;
    }
}

export default AnalyticsPanel;
