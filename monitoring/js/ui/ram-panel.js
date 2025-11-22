/**
 * RAM Panel Component
 * Displays detailed system memory stats in the bottom panel
 */

import CONFIG, { COLORS } from '../core/config.js';
import { formatNumber } from '../utils/formatters.js';

export class RAMPanel {
    constructor() {
        this.elements = {
            ramStatus: document.getElementById('ramStatus'),
            ramUsed: document.getElementById('ramUsed'),
            trainingRAM: document.getElementById('trainingRAM'),
            ramAvailable: document.getElementById('ramAvailable'),
            oomRisk: document.getElementById('oomRisk'),
            // Gauges
            ramGauge: document.getElementById('ramGauge'),
            ramLabel: document.getElementById('ramLabel'),
            trainingRAMGauge: document.getElementById('trainingRAMGauge'),
            trainingRAMLabel: document.getElementById('trainingRAMLabel'),
            swapGauge: document.getElementById('swapGauge'),
            swapLabel: document.getElementById('swapLabel')
        };
    }

    /**
     * Update RAM panel with new data
     */
    update(data) {
        if (!data) return;

        const ramPercent = data.ramPercent || 0;
        const ramUsedGB = data.ramUsed || 0;
        const ramTotalGB = data.ramTotal || 0;
        const ramAvailableGB = data.ramAvailable || 0;
        const trainingProcessGB = data.trainingProcessGB || 0;
        const swapUsedGB = data.swapUsed || 0;
        const swapTotalGB = data.swapTotal || 0;
        const swapPercent = data.swapPercent || 0;

        // Status text
        if (this.elements.ramStatus) {
            let status = 'Normal';
            if (ramPercent > 85) {
                status = 'Critical';
            } else if (ramPercent > 70) {
                status = 'High';
            }
            this.elements.ramStatus.textContent = status;
            
            // Color code
            if (ramPercent > 85) {
                this.elements.ramStatus.style.color = COLORS.ACCENT_RED;
            } else if (ramPercent > 70) {
                this.elements.ramStatus.style.color = COLORS.ACCENT_YELLOW;
            } else {
                this.elements.ramStatus.style.color = COLORS.ACCENT_GREEN;
            }
        }

        // RAM Used
        if (this.elements.ramUsed) {
            this.elements.ramUsed.textContent = formatNumber(ramUsedGB, 1) + ' GB';
        }

        // Training Process RAM
        if (this.elements.trainingRAM) {
            this.elements.trainingRAM.textContent = formatNumber(trainingProcessGB, 1) + ' GB';
        }

        // Available RAM
        if (this.elements.ramAvailable) {
            this.elements.ramAvailable.textContent = formatNumber(ramAvailableGB, 1) + ' GB';
        }

        // OOM Risk
        if (this.elements.oomRisk) {
            let risk = 'LOW';
            let color = COLORS.ACCENT_GREEN;
            
            if (ramPercent > 85) {
                risk = 'HIGH';
                color = COLORS.ACCENT_RED;
            } else if (ramPercent > 70) {
                risk = 'MEDIUM';
                color = COLORS.ACCENT_YELLOW;
            }
            
            this.elements.oomRisk.textContent = risk;
            this.elements.oomRisk.style.color = color;
        }

        // System RAM Gauge
        if (this.elements.ramGauge && this.elements.ramLabel) {
            this.elements.ramGauge.style.width = ramPercent + '%';
            this.elements.ramLabel.textContent = 
                formatNumber(ramUsedGB, 1) + ' GB / ' + formatNumber(ramTotalGB, 1) + ' GB';

            // Color code
            if (ramPercent > 85) {
                this.elements.ramGauge.style.background = COLORS.ACCENT_RED;
            } else if (ramPercent > 70) {
                this.elements.ramGauge.style.background = COLORS.ACCENT_YELLOW;
            } else {
                this.elements.ramGauge.style.background = COLORS.ACCENT_GREEN;
            }
        }

        // Training Process RAM Gauge
        if (this.elements.trainingRAMGauge && this.elements.trainingRAMLabel) {
            // Calculate percentage relative to total RAM
            const trainingPercent = (trainingProcessGB / ramTotalGB) * 100;
            this.elements.trainingRAMGauge.style.width = trainingPercent + '%';
            this.elements.trainingRAMLabel.textContent = formatNumber(trainingProcessGB, 1) + ' GB';
        }

        // Swap Gauge
        if (this.elements.swapGauge && this.elements.swapLabel) {
            this.elements.swapGauge.style.width = swapPercent + '%';
            this.elements.swapLabel.textContent = 
                formatNumber(swapUsedGB, 1) + ' GB / ' + formatNumber(swapTotalGB, 1) + ' GB';

            // Color code swap (red if being used heavily)
            if (swapPercent > 50) {
                this.elements.swapGauge.style.background = COLORS.ACCENT_RED;
            } else if (swapPercent > 25) {
                this.elements.swapGauge.style.background = COLORS.ACCENT_YELLOW;
            } else {
                this.elements.swapGauge.style.background = COLORS.TEXT_SECONDARY;
            }
        }
    }
}

export default RAMPanel;
