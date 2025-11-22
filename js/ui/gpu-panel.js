/**
 * GPU Panel Component
 * Displays detailed GPU stats in the bottom panel
 */

import CONFIG, { COLORS } from '../core/config.js';
import { formatNumber } from '../utils/formatters.js';

export class GPUPanel {
    constructor() {
        this.elements = {
            gpuName: document.getElementById('gpuName'),
            gpuTemp: document.getElementById('gpuTemp'),
            gpuPower: document.getElementById('gpuPower'),
            gpuUtilPercent: document.getElementById('gpuUtilPercent'),
            gpuMemUsed: document.getElementById('gpuMemUsed'),
            // Gauges
            tempGauge: document.getElementById('tempGauge'),
            tempLabel: document.getElementById('tempLabel'),
            utilGauge: document.getElementById('utilGauge'),
            utilLabel: document.getElementById('utilLabel'),
            memGauge: document.getElementById('memGauge'),
            memLabel: document.getElementById('memLabel'),
            powerGauge: document.getElementById('powerGauge'),
            powerLabel: document.getElementById('powerLabel')
        };
    }

    /**
     * Update GPU panel with new data
     */
    update(data) {
        if (!data) return;

        // GPU Name
        if (this.elements.gpuName && data.gpuName) {
            this.elements.gpuName.textContent = data.gpuName;
        }

        // Temperature
        if (data.gpuTemp !== null && data.gpuTemp !== undefined) {
            const temp = Math.round(data.gpuTemp);
            
            if (this.elements.gpuTemp) {
                this.elements.gpuTemp.textContent = temp + '°C';
                
                // Color code
                if (temp > CONFIG.LIMITS.GPU_TEMP_DANGER) {
                    this.elements.gpuTemp.style.color = COLORS.ACCENT_RED;
                } else if (temp > CONFIG.LIMITS.GPU_TEMP_HOT) {
                    this.elements.gpuTemp.style.color = COLORS.ACCENT_YELLOW;
                } else {
                    this.elements.gpuTemp.style.color = COLORS.ACCENT_GREEN;
                }
            }

            // Temperature gauge (out of 85°C max)
            if (this.elements.tempGauge && this.elements.tempLabel) {
                const tempPercent = Math.min((temp / 85) * 100, 100);
                this.elements.tempGauge.style.width = tempPercent + '%';
                this.elements.tempLabel.textContent = temp + '°C / 85°C';

                // Color the gauge
                if (temp > CONFIG.LIMITS.GPU_TEMP_DANGER) {
                    this.elements.tempGauge.style.background = COLORS.ACCENT_RED;
                } else if (temp > CONFIG.LIMITS.GPU_TEMP_HOT) {
                    this.elements.tempGauge.style.background = COLORS.ACCENT_YELLOW;
                } else {
                    this.elements.tempGauge.style.background = COLORS.ACCENT_GREEN;
                }
            }
        }

        // GPU Utilization
        if (data.gpuUtil !== null && data.gpuUtil !== undefined) {
            const util = Math.round(data.gpuUtil);

            if (this.elements.gpuUtilPercent) {
                this.elements.gpuUtilPercent.textContent = util + '%';
            }

            if (this.elements.utilGauge && this.elements.utilLabel) {
                this.elements.utilGauge.style.width = util + '%';
                this.elements.utilLabel.textContent = util + '%';

                // Color code based on utilization
                if (util > 80) {
                    this.elements.utilGauge.style.background = COLORS.ACCENT_GREEN; // High is good!
                } else if (util > 50) {
                    this.elements.utilGauge.style.background = COLORS.ACCENT_YELLOW;
                } else {
                    this.elements.utilGauge.style.background = COLORS.ACCENT_RED; // Low is bad
                }
            }
        }

        // VRAM
        if (data.gpuMemUsed !== null && data.gpuMemUsed !== undefined && data.gpuMemTotal) {
            const vramUsedGB = data.gpuMemUsed / 1024;
            const vramTotalGB = data.gpuMemTotal / 1024;
            const vramPercent = (data.gpuMemUsed / data.gpuMemTotal) * 100;

            if (this.elements.gpuMemUsed) {
                this.elements.gpuMemUsed.textContent = formatNumber(vramUsedGB, 1) + ' GB';
            }

            if (this.elements.memGauge && this.elements.memLabel) {
                this.elements.memGauge.style.width = vramPercent + '%';
                this.elements.memLabel.textContent = 
                    formatNumber(vramUsedGB, 1) + ' GB / ' + formatNumber(vramTotalGB, 1) + ' GB';

                // Color code
                if (vramPercent > 90) {
                    this.elements.memGauge.style.background = COLORS.ACCENT_RED;
                } else if (vramPercent > 75) {
                    this.elements.memGauge.style.background = COLORS.ACCENT_YELLOW;
                } else {
                    this.elements.memGauge.style.background = COLORS.ACCENT_GREEN;
                }
            }
        }

        // Power Draw
        if (data.gpuPowerDraw !== null && data.gpuPowerDraw !== undefined && data.gpuPowerLimit) {
            const powerPercent = (data.gpuPowerDraw / data.gpuPowerLimit) * 100;

            if (this.elements.gpuPower) {
                this.elements.gpuPower.textContent = Math.round(data.gpuPowerDraw) + ' W';
            }

            if (this.elements.powerGauge && this.elements.powerLabel) {
                this.elements.powerGauge.style.width = powerPercent + '%';
                this.elements.powerLabel.textContent = 
                    Math.round(data.gpuPowerDraw) + ' W / ' + Math.round(data.gpuPowerLimit) + ' W';
            }
        }
    }
}

export default GPUPanel;
