/* ========================================
   MONITOR CHARTS - Rendering Logic
   All chart and visualization code
   ======================================== */

const ChartsRenderer = {
    // ========== LOSS SPARKLINE ==========

    drawLossSparkline(lossData) {
        const canvas = document.getElementById('lossSparkline');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        // Scale backing store to device pixels
        if (canvas.width !== Math.round(rect.width * dpr) ||
            canvas.height !== Math.round(rect.height * dpr)) {
            canvas.width = Math.round(rect.width * dpr);
            canvas.height = Math.round(rect.height * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const width = rect.width;
        const height = rect.height;

        ctx.clearRect(0, 0, width, height);
        if (!lossData || lossData.length < 2) return;

        const minLoss = Math.min(...lossData);
        const maxLoss = Math.max(...lossData);
        const range = (maxLoss - minLoss) || 1;

        // Draw grid
        ctx.strokeStyle = '#2a3f5f';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = (height / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw line
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const stepX = width / (lossData.length - 1);
        lossData.forEach((loss, i) => {
            const x = i * stepX;
            const y = height - ((loss - minLoss) / range) * (height - 10) - 5;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        // Draw labels
        ctx.fillStyle = '#888';
        ctx.font = '10px monospace';
        ctx.fillText(maxLoss.toFixed(3), 5, 10);
        ctx.fillText(minLoss.toFixed(3), 5, height - 2);
    },

    // ========== SMART ALERTS (NEW) ==========

    renderSmartAlerts(alerts) {
        const container = document.getElementById('smartAlerts');
        if (!container) return;

        if (!alerts || !alerts.length) {
            container.innerHTML = '';
            return;
        }

        const levelColors = {
            'critical': 'background: linear-gradient(135deg, #ff4444, #cc0000);',
            'warning': 'background: linear-gradient(135deg, #ffaa00, #ff8800);',
            'info': 'background: linear-gradient(135deg, #00d9ff, #0088cc);'
        };

        const levelIcons = {
            'critical': '⚠️',
            'warning': '📊',
            'info': 'ℹ️'
        };

        const html = alerts.map(alert => {
            const style = levelColors[alert.level] || levelColors.info;
            const icon = levelIcons[alert.level] || 'ℹ️';

            return `
                <div class="alert-banner" style="${style} color: white; padding: 15px 20px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                    <div style="display: flex; align-items: flex-start; gap: 15px;">
                        <div style="font-size: 2em;">${icon}</div>
                        <div style="flex: 1;">
                            <div style="font-weight: bold; font-size: 1.1em; margin-bottom: 8px;">
                                ${alert.message}
                            </div>
                            <div style="font-size: 0.95em; opacity: 0.95; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;">
                                <strong>💡 Action:</strong> ${alert.action}
                            </div>
                        </div>
                        <button onclick="this.parentElement.parentElement.remove()"
                                style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); color: white; padding: 5px 10px; border-radius: 5px; cursor: pointer; font-size: 1.2em;">
                            ✕
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;

        // Send notifications for critical alerts
        alerts.forEach(alert => {
            if (alert.level === 'critical' && typeof sendNotification === 'function') {
                sendNotification('Training Alert!', alert.message, 'urgent');
            }
            if (alert.level === 'critical' && typeof playSound === 'function') {
                playSound('critical');
            }
        });
    },

    // ========== CONFIDENCE BAR (EXISTING) ==========

    updateConfidenceBar(loss) {
        const confSection = document.getElementById('confidenceSection');
        if (!confSection) return;

        if (!loss || loss <= 0) return;

        // Calculate confidence (inverse of loss)
        const confidence = Math.min(Math.exp(-loss) * 100, 100);

        const confBar = document.getElementById('confidenceBar');
        const confText = document.getElementById('confidenceText');
        const confInterp = document.getElementById('confidenceInterpret');

        if (!confBar || !confText || !confInterp) return;

        confBar.style.width = confidence + '%';
        confText.textContent = confidence.toFixed(1) + '%';

        // Color coding and interpretation
        if (confidence > 80) {
            confBar.className = 'confidence-bar conf-high';
            confInterp.textContent = '🟢 High confidence - model is certain';
        } else if (confidence > 50) {
            confBar.className = 'confidence-bar conf-medium';
            confInterp.textContent = '🟡 Medium confidence - some uncertainty';
        } else {
            confBar.className = 'confidence-bar conf-low';
            confInterp.textContent = '🔴 Low confidence - model is guessing';
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChartsRenderer;
}
