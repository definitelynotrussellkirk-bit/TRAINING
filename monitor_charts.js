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

    // ========== PATTERN×LENGTH HEATMAP (NEW) ==========

    renderPatternHeatmap(matrix) {
        const el = document.getElementById('bucketHeatmap');
        if (!el) return;

        if (!matrix || !matrix.rows || !matrix.cols || !matrix.data) {
            el.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No pattern data yet...</p>';
            return;
        }

        const {rows, cols, data} = matrix;

        let html = '<table class="heatmap"><thead><tr>';
        html += '<th style="min-width:100px; text-align:left;">Pattern \\ Length</th>';
        cols.forEach(c => {
            html += `<th style="min-width:70px; text-align:center;">${c}</th>`;
        });
        html += '</tr></thead><tbody>';

        rows.forEach((rowName, ri) => {
            html += '<tr>';
            html += `<td style="text-align:left; font-weight:bold; color: #00d9ff;">${rowName}</td>`;

            cols.forEach((colName, ci) => {
                const cell = data[ri][ci] || {seen: 0, em: 0};
                const emPct = Math.round(cell.em * 100);

                let bgColor, textColor, displayText;

                if (cell.seen === 0) {
                    // No data
                    bgColor = '#1a1f3a';
                    textColor = '#666';
                    displayText = '—';
                } else {
                    // Has data - gradient based on EM
                    bgColor = `linear-gradient(90deg, #00ff88 ${emPct}%, #2a3f5f ${emPct}%)`;
                    textColor = '#fff';
                    displayText = cell.seen;
                }

                html += `<td title="Seen: ${cell.seen}, EM: ${emPct}%" ` +
                    `style="background: ${bgColor}; color: ${textColor}; ` +
                    `text-align: center; padding: 8px 5px; font-weight: bold;">${displayText}</td>`;
            });

            html += '</tr>';
        });

        html += '</tbody></table>';

        // Add legend
        html += `
            <div style="margin-top: 10px; font-size: 0.8em; color: #888; text-align: center;">
                <strong>Legend:</strong>
                Numbers show sample count.
                Green gradient = % Exact Match.
                Darker = lower accuracy.
            </div>
        `;

        el.innerHTML = html;
    },

    // ========== LORA LAYER DELTAS (NEW) ==========

    renderLoRALayers(layerDeltas) {
        const el = document.getElementById('loraTopLayers');
        if (!el) return;

        if (!layerDeltas || !layerDeltas.length) {
            el.innerHTML = '<p style="color: #888;">Waiting for first measurement...</p>';
            return;
        }

        // Sort by delta_norm descending, take top 5
        const top5 = [...layerDeltas]
            .sort((a, b) => b.delta_norm - a.delta_norm)
            .slice(0, 5);

        const html = top5.map((layer, index) => {
            const barWidth = Math.min((layer.delta_norm / top5[0].delta_norm) * 100, 100);

            return `
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                        <span style="color: #00d9ff; font-weight: bold;">${index + 1}. Layer ${layer.layer}</span>
                        <span style="color: #00ff88; font-size: 0.9em;">Δ=${layer.delta_norm.toFixed(3)}</span>
                    </div>
                    <div style="display: flex; gap: 5px; font-size: 0.85em; color: #888;">
                        <span>QKV: ${layer.qkv.toFixed(3)}</span>
                        <span>|</span>
                        <span>MLP: ${layer.mlp.toFixed(3)}</span>
                    </div>
                    <div style="height: 4px; background: #2a3f5f; border-radius: 2px; margin-top: 4px; overflow: hidden;">
                        <div style="height: 100%; width: ${barWidth}%; background: linear-gradient(90deg, #00ff88, #00d9ff);"></div>
                    </div>
                </div>
            `;
        }).join('');

        el.innerHTML = html;
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
