/* ========================================
   MONITOR METRICS - Organized Update Functions
   All metric update logic extracted for maintainability
   ======================================== */

const MetricsUpdater = {
    // ========== HELPER UTILITIES ==========

    formatNumber(num, decimals = 2) {
        if (num === null || num === undefined || !Number.isFinite(num)) return '-';
        return num.toFixed(decimals);
    },

    formatPercent(value, decimals = 1) {
        if (value === null || value === undefined || !Number.isFinite(value)) return '-%';
        return (value * 100).toFixed(decimals) + '%';
    },

    formatLargeNumber(num) {
        if (!num) return '-';
        return num.toLocaleString();
    },

    setElementText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    },

    setElementHTML(id, html) {
        const el = document.getElementById(id);
        if (el) el.innerHTML = html;
    },

    colorCodeValue(elementId, value, thresholds) {
        const el = document.getElementById(elementId);
        if (!el) return;

        if (value >= thresholds.danger) {
            el.style.color = '#ff4444';
        } else if (value >= thresholds.warning) {
            el.style.color = '#ffaa00';
        } else {
            el.style.color = '#00ff88';
        }
    },

    // ========== TRAINING STATUS ==========

    updateTrainingStatus(data) {
        if (!data) return;

        // Status
        const statusEl = document.getElementById('status');
        if (statusEl) {
            statusEl.textContent = data.status.toUpperCase();
            statusEl.className = 'metric-value status-' + data.status;
        }

        // Steps
        this.setElementText('currentStep', this.formatLargeNumber(data.current_step));
        this.setElementText('totalSteps', this.formatLargeNumber(data.total_steps));
        this.setElementText('epoch', data.epoch || '-');

        // Progress
        const progress = (data.current_step / data.total_steps * 100) || 0;
        this.setElementText('progressPercent', progress.toFixed(1) + '%');

        // Progress bar
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }

        // Overall progress bar
        const overallProgressBar = document.getElementById('overallProgressBar');
        const overallProgressText = document.getElementById('overallProgressText');
        const overallProgressLabel = document.getElementById('overallProgressLabel');
        if (overallProgressBar) {
            overallProgressBar.style.width = progress + '%';
        }
        if (overallProgressText) {
            overallProgressText.textContent = progress.toFixed(1) + '%';
        }
        if (overallProgressLabel) {
            overallProgressLabel.textContent = progress.toFixed(1) + '%';
        }

        // Config display
        this.setElementText('loraRank', data.lora_r || '-');
        this.setElementText('loraAlpha', data.lora_alpha || '-');
        this.setElementText('trainableParams', data.trainable_params || '-');
        this.setElementText('modelName', data.model_name || '-');
        this.setElementText('maxOutputTokens', data.max_output_tokens || '-');
        this.setElementText('contextWindow', data.context_window || '-');
    },

    // ========== LOSS METRICS ==========

    updateLossMetrics(data) {
        if (!data) return;

        // Current loss
        const lossEl = document.getElementById('currentLoss');
        if (lossEl) {
            lossEl.textContent = this.formatNumber(data.loss, 4);

            // Color code based on trend
            if (data.loss < 1.0) {
                lossEl.style.color = '#00ff88';
            } else if (data.loss < 2.0) {
                lossEl.style.color = '#ffaa00';
            } else {
                lossEl.style.color = '#ff4444';
            }
        }

        // Learning rate
        this.setElementText('learningRate', data.learning_rate ?
            data.learning_rate.toExponential(2) : '-');

        // Loss trend (computed from history)
        // This will be set by the main update loop
    },

    // ========== SPEED & ETA ==========

    updateSpeedMetrics(data, stepsPerSec) {
        if (!data) return;

        this.setElementText('stepsPerSec', this.formatNumber(stepsPerSec, 2));

        // Examples/sec (steps/sec × effective batch size)
        const effectiveBatchSize = 8; // TODO: Get from config
        const examplesPerSec = stepsPerSec * effectiveBatchSize;
        this.setElementText('examplesPerSec', this.formatNumber(examplesPerSec, 2));

        // Time elapsed
        if (data.time_elapsed) {
            this.setElementText('timeElapsed', this.formatTime(data.time_elapsed));
        }

        // ETA
        if (data.eta_seconds) {
            this.setElementText('etaRemaining', this.formatTime(data.eta_seconds));
            this.setElementText('etaTime', data.eta_time || '-');
        }
    },

    formatTime(seconds) {
        if (!Number.isFinite(seconds) || seconds < 0) return '-';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}h ${m}m ${s}s`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
    },

    // ========== ACCURACY METRICS ==========

    updateAccuracyMetrics(data, accuracyHistory) {
        if (!data || !data.validation_example) return;

        // Running accuracy
        const totalEvals = accuracyHistory.length;
        const correctEvals = accuracyHistory.filter(x => x.correct).length;
        const accuracy = totalEvals > 0 ? (correctEvals / totalEvals) : 0;

        this.setElementText('accuracyPercent', this.formatPercent(accuracy, 1));
        this.setElementText('accuracyFraction', `${correctEvals} / ${totalEvals}`);

        // Accuracy trends (Last 20, Last 50, Overall)
        const last20 = accuracyHistory.slice(-20);
        const last50 = accuracyHistory.slice(-50);

        const acc20 = last20.length > 0 ?
            (last20.filter(x => x.correct).length / last20.length) : 0;
        const acc50 = last50.length > 0 ?
            (last50.filter(x => x.correct).length / last50.length) : 0;

        this.setElementText('accuracy20',
            `${this.formatPercent(acc20, 1)} (${last20.filter(x => x.correct).length}/${last20.length})`);
        this.setElementText('accuracy50',
            `${this.formatPercent(acc50, 1)} (${last50.filter(x => x.correct).length}/${last50.length})`);

        // Trend analysis
        const trendEl = document.getElementById('accuracyTrend');
        if (trendEl && last20.length >= 10) {
            let trendText = '';
            let trendClass = '';

            if (acc20 > accuracy + 0.05) {
                trendText = '↑ Improving';
                trendClass = 'success';
            } else if (acc20 < accuracy - 0.05) {
                trendText = '↓ Regressing';
                trendClass = 'error';
            } else {
                trendText = '→ Stable';
                trendClass = '';
            }

            trendEl.textContent = trendText;
            trendEl.className = 'metric-value ' + trendClass;
        }
    },

    // ========== CURRENT EXAMPLE ==========

    updateCurrentExample(data) {
        if (!data || !data.validation_example) return;

        const example = data.validation_example;

        // Step number
        this.setElementText('currentExampleStep', this.formatLargeNumber(example.step));

        // Prompt
        const promptEl = document.getElementById('prompt');
        if (promptEl) {
            promptEl.textContent = example.prompt || 'Waiting for validation step...';
        }

        // Golden answer
        const goldenEl = document.getElementById('golden');
        if (goldenEl) {
            goldenEl.textContent = example.golden || 'Waiting for validation step...';
        }

        // Model output
        const modelEl = document.getElementById('modelAnswer');
        if (modelEl) {
            modelEl.textContent = example.model_output || 'Waiting for validation step...';
        }

        // Match badge
        const badgeEl = document.getElementById('matchBadge');
        if (badgeEl && example.matches !== undefined) {
            if (example.matches) {
                badgeEl.innerHTML = ' <span class="match-badge match">✓ MATCH</span>';
            } else {
                badgeEl.innerHTML = ' <span class="match-badge no-match">✗ NO MATCH</span>';
            }
        }
    },

    // ========== RECENT EXAMPLES ==========

    updateRecentExamples(accuracyHistory) {
        const container = document.getElementById('recentExamples');
        if (!container || !accuracyHistory || accuracyHistory.length === 0) return;

        const recent = accuracyHistory.slice(-5).reverse();

        const html = recent.map(ex => {
            const matchIcon = ex.correct ? '✓' : '✗';
            const matchClass = ex.correct ? 'success' : 'error';
            const lossDisplay = ex.loss ? `Loss: ${ex.loss.toFixed(3)}` : '';

            return `
                <div style="margin-bottom: 10px; padding: 8px; background: rgba(42, 63, 95, 0.3); border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #00d9ff;">Step ${ex.step.toLocaleString()}</span>
                        <span class="${matchClass}" style="font-weight: bold;">${matchIcon} ${ex.correct ? 'Correct' : 'Wrong'}</span>
                    </div>
                    <div style="font-size: 0.85em; color: #888;">
                        ${lossDisplay}
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
    },

    // ========== GPU STATS ==========

    updateGPUStats(data) {
        if (!data) return;

        this.setElementText('gpuName', data.name || 'Loading...');
        this.setElementText('gpuTemp', data.temperature ? `${data.temperature}°C` : '-');
        this.setElementText('gpuPower', data.power ? `${Math.round(data.power)} W` : '-');
        this.setElementText('gpuUtilPercent', data.utilization ? `${data.utilization}%` : '-');
        this.setElementText('gpuMemUsed', data.memory_used ? `${data.memory_used.toFixed(1)} GB` : '-');

        // Update labels
        this.setElementText('tempLabel', data.temperature ? `${data.temperature}°C / 85°C` : '-');
        this.setElementText('utilLabel', data.utilization ? `${data.utilization}%` : '-');
        this.setElementText('memLabel',
            data.memory_used && data.memory_total ?
            `${data.memory_used.toFixed(1)} GB / ${data.memory_total.toFixed(1)} GB` : '-');
        this.setElementText('powerLabel',
            data.power && data.power_limit ?
            `${Math.round(data.power)} W / ${Math.round(data.power_limit)} W` : '-');

        // Update gauges
        this.updateGauge('tempGauge', data.temperature, 85);
        this.updateGauge('utilGauge', data.utilization, 100);
        this.updateGauge('memGauge',
            data.memory_used && data.memory_total ? (data.memory_used / data.memory_total * 100) : 0, 100);
        this.updateGauge('powerGauge',
            data.power && data.power_limit ? (data.power / data.power_limit * 100) : 0, 100);

        // Color code temperature
        this.colorCodeValue('gpuTemp', data.temperature || 0, {
            warning: 70,
            danger: 80
        });
    },

    // ========== MEMORY STATS ==========

    updateMemoryStats(data) {
        if (!data) return;

        this.setElementText('ramUsed', data.used_gb ? `${data.used_gb.toFixed(1)} GB` : '- GB');
        this.setElementText('trainingRAM', data.training_process_gb ? `${data.training_process_gb.toFixed(1)} GB` : '- GB');
        this.setElementText('ramAvailable', data.available_gb ? `${data.available_gb.toFixed(1)} GB` : '- GB');
        this.setElementText('oomRisk', data.oom_risk ? data.oom_risk.toUpperCase() : 'LOW');

        // Labels
        this.setElementText('ramLabel',
            data.used_gb && data.total_gb ?
            `${data.used_gb.toFixed(1)} GB / ${data.total_gb.toFixed(1)} GB` : '- GB / - GB');
        this.setElementText('trainingRAMLabel',
            data.training_process_gb ? `${data.training_process_gb.toFixed(1)} GB` : '- GB');
        this.setElementText('swapLabel',
            data.swap_used_gb && data.swap_total_gb ?
            `${data.swap_used_gb.toFixed(2)} GB / ${data.swap_total_gb.toFixed(1)} GB` : '- GB / - GB');

        // Gauges
        this.updateGauge('ramGauge', data.percent || 0, 100);
        this.updateGauge('trainingRAMGauge',
            data.training_process_gb && data.total_gb ? (data.training_process_gb / data.total_gb * 100) : 0, 100);
        this.updateGauge('swapGauge', data.swap_percent || 0, 100);

        // RAM status
        const ramStatusEl = document.getElementById('ramStatus');
        if (ramStatusEl) {
            if (data.percent > 85) {
                ramStatusEl.textContent = 'CRITICAL';
                ramStatusEl.style.color = '#ff4444';
            } else if (data.percent > 70) {
                ramStatusEl.textContent = 'HIGH';
                ramStatusEl.style.color = '#ffaa00';
            } else {
                ramStatusEl.textContent = 'NORMAL';
                ramStatusEl.style.color = '#00ff88';
            }
        }

        // OOM Risk color
        const oomEl = document.getElementById('oomRisk');
        if (oomEl) {
            const risk = (data.oom_risk || 'low').toLowerCase();
            if (risk === 'high') {
                oomEl.style.color = '#ff4444';
                oomEl.style.fontWeight = 'bold';
            } else if (risk === 'medium') {
                oomEl.style.color = '#ffaa00';
            } else {
                oomEl.style.color = '#00ff88';
            }
        }
    },

    updateGauge(gaugeId, value, max) {
        const gauge = document.getElementById(gaugeId);
        if (!gauge) return;

        const percent = Math.min((value / max) * 100, 100);
        gauge.style.width = percent + '%';

        // Color coding for gauges
        if (percent > 85) {
            gauge.style.background = 'linear-gradient(90deg, #ff4444, #ff8888)';
        } else if (percent > 70) {
            gauge.style.background = 'linear-gradient(90deg, #ffaa00, #ffdd00)';
        } else {
            gauge.style.background = 'linear-gradient(90deg, #00ff88, #00d9ff)';
        }
    },

    // ========== QUEUE ESTIMATOR ==========

    updateQueueEstimator(data) {
        if (!data || !data.queued_files) return;

        const container = document.getElementById('queueEstimator');
        if (!container) return;

        if (data.queued_files.count > 0) {
            container.style.display = 'block';

            this.setElementText('queueCount', data.queued_files.count);
            this.setElementText('queueSize', `${data.queued_files.total_size_mb.toFixed(1)} MB`);
            this.setElementText('queueETA', this.formatTime(data.queued_files.eta_seconds || 0));
            this.setElementText('queueCompleteTime', data.queued_files.completion_time || '---');
        } else {
            container.style.display = 'none';
        }
    },

    // ========== FIXED EVAL (NEW) ==========

    updateFixedEval(data) {
        if (!data) return;

        // Use actual field names from training_status.py
        if (data.fixed_eval_em != null) {
            this.setElementText('fixedEvalEM', this.formatPercent(data.fixed_eval_em, 1));
        }
        if (data.fixed_eval_ce != null) {
            this.setElementText('fixedEvalCE', this.formatNumber(data.fixed_eval_ce, 3));
        }
        if (data.fixed_eval_ece != null) {
            this.setElementText('fixedEvalECE', this.formatNumber(data.fixed_eval_ece, 3));
        }

        // Trend with color coding
        const trendEl = document.getElementById('fixedEvalTrend');
        if (trendEl && data.fixed_eval_trend) {
            const trends = {
                'improving': { text: '↑ Improving', class: 'success' },
                'stable': { text: '→ Stable', class: '' },
                'degrading': { text: '↓ Degrading', class: 'error' }
            };
            const trend = trends[data.fixed_eval_trend] || { text: '-', class: '' };
            trendEl.textContent = trend.text;
            trendEl.className = 'metric-value ' + trend.class;
        }
    },

    // ========== STREAMING METRICS (NEW) ==========

    updateStreamingMetrics(data) {
        if (!data) return;

        // Streaming cross-entropy
        if (data.streaming_ce != null) {
            this.setElementText('streamingCE', this.formatNumber(data.streaming_ce, 3));
        }

        // Token entropy
        if (data.token_entropy != null) {
            this.setElementText('tokenEntropy', this.formatNumber(data.token_entropy, 2));
        }

        // Loss variance
        if (data.loss_variance != null) {
            this.setElementText('lossVariance', this.formatNumber(data.loss_variance, 4));
        }

        // Loss trend
        const lossTrendEl = document.getElementById('lossTrend');
        if (lossTrendEl && data.loss_trend) {
            const trends = {
                'improving': { text: '↑ Improving', class: 'success' },
                'stable': { text: '→ Stable', class: '' },
                'degrading': { text: '↓ Degrading', class: 'error' }
            };
            const trend = trends[data.loss_trend] || { text: '-', class: '' };
            lossTrendEl.textContent = trend.text;
            lossTrendEl.className = 'metric-value ' + trend.class;
        }
    },

    // ========== THROUGHPUT (NEW) ==========

    updateThroughput(data) {
        if (!data) return;

        // Current throughput
        if (data.tokens_per_sec != null) {
            this.setElementText('tokensPerSec', this.formatLargeNumber(Math.round(data.tokens_per_sec)));
        }

        // Average throughput
        if (data.tokens_per_sec_avg != null) {
            this.setElementText('tokensPerSecAvg', this.formatLargeNumber(Math.round(data.tokens_per_sec_avg)));
        }

        // Throughput trend
        const throughputTrendEl = document.getElementById('throughputTrend');
        if (throughputTrendEl && data.throughput_trend) {
            const trends = {
                'improving': { text: '↑ Faster', class: 'success' },
                'stable': { text: '→ Stable', class: '' },
                'degrading': { text: '↓ Slower', class: 'error' }
            };
            const trend = trends[data.throughput_trend] || { text: '-', class: '' };
            throughputTrendEl.textContent = trend.text;
            throughputTrendEl.className = 'metric-value ' + trend.class;
        }
    },

    // ========== LORA MONITORING (NEW) ==========

    updateLoRAStats(data) {
        if (!data || !data.lora_summary) return;

        const summary = data.lora_summary;

        // Update summary stats
        if (summary.total_layers != null) {
            this.setElementText('loraLayers', summary.total_layers);
        }
        if (summary.active_layers != null) {
            this.setElementText('loraActiveLayers', summary.active_layers);
        }

        // Top layers (could render a mini chart or list)
        if (summary.top_layers && summary.top_layers.length > 0) {
            const container = document.getElementById('loraTopLayers');
            if (container) {
                const html = summary.top_layers.slice(0, 5).map((layer, idx) => `
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.9em;">
                        <span style="color: #00d9ff;">${layer.name.split('.').pop()}</span>
                        <span style="color: #00ff88;">${layer.grad_norm.toExponential(2)}</span>
                    </div>
                `).join('');
                container.innerHTML = html || '<div style="color: #888;">No data yet</div>';
            }
        }

        // Dead layers warning
        if (summary.dead_layers && summary.dead_layers.length > 0) {
            const warningEl = document.getElementById('loraDeadLayersWarning');
            if (warningEl) {
                warningEl.textContent = `⚠️ ${summary.dead_layers.length} dead layers detected`;
                warningEl.style.display = 'block';
                warningEl.style.color = '#ffaa00';
            }
        }
    },

    // ========== SMART ALERTS (NEW) ==========

    updateAlerts(data) {
        if (!data || !data.active_alerts) return;

        const container = document.getElementById('alertsContainer');
        if (!container) return;

        const alerts = data.active_alerts;

        if (alerts.length === 0) {
            container.innerHTML = '<div style="color: #00ff88; text-align: center; padding: 20px;">✓ No alerts - training looks good!</div>';
            return;
        }

        // Render alerts
        const html = alerts.map(alert => {
            const severityColors = {
                'critical': '#ff4444',
                'warning': '#ffaa00',
                'info': '#00d9ff'
            };
            const color = severityColors[alert.severity] || '#888';
            const icon = alert.severity === 'critical' ? '🚨' : alert.severity === 'warning' ? '⚠️' : 'ℹ️';

            return `
                <div style="background: rgba(42, 63, 95, 0.3); padding: 12px; margin-bottom: 10px; border-left: 4px solid ${color}; border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: bold; color: ${color};">${icon} ${alert.severity.toUpperCase()}</span>
                        <span style="color: #888; font-size: 0.85em;">${alert.category}</span>
                    </div>
                    <div style="margin-bottom: 8px;">${alert.message}</div>
                    ${alert.recommendation ? `<div style="color: #00d9ff; font-size: 0.9em; font-style: italic;">💡 ${alert.recommendation}</div>` : ''}
                </div>
            `;
        }).join('');

        container.innerHTML = html;

        // Update alert summary badge
        if (data.alert_summary) {
            const summaryEl = document.getElementById('alertSummary');
            if (summaryEl) {
                const { critical, warning, info } = data.alert_summary;
                const total = critical + warning + info;
                if (total > 0) {
                    summaryEl.textContent = `${total} active`;
                    summaryEl.style.display = 'inline-block';
                    if (critical > 0) {
                        summaryEl.style.background = '#ff4444';
                    } else if (warning > 0) {
                        summaryEl.style.background = '#ffaa00';
                    } else {
                        summaryEl.style.background = '#00d9ff';
                    }
                } else {
                    summaryEl.style.display = 'none';
                }
            }
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MetricsUpdater;
}
