// Systems Dashboard JavaScript - Real-time monitoring
// Updates every 30 seconds

const REFRESH_INTERVAL = 30000; // 30 seconds
let refreshTimer = null;

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Systems Dashboard initializing...');
    fetchSystemsStatus();
    startAutoRefresh();
});

// Start automatic refresh
function startAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
    }
    refreshTimer = setInterval(fetchSystemsStatus, REFRESH_INTERVAL);
    console.log(`Auto-refresh enabled (${REFRESH_INTERVAL / 1000}s interval)`);
}

// Fetch systems status from API
async function fetchSystemsStatus() {
    try {
        const response = await fetch('/api/systems_status');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        console.log('Systems status updated:', data);
        updateDashboard(data);
    } catch (error) {
        console.error('Error fetching systems status:', error);
        showError('Unable to fetch systems status. Check if server is running.');
    }
}

// Update all dashboard elements
function updateDashboard(data) {
    updateHeader(data);
    updateCurriculumHero(data);
    updateMonitoringSystems(data);
    updateSkills(data);
    updateQueue(data);
}

// Update header stats
function updateHeader(data) {
    // Count running systems
    const systems = data.monitoring_systems || {};
    const runningCount = Object.values(systems).filter(s => s.status === 'running').length;
    document.getElementById('systemsRunning').textContent = `${runningCount}/7`;

    // Last update time
    if (data.last_updated) {
        const timestamp = new Date(data.last_updated);
        const timeStr = timestamp.toLocaleTimeString();
        document.getElementById('lastUpdate').textContent = timeStr;
    }

    // Queue total
    const queue = data.queue || {};
    const queueTotal = Object.values(queue).reduce((sum, count) => sum + count, 0);
    document.getElementById('queueTotal').textContent = queueTotal;
}

// Update curriculum hero section
function updateCurriculumHero(data) {
    const curriculum = data.monitoring_systems?.curriculum_optimizer || {};

    if (curriculum.metrics) {
        const metrics = curriculum.metrics;

        // Update accuracy values and progress bars
        updateAccuracy('Easy', metrics.easy_accuracy || 0);
        updateAccuracy('Medium', metrics.medium_accuracy || 0);
        updateAccuracy('Hard', metrics.hard_accuracy || 0);

        // Update checkpoint
        document.getElementById('curriculumCheckpoint').textContent =
            curriculum.last_checkpoint || '--';

        // Update status
        const statusBadge = document.getElementById('curriculumStatus');
        statusBadge.textContent = curriculum.status || 'unknown';
        statusBadge.className = `status-badge ${curriculum.status || 'unknown'}`;
    } else {
        // No data available
        document.getElementById('curriculumEasy').textContent = '--';
        document.getElementById('curriculumMedium').textContent = '--';
        document.getElementById('curriculumHard').textContent = '--';
        document.getElementById('curriculumCheckpoint').textContent = '--';

        const statusBadge = document.getElementById('curriculumStatus');
        statusBadge.textContent = 'no data';
        statusBadge.className = 'status-badge unknown';
    }
}

// Update individual accuracy metric
function updateAccuracy(difficulty, accuracy) {
    const percent = Math.round(accuracy * 100);
    const id = `curriculum${difficulty}`;
    const barId = `curriculum${difficulty}Bar`;

    document.getElementById(id).textContent = `${percent}%`;

    const bar = document.getElementById(barId);
    if (bar) {
        bar.style.width = `${percent}%`;

        // Color based on performance
        bar.classList.remove('high', 'medium', 'low');
        if (percent >= 70) {
            bar.classList.add('high');
        } else if (percent >= 40) {
            bar.classList.add('medium');
        } else {
            bar.classList.add('low');
        }
    }
}

// Update monitoring systems cards
function updateMonitoringSystems(data) {
    const systems = data.monitoring_systems || {};

    // 1. Checkpoint Sync
    updateSystemCard('checkpointSync', systems.checkpoint_sync, {
        lastStep: s => s.last_synced || '--',
        checkpoints: s => `${s.local_checkpoints || 0} / ${s.remote_checkpoints || 0}`,
        timestamp: s => formatTimestamp(s.timestamp)
    });

    // 2. Adversarial Miner
    updateSystemCard('adversarialMiner', systems.adversarial_miner, {
        found: s => s.examples_found || 0,
        tested: s => s.examples_tested || 0,
        timestamp: s => formatTimestamp(s.last_updated)
    });

    // 3. Regression Monitor
    updateSystemCard('regressionMonitor', systems.regression_monitor, {
        detected: s => s.regressions_detected || 0,
        tested: s => s.checkpoints_tested || 0,
        timestamp: s => formatTimestamp(s.last_updated)
    });

    // 4. Model Comparison
    updateSystemCard('modelComparison', systems.model_comparison, {
        status: s => s.status || 'unknown',
        data: s => s.has_data ? 'Yes' : 'No'
    });

    // 5. Confidence Calibrator
    updateSystemCard('confidenceCalibrator', systems.confidence_calibrator, {
        bins: s => s.bins || '--',
        timestamp: s => formatTimestamp(s.last_updated)
    });

    // 6. Automated Testing
    updateSystemCard('automatedTesting', systems.automated_testing, {
        status: s => s.status || 'unknown',
        data: s => s.has_data ? 'Yes' : 'No'
    });
}

// Update individual system card
function updateSystemCard(cardId, systemData, fields) {
    const card = document.getElementById(cardId);
    if (!card) return;

    // Update status indicator
    const indicator = card.querySelector('.status-indicator');
    if (indicator && systemData) {
        indicator.className = 'status-indicator';
        indicator.classList.add(systemData.status || 'unknown');
    }

    // Update specific fields
    if (systemData && !systemData.error) {
        // Checkpoint Sync
        if (fields.lastStep) {
            const el = document.getElementById('syncLastStep');
            if (el) el.textContent = fields.lastStep(systemData);
        }
        if (fields.checkpoints) {
            const el = document.getElementById('syncCheckpoints');
            if (el) el.textContent = fields.checkpoints(systemData);
        }
        if (fields.timestamp && cardId === 'checkpointSync') {
            const el = document.getElementById('syncTimestamp');
            if (el) el.textContent = fields.timestamp(systemData);
        }

        // Adversarial Miner
        if (fields.found) {
            const el = document.getElementById('adversarialFound');
            if (el) el.textContent = fields.found(systemData);
        }
        if (fields.tested) {
            const el = document.getElementById('adversarialTested');
            if (el) el.textContent = fields.tested(systemData);
        }
        if (fields.timestamp && cardId === 'adversarialMiner') {
            const el = document.getElementById('adversarialTimestamp');
            if (el) el.textContent = fields.timestamp(systemData);
        }

        // Regression Monitor
        if (fields.detected) {
            const el = document.getElementById('regressionsDetected');
            if (el) el.textContent = fields.detected(systemData);
        }
        if (fields.tested && cardId === 'regressionMonitor') {
            const el = document.getElementById('regressionsTested');
            if (el) el.textContent = fields.tested(systemData);
        }
        if (fields.timestamp && cardId === 'regressionMonitor') {
            const el = document.getElementById('regressionTimestamp');
            if (el) el.textContent = fields.timestamp(systemData);
        }

        // Model Comparison
        if (fields.status && cardId === 'modelComparison') {
            const el = document.getElementById('comparisonStatus');
            if (el) el.textContent = fields.status(systemData);
        }
        if (fields.data && cardId === 'modelComparison') {
            const el = document.getElementById('comparisonData');
            if (el) el.textContent = fields.data(systemData);
        }

        // Confidence Calibrator
        if (fields.bins) {
            const el = document.getElementById('confidenceBins');
            if (el) el.textContent = fields.bins(systemData);
        }
        if (fields.timestamp && cardId === 'confidenceCalibrator') {
            const el = document.getElementById('confidenceTimestamp');
            if (el) el.textContent = fields.timestamp(systemData);
        }

        // Automated Testing
        if (fields.status && cardId === 'automatedTesting') {
            const el = document.getElementById('testingStatus');
            if (el) el.textContent = fields.status(systemData);
        }
        if (fields.data && cardId === 'automatedTesting') {
            const el = document.getElementById('testingData');
            if (el) el.textContent = fields.data(systemData);
        }
    }
}

// Update skills section
function updateSkills(data) {
    const skills = data.skills || {};

    // For now, skills just show static info
    // In future, could check API health
    console.log('Skills:', skills);
}

// Update queue pipeline
function updateQueue(data) {
    const queue = data.queue || {};

    document.getElementById('queueHigh').textContent = queue.high || 0;
    document.getElementById('queueNormal').textContent = queue.normal || 0;
    document.getElementById('queueLow').textContent = queue.low || 0;
    document.getElementById('queueProcessing').textContent = queue.processing || 0;
    document.getElementById('queueFailed').textContent = queue.failed || 0;

    // Highlight if files in failed
    const failedCard = document.querySelector('.queue-card.failed');
    if (failedCard && queue.failed > 0) {
        failedCard.classList.add('has-items');
    } else if (failedCard) {
        failedCard.classList.remove('has-items');
    }
}

// Format timestamp for display
function formatTimestamp(timestamp) {
    if (!timestamp) return '--';

    try {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);

        if (diffMins < 1) {
            return 'Just now';
        } else if (diffMins < 60) {
            return `${diffMins}m ago`;
        } else {
            const diffHours = Math.floor(diffMins / 60);
            if (diffHours < 24) {
                return `${diffHours}h ago`;
            } else {
                return date.toLocaleDateString();
            }
        }
    } catch (e) {
        return '--';
    }
}

// Show error message
function showError(message) {
    console.error(message);
    // Could add visual error indicator
}

// Manual refresh button (if added to UI)
function manualRefresh() {
    console.log('Manual refresh triggered');
    fetchSystemsStatus();
}
