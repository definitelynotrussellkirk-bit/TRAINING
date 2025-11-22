// Evolution Viewer JavaScript
let currentDataset = null;
let allSnapshots = [];
let currentSnapshot = null;
let learningChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadDatasets();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    document.getElementById('sortSelect').addEventListener('change', renderExamples);
    document.getElementById('filterSelect').addEventListener('change', renderExamples);
}

// Load available datasets
async function loadDatasets() {
    try {
        const response = await fetch('/api/evolution/datasets');
        const data = await response.json();

        const container = document.getElementById('datasetsContainer');

        if (!data.datasets || data.datasets.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <div class="empty-state-message">No Evolution Data Found</div>
                    <div class="empty-state-hint">
                        Evolution snapshots will appear here after training runs.<br>
                        They are saved in: data/evolution_snapshots/
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = '<div class="dataset-grid"></div>';
        const grid = container.querySelector('.dataset-grid');

        data.datasets.forEach(dataset => {
            const card = document.createElement('div');
            card.className = 'dataset-card';
            card.innerHTML = `
                <div class="dataset-name">${dataset.name}</div>
                <div class="dataset-stats">
                    <div>📸 ${dataset.snapshot_count} snapshots</div>
                    <div>📊 Steps ${dataset.first_step} → ${dataset.last_step}</div>
                </div>
            `;
            card.addEventListener('click', () => selectDataset(dataset));
            grid.appendChild(card);
        });
    } catch (error) {
        console.error('Error loading datasets:', error);
        document.getElementById('datasetsContainer').innerHTML = `
            <div class="error-message">
                Error loading datasets: ${error.message}<br>
                Make sure the server is running and evolution data exists.
            </div>
        `;
    }
}

// Select a dataset
async function selectDataset(dataset) {
    // Update UI to show selected
    document.querySelectorAll('.dataset-card').forEach(card => {
        card.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');

    currentDataset = dataset;
    await loadSnapshots(dataset.name);
}

// Load all snapshots for a dataset
async function loadSnapshots(datasetName) {
    try {
        const response = await fetch(`/api/evolution/${datasetName}/snapshots`);
        const data = await response.json();

        if (!data.snapshots || data.snapshots.length === 0) {
            return;
        }

        allSnapshots = data.snapshots.sort((a, b) => a.step - b.step);

        // Show sections
        document.getElementById('overviewSection').style.display = 'block';
        document.getElementById('snapshotsSection').style.display = 'block';

        renderOverview();
        renderSnapshotsTimeline();
        renderLearningCurve();

        // Select latest snapshot by default
        if (allSnapshots.length > 0) {
            selectSnapshot(allSnapshots[allSnapshots.length - 1]);
        }
    } catch (error) {
        console.error('Error loading snapshots:', error);
    }
}

// Render overview stats
function renderOverview() {
    const container = document.getElementById('overviewStats');

    if (allSnapshots.length === 0) {
        container.innerHTML = '<div class="empty-state">No snapshot data available</div>';
        return;
    }

    const latest = allSnapshots[allSnapshots.length - 1];
    const first = allSnapshots[0];

    const accuracyImprovement = ((latest.summary.accuracy - first.summary.accuracy) * 100).toFixed(1);
    const lossImprovement = ((first.summary.avg_loss - latest.summary.avg_loss) / first.summary.avg_loss * 100).toFixed(1);

    container.innerHTML = `
        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-value">${allSnapshots.length}</div>
                <div class="stat-label">Total Snapshots</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${(latest.summary.accuracy * 100).toFixed(1)}%</div>
                <div class="stat-label">Current Accuracy</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${latest.summary.accuracy * 100}%"></div>
                </div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${latest.summary.avg_loss.toFixed(3)}</div>
                <div class="stat-label">Current Avg Loss</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${accuracyImprovement > 0 ? '+' : ''}${accuracyImprovement}%</div>
                <div class="stat-label">Accuracy Improvement</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${lossImprovement > 0 ? '+' : ''}${lossImprovement}%</div>
                <div class="stat-label">Loss Improvement</div>
            </div>
        </div>
    `;
}

// Render snapshots timeline
function renderSnapshotsTimeline() {
    const container = document.getElementById('snapshotsTimeline');
    container.innerHTML = '';

    allSnapshots.forEach(snapshot => {
        const marker = document.createElement('div');
        marker.className = 'snapshot-marker';
        marker.innerHTML = `
            <div class="snapshot-step">Step ${snapshot.step}</div>
            <div class="snapshot-acc">${(snapshot.summary.accuracy * 100).toFixed(1)}% acc</div>
        `;
        marker.addEventListener('click', () => selectSnapshot(snapshot));
        container.appendChild(marker);
    });
}

// Render learning curve chart
function renderLearningCurve() {
    const canvas = document.getElementById('learningCurveChart');
    const ctx = canvas.getContext('2d');

    // Destroy existing chart if any
    if (learningChart) {
        learningChart.destroy();
    }

    const steps = allSnapshots.map(s => s.step);
    const accuracies = allSnapshots.map(s => s.summary.accuracy * 100);
    const losses = allSnapshots.map(s => s.summary.avg_loss);

    learningChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracies,
                    borderColor: '#00ffaa',
                    backgroundColor: 'rgba(0, 255, 170, 0.1)',
                    yAxisID: 'y',
                    tension: 0.3,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Average Loss',
                    data: losses,
                    borderColor: '#ff6666',
                    backgroundColor: 'rgba(255, 102, 102, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.3,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Learning Curve: Accuracy & Loss Over Training Steps',
                    color: '#e0e0e0',
                    font: { size: 16 }
                },
                legend: {
                    labels: { color: '#e0e0e0' }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#00ffaa',
                    bodyColor: '#e0e0e0',
                    borderColor: '#00ffaa',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Training Step',
                        color: '#888'
                    },
                    ticks: { color: '#888' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        color: '#00ffaa'
                    },
                    ticks: { color: '#00ffaa' },
                    grid: { color: 'rgba(0, 255, 170, 0.1)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Loss',
                        color: '#ff6666'
                    },
                    ticks: { color: '#ff6666' },
                    grid: { display: false }
                }
            }
        }
    });
}

// Select a specific snapshot
async function selectSnapshot(snapshot) {
    // Update timeline UI
    document.querySelectorAll('.snapshot-marker').forEach(marker => {
        marker.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');

    currentSnapshot = snapshot;
    document.getElementById('examplesSection').style.display = 'block';

    // Load full snapshot data
    try {
        const response = await fetch(`/api/evolution/${currentDataset.name}/snapshot/${snapshot.step}`);
        const fullSnapshot = await response.json();
        currentSnapshot.examples = fullSnapshot.examples;
        renderExamples();
    } catch (error) {
        console.error('Error loading snapshot details:', error);
    }
}

// Render examples grid
function renderExamples() {
    if (!currentSnapshot || !currentSnapshot.examples) {
        return;
    }

    const container = document.getElementById('examplesGrid');
    const sortBy = document.getElementById('sortSelect').value;
    const filterBy = document.getElementById('filterSelect').value;

    // Filter examples
    let examples = [...currentSnapshot.examples];
    if (filterBy === 'correct') {
        examples = examples.filter(ex => ex.exact_match);
    } else if (filterBy === 'incorrect') {
        examples = examples.filter(ex => !ex.exact_match);
    }

    // Sort examples
    if (sortBy === 'loss') {
        examples.sort((a, b) => b.loss - a.loss);
    } else if (sortBy === 'similarity') {
        examples.sort((a, b) => (b.similarity || 0) - (a.similarity || 0));
    }

    if (examples.length === 0) {
        container.innerHTML = '<div class="empty-state">No examples match the current filter</div>';
        return;
    }

    container.innerHTML = examples.slice(0, 20).map((example, idx) => `
        <div class="example-card">
            <div class="example-header">
                <div class="example-id">Example ${example.example_id || idx}</div>
                <div class="example-metrics">
                    <div class="metric">
                        <span class="metric-label">Loss:</span>
                        <span class="metric-value">${example.loss.toFixed(3)}</span>
                    </div>
                    ${example.similarity !== undefined ? `
                        <div class="metric">
                            <span class="metric-label">Similarity:</span>
                            <span class="metric-value">${(example.similarity * 100).toFixed(1)}%</span>
                        </div>
                    ` : ''}
                    <span class="match-indicator ${example.exact_match ? 'match-yes' : 'match-no'}">
                        ${example.exact_match ? '✓ Match' : '✗ No Match'}
                    </span>
                </div>
            </div>
            <div class="example-content">
                <div class="content-label">Input:</div>
                <div class="content-text">${escapeHtml(example.input || 'N/A')}</div>
            </div>
            <div class="example-content">
                <div class="content-label">Expected Output:</div>
                <div class="content-text">${escapeHtml(example.expected_output || 'N/A')}</div>
            </div>
            <div class="example-content">
                <div class="content-label">Model Prediction:</div>
                <div class="content-text">${escapeHtml(example.model_output || 'N/A')}</div>
            </div>
        </div>
    `).join('');

    if (examples.length > 20) {
        container.innerHTML += `
            <div style="text-align: center; padding: 20px; color: #888;">
                Showing first 20 of ${examples.length} examples
            </div>
        `;
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-refresh datasets every 30 seconds
setInterval(loadDatasets, 30000);
