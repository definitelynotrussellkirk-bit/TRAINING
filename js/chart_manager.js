// ===== CHART MANAGER =====
// Manages Chart.js instances for loss and think tag charts

class ChartManager {
    constructor() {
        this.lossChart = null;
        this.thinkTagChart = null;
        this.maxDataPoints = 50;
        this.lossHistory = {
            steps: [],
            trainLoss: [],
            valLoss: []
        };
        this.thinkHistory = {
            steps: [],
            percentage: []
        };
    }

    async init() {
        this.createLossChart();
        this.createThinkTagChart();
    }

    createLossChart() {
        const ctx = document.getElementById('lossChart');

        this.lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#00d9ff',
                        backgroundColor: 'rgba(0, 217, 255, 0.1)',
                        tension: 0.4,
                        pointRadius: 2
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        tension: 0.4,
                        pointRadius: 2,
                        borderDash: [5, 5]
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#888' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#888' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        beginAtZero: false
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    createThinkTagChart() {
        const ctx = document.getElementById('thinkTagChart');

        this.thinkTagChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Think Tag %',
                    data: [],
                    borderColor: '#ffaa00',
                    backgroundColor: 'rgba(255, 170, 0, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.parsed.y.toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#888' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        ticks: {
                            color: '#888',
                            callback: (value) => value + '%'
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    updateCharts(data) {
        if (!data || !data.current_step) return;

        // Update loss chart
        if (data.loss !== null && data.loss !== undefined) {
            this.updateLossChart(data);
        }

        // Update think tag chart
        if (data.think_tag_percent !== null && data.think_tag_percent !== undefined) {
            this.updateThinkTagChart(data);
        }
    }

    updateLossChart(data) {
        const step = data.current_step;
        const trainLoss = data.loss;
        const valLoss = data.validation_loss;

        // Add new data point
        this.lossHistory.steps.push(step);
        this.lossHistory.trainLoss.push(trainLoss);
        this.lossHistory.valLoss.push(valLoss);

        // Keep only last N points
        if (this.lossHistory.steps.length > this.maxDataPoints) {
            this.lossHistory.steps.shift();
            this.lossHistory.trainLoss.shift();
            this.lossHistory.valLoss.shift();
        }

        // Update chart
        this.lossChart.data.labels = this.lossHistory.steps;
        this.lossChart.data.datasets[0].data = this.lossHistory.trainLoss;
        this.lossChart.data.datasets[1].data = this.lossHistory.valLoss;
        this.lossChart.update('none'); // No animation for smoother updates
    }

    updateThinkTagChart(data) {
        const step = data.current_step;
        const thinkPct = data.think_tag_percent;

        // Add new data point
        this.thinkHistory.steps.push(step);
        this.thinkHistory.percentage.push(thinkPct);

        // Keep only last N points
        if (this.thinkHistory.steps.length > this.maxDataPoints) {
            this.thinkHistory.steps.shift();
            this.thinkHistory.percentage.shift();
        }

        // Update chart
        this.thinkTagChart.data.labels = this.thinkHistory.steps;
        this.thinkTagChart.data.datasets[0].data = this.thinkHistory.percentage;
        this.thinkTagChart.update('none');
    }
}
