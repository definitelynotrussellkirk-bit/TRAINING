// ===== DATA BROWSER =====
// Allows browsing training data, seeing system prompts and golden answers

class DataBrowser {
    constructor() {
        this.examples = [];
        this.filteredExamples = [];
        this.currentFilter = 'all';
        this.searchTerm = '';
    }

    async loadExamples() {
        // For now, load from current training status
        // In future, this could fetch actual training files
        try {
            const response = await fetch('/status/training_status.json');
            const data = await response.json();

            // Create examples from recent data
            this.examples = (data.recent_examples || []).map(ex => ({
                step: ex.step,
                prompt: ex.prompt,
                golden: ex.golden,
                modelOutput: ex.model_output,
                matches: ex.matches,
                loss: ex.loss
            }));

            this.applyFilters();
            this.renderExamples();

        } catch (error) {
            console.error('Failed to load examples:', error);
        }
    }

    applyFilters() {
        this.filteredExamples = this.examples.filter(ex => {
            // Search filter
            if (this.searchTerm) {
                const searchLower = this.searchTerm.toLowerCase();
                const inPrompt = ex.prompt?.toLowerCase().includes(searchLower);
                const inGolden = ex.golden?.toLowerCase().includes(searchLower);
                if (!inPrompt && !inGolden) return false;
            }

            // Pattern filter (placeholder - would need actual metadata)
            if (this.currentFilter !== 'all') {
                // Future: filter by difficulty, pattern type, etc.
            }

            return true;
        });
    }

    renderExamples() {
        const container = document.getElementById('examplesList');

        if (this.filteredExamples.length === 0) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
                    No examples found. Try adjusting filters or wait for training to progress.
                </div>
            `;
            return;
        }

        container.innerHTML = this.filteredExamples.map(ex => `
            <div class="example-card" onclick="dataBrowser.showExampleDetail(${ex.step})">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <strong style="color: var(--accent-blue);">Step ${ex.step}</strong>
                    <span style="color: ${ex.matches ? 'var(--accent-green)' : 'var(--accent-red)'};">
                        ${ex.matches ? '✅ Match' : '❌ No Match'}
                    </span>
                </div>
                <div style="font-size: 0.9em; color: var(--text-secondary); margin-bottom: 8px;">
                    <strong>Prompt:</strong> ${this.truncate(ex.prompt, 100)}
                </div>
                <div style="font-size: 0.9em; color: var(--text-secondary);">
                    <strong>Golden:</strong> ${this.truncate(ex.golden, 100)}
                </div>
                <div style="margin-top: 8px; font-size: 0.85em; color: var(--accent-blue);">
                    Loss: ${this.formatLoss(ex.loss)}
                </div>
            </div>
        `).join('');
    }

    showExampleDetail(step) {
        const example = this.examples.find(ex => ex.step === step);
        if (!example) return;

        // Show in expanded modal
        const modal = document.getElementById('expandModal');
        modal.classList.add('active');

        const content = document.getElementById('expandedContent');
        content.innerHTML = `
            <div style="margin-bottom: 25px;">
                <h3 style="color: var(--accent-blue); margin-bottom: 10px;">
                    📋 User Prompt
                </h3>
                <div class="content-box">
                    <pre>${this.escapeHtml(example.prompt || 'N/A')}</pre>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;">
                <div>
                    <h3 style="color: var(--accent-green); margin-bottom: 10px;">
                        ✅ Golden Answer
                    </h3>
                    <div class="content-box">
                        <pre>${this.escapeHtml(this.formatJson(example.golden || 'N/A'))}</pre>
                    </div>
                </div>
                <div>
                    <h3 style="color: var(--accent-blue); margin-bottom: 10px;">
                        🤖 Model Output
                    </h3>
                    <div class="content-box">
                        <pre>${this.escapeHtml(this.formatJson(example.modelOutput || 'N/A'))}</pre>
                    </div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div class="metric-card">
                    <span class="metric-label">Match</span>
                    <span class="metric-value" style="color: ${example.matches ? 'var(--accent-green)' : 'var(--accent-red)'};">
                        ${example.matches ? '✅ Yes' : '❌ No'}
                    </span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Loss</span>
                    <span class="metric-value">${this.formatLoss(example.loss)}</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Step</span>
                    <span class="metric-value">${example.step}</span>
                </div>
            </div>
        `;

        document.getElementById('modalStep').textContent = step;
    }

    truncate(text, maxLength) {
        if (!text) return 'N/A';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    formatJson(text) {
        try {
            const parsed = JSON.parse(text);
            return JSON.stringify(parsed, null, 2);
        } catch {
            return text;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatLoss(value) {
        if (value === null || value === undefined) return 'N/A';
        const num = Number(value);
        if (!Number.isFinite(num)) return 'N/A';
        return num.toString();
    }
}

// Make dataBrowser globally accessible
let dataBrowser;
