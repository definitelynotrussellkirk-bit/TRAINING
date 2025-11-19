// ===== INFERENCE HISTORY VIEWER =====
// Fetches and displays recent entries from the inference log

class HistoryViewer {
    constructor() {
        this.container = null;
    }

    async open() {
        const modal = document.getElementById('expandModal');
        modal.classList.add('active');
        document.getElementById('modalStep').textContent = 'History (last 10)';
        const content = document.getElementById('expandedContent');
        content.innerHTML = '<div style="padding: 10px; color: var(--text-secondary);">Loading history...</div>';

        try {
            const entries = await this.fetchRecentHistory(10);
            if (!entries.length) {
                content.innerHTML = '<div style="padding: 10px;">No history entries found.</div>';
                return;
            }

            content.innerHTML = entries.map(e => this.renderEntry(e)).join('');
        } catch (err) {
            console.error('History load error:', err);
            content.innerHTML = '<div style="padding: 10px; color: var(--accent-red);">Failed to load history.</div>';
        }
    }

    async fetchRecentHistory(limit = 10) {
        // Try to fetch today’s log; if not available, return empty
        const today = new Date();
        const ymd = today.toISOString().slice(0, 10).replace(/-/g, '');
        const url = `/status/logs/inference_${ymd}.log`;
        const resp = await fetch(url);
        if (!resp.ok) return [];
        const text = await resp.text();
        const lines = text.trim().split('\n').filter(Boolean);
        const lastLines = lines.slice(-limit);
        return lastLines.map(line => {
            try {
                return JSON.parse(line);
            } catch {
                return null;
            }
        }).filter(Boolean).reverse(); // newest first
    }

    renderEntry(e) {
        const prompt = this.escape(e.prompt || 'N/A');
        const golden = this.formatJson(e.golden || 'N/A');
        const model = this.formatJson(e.model_output || 'N/A');
        const sys = this.escape(e.system_prompt || 'N/A');
        const match = e.matches ? '✅' : (e.matches === false ? '❌' : '--');
        return `
            <div class="example-card" style="margin-bottom: 20px;">
                <div style="display:flex; justify-content: space-between; margin-bottom: 6px;">
                    <strong style="color: var(--accent-blue);">Step ${e.step ?? '--'}</strong>
                    <span>${match}</span>
                </div>
                <div style="margin-bottom:8px; font-size:0.9em; color: var(--text-secondary);">
                    <strong>System:</strong> ${sys}
                </div>
                <div style="margin-bottom:8px; font-size:0.9em; color: var(--text-secondary);">
                    <strong>Prompt:</strong> ${prompt}
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                    <div>
                        <div style="color: var(--accent-green); font-weight: 600; margin-bottom:4px;">Golden</div>
                        <pre class="content-box" style="white-space: pre-wrap;">${golden}</pre>
                    </div>
                    <div>
                        <div style="color: var(--accent-blue); font-weight: 600; margin-bottom:4px;">Model</div>
                        <pre class="content-box" style="white-space: pre-wrap;">${model}</pre>
                    </div>
                </div>
            </div>
        `;
    }

    formatJson(text) {
        try {
            return this.escape(JSON.stringify(JSON.parse(text), null, 2));
        } catch {
            return this.escape(text);
        }
    }

    escape(str) {
        return (str || '').replace(/[&<>"']/g, m => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[m]));
    }
}

// Global hook
const historyViewer = new HistoryViewer();
function openHistoryLog() {
    historyViewer.open();
}
