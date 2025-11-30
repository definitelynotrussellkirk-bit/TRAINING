/**
 * Evals Page - Checkpoint Evaluation History
 *
 * Displays eval results from the evaluation ledger with:
 * - Summary stats (total evals, per-skill breakdown)
 * - Filterable table of eval results
 * - Delete functionality for individual evals
 */

// State
let allEvals = [];
let summary = null;

// Configuration
const VAULT_API = '/api/evals';  // Tavern proxies to VaultKeeper

/**
 * Fetch JSON from API
 */
async function fetchJson(url) {
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`HTTP ${res.status} for ${url}`);
    }
    return await res.json();
}

/**
 * Load summary statistics
 */
async function loadSummary() {
    try {
        const data = await fetchJson(`${VAULT_API}/summary`);
        summary = data;

        // Update total evals
        document.getElementById('total-evals').textContent = data.total_evaluations || 0;

        // Update skills count
        const skillCount = data.by_skill ? Object.keys(data.by_skill).length : 0;
        document.getElementById('skills-count').textContent = skillCount;

        // Update latest eval info
        if (data.latest) {
            const acc = (data.latest.accuracy * 100).toFixed(1);
            document.getElementById('latest-accuracy').textContent = `${acc}%`;
            document.getElementById('latest-info').textContent =
                `${data.latest.skill} L${data.latest.level}`;
        }

        // Render per-skill summaries
        renderSkillSummaries(data.by_skill || {});

    } catch (err) {
        console.error('Failed to load eval summary:', err);
    }
}

/**
 * Render per-skill summary cards
 */
function renderSkillSummaries(bySkill) {
    const container = document.getElementById('skill-summaries');
    container.innerHTML = '';

    const skillIcons = {
        'bin': { icon: '0101', color: '#818cf8' },
        'sy': { icon: 'ABC', color: '#34d399' },
    };

    for (const [skillId, stats] of Object.entries(bySkill)) {
        const info = skillIcons[skillId] || { icon: skillId.toUpperCase(), color: '#888' };
        const bestAcc = ((stats.best_accuracy || 0) * 100).toFixed(1);
        const levels = Array.isArray(stats.levels_evaluated)
            ? stats.levels_evaluated.join(', ')
            : '-';

        const card = document.createElement('div');
        card.className = 'skill-summary-card';
        card.innerHTML = `
            <div class="skill-header">
                <span class="skill-icon" style="color: ${info.color}">${info.icon}</span>
                <span class="skill-name">${skillId.toUpperCase()}</span>
            </div>
            <div class="stats">
                <div class="stat">
                    <span class="label">Evals:</span>
                    <span class="value">${stats.count || 0}</span>
                </div>
                <div class="stat">
                    <span class="label">Best:</span>
                    <span class="value">${bestAcc}%</span>
                </div>
                <div class="stat">
                    <span class="label">Best Ckpt:</span>
                    <span class="value">${stats.best_checkpoint || '-'}</span>
                </div>
                <div class="stat">
                    <span class="label">Levels:</span>
                    <span class="value">${levels}</span>
                </div>
            </div>
        `;
        container.appendChild(card);
    }
}

/**
 * Load evaluations list
 */
async function loadEvals() {
    const skill = document.getElementById('filter-skill').value;
    const checkpoint = document.getElementById('filter-checkpoint').value;
    const level = document.getElementById('filter-level').value;

    let url = `${VAULT_API}?limit=200`;

    // Apply filters
    if (checkpoint) {
        url = `${VAULT_API}/checkpoint/${checkpoint}`;
    } else if (skill) {
        url = `${VAULT_API}/skill/${skill}`;
        if (level) {
            url += `?level=${level}`;
        }
    }

    try {
        const data = await fetchJson(url);
        allEvals = data.evaluations || [];
        renderEvals(allEvals);
    } catch (err) {
        console.error('Failed to load evals:', err);
        renderEmpty('Failed to load evaluations');
    }
}

/**
 * Render evaluations table
 */
function renderEvals(evals) {
    const tbody = document.getElementById('eval-rows');
    tbody.innerHTML = '';

    if (!evals || evals.length === 0) {
        renderEmpty('No evaluations found');
        return;
    }

    // Sort by timestamp descending (newest first)
    const sorted = [...evals].sort((a, b) => {
        const ta = a.timestamp || '';
        const tb = b.timestamp || '';
        return tb.localeCompare(ta);
    });

    for (const eval_ of sorted) {
        const row = document.createElement('tr');

        // Parse timestamp
        const ts = eval_.timestamp || '';
        const timeStr = ts ? formatTimestamp(ts) : '-';

        // Checkpoint
        const ckpt = eval_.checkpoint_step || '-';

        // Skill badge
        const skill = eval_.skill || '-';
        const skillClass = skill.toLowerCase();

        // Level
        const level = eval_.level || '-';

        // Accuracy with color coding
        const acc = eval_.accuracy;
        const accPct = acc != null ? (acc * 100).toFixed(1) : '-';
        const accClass = acc >= 0.8 ? 'high' : acc >= 0.6 ? 'medium' : 'low';

        // Correct/Total
        const correct = eval_.correct || 0;
        const total = eval_.total || 0;

        // Type
        const evalType = eval_.eval_type || eval_.validation_type || '-';

        row.innerHTML = `
            <td class="timestamp">${timeStr}</td>
            <td>
                <a href="/checkpoint/${ckpt}" class="checkpoint-link">${ckpt}</a>
            </td>
            <td>
                <span class="skill-badge ${skillClass}">${skill.toUpperCase()}</span>
            </td>
            <td>${level}</td>
            <td class="accuracy ${accClass}">${accPct}%</td>
            <td>${correct}/${total}</td>
            <td><span class="type-badge">${evalType}</span></td>
            <td>
                <button class="btn btn-danger btn-small"
                        onclick="deleteEval(${ckpt}, '${skill}', ${level})"
                        title="Delete this eval">
                    Delete
                </button>
            </td>
        `;

        tbody.appendChild(row);
    }
}

/**
 * Render empty state
 */
function renderEmpty(message) {
    const tbody = document.getElementById('eval-rows');
    tbody.innerHTML = `
        <tr>
            <td colspan="8">
                <div class="empty-state">
                    <div class="icon">📊</div>
                    <div>${message}</div>
                </div>
            </td>
        </tr>
    `;
}

/**
 * Format timestamp for display
 */
function formatTimestamp(ts) {
    try {
        const date = new Date(ts);
        const now = new Date();
        const diff = now - date;

        // If less than 24 hours, show relative time
        if (diff < 86400000) {
            const hours = Math.floor(diff / 3600000);
            if (hours < 1) {
                const mins = Math.floor(diff / 60000);
                return `${mins}m ago`;
            }
            return `${hours}h ago`;
        }

        // Otherwise show date
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (e) {
        return ts;
    }
}

/**
 * Delete eval for a checkpoint/skill/level
 */
async function deleteEval(checkpoint, skill, level) {
    if (!confirm(`Delete eval for checkpoint ${checkpoint}, ${skill} L${level}?`)) {
        return;
    }

    try {
        // Call VaultKeeper DELETE endpoint via fetch
        const url = `http://localhost:8767/api/evals/checkpoint/${checkpoint}?skill=${skill}&level=${level}`;
        const res = await fetch(url, { method: 'DELETE' });
        const data = await res.json();

        if (data.ok) {
            console.log(`Deleted ${data.removed} eval(s)`);
            // Reload data
            await loadSummary();
            await loadEvals();
        } else {
            alert('Failed to delete: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('Delete failed:', err);
        alert('Failed to delete eval: ' + err.message);
    }
}

/**
 * Delete all evals for a checkpoint
 */
async function deleteAllEvalsForCheckpoint(checkpoint) {
    if (!confirm(`Delete ALL evals for checkpoint ${checkpoint}?`)) {
        return;
    }

    try {
        const url = `http://localhost:8767/api/evals/checkpoint/${checkpoint}`;
        const res = await fetch(url, { method: 'DELETE' });
        const data = await res.json();

        if (data.ok) {
            console.log(`Deleted ${data.removed} eval(s)`);
            await loadSummary();
            await loadEvals();
        } else {
            alert('Failed to delete: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('Delete failed:', err);
        alert('Failed to delete evals: ' + err.message);
    }
}

/**
 * Clear all filters
 */
function clearFilters() {
    document.getElementById('filter-skill').value = '';
    document.getElementById('filter-checkpoint').value = '';
    document.getElementById('filter-level').value = '';
    loadEvals();
}

/**
 * Initialize page
 */
document.addEventListener('DOMContentLoaded', () => {
    // Load initial data
    loadSummary();
    loadEvals();

    // Setup filter handlers
    document.getElementById('apply-filters').addEventListener('click', loadEvals);
    document.getElementById('clear-filters').addEventListener('click', clearFilters);

    // Enter key in filter inputs
    ['filter-skill', 'filter-checkpoint', 'filter-level'].forEach(id => {
        document.getElementById(id).addEventListener('keypress', (e) => {
            if (e.key === 'Enter') loadEvals();
        });
    });

    // Auto-refresh every 30 seconds
    setInterval(() => {
        loadSummary();
        loadEvals();
    }, 30000);
});
