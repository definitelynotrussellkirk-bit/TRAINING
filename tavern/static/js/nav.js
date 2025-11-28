/**
 * Shared Navigation Component
 * Include this in all pages for consistent navigation
 */

const NAV_ITEMS = [
    { href: '/', icon: '⚔️', label: 'Battle', id: 'battle' },
    { href: '/guild', icon: '🏰', label: 'Guild', id: 'guild' },
    { href: '/quests', icon: '📜', label: 'Quests', id: 'quests' },
    { href: '/vault', icon: '🗃️', label: 'Vault', id: 'vault' },
    { href: '/oracle', icon: '🔮', label: 'Oracle', id: 'oracle' },
    { href: '/settings', icon: '⚙️', label: 'Settings', id: 'settings' },
];

// Weaver status (shown as indicator, not a nav destination)
let weaverStatus = { running: false, healthy: 0, total: 0 };

/**
 * Render the bottom navigation bar
 * @param {string} activeId - Which nav item is active (e.g., 'battle', 'vault')
 * @param {string} containerId - ID of container element (default: 'bottomNav')
 */
function renderBottomNav(activeId, containerId = 'bottomNav') {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Build nav items HTML
    const navHtml = NAV_ITEMS.map(item => {
        const isActive = item.id === activeId;
        return `
            <button class="nav-btn${isActive ? ' active' : ''}"
                    onclick="${isActive ? '' : `window.location.href='${item.href}'`}"
                    ${isActive ? 'disabled' : ''}>
                <span class="nav-icon">${item.icon}</span>
                <span class="nav-label">${item.label}</span>
            </button>
        `;
    }).join('');

    // Add Weaver status indicator
    const weaverHtml = `
        <div class="weaver-indicator" id="weaverIndicator" title="Weaver Status">
            <span class="weaver-icon">🕸️</span>
            <span class="weaver-status" id="weaverStatusDot">●</span>
        </div>
    `;

    container.innerHTML = navHtml + weaverHtml;

    // Fetch and update weaver status
    updateWeaverStatus();
}

/**
 * Fetch and update the Weaver status indicator
 */
async function updateWeaverStatus() {
    try {
        const resp = await fetch('/api/weaver/status');
        const data = await resp.json();

        weaverStatus = {
            running: data.weaver_running,
            healthy: data.healthy_count || 0,
            total: data.total_threads || 0
        };

        const dot = document.getElementById('weaverStatusDot');
        const indicator = document.getElementById('weaverIndicator');

        if (dot && indicator) {
            if (data.weaver_running) {
                if (data.healthy_count === data.total_threads) {
                    dot.style.color = '#10B981'; // Green - all healthy
                    indicator.title = `Weaver: ${data.healthy_count}/${data.total_threads} threads healthy`;
                } else {
                    dot.style.color = '#F59E0B'; // Yellow - some unhealthy
                    indicator.title = `Weaver: ${data.healthy_count}/${data.total_threads} threads healthy`;
                }
            } else {
                dot.style.color = '#EF4444'; // Red - not running
                indicator.title = 'Weaver: Not running';
            }
        }
    } catch (e) {
        const dot = document.getElementById('weaverStatusDot');
        if (dot) {
            dot.style.color = '#6B7280'; // Gray - unknown
        }
    }
}

// Update weaver status periodically
setInterval(updateWeaverStatus, 30000);

/**
 * Auto-detect current page and render nav
 */
function autoRenderNav() {
    const path = window.location.pathname;
    let activeId = 'battle';

    if (path === '/' || path === '/game' || path === '/game.html') {
        activeId = 'battle';
    } else if (path.startsWith('/guild') || path.startsWith('/skill')) {
        activeId = 'guild';
    } else if (path.startsWith('/quests')) {
        activeId = 'quests';
    } else if (path.startsWith('/vault') || path.startsWith('/checkpoint')) {
        activeId = 'vault';
    } else if (path.startsWith('/oracle')) {
        activeId = 'oracle';
    } else if (path.startsWith('/settings') || path.startsWith('/scheduler')) {
        activeId = 'settings';
    }

    renderBottomNav(activeId);
}

// Auto-render on DOMContentLoaded if bottomNav container exists
document.addEventListener('DOMContentLoaded', autoRenderNav);
