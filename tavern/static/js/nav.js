/**
 * Shared Navigation Component
 * Include this in all pages for consistent navigation
 */

const NAV_ITEMS = [
    { href: '/', icon: '⚔️', label: 'Battle', id: 'battle' },
    { href: '/guild', icon: '🏰', label: 'Guild', id: 'guild' },
    { href: '/campaign', icon: '🗺️', label: 'Campaign', id: 'campaign' },
    { href: '/quests', icon: '📜', label: 'Quests', id: 'quests' },
    { href: '/forge', icon: '🔥', label: 'Forge', id: 'forge' },
    { href: '/vault', icon: '🗝️', label: 'Vault', id: 'vault' },
    { href: '/ledger', icon: '📖', label: 'Ledger', id: 'ledger' },
    { href: '/oracle', icon: '🔮', label: 'Oracle', id: 'oracle' },
    { href: '/arcana', icon: '🌀', label: 'Arcana', id: 'arcana' },
    { href: '/temple', icon: '🏛️', label: 'Temple', id: 'temple' },
    { href: '/garrison', icon: '🛡️', label: 'Garrison', id: 'garrison' },
    { href: '/settings', icon: '⚙️', label: 'Settings', id: 'settings' },
];

// Weaver status (shown as indicator, not a nav destination)
let weaverStatus = { running: false, healthy: 0, total: 0, pid: null };

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

    // Add Weaver status indicator (clickable)
    const weaverHtml = `
        <div class="weaver-indicator" id="weaverIndicator" title="Click to manage Weaver" onclick="toggleWeaverMenu(event)">
            <span class="weaver-icon">🕸️</span>
            <span class="weaver-status" id="weaverStatusDot">●</span>
        </div>
        <div class="weaver-menu" id="weaverMenu" style="display: none;">
            <div class="weaver-menu-header">
                <span>🕸️ Weaver</span>
                <span class="weaver-menu-status" id="weaverMenuStatus">Unknown</span>
            </div>
            <div class="weaver-menu-threads" id="weaverMenuThreads"></div>
            <div class="weaver-menu-actions">
                <button class="weaver-btn weaver-btn-start" id="weaverStartBtn" onclick="weaverStart()">
                    ▶️ Start
                </button>
                <button class="weaver-btn weaver-btn-stop" id="weaverStopBtn" onclick="weaverStop()">
                    ⏹️ Stop
                </button>
            </div>
        </div>
    `;

    container.innerHTML = navHtml + weaverHtml;

    // Add styles for menu if not already added
    if (!document.getElementById('weaverMenuStyles')) {
        const style = document.createElement('style');
        style.id = 'weaverMenuStyles';
        style.textContent = `
            .weaver-indicator {
                cursor: pointer;
                user-select: none;
            }
            .weaver-indicator:hover {
                opacity: 1 !important;
            }
            .weaver-menu {
                position: absolute;
                bottom: 100%;
                right: 0.5rem;
                background: var(--bg-panel, #1a1a2e);
                border: 1px solid var(--border-gold, #d4a574);
                border-radius: 8px;
                padding: 0.75rem;
                min-width: 200px;
                box-shadow: 0 -4px 12px rgba(0,0,0,0.4);
                z-index: 1000;
            }
            .weaver-menu-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--border-gold, #d4a574);
                font-weight: bold;
                color: var(--text-gold, #d4a574);
            }
            .weaver-menu-status {
                font-size: 0.8rem;
                padding: 0.15rem 0.5rem;
                border-radius: 4px;
                background: var(--bg-card, #252540);
            }
            .weaver-menu-status.running {
                color: #10B981;
            }
            .weaver-menu-status.stopped {
                color: #EF4444;
            }
            .weaver-menu-threads {
                font-size: 0.85rem;
                margin-bottom: 0.75rem;
                color: var(--text-secondary, #a0a0a0);
            }
            .weaver-menu-threads .thread-item {
                display: flex;
                justify-content: space-between;
                padding: 0.2rem 0;
            }
            .weaver-menu-threads .thread-status {
                font-weight: bold;
            }
            .weaver-menu-threads .thread-status.alive { color: #10B981; }
            .weaver-menu-threads .thread-status.dead { color: #EF4444; }
            .weaver-menu-actions {
                display: flex;
                gap: 0.5rem;
            }
            .weaver-btn {
                flex: 1;
                padding: 0.5rem;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.85rem;
                transition: opacity 0.2s;
            }
            .weaver-btn:hover {
                opacity: 0.9;
            }
            .weaver-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .weaver-btn-start {
                background: #10B981;
                color: white;
            }
            .weaver-btn-stop {
                background: #EF4444;
                color: white;
            }
        `;
        document.head.appendChild(style);
    }

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        const menu = document.getElementById('weaverMenu');
        const indicator = document.getElementById('weaverIndicator');
        if (menu && indicator && !indicator.contains(e.target) && !menu.contains(e.target)) {
            menu.style.display = 'none';
        }
    });

    // Fetch and update weaver status
    updateWeaverStatus();
}

/**
 * Toggle Weaver control menu
 */
function toggleWeaverMenu(event) {
    event.stopPropagation();
    const menu = document.getElementById('weaverMenu');
    if (menu) {
        const isVisible = menu.style.display !== 'none';
        menu.style.display = isVisible ? 'none' : 'block';
        if (!isVisible) {
            updateWeaverStatus(); // Refresh status when opening
        }
    }
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
            total: data.total_threads || 0,
            pid: data.weaver_pid,
            threads: data.threads || {}
        };

        const dot = document.getElementById('weaverStatusDot');
        const indicator = document.getElementById('weaverIndicator');
        const menuStatus = document.getElementById('weaverMenuStatus');
        const menuThreads = document.getElementById('weaverMenuThreads');
        const startBtn = document.getElementById('weaverStartBtn');
        const stopBtn = document.getElementById('weaverStopBtn');

        if (dot && indicator) {
            if (data.weaver_running) {
                if (data.healthy_count === data.total_threads) {
                    dot.style.color = '#10B981'; // Green - all healthy
                    indicator.title = `Weaver: ${data.healthy_count}/${data.total_threads} threads healthy (click to manage)`;
                } else {
                    dot.style.color = '#F59E0B'; // Yellow - some unhealthy
                    indicator.title = `Weaver: ${data.healthy_count}/${data.total_threads} threads healthy (click to manage)`;
                }
            } else {
                dot.style.color = '#EF4444'; // Red - not running
                indicator.title = 'Weaver: Not running (click to start)';
            }
        }

        // Update menu
        if (menuStatus) {
            if (data.weaver_running) {
                menuStatus.textContent = `Running (PID ${data.weaver_pid})`;
                menuStatus.className = 'weaver-menu-status running';
            } else {
                menuStatus.textContent = 'Stopped';
                menuStatus.className = 'weaver-menu-status stopped';
            }
        }

        if (menuThreads) {
            if (data.weaver_running && Object.keys(data.threads).length > 0) {
                menuThreads.innerHTML = Object.entries(data.threads).map(([key, t]) => `
                    <div class="thread-item">
                        <span>${t.name}</span>
                        <span class="thread-status ${t.alive ? 'alive' : 'dead'}">${t.alive ? '✓' : '✗'}</span>
                    </div>
                `).join('');
            } else {
                menuThreads.innerHTML = '<div style="color: #666; font-style: italic;">No threads</div>';
            }
        }

        // Update buttons
        if (startBtn) startBtn.disabled = data.weaver_running;
        if (stopBtn) stopBtn.disabled = !data.weaver_running;

    } catch (e) {
        const dot = document.getElementById('weaverStatusDot');
        if (dot) {
            dot.style.color = '#6B7280'; // Gray - unknown
        }
    }
}

/**
 * Start the Weaver daemon
 */
async function weaverStart() {
    const startBtn = document.getElementById('weaverStartBtn');
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = '⏳ Starting...';
    }

    try {
        const resp = await fetch('/api/weaver/start', { method: 'POST' });
        const data = await resp.json();

        if (data.success) {
            // Refresh status
            setTimeout(updateWeaverStatus, 1000);
        } else {
            alert(data.message || 'Failed to start Weaver');
        }
    } catch (e) {
        alert('Error starting Weaver: ' + e.message);
    } finally {
        if (startBtn) {
            startBtn.textContent = '▶️ Start';
        }
        setTimeout(updateWeaverStatus, 500);
    }
}

/**
 * Stop the Weaver daemon
 */
async function weaverStop() {
    if (!confirm('Stop Weaver? This will stop monitoring all services.')) {
        return;
    }

    const stopBtn = document.getElementById('weaverStopBtn');
    if (stopBtn) {
        stopBtn.disabled = true;
        stopBtn.textContent = '⏳ Stopping...';
    }

    try {
        const resp = await fetch('/api/weaver/stop', { method: 'POST' });
        const data = await resp.json();

        if (data.success) {
            setTimeout(updateWeaverStatus, 1000);
        } else {
            alert(data.message || 'Failed to stop Weaver');
        }
    } catch (e) {
        alert('Error stopping Weaver: ' + e.message);
    } finally {
        if (stopBtn) {
            stopBtn.textContent = '⏹️ Stop';
        }
        setTimeout(updateWeaverStatus, 500);
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
    } else if (path.startsWith('/campaign')) {
        activeId = 'campaign';
    } else if (path.startsWith('/quests')) {
        activeId = 'quests';
    } else if (path.startsWith('/forge')) {
        activeId = 'forge';
    } else if (path.startsWith('/vault') || path.startsWith('/checkpoint')) {
        activeId = 'vault';
    } else if (path.startsWith('/ledger')) {
        activeId = 'ledger';
    } else if (path.startsWith('/oracle')) {
        activeId = 'oracle';
    } else if (path.startsWith('/arcana')) {
        activeId = 'arcana';
    } else if (path.startsWith('/temple')) {
        activeId = 'temple';
    } else if (path.startsWith('/garrison')) {
        activeId = 'garrison';
    } else if (path.startsWith('/settings') || path.startsWith('/scheduler')) {
        activeId = 'settings';
    }

    renderBottomNav(activeId);
}

// Auto-render on DOMContentLoaded if bottomNav container exists
document.addEventListener('DOMContentLoaded', autoRenderNav);
