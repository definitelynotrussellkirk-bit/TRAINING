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

/**
 * Render the bottom navigation bar
 * @param {string} activeId - Which nav item is active (e.g., 'battle', 'vault')
 * @param {string} containerId - ID of container element (default: 'bottomNav')
 */
function renderBottomNav(activeId, containerId = 'bottomNav') {
    const container = document.getElementById(containerId);
    if (!container) return;

    const html = NAV_ITEMS.map(item => {
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

    container.innerHTML = html;
}

/**
 * Auto-detect current page and render nav
 */
function autoRenderNav() {
    const path = window.location.pathname;
    let activeId = 'battle';

    if (path === '/' || path === '/game' || path === '/game.html') {
        activeId = 'battle';
    } else if (path.startsWith('/guild')) {
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
