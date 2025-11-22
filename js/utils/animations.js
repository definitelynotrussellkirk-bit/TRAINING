/**
 * Animation Utilities
 * Functions for visual feedback and UI animations
 */

/**
 * Animate a value change with a flash effect
 * @param {string} elementId - ID of element to animate
 * @param {string} color - Flash color (default: yellow)
 */
export function animateValueChange(elementId, color = '#ffaa00') {
    const element = document.getElementById(elementId);
    if (!element) return;

    // Add flash animation
    element.style.transition = 'background-color 0.3s ease';
    element.style.backgroundColor = color;

    setTimeout(() => {
        element.style.backgroundColor = 'transparent';
    }, 300);
}

/**
 * Update delta indicator (up/down arrows with color)
 * @param {string} elementId - ID of delta element
 * @param {number} currentValue - Current value
 * @param {number} previousValue - Previous value
 * @param {boolean} lowerIsBetter - If true, lower values are good (green)
 */
export function updateDelta(elementId, currentValue, previousValue, lowerIsBetter = true) {
    const element = document.getElementById(elementId);
    if (!element) return;

    if (previousValue === null || previousValue === undefined) {
        element.textContent = '';
        return;
    }

    const delta = currentValue - previousValue;
    if (Math.abs(delta) < 0.001) {
        element.textContent = '';
        return;
    }

    const isImprovement = lowerIsBetter ? delta < 0 : delta > 0;
    const arrow = delta > 0 ? '↑' : '↓';
    const color = isImprovement ? '#00ff88' : '#ff4444';

    element.textContent = ` ${arrow}${Math.abs(delta).toFixed(3)}`;
    element.style.color = color;
}

/**
 * Show a toast notification
 * @param {string} message - Message to display
 * @param {string} type - Type: 'success', 'warning', 'error', 'info'
 * @param {number} duration - Duration in ms (default: 3000)
 */
export function showToast(message, type = 'info', duration = 3000) {
    // Remove existing toasts
    const existing = document.querySelectorAll('.toast-notification');
    existing.forEach(el => el.remove());

    // Create toast
    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;
    toast.textContent = message;

    // Style
    Object.assign(toast.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '15px 20px',
        borderRadius: '8px',
        color: '#fff',
        fontSize: '14px',
        fontWeight: '500',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        zIndex: '10000',
        animation: 'slideInRight 0.3s ease',
        maxWidth: '400px'
    });

    // Type-specific colors
    const colors = {
        success: '#00ff88',
        warning: '#ffaa00',
        error: '#ff4444',
        info: '#00d9ff'
    };
    toast.style.backgroundColor = colors[type] || colors.info;

    document.body.appendChild(toast);

    // Auto-remove
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Pulse an element to draw attention
 * @param {string} elementId - ID of element to pulse
 * @param {number} count - Number of pulses (default: 3)
 */
export function pulseElement(elementId, count = 3) {
    const element = document.getElementById(elementId);
    if (!element) return;

    let pulseCount = 0;
    const interval = setInterval(() => {
        element.style.transform = 'scale(1.1)';
        element.style.transition = 'transform 0.2s ease';

        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 200);

        pulseCount++;
        if (pulseCount >= count) {
            clearInterval(interval);
        }
    }, 400);
}

/**
 * Shake an element (for errors)
 * @param {string} elementId - ID of element to shake
 */
export function shakeElement(elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.style.animation = 'shake 0.5s ease';

    setTimeout(() => {
        element.style.animation = '';
    }, 500);
}

/**
 * Smooth scroll to element
 * @param {string} elementId - ID of element to scroll to
 */
export function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
    });
}

/**
 * Highlight element temporarily
 * @param {string} elementId - ID of element to highlight
 * @param {number} duration - Duration in ms (default: 2000)
 */
export function highlightElement(elementId, duration = 2000) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const originalBg = element.style.backgroundColor;
    element.style.backgroundColor = 'rgba(0, 217, 255, 0.2)';
    element.style.transition = 'background-color 0.3s ease';

    setTimeout(() => {
        element.style.backgroundColor = originalBg;
    }, duration);
}

/**
 * Progress bar animation
 * @param {string} elementId - ID of progress bar element
 * @param {number} percent - Percentage (0-100)
 * @param {boolean} animated - Whether to animate (default: true)
 */
export function updateProgressBar(elementId, percent, animated = true) {
    const element = document.getElementById(elementId);
    if (!element) return;

    if (animated) {
        element.style.transition = 'width 0.5s ease';
    } else {
        element.style.transition = 'none';
    }

    element.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

// Add CSS animations if not already present
if (typeof document !== 'undefined') {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
    `;
    document.head.appendChild(style);
}
