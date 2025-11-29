/**
 * Global Event Stream - Real-time announcements from the training system
 *
 * Connects to /api/events/stream via Server-Sent Events and displays
 * notifications in a toast-style UI.
 */

class EventStream {
    constructor(options = {}) {
        this.maxNotifications = options.maxNotifications || 5;
        this.notificationDuration = options.notificationDuration || 8000;
        this.eventSource = null;
        this.notifications = [];
        this.container = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;

        // Callback handlers for game state updates
        this.handlers = {};

        // Severity colors
        this.severityColors = {
            debug: '#6b7280',
            info: '#3b82f6',
            warning: '#f59e0b',
            error: '#ef4444',
            success: '#10b981'
        };

        // Event icons
        this.eventIcons = {
            'queue.empty': '📭',
            'queue.low': '📬',
            'data.need': '📋',
            'data.generating': '⚙️',
            'data.generated': '✅',
            'data.queued': '📥',
            'data.quality_pass': '✅',
            'data.quality_fail': '❌',
            'training.started': '⚔️',
            'training.step': '⚡',
            'training.completed': '🎉',
            'training.checkpoint': '💾',
            'hero.level_up': '🆙',
            'daemon.heartbeat': '💓',
            'default': '📢'
        };

        // Events that should NOT show notifications (high-frequency)
        this.silentEvents = new Set(['training.step', 'daemon.heartbeat']);
    }

    /**
     * Register a handler for a specific event type
     * @param {string} eventType - The event type to handle
     * @param {function} handler - Callback function(eventData)
     */
    on(eventType, handler) {
        if (!this.handlers[eventType]) {
            this.handlers[eventType] = [];
        }
        this.handlers[eventType].push(handler);
    }

    /**
     * Remove a handler for an event type
     */
    off(eventType, handler) {
        if (this.handlers[eventType]) {
            this.handlers[eventType] = this.handlers[eventType].filter(h => h !== handler);
        }
    }

    /**
     * Initialize the event stream
     */
    init() {
        this.createContainer();
        this.connect();
        console.log('[EventStream] Initialized');
    }

    /**
     * Create the notification container
     */
    createContainer() {
        // Check if container already exists
        this.container = document.getElementById('event-notifications');
        if (this.container) return;

        this.container = document.createElement('div');
        this.container.id = 'event-notifications';
        this.container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            max-height: 400px;
            overflow-y: auto;
            z-index: 9999;
            display: flex;
            flex-direction: column-reverse;
            gap: 10px;
            pointer-events: none;
        `;
        document.body.appendChild(this.container);

        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .event-notification {
                background: rgba(30, 30, 40, 0.95);
                border-left: 4px solid #3b82f6;
                border-radius: 8px;
                padding: 12px 16px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                pointer-events: auto;
                animation: slideIn 0.3s ease-out;
                font-family: 'JetBrains Mono', monospace;
            }

            .event-notification.closing {
                animation: slideOut 0.3s ease-in forwards;
            }

            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }

            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }

            .event-notification .header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 6px;
            }

            .event-notification .icon {
                font-size: 18px;
            }

            .event-notification .source {
                font-size: 11px;
                color: #888;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .event-notification .time {
                font-size: 10px;
                color: #666;
                margin-left: auto;
            }

            .event-notification .message {
                font-size: 13px;
                color: #e0e0e0;
                line-height: 1.4;
            }

            .event-notification .close-btn {
                position: absolute;
                top: 8px;
                right: 8px;
                background: none;
                border: none;
                color: #666;
                cursor: pointer;
                font-size: 16px;
                padding: 0;
                line-height: 1;
            }

            .event-notification .close-btn:hover {
                color: #fff;
            }

            /* Severity-specific border colors */
            .event-notification.severity-warning {
                border-left-color: #f59e0b;
            }

            .event-notification.severity-error {
                border-left-color: #ef4444;
            }

            .event-notification.severity-success {
                border-left-color: #10b981;
            }

            .event-notification.severity-debug {
                border-left-color: #6b7280;
            }
        `;
        document.head.appendChild(style);
    }

    /**
     * Connect to the event stream
     */
    connect() {
        if (this.eventSource) {
            this.eventSource.close();
        }

        try {
            this.eventSource = new EventSource('/api/events/stream');

            this.eventSource.onopen = () => {
                console.log('[EventStream] Connected');
                this.reconnectAttempts = 0;
            };

            this.eventSource.onmessage = (event) => {
                // Default event handler (shouldn't be called with named events)
                console.log('[EventStream] Message:', event.data);
            };

            // Listen for all event types
            this.eventSource.addEventListener('queue.empty', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('queue.low', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('data.need', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('data.generating', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('data.generated', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('data.queued', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('data.quality_pass', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('data.quality_fail', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('training.started', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('training.step', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('training.completed', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('training.checkpoint', (e) => this.handleEvent(e));
            this.eventSource.addEventListener('hero.level_up', (e) => this.handleEvent(e));

            this.eventSource.onerror = (error) => {
                console.error('[EventStream] Error:', error);
                this.eventSource.close();
                this.scheduleReconnect();
            };

        } catch (error) {
            console.error('[EventStream] Failed to connect:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Schedule a reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('[EventStream] Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * this.reconnectAttempts;
        console.log(`[EventStream] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => this.connect(), delay);
    }

    /**
     * Handle an incoming event
     */
    handleEvent(event) {
        try {
            const data = JSON.parse(event.data);

            // Call registered handlers for this event type
            const eventType = data.type;
            if (this.handlers[eventType]) {
                for (const handler of this.handlers[eventType]) {
                    try {
                        handler(data);
                    } catch (err) {
                        console.error(`[EventStream] Handler error for ${eventType}:`, err);
                    }
                }
            }

            // Show notification unless it's a silent event
            if (!this.silentEvents.has(eventType)) {
                this.showNotification(data);
            }
        } catch (error) {
            console.error('[EventStream] Failed to parse event:', error);
        }
    }

    /**
     * Show a notification
     */
    showNotification(event) {
        const notification = document.createElement('div');
        notification.className = `event-notification severity-${event.severity || 'info'}`;
        notification.style.position = 'relative';

        const icon = this.eventIcons[event.type] || this.eventIcons.default;
        const time = new Date(event.timestamp).toLocaleTimeString();

        notification.innerHTML = `
            <button class="close-btn">&times;</button>
            <div class="header">
                <span class="icon">${icon}</span>
                <span class="source">${event.source || 'system'}</span>
                <span class="time">${time}</span>
            </div>
            <div class="message">${event.message}</div>
        `;

        // Close button handler
        const closeBtn = notification.querySelector('.close-btn');
        closeBtn.addEventListener('click', () => this.closeNotification(notification));

        // Add to container
        this.container.appendChild(notification);
        this.notifications.push(notification);

        // Limit number of notifications
        while (this.notifications.length > this.maxNotifications) {
            const oldest = this.notifications.shift();
            this.closeNotification(oldest);
        }

        // Auto-close after duration
        setTimeout(() => {
            if (notification.parentNode) {
                this.closeNotification(notification);
            }
        }, this.notificationDuration);
    }

    /**
     * Close a notification with animation
     */
    closeNotification(notification) {
        notification.classList.add('closing');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
            const idx = this.notifications.indexOf(notification);
            if (idx > -1) {
                this.notifications.splice(idx, 1);
            }
        }, 300);
    }

    /**
     * Disconnect from the event stream
     */
    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.eventStream = new EventStream();
    window.eventStream.init();
});

// Clean up on page navigation to prevent lingering connections
window.addEventListener('beforeunload', () => {
    if (window.eventStream) {
        window.eventStream.disconnect();
    }
});

window.addEventListener('pagehide', () => {
    if (window.eventStream) {
        window.eventStream.disconnect();
    }
});
