/* ========================================
   TIER 1 UI/UX IMPROVEMENTS
   2025-11-12 - Ultrathink Session

   Features:
   1. Browser Notifications API
   2. Better Error Display
   3. Collapsible Panels
   4. Keyboard Shortcut Help Modal
   5. Enhanced Sound Alerts
   6. Training Velocity Indicator
   7. Loss Plateau Detection
   8. Confidence Bars
   ======================================== */

// ========== BROWSER NOTIFICATIONS ==========

let notificationsEnabled = false;

function initializeNotifications() {
    if ('Notification' in window) {
        if (Notification.permission === 'granted') {
            notificationsEnabled = true;
            console.log('✅ Browser notifications enabled');
        } else if (Notification.permission === 'default') {
            // Show subtle prompt after 5 seconds
            setTimeout(() => {
                const banner = document.createElement('div');
                banner.className = 'toast success';
                banner.innerHTML = `
                    <div style="text-align: center;">
                        <strong>Enable Desktop Notifications?</strong><br>
                        <small>Get alerts for critical events even when tab is not focused</small><br>
                        <button onclick="requestNotificationPermission()" style="margin-top: 10px; padding: 5px 15px;">
                            Enable Notifications
                        </button>
                        <button onclick="this.parentElement.parentElement.remove()" style="margin-top: 10px; margin-left: 10px; padding: 5px 15px;">
                            No Thanks
                        </button>
                    </div>
                `;
                document.body.appendChild(banner);

                setTimeout(() => {
                    if (banner.parentElement) {
                        banner.remove();
                    }
                }, 10000);
            }, 5000);
        }
    }
}

function requestNotificationPermission() {
    if ('Notification' in window) {
        Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
                notificationsEnabled = true;
                showToast('✅ Notifications enabled!', 'success');

                // Show test notification
                sendNotification('Training Monitor', 'Notifications are now enabled! You\'ll be alerted for critical events.', 'normal');
            } else {
                showToast('ℹ️ Notifications not enabled', 'warning');
            }

            // Remove any permission prompts
            document.querySelectorAll('.toast').forEach(t => {
                if (t.textContent.includes('Enable Desktop Notifications')) {
                    t.remove();
                }
            });
        });
    }
}

function sendNotification(title, body, urgency = 'normal') {
    if (!notificationsEnabled || Notification.permission !== 'granted') {
        return;
    }

    const options = {
        body: body,
        icon: urgency === 'urgent' ? '⚠️' : '✅',
        tag: 'training-monitor-' + Date.now(),
        requireInteraction: urgency === 'urgent',
        silent: false
    };

    try {
        const notification = new Notification(title, options);

        // Auto-close after 10 seconds for normal priority
        if (urgency !== 'urgent') {
            setTimeout(() => notification.close(), 10000);
        }

        // Focus tab when clicked
        notification.onclick = () => {
            window.focus();
            notification.close();
        };
    } catch (e) {
        console.error('Failed to send notification:', e);
    }
}

// ========== BETTER ERROR DISPLAY ==========

function showErrorBanner(error) {
    // Remove any existing error banners
    document.querySelectorAll('.error-banner').forEach(b => b.remove());

    const banner = document.createElement('div');
    banner.className = 'error-banner';
    banner.innerHTML = `
        <div class="error-banner-content">
            <div class="error-icon">⚠️</div>
            <div class="error-details">
                <div class="error-title">Training Error</div>
                <div class="error-message">${escapeHTML(error.message || error)}</div>
                ${error.file ? `<div class="error-file">File: ${escapeHTML(error.file)}</div>` : ''}
            </div>
            <div class="error-actions">
                <button onclick="viewFullLog()">View Log</button>
                <button onclick="copyErrorToClipboard()">Copy</button>
                <button onclick="dismissError()">Dismiss</button>
            </div>
            <button class="error-close" onclick="dismissError()">✕</button>
        </div>
    `;

    // Insert at top of container
    const container = document.querySelector('.container');
    container.insertBefore(banner, container.firstChild);

    // Play error sound
    playSound('error');

    // Send browser notification
    sendNotification('Training Error!', error.message || error, 'urgent');

    // Flash animation
    setTimeout(() => banner.classList.add('flash'), 100);
}

function dismissError() {
    document.querySelectorAll('.error-banner').forEach(b => {
        b.style.animation = 'slideUp 0.3s ease';
        setTimeout(() => b.remove(), 300);
    });
}

function copyErrorToClipboard() {
    const errorText = document.querySelector('.error-message')?.textContent || '';
    navigator.clipboard.writeText(errorText).then(() => {
        showToast('✅ Error copied to clipboard', 'success');
    });
}

function viewFullLog() {
    // This could open a log viewer or download logs
    // For now, just show a toast
    showToast('📜 Opening logs... (Feature coming soon!)', 'warning');
}

// ========== COLLAPSIBLE PANELS ==========

function initializeCollapsiblePanels() {
    // Add collapse icons to all h2 elements in panels
    const panels = document.querySelectorAll('.panel > h2, .panel > h3');

    panels.forEach((header, index) => {
        // Make header collapsible
        header.classList.add('panel-header');
        header.dataset.panelId = 'panel-' + index;

        // Add collapse icon
        const icon = document.createElement('span');
        icon.className = 'collapse-icon';
        icon.textContent = '▼';
        header.appendChild(icon);

        // Wrap subsequent content in panel-content div
        let content = [];
        let sibling = header.nextElementSibling;

        while (sibling && !['H2', 'H3'].includes(sibling.tagName)) {
            content.push(sibling);
            sibling = sibling.nextElementSibling;
        }

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'panel-content';
        contentWrapper.dataset.panelId = 'panel-' + index;

        content.forEach(el => contentWrapper.appendChild(el));
        header.after(contentWrapper);

        // Restore saved state
        const savedState = localStorage.getItem('tlm:panel:' + index);
        if (savedState === 'collapsed') {
            header.classList.add('collapsed');
            contentWrapper.classList.add('collapsed');
        }

        // Click handler
        header.addEventListener('click', () => {
            const isCollapsed = header.classList.toggle('collapsed');
            contentWrapper.classList.toggle('collapsed');

            // Save state
            localStorage.setItem('tlm:panel:' + index, isCollapsed ? 'collapsed' : 'expanded');
        });
    });

    console.log('✅ Collapsible panels initialized');
}

// ========== KEYBOARD SHORTCUT HELP MODAL ==========

function showShortcutsModal() {
    const modal = document.getElementById('shortcutsModal');
    if (modal) {
        modal.classList.add('active');
    }
}

function closeShortcutsModal() {
    const modal = document.getElementById('shortcutsModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

// Close modal on background click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        closeShortcutsModal();
    }
});

// ========== ENHANCED SOUND ALERTS ==========

const SOUNDS = {
    complete: { freq: 800, duration: 0.3 },
    error: { freq: 200, duration: 0.5 },
    critical: { freq: 400, duration: 0.2, pulses: 3 },
    bestModel: { freq: 1000, duration: 0.2, pulses: 2 },
    milestone: { freq: 600, duration: 0.15 }
};

function playSingleTone(audioCtx, freq, duration) {
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.frequency.value = freq;
    oscillator.type = 'sine';
    gainNode.gain.value = 0.3;

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + duration);
}

function playSound(type) {
    if (!soundEnabled || !SOUNDS[type]) return;

    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const sound = SOUNDS[type];

        if (sound.pulses) {
            for (let i = 0; i < sound.pulses; i++) {
                setTimeout(() => {
                    playSingleTone(audioCtx, sound.freq, sound.duration);
                }, i * (sound.duration * 1000 + 100));
            }
        } else {
            playSingleTone(audioCtx, sound.freq, sound.duration);
        }
    } catch (e) {
        console.error('Failed to play sound:', e);
    }
}

// ========== TRAINING VELOCITY INDICATOR ==========

let velocityHistory = [];

function updateVelocity(stepsPerSec) {
    if (!stepsPerSec || stepsPerSec <= 0) return;

    velocityHistory.push({
        timestamp: Date.now(),
        stepsPerSec: stepsPerSec
    });

    // Keep last 100 measurements
    if (velocityHistory.length > 100) velocityHistory.shift();

    if (velocityHistory.length < 20) return;

    // Calculate acceleration
    const recent = velocityHistory.slice(-10);
    const older = velocityHistory.slice(-20, -10);

    const recentAvg = recent.reduce((a, b) => a + b.stepsPerSec, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b.stepsPerSec, 0) / older.length;

    const acceleration = (recentAvg - olderAvg) / olderAvg * 100;

    // Display (add to DOM if needed)
    let velocityEl = document.getElementById('velocityIndicator');
    if (!velocityEl) {
        // Add velocity indicator to Speed & ETA panel
        const speedPanel = document.querySelector('.panel:has(#stepsPerSec)');
        if (speedPanel) {
            velocityEl = document.createElement('div');
            velocityEl.id = 'velocityIndicator';
            velocityEl.className = 'velocity-indicator velocity-steady';
            speedPanel.querySelector('h2').appendChild(velocityEl);
        } else {
            return;
        }
    }

    if (Math.abs(acceleration) > 5) {
        if (acceleration > 0) {
            velocityEl.innerHTML = `🚀 +${acceleration.toFixed(1)}%`;
            velocityEl.className = 'velocity-indicator velocity-up';
        } else {
            velocityEl.innerHTML = `🐌 ${acceleration.toFixed(1)}%`;
            velocityEl.className = 'velocity-indicator velocity-down';
        }
    } else {
        velocityEl.innerHTML = `➡️ Steady`;
        velocityEl.className = 'velocity-indicator velocity-steady';
    }
}

// ========== LOSS PLATEAU DETECTION ==========

function detectLossPlateau(lossHistory) {
    if (lossHistory.length < 500) return false;

    const recent100 = lossHistory.slice(-100);
    const older500 = lossHistory.slice(-600, -500);

    if (older500.length === 0) return false;

    const recentAvg = recent100.reduce((a, b) => a + b, 0) / recent100.length;
    const olderAvg = older500.reduce((a, b) => a + b, 0) / older500.length;

    const improvement = (olderAvg - recentAvg) / olderAvg * 100;

    if (improvement < 1) {
        // Plateau detected - only show once
        if (!window.plateauWarningShown) {
            showPlateauWarning(improvement);
            window.plateauWarningShown = true;

            // Reset flag after 10 minutes
            setTimeout(() => {
                window.plateauWarningShown = false;
            }, 600000);
        }
        return true;
    }
    return false;
}

function showPlateauWarning(improvement) {
    const banner = document.createElement('div');
    banner.className = 'warning-banner';
    banner.innerHTML = `
        <div class="warning-content">
            <div class="warning-icon">📊</div>
            <div class="warning-text">
                <strong>Loss Plateaued</strong><br>
                Loss improved only ${improvement.toFixed(1)}% in last 500 steps.
                <br><br>
                <strong>Suggestions:</strong>
                <ul>
                    <li>Continue training (may break through plateau)</li>
                    <li>Consider increasing learning rate by 1.5x</li>
                    <li>Or reduce learning rate by 0.5x</li>
                    <li>Check if model capacity saturated (increase LoRA rank)</li>
                </ul>
            </div>
        </div>
        <button onclick="dismissWarning(this)">Got it</button>
    `;

    const container = document.getElementById('errorContainer');
    if (container) {
        container.appendChild(banner);
    }

    // Send notification
    sendNotification('Loss Plateaued', `Training progress slowed. Loss improved only ${improvement.toFixed(1)}% recently.`, 'normal');

    // Play milestone sound
    playSound('milestone');
}

function dismissWarning(button) {
    const banner = button.closest('.warning-banner');
    if (banner) {
        banner.style.animation = 'slideUp 0.3s ease';
        setTimeout(() => banner.remove(), 300);
    }
}

// ========== CONFIDENCE BARS ==========

function updateConfidence(loss) {
    if (!loss || loss <= 0) return;

    // Calculate confidence (inverse of loss)
    const confidence = Math.exp(-loss) * 100;

    let confSection = document.getElementById('confidenceSection');
    if (!confSection) {
        // Add confidence section to current example panel
        const answerPanel = document.querySelector('.panel:has(#modelAnswer)');
        if (answerPanel) {
            confSection = document.createElement('div');
            confSection.id = 'confidenceSection';
            confSection.className = 'confidence-section';
            confSection.innerHTML = `
                <div class="confidence-label">Model Confidence:</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar" id="confidenceBar"></div>
                    <div class="confidence-text" id="confidenceText">--%</div>
                </div>
                <div class="confidence-interpretation" id="confidenceInterpret"></div>
            `;
            answerPanel.appendChild(confSection);
        } else {
            return;
        }
    }

    const confBar = document.getElementById('confidenceBar');
    const confText = document.getElementById('confidenceText');
    const confInterp = document.getElementById('confidenceInterpret');

    if (!confBar || !confText || !confInterp) return;

    confBar.style.width = Math.min(confidence, 100) + '%';
    confText.textContent = confidence.toFixed(1) + '%';

    // Color coding and interpretation
    if (confidence > 80) {
        confBar.className = 'confidence-bar conf-high';
        confInterp.textContent = '🟢 High confidence - model is certain';
    } else if (confidence > 50) {
        confBar.className = 'confidence-bar conf-medium';
        confInterp.textContent = '🟡 Medium confidence - some uncertainty';
    } else {
        confBar.className = 'confidence-bar conf-low';
        confInterp.textContent = '🔴 Low confidence - model is guessing';
    }
}

// ========== TOAST NOTIFICATIONS ==========

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideInFromRight 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// ========== INITIALIZE EVERYTHING ==========

function initializeImprovements() {
    console.log('🚀 Initializing Tier 1 UI/UX Improvements...');

    // Request notification permission after a delay
    setTimeout(initializeNotifications, 3000);

    // Initialize collapsible panels
    initializeCollapsiblePanels();

    // Add keyboard shortcut for help modal
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        // ? or / to show shortcuts modal
        if (e.key === '?' || e.key === '/') {
            e.preventDefault();
            showShortcutsModal();
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            closeShortcutsModal();
            dismissError();
        }
    });

    console.log('✅ All improvements initialized!');
    showToast('✨ UI/UX improvements loaded!', 'success');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeImprovements);
} else {
    initializeImprovements();
}

// Make functions globally available
window.initializeNotifications = initializeNotifications;
window.requestNotificationPermission = requestNotificationPermission;
window.sendNotification = sendNotification;
window.showErrorBanner = showErrorBanner;
window.dismissError = dismissError;
window.copyErrorToClipboard = copyErrorToClipboard;
window.viewFullLog = viewFullLog;
window.showShortcutsModal = showShortcutsModal;
window.closeShortcutsModal = closeShortcutsModal;
window.playSound = playSound;
window.updateVelocity = updateVelocity;
window.detectLossPlateau = detectLossPlateau;
window.dismissWarning = dismissWarning;
window.updateConfidence = updateConfidence;
window.showToast = showToast;
