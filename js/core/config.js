/**
 * Configuration Module
 * Central location for all constants, API endpoints, and thresholds
 */

export const CONFIG = {
    // API Endpoints
    API: {
        STATUS: '/status/training_status.json',
        GPU_STATS: '/api/gpu_stats',
        MEMORY_STATS: 'http://localhost:8081/api/memory_stats',
        INBOX_FILES: '/api/inbox_files',
        CONFIG: '/api/config',
        QUEUE_SAMPLES: '/queue/normal/'
    },

    // Timing
    TIMING: {
        BASE_REFRESH_MS: 2000,      // 2 seconds
        STALE_DATA_SEC: 10,          // Consider data stale after 10s
        MAX_BACKOFF_MS: 30000        // Max 30s backoff on errors
    },

    // Training Parameters
    TRAINING: {
        EFFECTIVE_BATCH_SIZE: 8,
        EMA_ALPHA: 0.25              // For steps/sec smoothing
    },

    // System Thresholds
    LIMITS: {
        // GPU
        GPU_TEMP_WARN: 70,
        GPU_TEMP_HOT: 80,
        GPU_TEMP_DANGER: 85,

        // RAM
        RAM_WARN: 70,
        RAM_DANGER: 85,

        // Training Memory
        TRAINING_MEM_WARN: 30,
        TRAINING_MEM_DANGER: 40,

        // Errors
        MAX_ERROR_RETRIES: 5
    },

    // Loss Analysis
    LOSS: {
        GOOD_GAP: 0.3,               // train-val gap < 0.3 is excellent
        WARNING_GAP: 0.5,            // gap 0.3-0.5 is warning
        DANGER_GAP: 0.5              // gap > 0.5 is overfitting
    },

    // Think Tag Thresholds
    THINK_TAG: {
        EXCELLENT: 20,               // < 20% is excellent
        WARNING: 60                  // > 60% is concerning
    },

    // Queue Status
    QUEUE: {
        LOW: 5,                      // < 5 files is low
        EMPTY: 0                     // 0 files needs attention
    },

    // UI Settings
    UI: {
        CHART_MAX_POINTS: 100,
        RECENT_EXAMPLES_LIMIT: 10,
        FLAGGED_DISPLAY_LIMIT: 50,
        LOSS_HISTORY_LIMIT: 50
    },

    // LocalStorage Keys
    STORAGE_KEYS: {
        COMPACT_MODE: 'tlm:compactMode',
        DARK_THEME: 'tlm:darkTheme',
        SOUND_ENABLED: 'tlm:soundEnabled',
        NOTIFICATIONS_ENABLED: 'tlm:notificationsEnabled'
    }
};

// Color schemes
export const COLORS = {
    ACCENT_GREEN: '#00ff88',
    ACCENT_BLUE: '#00d9ff',
    ACCENT_YELLOW: '#ffaa00',
    ACCENT_RED: '#ff4444',
    TEXT_PRIMARY: '#e0e0e0',
    TEXT_SECONDARY: '#888',
    BG_PANEL: 'rgba(26, 39, 59, 0.95)'
};

// Export default for convenience
export default CONFIG;
