/**
 * Shared formatting utilities for Realm of Training UI
 *
 * All formatting functions handle null/undefined gracefully and return '--' for invalid values.
 * Use these instead of inline .toFixed() calls for consistency.
 */

const Format = (() => {
    /**
     * Format training loss (strain) - 4 decimal places
     * @param {number} loss - Training loss value
     * @returns {string} Formatted loss or '--'
     */
    function loss(x) {
        if (x == null || x <= 0 || !isFinite(x)) return '--';
        return x.toFixed(4);
    }

    /**
     * Format generic metric with configurable precision
     * @param {number} x - Metric value
     * @param {number} digits - Decimal places (default 3)
     * @returns {string} Formatted metric or '--'
     */
    function metric(x, digits = 3) {
        if (x == null || !isFinite(x)) return '--';
        return x.toFixed(digits);
    }

    /**
     * Format percentage (multiplies decimal by 100)
     * @param {number} x - Decimal value (0.95 = 95%)
     * @param {number} digits - Decimal places (default 1)
     * @returns {string} Formatted percentage with % sign or '--'
     */
    function percent(x, digits = 1) {
        if (x == null || !isFinite(x)) return '--';
        return (x * 100).toFixed(digits) + '%';
    }

    /**
     * Format percentage when already in 0-100 range
     * @param {number} x - Percentage value (95 = 95%)
     * @param {number} digits - Decimal places (default 1)
     * @returns {string} Formatted percentage with % sign or '--'
     */
    function percentRaw(x, digits = 1) {
        if (x == null || !isFinite(x)) return '--';
        return x.toFixed(digits) + '%';
    }

    /**
     * Format integer with locale separators (1,234,567)
     * @param {number} x - Integer value
     * @returns {string} Formatted integer or '--'
     */
    function integer(x) {
        if (x == null || !isFinite(x)) return '--';
        return Math.round(x).toLocaleString();
    }

    /**
     * Format large numbers with K/M abbreviations
     * @param {number} num - Number to format
     * @returns {string} Abbreviated number (e.g., "1.5M")
     */
    function compact(num) {
        if (num == null || !isFinite(num)) return '--';
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return Math.round(num).toString();
    }

    /**
     * Format duration in seconds to human-readable
     * @param {number} seconds - Duration in seconds
     * @returns {string} Human-readable duration (e.g., "2h 34m")
     */
    function eta(seconds) {
        if (seconds == null || seconds <= 0 || !isFinite(seconds)) return '--';
        if (seconds < 60) return `${Math.round(seconds)}s`;
        if (seconds < 3600) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            return `${mins}m ${secs}s`;
        }
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    }

    /**
     * Format Date object to HH:MM:SS
     * @param {Date} date - Date object
     * @returns {string} Time string
     */
    function timestamp(date) {
        if (!date || !(date instanceof Date)) return '--:--:--';
        return date.toLocaleTimeString('en-US', { hour12: false });
    }

    /**
     * Format ISO date string to localized date
     * @param {string} isoString - ISO date string
     * @returns {string} Localized date string
     */
    function dateFromISO(isoString) {
        if (!isoString) return '--';
        try {
            const date = new Date(isoString);
            return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            });
        } catch {
            return '--';
        }
    }

    /**
     * Format file size in MB to human-readable
     * @param {number} sizeMb - Size in megabytes
     * @returns {string} Human-readable size (e.g., "1.5 GB")
     */
    function size(sizeMb) {
        if (sizeMb == null || !isFinite(sizeMb)) return '--';
        if (sizeMb >= 1024) return (sizeMb / 1024).toFixed(1) + ' GB';
        if (sizeMb >= 1) return sizeMb.toFixed(1) + ' MB';
        return (sizeMb * 1024).toFixed(0) + ' KB';
    }

    /**
     * Format VRAM usage
     * @param {number} gb - VRAM in gigabytes
     * @returns {string} Formatted VRAM (e.g., "12.5 GB")
     */
    function vram(gb) {
        if (gb == null || !isFinite(gb)) return '-- GB';
        return gb.toFixed(1) + ' GB';
    }

    /**
     * Format temperature
     * @param {number} celsius - Temperature in Celsius
     * @returns {string} Formatted temperature (e.g., "65°C")
     */
    function temp(celsius) {
        if (celsius == null || !isFinite(celsius)) return '--°C';
        return Math.round(celsius) + '°C';
    }

    /**
     * Format clarity (1/perplexity)
     * @param {number} perplexity - Perplexity value
     * @returns {string} Clarity value or '--'
     */
    function clarity(perplexity) {
        if (!perplexity || perplexity <= 0 || !isFinite(perplexity)) return '--';
        return (1 / perplexity).toFixed(3);
    }

    /**
     * Format training speed
     * @param {number} stepsPerSec - Steps per second
     * @returns {string} Formatted speed (e.g., "2.50/s")
     */
    function speed(stepsPerSec) {
        if (stepsPerSec == null || stepsPerSec <= 0 || !isFinite(stepsPerSec)) return '--';
        return stepsPerSec.toFixed(2) + '/s';
    }

    /**
     * Format step counter (current/total)
     * @param {number} current - Current step
     * @param {number} total - Total steps
     * @returns {string} Formatted step counter
     */
    function steps(current, total) {
        const c = current != null && isFinite(current) ? current.toLocaleString() : '--';
        const t = total != null && isFinite(total) ? total.toLocaleString() : '--';
        return `${c}/${t}`;
    }

    return {
        loss,
        metric,
        percent,
        percentRaw,
        integer,
        compact,
        eta,
        timestamp,
        dateFromISO,
        size,
        vram,
        temp,
        clarity,
        speed,
        steps
    };
})();

// Export for module systems if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Format;
}
