/**
 * Formatting Utilities
 * Functions for formatting numbers, time, percentages, and HTML
 */

/**
 * Format seconds into human-readable time
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time (e.g., "2h 34m")
 */
export function formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return '--';

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

/**
 * Format seconds into HH:MM:SS format
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time (e.g., "02:34:15")
 */
export function formatTimeHHMMSS(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return '00:00:00';

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    return [hours, minutes, secs]
        .map(v => v.toString().padStart(2, '0'))
        .join(':');
}

/**
 * Escape HTML to prevent XSS
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
export function escapeHTML(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Format a number with specified decimal places
 * @param {number} num - Number to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted number
 */
export function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || !Number.isFinite(num)) {
        return '--';
    }
    return num.toFixed(decimals);
}

/**
 * Format a number as a percentage
 * @param {number} value - Value (0-100 or 0-1)
 * @param {number} decimals - Decimal places
 * @param {boolean} isDecimal - If true, value is 0-1, else 0-100
 * @returns {string} Formatted percentage
 */
export function formatPercent(value, decimals = 1, isDecimal = false) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
        return '--%';
    }
    const percent = isDecimal ? value * 100 : value;
    return `${percent.toFixed(decimals)}%`;
}

/**
 * Format large numbers with commas
 * @param {number} num - Number to format
 * @returns {string} Formatted number with commas
 */
export function formatLargeNumber(num) {
    if (num === null || num === undefined || !Number.isFinite(num)) {
        return '--';
    }
    return num.toLocaleString();
}

/**
 * Format bytes to human-readable size
 * @param {number} bytes - Bytes
 * @returns {string} Formatted size (e.g., "1.5 GB")
 */
export function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return '--';

    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }

    return `${size.toFixed(unitIndex === 0 ? 0 : 2)} ${units[unitIndex]}`;
}

/**
 * Format a loss value (preserves full precision)
 * @param {number} loss - Loss value
 * @returns {string} Formatted loss
 */
export function formatLoss(loss) {
    if (loss === null || loss === undefined || !Number.isFinite(loss)) {
        return '--';
    }
    // Use full precision for losses
    return loss.toString();
}

/**
 * Format JSON with syntax highlighting (simple version)
 * @param {string} text - Text that might contain JSON
 * @returns {string} Formatted text
 */
export function formatMaybeJson(text) {
    if (!text) return '';
    const trimmed = text.trim();

    const firstBrace = trimmed.indexOf('{');
    const firstBracket = trimmed.indexOf('[');
    const firstJsonIdx = Math.min(
        firstBrace >= 0 ? firstBrace : Infinity,
        firstBracket >= 0 ? firstBracket : Infinity
    );

    if (firstJsonIdx === Infinity) return escapeHTML(text);

    try {
        const prefix = trimmed.substring(0, firstJsonIdx);
        const jsonPart = trimmed.substring(firstJsonIdx);
        const parsed = JSON.parse(jsonPart);
        const formatted = JSON.stringify(parsed, null, 2);
        return escapeHTML(prefix) + formatted;
    } catch {
        return escapeHTML(text);
    }
}

/**
 * Truncate text with ellipsis
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
export function truncate(text, maxLength = 100) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

/**
 * Format a step count with ordinal suffix
 * @param {number} step - Step number
 * @returns {string} Formatted step (e.g., "1st", "2nd", "3rd", "4th")
 */
export function formatStep(step) {
    if (!Number.isFinite(step)) return '--';

    const suffix = ['th', 'st', 'nd', 'rd'];
    const v = step % 100;
    return step + (suffix[(v - 20) % 10] || suffix[v] || suffix[0]);
}

/**
 * Format duration in ms to human readable
 * @param {number} ms - Duration in milliseconds
 * @returns {string} Formatted duration
 */
export function formatDuration(ms) {
    if (!Number.isFinite(ms) || ms < 0) return '--';
    return formatTime(ms / 1000);
}

/**
 * Format number in scientific notation
 * @param {number} num - Number to format
 * @returns {string} Formatted number (e.g., "1.23e-4")
 */
export function formatScientific(num) {
    if (num === null || num === undefined || !Number.isFinite(num)) {
        return '--';
    }
    return num.toExponential(2);
}
