/**
 * Shared Context Helpers
 *
 * Common data fetching patterns used across multiple pages.
 * DRYs up campaign/hero selection dropdowns and model info loading.
 */

/**
 * Fetch campaign context (active campaign + all campaigns by hero).
 *
 * @returns {Promise<{active: Object|null, campaignsByHero: Object}>}
 */
window.fetchCampaignContext = async function() {
    const resp = await fetch('/api/campaigns');
    if (!resp.ok) throw new Error('Failed to load campaigns');
    const data = await resp.json();
    return {
        active: data.active || null,
        campaignsByHero: data.campaigns || {},
    };
};

/**
 * Fetch merged hero + model info (single endpoint).
 *
 * Returns all info needed for settings/status pages:
 * - hero_id, hero_name
 * - model_name, architecture, context_length, vocab_size
 * - campaign_id, campaign_path, campaign_name
 * - peak_skill_levels, peak_metrics, journey_summary
 *
 * @returns {Promise<Object>}
 */
window.fetchHeroModelInfo = async function() {
    const resp = await fetch('/api/hero-model-info');
    if (!resp.ok) throw new Error('Failed to load hero model info');
    return await resp.json();
};

/**
 * Fetch active campaign with peak tracking.
 *
 * @returns {Promise<Object|null>}
 */
window.fetchActiveCampaign = async function() {
    const resp = await fetch('/api/campaigns/active');
    if (!resp.ok) throw new Error('Failed to load active campaign');
    return await resp.json();
};

/**
 * Build hero dropdown options from campaign context.
 *
 * @param {Object} campaignsByHero - Map of hero_id -> campaigns[]
 * @param {string} selectedHeroId - Currently selected hero
 * @returns {string} HTML options string
 */
window.buildHeroOptions = function(campaignsByHero, selectedHeroId) {
    const heroes = Object.keys(campaignsByHero || {});
    if (heroes.length === 0) {
        return '<option value="">No heroes available</option>';
    }
    return heroes.map(heroId => {
        const displayName = heroId
            .replace(/-/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
        const selected = heroId === selectedHeroId ? 'selected' : '';
        return `<option value="${heroId}" ${selected}>${displayName}</option>`;
    }).join('');
};

/**
 * Build campaign dropdown options from campaign context.
 *
 * @param {Object} campaignsByHero - Map of hero_id -> campaigns[]
 * @param {string} selectedCampaignId - Currently selected campaign
 * @param {boolean} includeArchived - Include archived campaigns (default: false)
 * @returns {string} HTML options string
 */
window.buildCampaignOptions = function(campaignsByHero, selectedCampaignId, includeArchived = false) {
    const campaigns = [];
    for (const [heroId, heroCampaigns] of Object.entries(campaignsByHero || {})) {
        for (const c of heroCampaigns) {
            if (includeArchived || c.status !== 'archived') {
                campaigns.push({ ...c, hero_id: heroId });
            }
        }
    }

    if (campaigns.length === 0) {
        return '<option value="">No campaigns available</option>';
    }

    return campaigns.map(c => {
        const selected = c.id === selectedCampaignId ? 'selected' : '';
        return `<option value="${c.id}" data-hero="${c.hero_id}" ${selected}>${c.name}</option>`;
    }).join('');
};

/**
 * Initialize hero/campaign dropdowns with auto-sync.
 *
 * When campaign changes, automatically updates hero dropdown to match.
 *
 * @param {HTMLSelectElement} heroSelect - Hero dropdown element
 * @param {HTMLSelectElement} campaignSelect - Campaign dropdown element
 * @param {Object} context - Campaign context from fetchCampaignContext()
 * @param {Function} onSelectionChange - Callback when selection changes
 */
window.initCampaignDropdowns = async function(heroSelect, campaignSelect, context, onSelectionChange) {
    const { active, campaignsByHero } = context;

    // Determine initial selection
    let heroId = active?.hero_id || Object.keys(campaignsByHero)[0] || '';
    let campaignId = active?.id || '';

    // Build dropdowns
    heroSelect.innerHTML = window.buildHeroOptions(campaignsByHero, heroId);
    campaignSelect.innerHTML = window.buildCampaignOptions(campaignsByHero, campaignId);

    // Sync hero when campaign changes
    campaignSelect.addEventListener('change', () => {
        const selectedOption = campaignSelect.selectedOptions[0];
        if (selectedOption && selectedOption.dataset.hero) {
            heroSelect.value = selectedOption.dataset.hero;
        }
        if (onSelectionChange) {
            onSelectionChange({
                heroId: heroSelect.value,
                campaignId: campaignSelect.value,
            });
        }
    });

    // Fire callback on hero change too
    heroSelect.addEventListener('change', () => {
        if (onSelectionChange) {
            onSelectionChange({
                heroId: heroSelect.value,
                campaignId: campaignSelect.value,
            });
        }
    });

    return { heroId, campaignId };
};

/**
 * Set text content of element by selector, with fallback.
 *
 * @param {string} selector - CSS selector
 * @param {string} text - Text to set
 * @param {string} fallback - Fallback if text is null/undefined (default: '--')
 */
window.setText = function(selector, text, fallback = '--') {
    const el = document.querySelector(selector);
    if (el) {
        el.textContent = text ?? fallback;
    }
};

/**
 * Show empty state message in container.
 *
 * @param {string|HTMLElement} container - Selector or element
 * @param {string} message - Message to show
 * @param {string} icon - Optional icon (default: info circle)
 */
window.showEmptyState = function(container, message, icon = 'ℹ️') {
    const el = typeof container === 'string' ? document.querySelector(container) : container;
    if (el) {
        el.innerHTML = `
            <div class="empty-state" style="text-align: center; padding: 2rem; color: #888;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">${icon}</div>
                <div>${message}</div>
            </div>
        `;
    }
};

/**
 * Format number with locale-aware thousands separator.
 *
 * @param {number} num - Number to format
 * @returns {string} Formatted string
 */
window.formatNumber = function(num) {
    if (num === null || num === undefined) return '--';
    return num.toLocaleString();
};

/**
 * Format a metric value for display.
 *
 * @param {string} name - Metric name
 * @param {number} value - Metric value
 * @returns {string} Formatted string
 */
window.formatMetric = function(name, value) {
    if (value === null || value === undefined) return '--';

    if (name.includes('loss')) {
        return value.toFixed(4);
    } else if (name.includes('accuracy') || name.includes('acc')) {
        return (value * 100).toFixed(1) + '%';
    } else {
        return value.toLocaleString();
    }
};

console.log('Context helpers loaded');
