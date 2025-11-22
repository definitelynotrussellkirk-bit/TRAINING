/**
 * Examples Panel Component
 * Displays current training example and recent examples
 */

import { formatNumber } from '../utils/formatters.js';
import { escapeHTML } from '../utils/formatters.js';

export class ExamplesPanel {
    constructor() {
        this.elements = {
            // Current example (match actual HTML IDs)
            systemPrompt: document.getElementById('systemPrompt'),
            prompt: document.getElementById('prompt'),
            golden: document.getElementById('golden'),
            modelAnswer: document.getElementById('modelAnswer'),

            // Match indicator
            matchBadge: document.getElementById('matchBadge'),

            // Recent examples
            recentExamples: document.getElementById('recentExamples')
        };
    }

    /**
     * Update all example displays
     */
    update(data) {
        if (!data) return;

        this.updateCurrentExample(data);
        this.updateRecentExamples(data.recentExamples);
    }

    /**
     * Update current training example
     */
    updateCurrentExample(data) {
        // System prompt
        if (this.elements.systemPrompt && data.currentSystemPrompt) {
            this.elements.systemPrompt.textContent = data.currentSystemPrompt;
        }

        // User prompt
        if (this.elements.prompt && data.currentPrompt) {
            this.elements.prompt.textContent = data.currentPrompt;
        }

        // Golden answer (expected)
        if (this.elements.golden && data.goldenAnswer) {
            this.elements.golden.textContent = data.goldenAnswer;
        }

        // Model answer (actual)
        if (this.elements.modelAnswer && data.modelAnswer) {
            this.elements.modelAnswer.textContent = data.modelAnswer;
        }

        // Match badge
        if (this.elements.matchBadge && data.answerMatches !== null) {
            const matched = data.answerMatches;
            this.elements.matchBadge.innerHTML = matched ?
                ' <span style="color: #00ff88; font-weight: bold; margin-left: 10px;">✅ MATCH</span>' :
                ' <span style="color: #ff4444; font-weight: bold; margin-left: 10px;">❌ MISMATCH</span>';
        }
    }

    /**
     * Update recent examples list
     */
    updateRecentExamples(examples) {
        if (!this.elements.recentExamples || !examples || !Array.isArray(examples)) {
            return;
        }

        if (examples.length === 0) {
            this.elements.recentExamples.innerHTML = '<p style="color: #888;">No model evaluations yet (waiting for next eval step)...</p>';
            return;
        }

        // Build HTML for recent examples
        const html = examples.slice(0, 10).map(ex => {
            const matchIcon = ex.matches ? '✅' : '❌';
            const matchColor = ex.matches ? '#00ff88' : '#ff4444';
            const lossColor = ex.loss < 0.5 ? '#00ff88' : ex.loss < 1.0 ? '#ffaa00' : '#ff4444';

            return `
                <div style="
                    background: rgba(26, 39, 59, 0.3);
                    padding: 8px 12px;
                    margin: 4px 0;
                    border-radius: 5px;
                    border-left: 3px solid ${matchColor};
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 0.9em;
                ">
                    <div style="display: flex; gap: 15px; align-items: center;">
                        <span style="color: #888;">Step ${ex.step || '-'}</span>
                        <span style="color: ${matchColor};">${matchIcon}</span>
                        <span style="color: ${lossColor};">Loss: ${formatNumber(ex.loss, 3)}</span>
                    </div>
                    ${ex.current_file ? `<span style="color: #888; font-size: 0.8em;">${this.truncateFilename(ex.current_file)}</span>` : ''}
                </div>
            `;
        }).join('');

        this.elements.recentExamples.innerHTML = html;
    }

    /**
     * Truncate long filenames
     */
    truncateFilename(filename) {
        if (!filename) return '';
        if (filename.length <= 30) return filename;
        return '...' + filename.slice(-27);
    }
}

export default ExamplesPanel;
