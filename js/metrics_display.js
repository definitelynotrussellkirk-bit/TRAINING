// ===== METRICS DISPLAY =====
// Handles rendering of prompts, responses, and example metrics

class MetricsDisplay {
    constructor() {
        this.systemPrompt = "You are a helpful AI assistant trained to solve syllogism puzzles.";
    }

    updatePromptDisplay(data) {
        const example = this.getActiveExample(data);

        // Step indicator
        const stepVal = example?.step ?? data.current_step;
        document.getElementById('exampleStep').textContent = (stepVal || '--').toLocaleString();

        // System prompt (extract from training data or use default)
        this.updateSystemPrompt(data, example);

        // User prompt
        this.updateUserPrompt(data, example);

        // Responses
        this.updateResponses(data, example);

        // Example metrics
        this.updateExampleMetrics(data, example);
    }

    getActiveExample(data) {
        const recents = data.recent_examples || [];
        // Use the latest example that has a model_output (inference preview); fallback to last recent
        const withModel = [...recents].reverse().find(ex => ex.model_output);
        if (withModel) return withModel;
        if (recents.length > 0) return recents[recents.length - 1];
        return null;
    }

    updateSystemPrompt(data, example) {
        const container = document.getElementById('systemPromptText');

        const systemPrompt = example?.system_prompt || data.current_system_prompt || this.systemPrompt;
        container.textContent = systemPrompt || 'N/A';
    }

    updateUserPrompt(data, example) {
        const container = document.getElementById('userPromptText');
        const prompt = example?.prompt || data.current_prompt;

        if (prompt) {
            // Truncate very long prompts for display
            const maxLength = 2000;
            const displayText = prompt.length > maxLength
                ? prompt.substring(0, maxLength) + '\n\n... (truncated, click "Expand" to see full prompt)'
                : prompt;

            container.textContent = displayText;
        } else {
            container.textContent = 'Waiting for training to start...';
        }
    }

    updateResponses(data, example) {
        const goldenContainer = document.getElementById('goldenAnswer');
        const modelContainer = document.getElementById('modelAnswer');
        const matchIndicator = document.getElementById('matchIndicator');

        const golden = example?.golden || data.golden_answer;
        const modelAnswer = example?.model_output || data.model_answer;
        const matches = example?.matches ?? data.answer_matches;

        // Golden answer
        if (golden) {
            goldenContainer.textContent = this.formatResponse(golden);
        } else {
            goldenContainer.textContent = 'N/A';
        }

        // Model answer
        if (modelAnswer) {
            modelContainer.textContent = this.formatResponse(modelAnswer);

            // Highlight <think> tags if present
            if (modelAnswer.includes('<think>')) {
                const highlighted = modelAnswer.replace(
                    /(<think>.*?<\/think>)/gs,
                    '🤔 $1'
                );
                modelContainer.textContent = highlighted;
            }
        } else {
            modelContainer.textContent = 'N/A';
        }

        // Match indicator
        if (matches !== null && matches !== undefined) {
            if (matches) {
                matchIndicator.textContent = '✅ MATCH';
                matchIndicator.className = 'match-indicator match';
            } else {
                matchIndicator.textContent = '❌ NO MATCH';
                matchIndicator.className = 'match-indicator no-match';
            }
        } else {
            matchIndicator.textContent = '--';
            matchIndicator.className = 'match-indicator';
        }
    }

    updateExampleMetrics(data, example) {
        // Loss
        const lossEl = document.getElementById('exampleLoss');
        const lossVal = example?.loss ?? data.loss;
        if (lossVal !== null && lossVal !== undefined) {
            lossEl.textContent = lossVal.toString();
        } else {
            lossEl.textContent = '--';
        }

        // Think tags
        const thinkEl = document.getElementById('exampleHasThink');
        const modelAnswer = example?.model_output || data.model_answer;
        if (modelAnswer) {
            const hasThink = modelAnswer.includes('<think>');
            thinkEl.textContent = hasThink ? '⚠️ Yes' : '✅ No';
            thinkEl.style.color = hasThink ? 'var(--accent-yellow)' : 'var(--accent-green)';
        } else {
            thinkEl.textContent = '--';
            thinkEl.style.color = 'var(--text-secondary)';
        }

        // Length (approximate token count)
        const lengthEl = document.getElementById('exampleLength');
        if (modelAnswer) {
            // Rough estimate: ~4 chars per token
            const approxTokens = Math.round(modelAnswer.length / 4);
            lengthEl.textContent = `~${approxTokens} tokens`;
        } else {
            lengthEl.textContent = '--';
        }
    }

    formatResponse(text) {
        // Pretty-print JSON if it looks like JSON
        try {
            const parsed = JSON.parse(text);
            return JSON.stringify(parsed, null, 2);
        } catch {
            return text;
        }
    }
}
