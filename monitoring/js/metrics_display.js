// ===== METRICS DISPLAY =====
// Handles rendering of prompts, responses, and example metrics

class MetricsDisplay {
    constructor() {
        this.systemPrompt = "You are a helpful AI assistant trained to solve syllogism puzzles.";
    }

    updatePromptDisplay(data) {
        const example = this.buildDisplayExample(data);

        // Step indicator
        const stepVal = example?.step;
        if (stepVal !== null && stepVal !== undefined) {
            document.getElementById('exampleStep').textContent = stepVal.toLocaleString();
        } else {
            document.getElementById('exampleStep').textContent = '--';
        }

        // System prompt / user prompt / responses all come from the same example snapshot
        this.updateSystemPrompt(example);
        this.updateUserPrompt(example);
        this.updateResponses(example);
        this.updateExampleMetrics(example);
    }

    buildDisplayExample(data) {
        const example = this.getActiveExample(data);
        if (example) {
            return {
                step: example.step ?? data.current_step ?? null,
                system_prompt: example.system_prompt ?? data.current_system_prompt ?? null,
                prompt: example.prompt ?? data.current_prompt ?? null,
                golden: example.golden ?? data.golden_answer ?? null,
                model_output: example.model_output ?? data.model_answer ?? null,
                matches: example.matches ?? data.answer_matches ?? null,
                loss: example.loss ?? data.loss ?? null
            };
        }

        return {
            step: data.current_step ?? null,
            system_prompt: data.current_system_prompt ?? null,
            prompt: data.current_prompt ?? null,
            golden: data.golden_answer ?? null,
            model_output: data.model_answer ?? null,
            matches: data.answer_matches ?? null,
            loss: data.loss ?? null
        };
    }

    getActiveExample(data) {
        const recents = data.recent_examples || [];
        // Use the latest example that has a model_output (inference preview); fallback to last recent
        const withModel = [...recents].reverse().find(ex => ex.model_output);
        if (withModel) return withModel;
        if (recents.length > 0) return recents[recents.length - 1];
        return null;
    }

    updateSystemPrompt(example) {
        const container = document.getElementById('systemPromptText');

        const systemPrompt = example?.system_prompt || this.systemPrompt;
        container.textContent = systemPrompt || 'N/A';
    }

    updateUserPrompt(example) {
        const container = document.getElementById('userPromptText');
        const prompt = example?.prompt;

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

    updateResponses(example) {
        const goldenContainer = document.getElementById('goldenAnswer');
        const modelContainer = document.getElementById('modelAnswer');
        const matchIndicator = document.getElementById('matchIndicator');

        const golden = example?.golden;
        const modelAnswer = example?.model_output;
        const matches = example?.matches;

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

    updateExampleMetrics(example) {
        // Loss
        const lossEl = document.getElementById('exampleLoss');
        const lossVal = example?.loss;
        if (lossVal !== null && lossVal !== undefined) {
            lossEl.textContent = lossVal.toString();
        } else {
            lossEl.textContent = '--';
        }

        // Think tags
        const thinkEl = document.getElementById('exampleHasThink');
        const modelAnswer = example?.model_output;
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
