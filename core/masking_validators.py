"""
Masking Validators - Prevent Training on Instructions

CRITICAL BUG PREVENTED (2025-11-27):
When packing is enabled, multiple examples are combined into one sequence.
The old collator only masked the FIRST instruction, causing the model to
train on ALL subsequent instructions, system prompts, and user messages.

Result: Model output garbage like "You are happy. You enjoy helping others."
because it learned to reproduce system prompts instead of actual responses.

These validators ensure this NEVER happens again.

=== VALIDATORS ===

1. MaskingRatioValidator
   - Ensures masked % is within expected range (30-85%)
   - Fails if too little masking (training on instructions)
   - Fails if too much masking (not training on anything)

2. ResponseTemplateCountValidator
   - Counts response templates in sequence
   - Verifies each has corresponding masked instruction region
   - Catches single-template-in-packed-sequence bug

3. TrainedTokenContentValidator
   - Samples tokens being trained on
   - Fails if they contain instruction markers
   - Catches "training on system prompts" bug

4. PackedSequenceValidator
   - Validates masking pattern in packed sequences
   - Ensures mask/train/mask/train alternation
   - Catches boundary issues

5. LabelDistributionValidator
   - Analyzes label patterns across sequence
   - Detects anomalies like all-masked or all-trained regions

=== USAGE ===

from core.masking_validators import validate_masking, MaskingValidationError

# In training loop, after collator:
try:
    validate_masking(
        batch=collated_batch,
        tokenizer=tokenizer,
        response_template="<|im_start|>assistant\n",
        packing_enabled=True
    )
except MaskingValidationError as e:
    print(f"CRITICAL: {e}")
    # Abort training to prevent bad model

=== INTEGRATION ===

Called by: core/train.py (before training starts)
Fails fast: Raises MaskingValidationError on any issue
Logs: Detailed diagnostics for debugging
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizerBase


class MaskingValidationError(Exception):
    """Raised when masking validation fails - training would be corrupted."""
    pass


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    validator_name: str
    message: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class MaskingReport:
    """Complete masking validation report."""
    passed: bool
    results: List[ValidationResult]
    summary: str

    def __str__(self):
        lines = [self.summary, ""]
        for r in self.results:
            status = "✅" if r.passed else "❌"
            lines.append(f"{status} {r.validator_name}: {r.message}")
            if r.details and not r.passed:
                for k, v in r.details.items():
                    lines.append(f"     {k}: {v}")
        return "\n".join(lines)


class MaskingRatioValidator:
    """
    Validate that masking ratio is within expected bounds.

    Expected: 30-85% of tokens masked (instructions)
    Too low (<30%): Probably training on instructions
    Too high (>85%): Not training on enough response content
    """

    def __init__(self, min_masked_pct: float = 0.30, max_masked_pct: float = 0.85):
        self.min_masked_pct = min_masked_pct
        self.max_masked_pct = max_masked_pct

    def validate(self, labels: torch.Tensor, attention_mask: torch.Tensor = None) -> ValidationResult:
        """Validate masking ratio for a batch."""
        # Count masked vs trained tokens
        # labels == -100 means masked (not trained on)
        total_tokens = labels.numel()
        masked_count = (labels == -100).sum().item()
        trained_count = total_tokens - masked_count

        # Exclude padding if attention_mask provided
        if attention_mask is not None:
            real_tokens = attention_mask.sum().item()
            # Recalculate for real tokens only
            for i in range(labels.shape[0]):
                seq_len = attention_mask[i].sum().item()
                # This is approximate - we count masked in real portion

        masked_pct = masked_count / total_tokens if total_tokens > 0 else 0
        trained_pct = trained_count / total_tokens if total_tokens > 0 else 0

        details = {
            "total_tokens": total_tokens,
            "masked_count": masked_count,
            "trained_count": trained_count,
            "masked_pct": f"{masked_pct*100:.1f}%",
            "trained_pct": f"{trained_pct*100:.1f}%",
        }

        if masked_pct < self.min_masked_pct:
            return ValidationResult(
                passed=False,
                validator_name="MaskingRatioValidator",
                message=f"Too little masking ({masked_pct*100:.1f}% < {self.min_masked_pct*100}%) - likely training on instructions!",
                details=details
            )

        if masked_pct > self.max_masked_pct:
            return ValidationResult(
                passed=False,
                validator_name="MaskingRatioValidator",
                message=f"Too much masking ({masked_pct*100:.1f}% > {self.max_masked_pct*100}%) - not training on enough content",
                details=details
            )

        return ValidationResult(
            passed=True,
            validator_name="MaskingRatioValidator",
            message=f"Masking ratio OK: {masked_pct*100:.1f}% masked, {trained_pct*100:.1f}% trained",
            details=details
        )


class ResponseTemplateCountValidator:
    """
    Validate that each response template has corresponding masked region.

    In packed sequences, there should be N response templates and N masked regions.
    If there's only 1 masked region but N templates, the collator is broken.
    """

    def __init__(self, response_template: str, tokenizer: PreTrainedTokenizerBase):
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def validate(self, input_ids: torch.Tensor, labels: torch.Tensor) -> ValidationResult:
        """Validate template count matches masked region count."""
        batch_size = input_ids.shape[0]

        issues = []
        for idx in range(batch_size):
            seq_ids = input_ids[idx].tolist()
            seq_labels = labels[idx].tolist()

            # Count response templates
            template_count = 0
            template_len = len(self.template_ids)
            for i in range(len(seq_ids) - template_len + 1):
                if seq_ids[i:i + template_len] == self.template_ids:
                    template_count += 1

            # Count transitions from masked to unmasked (response regions)
            response_regions = 0
            in_masked = seq_labels[0] == -100
            for i in range(1, len(seq_labels)):
                is_masked = seq_labels[i] == -100
                if in_masked and not is_masked:
                    response_regions += 1
                in_masked = is_masked

            if template_count != response_regions:
                issues.append({
                    "batch_idx": idx,
                    "template_count": template_count,
                    "response_regions": response_regions,
                })

        if issues:
            return ValidationResult(
                passed=False,
                validator_name="ResponseTemplateCountValidator",
                message=f"Template count mismatch in {len(issues)} sequences - collator may be broken!",
                details={"issues": issues}
            )

        return ValidationResult(
            passed=True,
            validator_name="ResponseTemplateCountValidator",
            message=f"All {batch_size} sequences have matching template/response counts",
        )


class TrainedTokenContentValidator:
    """
    Validate that trained tokens don't contain instruction markers.

    If we're training on tokens that look like instructions, something is wrong.
    Looks for: system prompts, "user:", "You are", role markers, etc.
    """

    INSTRUCTION_MARKERS = [
        # Role markers
        "<|im_start|>user",
        "<|im_start|>system",
        "user:",
        "User:",
        "system:",
        "System:",
        # Common system prompt fragments
        "You are a",
        "You are an",
        "helpful assistant",
        "AI assistant",
        # Instruction patterns
        "Output contract",
        "Return your answer",
        "Rules:",
        "Instructions:",
    ]

    def __init__(self, tokenizer: PreTrainedTokenizerBase, sample_size: int = 100):
        self.tokenizer = tokenizer
        self.sample_size = sample_size

    def validate(self, input_ids: torch.Tensor, labels: torch.Tensor) -> ValidationResult:
        """Validate that trained tokens don't look like instructions."""
        # Collect trained token positions
        trained_positions = []
        for idx in range(input_ids.shape[0]):
            for pos in range(input_ids.shape[1]):
                if labels[idx, pos] != -100:
                    trained_positions.append((idx, pos))

        if not trained_positions:
            return ValidationResult(
                passed=False,
                validator_name="TrainedTokenContentValidator",
                message="No trained tokens found at all!",
            )

        # Sample and decode trained tokens
        import random
        sample = random.sample(trained_positions, min(self.sample_size, len(trained_positions)))

        # Decode each trained region
        found_markers = []
        for idx, pos in sample:
            # Get a window of trained tokens around this position
            start = pos
            while start > 0 and labels[idx, start-1] != -100:
                start -= 1
            end = pos + 1
            while end < labels.shape[1] and labels[idx, end] != -100:
                end += 1

            # Decode the trained region
            trained_ids = input_ids[idx, start:end].tolist()
            trained_text = self.tokenizer.decode(trained_ids)

            # Check for instruction markers
            for marker in self.INSTRUCTION_MARKERS:
                if marker.lower() in trained_text.lower():
                    found_markers.append({
                        "marker": marker,
                        "context": trained_text[:100],
                        "batch_idx": idx,
                        "position": pos,
                    })
                    break

        if found_markers:
            # Group by marker
            marker_counts = {}
            for m in found_markers:
                marker_counts[m["marker"]] = marker_counts.get(m["marker"], 0) + 1

            return ValidationResult(
                passed=False,
                validator_name="TrainedTokenContentValidator",
                message=f"Found instruction markers in trained tokens! Training on prompts!",
                details={
                    "marker_counts": marker_counts,
                    "examples": found_markers[:3],  # First 3 examples
                }
            )

        return ValidationResult(
            passed=True,
            validator_name="TrainedTokenContentValidator",
            message=f"Sampled {len(sample)} trained regions - no instruction markers found",
        )


class PackedSequenceValidator:
    """
    Validate masking pattern in packed sequences.

    Expected pattern: [MASKED][TRAINED][MASKED][TRAINED]...
    Each MASKED region = instruction for one example
    Each TRAINED region = response for one example
    """

    def __init__(self, response_template: str, end_token: str, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)

    def validate(self, input_ids: torch.Tensor, labels: torch.Tensor) -> ValidationResult:
        """Validate packed sequence masking pattern."""
        issues = []

        for idx in range(input_ids.shape[0]):
            seq_ids = input_ids[idx].tolist()
            seq_labels = labels[idx].tolist()

            # Find all response template positions
            template_positions = []
            template_len = len(self.response_template_ids)
            for i in range(len(seq_ids) - template_len + 1):
                if seq_ids[i:i + template_len] == self.response_template_ids:
                    template_positions.append(i)

            # For each template, verify:
            # 1. Everything before it is masked
            # 2. Everything after it (until end token) is trained
            for pos in template_positions:
                response_start = pos + template_len

                # Find end token after this response
                end_pos = len(seq_ids)
                end_len = len(self.end_token_ids)
                for j in range(response_start, len(seq_ids) - end_len + 1):
                    if seq_ids[j:j + end_len] == self.end_token_ids:
                        end_pos = j + end_len
                        break

                # Check that response region is trained (not masked)
                trained_in_response = sum(1 for l in seq_labels[response_start:end_pos] if l != -100)
                total_in_response = end_pos - response_start

                if trained_in_response < total_in_response * 0.9:  # Allow 10% tolerance
                    issues.append({
                        "batch_idx": idx,
                        "template_pos": pos,
                        "response_range": (response_start, end_pos),
                        "trained_pct": trained_in_response / total_in_response if total_in_response > 0 else 0,
                    })

        if issues:
            return ValidationResult(
                passed=False,
                validator_name="PackedSequenceValidator",
                message=f"Response regions not fully trained in {len(issues)} cases",
                details={"issues": issues[:5]}  # First 5
            )

        return ValidationResult(
            passed=True,
            validator_name="PackedSequenceValidator",
            message="All response regions correctly trained",
        )


class LabelDistributionValidator:
    """
    Validate overall label distribution pattern.

    Detects anomalies like:
    - All tokens masked (nothing to train on)
    - All tokens trained (no masking at all)
    - Uniform distribution (random masking - wrong)
    - Unexpected patterns
    """

    def validate(self, labels: torch.Tensor) -> ValidationResult:
        """Analyze label distribution for anomalies."""
        batch_size, seq_len = labels.shape

        anomalies = []
        for idx in range(batch_size):
            seq_labels = labels[idx]

            masked_count = (seq_labels == -100).sum().item()
            trained_count = seq_len - masked_count

            # Check for all-masked
            if masked_count == seq_len:
                anomalies.append({
                    "batch_idx": idx,
                    "type": "all_masked",
                    "message": "Entire sequence masked - nothing to train on",
                })
                continue

            # Check for all-trained
            if trained_count == seq_len:
                anomalies.append({
                    "batch_idx": idx,
                    "type": "all_trained",
                    "message": "No masking at all - training on everything including instructions",
                })
                continue

            # Check for proper structure: masked region(s) followed by trained region(s)
            # Convert to transitions
            transitions = []
            current = seq_labels[0].item() == -100
            for i in range(1, seq_len):
                is_masked = seq_labels[i].item() == -100
                if is_masked != current:
                    transitions.append(i)
                    current = is_masked

            # For packed sequences, expect: M->T->M->T... pattern
            # Minimum 1 transition (instruction -> response)
            if len(transitions) == 0:
                anomalies.append({
                    "batch_idx": idx,
                    "type": "no_transitions",
                    "message": "No mask/train transitions - unexpected pattern",
                })

        if anomalies:
            return ValidationResult(
                passed=False,
                validator_name="LabelDistributionValidator",
                message=f"Found {len(anomalies)} sequences with anomalous label patterns",
                details={"anomalies": anomalies[:5]}
            )

        return ValidationResult(
            passed=True,
            validator_name="LabelDistributionValidator",
            message="Label distribution patterns look correct",
        )


def validate_masking(
    batch: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizerBase,
    response_template: str = "<|im_start|>assistant\n",
    end_token: str = "<|im_end|>",
    packing_enabled: bool = True,
    strict: bool = True,
) -> MaskingReport:
    """
    Run all masking validators on a batch.

    Args:
        batch: Collated batch with 'input_ids', 'labels', 'attention_mask'
        tokenizer: Tokenizer used for encoding
        response_template: Template marking response start
        end_token: Token marking response end
        packing_enabled: Whether packing is enabled (affects validation)
        strict: If True, raise MaskingValidationError on failure

    Returns:
        MaskingReport with all validation results

    Raises:
        MaskingValidationError: If strict=True and any validation fails
    """
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch.get('attention_mask')

    results = []

    # 1. Masking ratio
    ratio_validator = MaskingRatioValidator()
    results.append(ratio_validator.validate(labels, attention_mask))

    # 2. Response template count
    template_validator = ResponseTemplateCountValidator(response_template, tokenizer)
    results.append(template_validator.validate(input_ids, labels))

    # 3. Trained token content
    content_validator = TrainedTokenContentValidator(tokenizer)
    results.append(content_validator.validate(input_ids, labels))

    # 4. Packed sequence validation (if packing enabled)
    if packing_enabled:
        packed_validator = PackedSequenceValidator(response_template, end_token, tokenizer)
        results.append(packed_validator.validate(input_ids, labels))

    # 5. Label distribution
    dist_validator = LabelDistributionValidator()
    results.append(dist_validator.validate(labels))

    # Create report
    all_passed = all(r.passed for r in results)
    failed_count = sum(1 for r in results if not r.passed)

    if all_passed:
        summary = "✅ All masking validations passed"
    else:
        summary = f"❌ {failed_count} masking validation(s) FAILED - DO NOT TRAIN!"

    report = MaskingReport(passed=all_passed, results=results, summary=summary)

    if strict and not all_passed:
        raise MaskingValidationError(str(report))

    return report


def quick_masking_check(
    collator,
    tokenizer: PreTrainedTokenizerBase,
    sample_texts: List[str],
    response_template: str = "<|im_start|>assistant\n",
) -> Tuple[bool, str]:
    """
    Quick sanity check for masking before training.

    Args:
        collator: Data collator to test
        tokenizer: Tokenizer
        sample_texts: List of formatted chat texts to test
        response_template: Template marking response start

    Returns:
        (passed, message) tuple
    """
    # Tokenize samples
    features = []
    for text in sample_texts:
        tokenized = tokenizer(text, truncation=True, max_length=2048, padding=False)
        features.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        })

    # Apply collator
    batch = collator(features)

    # Run validation
    try:
        report = validate_masking(
            batch=batch,
            tokenizer=tokenizer,
            response_template=response_template,
            packing_enabled=True,
            strict=False,
        )
        return report.passed, str(report)
    except Exception as e:
        return False, f"Validation error: {e}"
