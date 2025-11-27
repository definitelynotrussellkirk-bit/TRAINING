"""
Quest Quality Gate - Inspects quest batches before posting to the Quest Board.

The Quality Gate ensures:
- Quest format is valid (proper message structure)
- Quest length is appropriate (not too short/long)
- Quest diversity is sufficient (no repetitive content)
- Quest balance is correct (user/assistant turns)
- Quest content is clean (no gibberish or errors)

RPG Flavor:
    The Quality Gate is staffed by seasoned inspectors who examine
    each quest scroll before it reaches the Quest Board. They reject
    poorly-formed quests that would confuse or harm the hero.
"""

import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence
import json

from guild.dispatch.types import QuestVerdict, QualityReport


logger = logging.getLogger(__name__)


class QuestQualityGate:
    """
    Inspects quest batches for quality before they reach the Quest Board.

    Runs a series of checks and produces a QualityReport with verdict.

    Usage:
        gate = QuestQualityGate()
        report = gate.inspect(quest_batch)

        if report.verdict == QuestVerdict.APPROVED:
            # Safe to post to quest board
            dispatcher.post_to_board(quest_batch)
    """

    def __init__(
        self,
        min_tokens: int = 10,
        max_tokens: int = 4096,
        min_diversity: float = 0.70,
        max_issue_rate: float = 0.05,
    ):
        """
        Initialize the Quality Gate.

        Args:
            min_tokens: Minimum estimated tokens per quest
            max_tokens: Maximum estimated tokens per quest
            min_diversity: Minimum unique response ratio (0-1)
            max_issue_rate: Maximum acceptable content issue rate (0-1)
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_diversity = min_diversity
        self.max_issue_rate = max_issue_rate

        # Track inspection results
        self._last_report: QualityReport | None = None

    def inspect(self, quests: Sequence[dict[str, Any]]) -> QualityReport:
        """
        Inspect a batch of quests for quality.

        Runs all quality checks and returns a comprehensive report.

        Args:
            quests: Sequence of quest dicts with 'messages' key

        Returns:
            QualityReport with verdict and detailed results
        """
        checks = [
            ("scroll_format", self._check_scroll_format),
            ("scroll_length", self._check_scroll_length),
            ("quest_diversity", self._check_quest_diversity),
            ("turn_balance", self._check_turn_balance),
            ("content_quality", self._check_content_quality),
        ]

        all_passed = True
        check_results = {}

        for check_name, check_func in checks:
            passed, details = check_func(quests)
            check_results[check_name] = {
                "passed": passed,
                "details": details
            }

            if not passed:
                all_passed = False
                logger.warning(f"Quality check FAILED: {check_name} - {details.get('reason', 'Unknown')}")
            else:
                logger.debug(f"Quality check PASSED: {check_name}")

        # Determine verdict
        failed_checks = [name for name, r in check_results.items() if not r["passed"]]
        passed_count = len(checks) - len(failed_checks)

        if not failed_checks:
            verdict = QuestVerdict.APPROVED
            recommendation = "APPROVED: Quest quality is excellent. Safe to post to Quest Board."
        elif len(failed_checks) <= 2:
            verdict = QuestVerdict.CONDITIONAL
            recommendation = f"CONDITIONAL: {', '.join(failed_checks)} checks failed. Review before posting."
        else:
            verdict = QuestVerdict.REJECTED
            recommendation = f"REJECTED: Multiple quality issues ({', '.join(failed_checks)}). Do not post."

        report = QualityReport(
            verdict=verdict,
            total_checks=len(checks),
            passed_checks=passed_count,
            failed_checks=len(failed_checks),
            check_results=check_results,
            recommendation=recommendation,
        )

        self._last_report = report
        return report

    def _check_scroll_format(self, quests: Sequence[dict]) -> tuple[bool, dict]:
        """
        Check if quest scrolls have correct conversation format.

        Validates:
        - Each quest is a dict with 'messages' key
        - Messages is a list with at least 2 turns
        - Has both user and assistant messages
        """
        if not quests:
            return False, {"reason": "Empty quest batch", "count": 0}

        valid_count = 0
        errors = []
        sampled = min(100, len(quests))

        for i, quest in enumerate(quests[:sampled]):
            if not isinstance(quest, dict):
                errors.append(f"Quest {i}: Not a dict")
                continue

            if "messages" not in quest:
                errors.append(f"Quest {i}: Missing 'messages' key")
                continue

            messages = quest["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"Quest {i}: Invalid messages format")
                continue

            # Check for user/assistant pattern
            has_user = any(m.get("role") == "user" for m in messages)
            has_assistant = any(m.get("role") == "assistant" for m in messages)

            if not (has_user and has_assistant):
                errors.append(f"Quest {i}: Missing user or assistant message")
                continue

            valid_count += 1

        pass_rate = valid_count / sampled if sampled > 0 else 0
        passed = pass_rate >= 0.95  # 95% must be valid

        return passed, {
            "valid_count": valid_count,
            "sampled": sampled,
            "pass_rate": pass_rate,
            "errors": errors[:5],  # First 5 errors
            "total_quests": len(quests),
            "reason": None if passed else f"Only {pass_rate:.0%} valid format"
        }

    def _check_scroll_length(self, quests: Sequence[dict]) -> tuple[bool, dict]:
        """
        Check if quest scroll lengths are within bounds.

        Estimates tokens as chars/4 and checks against min/max.
        """
        lengths = []
        too_short = 0
        too_long = 0
        sampled = min(1000, len(quests))

        for quest in quests[:sampled]:
            if "messages" not in quest:
                continue

            # Estimate total tokens (rough: 4 chars per token)
            total_chars = sum(len(m.get("content", "")) for m in quest["messages"])
            estimated_tokens = total_chars // 4

            lengths.append(estimated_tokens)

            if estimated_tokens < self.min_tokens:
                too_short += 1
            elif estimated_tokens > self.max_tokens:
                too_long += 1

        if not lengths:
            return False, {"reason": "No valid quests to measure"}

        passed = (too_short / len(lengths) < 0.05 and  # <5% too short
                  too_long / len(lengths) < 0.05)      # <5% too long

        return passed, {
            "mean_tokens": statistics.mean(lengths),
            "median_tokens": statistics.median(lengths),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "too_short": too_short,
            "too_long": too_long,
            "sampled": len(lengths),
            "reason": None if passed else f"Too short: {too_short}, too long: {too_long}"
        }

    def _check_quest_diversity(self, quests: Sequence[dict]) -> tuple[bool, dict]:
        """
        Check diversity of quest responses.

        Ensures we're not generating repetitive content.
        """
        assistant_responses = []
        sampled = min(1000, len(quests))

        for quest in quests[:sampled]:
            if "messages" not in quest:
                continue

            for msg in quest["messages"]:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content:
                        # First 100 chars for comparison
                        assistant_responses.append(content[:100])

        if not assistant_responses:
            return False, {"reason": "No assistant responses found"}

        unique_responses = len(set(assistant_responses))
        total_responses = len(assistant_responses)
        diversity_ratio = unique_responses / total_responses if total_responses > 0 else 0

        # Check for repeated words
        word_counter: Counter[str] = Counter()
        for response in assistant_responses[:100]:
            words = response.lower().split()
            word_counter.update(words)

        most_common = word_counter.most_common(10)

        passed = diversity_ratio >= self.min_diversity

        return passed, {
            "unique_responses": unique_responses,
            "total_responses": total_responses,
            "diversity_ratio": diversity_ratio,
            "most_common_words": most_common,
            "sampled": len(assistant_responses),
            "reason": None if passed else f"Diversity {diversity_ratio:.0%} < {self.min_diversity:.0%}"
        }

    def _check_turn_balance(self, quests: Sequence[dict]) -> tuple[bool, dict]:
        """
        Check if quest turns are balanced (user/assistant ratio).

        Should be roughly 1:1 user to assistant messages.
        """
        user_count = 0
        assistant_count = 0
        system_count = 0
        sampled = min(1000, len(quests))

        for quest in quests[:sampled]:
            if "messages" not in quest:
                continue

            for msg in quest["messages"]:
                role = msg.get("role", "")
                if role == "user":
                    user_count += 1
                elif role == "assistant":
                    assistant_count += 1
                elif role == "system":
                    system_count += 1

        total = user_count + assistant_count + system_count

        if total == 0:
            return False, {"reason": "No messages found"}

        # Check ratio (should be roughly 1:1 user:assistant)
        if user_count > 0:
            ratio = assistant_count / user_count
        else:
            ratio = 0.0

        passed = 0.8 <= ratio <= 1.2  # Within 20% of 1:1

        return passed, {
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "system_messages": system_count,
            "user_assistant_ratio": ratio,
            "total_messages": total,
            "reason": None if passed else f"Ratio {ratio:.2f} outside 0.8-1.2 range"
        }

    def _check_content_quality(self, quests: Sequence[dict]) -> tuple[bool, dict]:
        """
        Check content quality (no empty, malformed, or gibberish).
        """
        issues = {
            "empty_content": 0,
            "too_short_content": 0,
            "repetitive_chars": 0,
            "suspicious_patterns": 0,
        }
        sampled = min(1000, len(quests))

        for quest in quests[:sampled]:
            if "messages" not in quest:
                continue

            for msg in quest["messages"]:
                content = msg.get("content", "")

                if not content or not content.strip():
                    issues["empty_content"] += 1
                    continue

                if len(content.strip()) < 5:
                    issues["too_short_content"] += 1
                    continue

                # Check for repetitive characters (aaaaaaaa...)
                if any(char * 10 in content for char in "abcdefghijklmnopqrstuvwxyz"):
                    issues["repetitive_chars"] += 1

                # Check for error patterns
                if content.count("ERROR") > 3 or content.count("FAILED") > 3:
                    issues["suspicious_patterns"] += 1

        total_issues = sum(issues.values())
        # Estimate messages sampled (2 per quest average)
        messages_sampled = sampled * 2

        issue_rate = total_issues / messages_sampled if messages_sampled > 0 else 0
        passed = issue_rate < self.max_issue_rate

        return passed, {
            "issues": issues,
            "total_issues": total_issues,
            "messages_sampled": messages_sampled,
            "issue_rate": issue_rate,
            "reason": None if passed else f"Issue rate {issue_rate:.1%} > {self.max_issue_rate:.0%}"
        }

    def save_report(self, report: QualityReport, output_path: Path):
        """Save quality report to JSON file."""
        from datetime import datetime

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "report": report.to_dict(),
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Quality report saved to {output_path}")

    @property
    def last_report(self) -> QualityReport | None:
        """Get the last inspection report."""
        return self._last_report
