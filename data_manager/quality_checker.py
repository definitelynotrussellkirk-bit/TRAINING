#!/usr/bin/env python3
"""
Quality Checker - Test suite to gauge data quality and training readiness

Evaluates generated data against quality criteria before queuing for training.
"""

import json
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import Counter
import statistics

logger = logging.getLogger(__name__)


class QualityChecker:
    """Test suite for data quality assessment"""

    def __init__(self, min_length: int = 10, max_length: int = 4096):
        self.min_length = min_length
        self.max_length = max_length
        self.test_results = []

    def check_all(self, data: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all quality checks on data

        Returns:
            (passed, report) - True if all checks pass, plus detailed report
        """
        self.test_results = []

        checks = [
            ("format", self._check_format),
            ("length", self._check_length),
            ("diversity", self._check_diversity),
            ("balance", self._check_balance),
            ("content", self._check_content_quality),
        ]

        all_passed = True
        results = {}

        for check_name, check_func in checks:
            passed, details = check_func(data)
            results[check_name] = {
                "passed": passed,
                "details": details
            }

            if not passed:
                all_passed = False
                logger.warning(f"Quality check FAILED: {check_name} - {details.get('reason', 'Unknown')}")
            else:
                logger.info(f"Quality check PASSED: {check_name}")

            self.test_results.append({
                "check": check_name,
                "passed": passed,
                "details": details
            })

        # Generate summary
        summary = {
            "total_checks": len(checks),
            "passed_checks": sum(1 for r in results.values() if r["passed"]),
            "failed_checks": sum(1 for r in results.values() if not r["passed"]),
            "overall_pass": all_passed,
            "results": results,
            "recommendation": self._get_recommendation(results)
        }

        return all_passed, summary

    def _check_format(self, data: List[Dict]) -> Tuple[bool, Dict]:
        """Check if data has correct conversation format"""
        if not data:
            return False, {"reason": "Empty dataset", "count": 0}

        valid_count = 0
        errors = []

        for i, example in enumerate(data[:100]):  # Sample first 100
            if not isinstance(example, dict):
                errors.append(f"Example {i}: Not a dict")
                continue

            if "messages" not in example:
                errors.append(f"Example {i}: Missing 'messages' key")
                continue

            messages = example["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"Example {i}: Invalid messages format")
                continue

            # Check for user/assistant pattern
            has_user = any(m.get("role") == "user" for m in messages)
            has_assistant = any(m.get("role") == "assistant" for m in messages)

            if not (has_user and has_assistant):
                errors.append(f"Example {i}: Missing user or assistant message")
                continue

            valid_count += 1

        sampled = min(100, len(data))
        pass_rate = valid_count / sampled if sampled > 0 else 0

        passed = pass_rate >= 0.95  # 95% must be valid

        return passed, {
            "valid_count": valid_count,
            "sampled": sampled,
            "pass_rate": pass_rate,
            "errors": errors[:5],  # First 5 errors
            "total_examples": len(data)
        }

    def _check_length(self, data: List[Dict]) -> Tuple[bool, Dict]:
        """Check if conversation lengths are within bounds"""
        lengths = []
        too_short = 0
        too_long = 0

        for example in data[:1000]:  # Sample first 1000
            if "messages" not in example:
                continue

            # Estimate total tokens (rough: 4 chars per token)
            total_chars = sum(len(m.get("content", "")) for m in example["messages"])
            estimated_tokens = total_chars // 4

            lengths.append(estimated_tokens)

            if estimated_tokens < self.min_length:
                too_short += 1
            elif estimated_tokens > self.max_length:
                too_long += 1

        if not lengths:
            return False, {"reason": "No valid examples to measure"}

        passed = (too_short / len(lengths) < 0.05 and  # <5% too short
                 too_long / len(lengths) < 0.05)       # <5% too long

        return passed, {
            "mean_tokens": statistics.mean(lengths),
            "median_tokens": statistics.median(lengths),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "too_short": too_short,
            "too_long": too_long,
            "sampled": len(lengths)
        }

    def _check_diversity(self, data: List[Dict]) -> Tuple[bool, Dict]:
        """Check diversity of responses"""
        assistant_responses = []

        for example in data[:1000]:
            if "messages" not in example:
                continue

            for msg in example["messages"]:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content:
                        assistant_responses.append(content[:100])  # First 100 chars

        if not assistant_responses:
            return False, {"reason": "No assistant responses found"}

        unique_responses = len(set(assistant_responses))
        total_responses = len(assistant_responses)
        diversity_ratio = unique_responses / total_responses if total_responses > 0 else 0

        # Check for repeated phrases
        word_counter = Counter()
        for response in assistant_responses[:100]:
            words = response.lower().split()
            word_counter.update(words)

        most_common = word_counter.most_common(10)

        passed = diversity_ratio >= 0.7  # At least 70% unique

        return passed, {
            "unique_responses": unique_responses,
            "total_responses": total_responses,
            "diversity_ratio": diversity_ratio,
            "most_common_words": most_common,
            "sampled": len(assistant_responses)
        }

    def _check_balance(self, data: List[Dict]) -> Tuple[bool, Dict]:
        """Check if dataset is balanced (user/assistant ratio)"""
        user_count = 0
        assistant_count = 0
        system_count = 0

        for example in data[:1000]:
            if "messages" not in example:
                continue

            for msg in example["messages"]:
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
            "total_messages": total
        }

    def _check_content_quality(self, data: List[Dict]) -> Tuple[bool, Dict]:
        """Check content quality (no empty, malformed, or gibberish)"""
        issues = {
            "empty_content": 0,
            "too_short_content": 0,
            "repetitive": 0,
            "suspicious_patterns": 0
        }

        for example in data[:1000]:
            if "messages" not in example:
                continue

            for msg in example["messages"]:
                content = msg.get("content", "")

                if not content or not content.strip():
                    issues["empty_content"] += 1
                    continue

                if len(content.strip()) < 5:
                    issues["too_short_content"] += 1
                    continue

                # Check for repetitive characters
                if any(char * 10 in content for char in "abcdefghijklmnopqrstuvwxyz"):
                    issues["repetitive"] += 1

                # Check for suspicious patterns
                if content.count("ERROR") > 3 or content.count("FAILED") > 3:
                    issues["suspicious_patterns"] += 1

        total_issues = sum(issues.values())
        sampled = min(1000, len(data)) * 2  # Rough estimate of messages

        issue_rate = total_issues / sampled if sampled > 0 else 0
        passed = issue_rate < 0.05  # Less than 5% issues

        return passed, {
            "issues": issues,
            "total_issues": total_issues,
            "sampled_messages": sampled,
            "issue_rate": issue_rate
        }

    def _get_recommendation(self, results: Dict) -> str:
        """Get recommendation based on test results"""
        failed = [name for name, r in results.items() if not r["passed"]]

        if not failed:
            return "✅ APPROVED: Data quality is excellent. Safe to queue for training."

        if len(failed) <= 2:
            return f"⚠️  CONDITIONAL: {', '.join(failed)} checks failed. Review before training."

        return f"❌ REJECTED: Multiple quality issues ({', '.join(failed)}). Do not train on this data."

    def save_report(self, report: Dict, output_path: Path):
        """Save quality report to JSON file"""
        report_data = {
            "timestamp": str(Path().cwd()),
            "report": report,
            "test_results": self.test_results
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Quality report saved to {output_path}")
