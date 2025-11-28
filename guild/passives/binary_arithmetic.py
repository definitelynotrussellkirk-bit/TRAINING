"""
Binary Arithmetic Passive - Binary operations eval module.

Tests: binary addition, subtraction, bitwise operations, conversions.
Matches the BIN skill primitives defined in configs/skills/bin.yaml.
"""

import random
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


class BinaryArithmeticPassive(PassiveModule):
    """Binary arithmetic and bitwise operations tests."""

    id = "binary_arithmetic"
    name = "Binary Arithmetic"
    category = "math"
    description = "Binary add/sub, bitwise AND/OR/XOR, conversions"
    version = "1.0.0"
    lite_count = 5
    full_count = 30

    def generate_problems(self, count: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._binary_add_no_carry,
            self._binary_add_with_carry,
            self._binary_sub_no_borrow,
            self._binary_sub_with_borrow,
            self._bitwise_and,
            self._bitwise_or,
            self._bitwise_xor,
            self._binary_compare,
            self._binary_to_decimal,
            self._decimal_to_binary,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _format_binary(self, n: int, min_bits: int = 4) -> str:
        """Format number as binary string without 0b prefix."""
        if n < 0:
            # Handle negative numbers - represent as signed
            return f"-{bin(abs(n))[2:].zfill(min_bits)}"
        return bin(n)[2:].zfill(min_bits)

    def _binary_add_no_carry(self) -> Dict[str, Any]:
        """Generate binary addition that doesn't require carry."""
        # Create numbers with non-overlapping bits (no carry needed)
        bits = random.randint(3, 6)
        # Use sparse bit patterns that don't overlap
        a = 0
        b = 0
        for i in range(bits):
            if random.random() < 0.4:
                if random.choice([True, False]):
                    a |= (1 << i)
                else:
                    b |= (1 << i)

        # Ensure at least one bit in each
        if a == 0:
            a = 1 << random.randint(0, bits - 1)
        if b == 0:
            pos = random.randint(0, bits - 1)
            while (a >> pos) & 1:  # Find non-overlapping position
                pos = random.randint(0, bits - 1)
            b = 1 << pos

        result = a + b
        a_bin = self._format_binary(a, bits)
        b_bin = self._format_binary(b, bits)
        result_bin = self._format_binary(result, bits)

        return {
            "prompt": f"What is {a_bin} + {b_bin} in binary?",
            "expected": result_bin,
            "primitive_id": "binary_add_no_carry",
            "type": "binary_add",
            "a": a,
            "b": b,
            "result": result,
        }

    def _binary_add_with_carry(self) -> Dict[str, Any]:
        """Generate binary addition that requires carry."""
        bits = random.randint(3, 6)
        # Create numbers with overlapping bits to force carry
        a = random.randint(1, (1 << bits) - 1)
        b = random.randint(1, (1 << bits) - 1)

        # Ensure at least one position carries
        while (a & b) == 0:  # No overlapping bits = no carry
            b = random.randint(1, (1 << bits) - 1)

        result = a + b
        a_bin = self._format_binary(a, bits)
        b_bin = self._format_binary(b, bits)
        result_bin = self._format_binary(result, bits + 1)  # May need extra bit

        return {
            "prompt": f"What is {a_bin} + {b_bin} in binary?",
            "expected": result_bin,
            "primitive_id": "binary_add_with_carry",
            "type": "binary_add",
            "a": a,
            "b": b,
            "result": result,
        }

    def _binary_sub_no_borrow(self) -> Dict[str, Any]:
        """Generate binary subtraction without borrowing."""
        bits = random.randint(3, 6)
        # Ensure a >= b and no borrowing needed (a has 1 everywhere b has 1)
        b = random.randint(1, (1 << bits) - 1)
        # Make a have all bits of b plus some extras
        a = b | (random.randint(1, (1 << bits) - 1))

        # Ensure a > b
        if a == b:
            a |= (1 << random.randint(0, bits - 1))

        result = a - b
        a_bin = self._format_binary(a, bits)
        b_bin = self._format_binary(b, bits)
        result_bin = self._format_binary(result, bits)

        return {
            "prompt": f"What is {a_bin} - {b_bin} in binary?",
            "expected": result_bin,
            "primitive_id": "binary_sub_no_borrow",
            "type": "binary_sub",
            "a": a,
            "b": b,
            "result": result,
        }

    def _binary_sub_with_borrow(self) -> Dict[str, Any]:
        """Generate binary subtraction that requires borrowing."""
        bits = random.randint(3, 6)
        # Create situation where borrowing is needed
        a = random.randint((1 << (bits - 1)), (1 << bits) - 1)  # Larger number
        b = random.randint(1, a - 1)  # Smaller number

        # Ensure borrow is needed (b has a 1 where a has 0)
        while (b & ~a) == 0:
            b = random.randint(1, a - 1)

        result = a - b
        a_bin = self._format_binary(a, bits)
        b_bin = self._format_binary(b, bits)
        result_bin = self._format_binary(result, bits)

        return {
            "prompt": f"What is {a_bin} - {b_bin} in binary?",
            "expected": result_bin,
            "primitive_id": "binary_sub_with_borrow",
            "type": "binary_sub",
            "a": a,
            "b": b,
            "result": result,
        }

    def _bitwise_and(self) -> Dict[str, Any]:
        """Generate bitwise AND problem."""
        bits = random.randint(4, 6)
        a = random.randint(1, (1 << bits) - 1)
        b = random.randint(1, (1 << bits) - 1)
        result = a & b

        a_bin = self._format_binary(a, bits)
        b_bin = self._format_binary(b, bits)
        result_bin = self._format_binary(result, bits)

        return {
            "prompt": f"What is {a_bin} AND {b_bin}?",
            "expected": result_bin,
            "primitive_id": "bitwise_and",
            "type": "bitwise",
            "a": a,
            "b": b,
            "result": result,
        }

    def _bitwise_or(self) -> Dict[str, Any]:
        """Generate bitwise OR problem."""
        bits = random.randint(4, 6)
        a = random.randint(1, (1 << bits) - 1)
        b = random.randint(1, (1 << bits) - 1)
        result = a | b

        a_bin = self._format_binary(a, bits)
        b_bin = self._format_binary(b, bits)
        result_bin = self._format_binary(result, bits)

        return {
            "prompt": f"What is {a_bin} OR {b_bin}?",
            "expected": result_bin,
            "primitive_id": "bitwise_or",
            "type": "bitwise",
            "a": a,
            "b": b,
            "result": result,
        }

    def _bitwise_xor(self) -> Dict[str, Any]:
        """Generate bitwise XOR problem."""
        bits = random.randint(4, 6)
        a = random.randint(1, (1 << bits) - 1)
        b = random.randint(1, (1 << bits) - 1)
        result = a ^ b

        a_bin = self._format_binary(a, bits)
        b_bin = self._format_binary(b, bits)
        result_bin = self._format_binary(result, bits)

        return {
            "prompt": f"What is {a_bin} XOR {b_bin}?",
            "expected": result_bin,
            "primitive_id": "bitwise_xor",
            "type": "bitwise",
            "a": a,
            "b": b,
            "result": result,
        }

    def _binary_compare(self) -> Dict[str, Any]:
        """Generate binary comparison problem."""
        bits = random.randint(4, 6)
        a = random.randint(1, (1 << bits) - 1)
        b = random.randint(1, (1 << bits) - 1)
        while a == b:
            b = random.randint(1, (1 << bits) - 1)

        a_bin = self._format_binary(a, bits)
        b_bin = self._format_binary(b, bits)
        answer = a_bin if a > b else b_bin

        return {
            "prompt": f"Which is larger: {a_bin} or {b_bin}?",
            "expected": answer,
            "primitive_id": "binary_compare",
            "type": "compare",
            "a": a,
            "b": b,
        }

    def _binary_to_decimal(self) -> Dict[str, Any]:
        """Generate binary to decimal conversion."""
        bits = random.randint(4, 8)
        n = random.randint(1, (1 << bits) - 1)
        n_bin = self._format_binary(n, bits)

        return {
            "prompt": f"What is {n_bin} in decimal?",
            "expected": str(n),
            "primitive_id": "binary_to_decimal",
            "type": "convert",
            "binary": n_bin,
            "decimal": n,
        }

    def _decimal_to_binary(self) -> Dict[str, Any]:
        """Generate decimal to binary conversion."""
        n = random.randint(1, 255)
        n_bin = self._format_binary(n)

        return {
            "prompt": f"What is {n} in binary?",
            "expected": n_bin,
            "primitive_id": "decimal_to_binary",
            "type": "convert",
            "decimal": n,
            "binary": n_bin,
        }

    def check_answer(self, expected: str, got: str) -> bool:
        """
        Check if model's answer matches expected.

        Handles various binary formats:
        - With or without leading zeros
        - With or without 0b prefix
        - Decimal equivalent in response
        """
        # Normalize expected
        expected_norm = expected.strip().lower().replace('0b', '').lstrip('0') or '0'

        # Normalize got
        got_norm = got.strip().lower()

        # Remove 0b prefix if present
        if '0b' in got_norm:
            # Extract the binary part after 0b
            parts = got_norm.split('0b')
            for part in parts[1:]:  # Skip text before first 0b
                # Get just the binary digits
                binary_part = ''
                for c in part:
                    if c in '01':
                        binary_part += c
                    else:
                        break
                if binary_part:
                    binary_norm = binary_part.lstrip('0') or '0'
                    if binary_norm == expected_norm:
                        return True

        # Direct check (strip leading zeros)
        got_stripped = got_norm.replace('0b', '').lstrip('0') or '0'
        if got_stripped == expected_norm:
            return True

        # Check if expected appears in response
        if expected_norm in got_norm:
            return True

        # For decimal answers, check if the number appears
        try:
            expected_int = int(expected_norm, 2)
            if str(expected_int) in got_norm:
                return True
        except ValueError:
            pass

        # For comparison answers, check if the expected binary appears
        if expected.strip().lower() in got_norm:
            return True

        return False
