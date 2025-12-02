"""
Code Trace Passive - Code execution tracing eval module.

Tests: variable assignment, conditionals, loops, boolean expressions, function calls.
These are foundational code reasoning skills.
"""

import random
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


class CodeTracePassive(PassiveModule):
    """Code execution tracing tests."""

    id = "code_trace"
    name = "Code Trace"
    category = "code"
    description = "Trace variable assignments, conditionals, loops, and function calls"
    version = "1.0.0"
    lite_count = 5
    full_count = 30

    def generate_problems(self, count: int, seed: Optional[int] = None, level: int = 1) -> List[Dict[str, Any]]:
        """Generate code trace problems. Level parameter accepted but not yet used."""
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._trace_assignment,
            self._trace_assignment_chain,
            self._trace_if_else,
            self._trace_loop_fixed,
            self._eval_bool_expr,
            self._function_call_trace,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _trace_assignment(self) -> Dict[str, Any]:
        """Simple variable assignment tracing."""
        a = random.randint(1, 20)
        b = random.randint(1, 20)

        op = random.choice(['+', '-', '*'])
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:
            result = a * b

        code = f"""x = {a}
y = {b}
z = x {op} y"""

        return {
            "prompt": f"What is the value of z after running this code?\n```\n{code}\n```",
            "expected": str(result),
            "primitive_id": "trace_assignment",
            "type": "assignment",
            "code": code,
        }

    def _trace_assignment_chain(self) -> Dict[str, Any]:
        """Chained variable assignment tracing."""
        a = random.randint(1, 10)
        b = random.randint(1, 10)

        # x = a, y = x + b, z = y * 2
        x = a
        y = x + b
        z = y * 2

        code = f"""x = {a}
y = x + {b}
z = y * 2"""

        return {
            "prompt": f"What is the value of z after running this code?\n```\n{code}\n```",
            "expected": str(z),
            "primitive_id": "trace_assignment",
            "type": "assignment_chain",
            "code": code,
        }

    def _trace_if_else(self) -> Dict[str, Any]:
        """Simple if-else tracing."""
        x = random.randint(1, 20)
        threshold = random.randint(5, 15)

        if x > threshold:
            result = "big"
        else:
            result = "small"

        code = f"""x = {x}
if x > {threshold}:
    result = "big"
else:
    result = "small\""""

        return {
            "prompt": f"What is the value of result after running this code?\n```\n{code}\n```",
            "expected": result,
            "primitive_id": "trace_if_else",
            "type": "conditional",
            "code": code,
        }

    def _trace_loop_fixed(self) -> Dict[str, Any]:
        """Fixed iteration loop tracing."""
        loop_type = random.choice(["sum", "count", "multiply"])

        if loop_type == "sum":
            nums = [random.randint(1, 5) for _ in range(random.randint(3, 5))]
            result = sum(nums)
            nums_str = ", ".join(map(str, nums))
            code = f"""total = 0
for n in [{nums_str}]:
    total = total + n"""
            var_name = "total"

        elif loop_type == "count":
            items = random.randint(3, 6)
            code = f"""count = 0
for i in range({items}):
    count = count + 1"""
            result = items
            var_name = "count"

        else:  # multiply
            base = random.randint(2, 3)
            times = random.randint(2, 4)
            code = f"""x = 1
for i in range({times}):
    x = x * {base}"""
            result = base ** times
            var_name = "x"

        return {
            "prompt": f"What is the value of {var_name} after running this code?\n```\n{code}\n```",
            "expected": str(result),
            "primitive_id": "trace_loop_fixed",
            "type": "loop",
            "code": code,
        }

    def _eval_bool_expr(self) -> Dict[str, Any]:
        """Evaluate boolean expressions."""
        expr_type = random.choice(["and", "or", "not", "combined"])

        if expr_type == "and":
            a = random.choice([True, False])
            b = random.choice([True, False])
            result = a and b
            expr = f"{a} and {b}"

        elif expr_type == "or":
            a = random.choice([True, False])
            b = random.choice([True, False])
            result = a or b
            expr = f"{a} or {b}"

        elif expr_type == "not":
            a = random.choice([True, False])
            result = not a
            expr = f"not {a}"

        else:  # combined
            a = random.choice([True, False])
            b = random.choice([True, False])
            c = random.choice([True, False])
            result = (a and b) or c
            expr = f"({a} and {b}) or {c}"

        return {
            "prompt": f"What is the result of this boolean expression: {expr}",
            "expected": str(result),
            "primitive_id": "eval_bool_expr",
            "type": "boolean",
            "expression": expr,
        }

    def _function_call_trace(self) -> Dict[str, Any]:
        """Simple function call tracing."""
        func_type = random.choice(["add", "double", "square"])

        if func_type == "add":
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            result = a + b
            code = f"""def add(x, y):
    return x + y

result = add({a}, {b})"""

        elif func_type == "double":
            n = random.randint(1, 20)
            result = n * 2
            code = f"""def double(x):
    return x * 2

result = double({n})"""

        else:  # square
            n = random.randint(1, 10)
            result = n * n
            code = f"""def square(x):
    return x * x

result = square({n})"""

        return {
            "prompt": f"What is the value of result after running this code?\n```\n{code}\n```",
            "expected": str(result),
            "primitive_id": "function_call_trace",
            "type": "function",
            "code": code,
        }

    def check_answer(self, expected: str, got: str) -> bool:
        """Check if model's answer matches expected."""
        expected_norm = expected.strip().lower()
        got_norm = got.strip().lower()

        # Direct match
        if expected_norm == got_norm:
            return True

        # Check if expected appears in response
        if expected_norm in got_norm:
            return True

        # For numeric answers, try to find the number
        if expected_norm.lstrip('-').isdigit():
            import re
            # Look for the exact number
            numbers = re.findall(r'-?\b\d+\b', got_norm)
            if expected_norm in numbers:
                return True

        # For boolean answers
        bool_map = {
            "true": ["true", "yes", "1"],
            "false": ["false", "no", "0"],
        }
        if expected_norm in bool_map:
            for variant in bool_map[expected_norm]:
                if variant in got_norm.split():
                    return True

        return False
