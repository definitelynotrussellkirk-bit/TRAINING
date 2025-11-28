"""
Primitive definitions - atomic testable concepts.

A Primitive is the smallest "unit test" of reasoning or knowledge.
Skills group multiple primitives together.

Example primitives:
- add_single_digit_no_carry (arithmetic)
- modus_ponens (logic)
- binary_add_with_carry (binary)
- reverse_string (string)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PrimitiveId:
    """
    Unique identifier for an atomic testable concept.

    Frozen (immutable) so it can be used as dict key or in sets.

    Attributes:
        name: Unique name within track (e.g., "add_single_digit_no_carry")
        track: Category/track (e.g., "arithmetic", "logic", "binary")
        version: Version string - bump when definition changes (e.g., "v1")
    """
    name: str
    track: str
    version: str = "v1"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.track}/{self.name}@{self.version}"

    def __repr__(self) -> str:
        return f"PrimitiveId({self.name!r}, {self.track!r}, {self.version!r})"

    @classmethod
    def from_string(cls, s: str) -> "PrimitiveId":
        """
        Parse from string format: "track/name@version" or "track/name".

        Examples:
            PrimitiveId.from_string("arithmetic/add_basic@v1")
            PrimitiveId.from_string("logic/modus_ponens")  # defaults to v1
        """
        # Handle version
        if "@" in s:
            base, version = s.rsplit("@", 1)
        else:
            base = s
            version = "v1"

        # Handle track/name
        if "/" in base:
            track, name = base.split("/", 1)
        else:
            # No track specified, use "general"
            track = "general"
            name = base

        return cls(name=name, track=track, version=version)


@dataclass
class PrimitiveMeta:
    """
    Metadata for a primitive - display info, difficulty, relationships.

    This is the full definition of a primitive including human-readable
    information for UI display and difficulty scaling.
    """
    id: PrimitiveId
    display_name: str                        # "Single-Digit Addition (No Carry)"
    description: str                         # What this tests
    difficulty: int = 1                      # 1-10 scale
    prerequisites: list[str] = field(default_factory=list)  # Primitive names
    tags: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Shortcut to id.name."""
        return self.id.name

    @property
    def track(self) -> str:
        """Shortcut to id.track."""
        return self.id.track

    def to_dict(self) -> dict:
        """Serialize to dict for JSON/YAML storage."""
        return {
            "name": self.id.name,
            "track": self.id.track,
            "version": self.id.version,
            "display_name": self.display_name,
            "description": self.description,
            "difficulty": self.difficulty,
            "prerequisites": self.prerequisites,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrimitiveMeta":
        """Deserialize from dict."""
        return cls(
            id=PrimitiveId(
                name=data["name"],
                track=data.get("track", "general"),
                version=data.get("version", "v1"),
            ),
            display_name=data.get("display_name", data["name"].replace("_", " ").title()),
            description=data.get("description", ""),
            difficulty=data.get("difficulty", 1),
            prerequisites=data.get("prerequisites", []),
            tags=data.get("tags", []),
        )


# =============================================================================
# PRIMITIVE CATALOG - Standard primitives organized by track
# =============================================================================

# These are reference definitions. Skills can define their own primitives
# in YAML configs, but these provide a standard vocabulary.

ARITHMETIC_PRIMITIVES = [
    PrimitiveMeta(
        id=PrimitiveId("add_single_digit_no_carry", "arithmetic"),
        display_name="Single Digit Add (No Carry)",
        description="Add two single digits where sum < 10",
        difficulty=1,
        tags=["foundation", "add"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("add_single_digit_with_carry", "arithmetic"),
        display_name="Single Digit Add (Carry)",
        description="Add two single digits where sum >= 10",
        difficulty=2,
        prerequisites=["add_single_digit_no_carry"],
        tags=["foundation", "add", "carry"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("add_two_digit", "arithmetic"),
        display_name="Two Digit Addition",
        description="Add two-digit numbers with potential carrying",
        difficulty=3,
        prerequisites=["add_single_digit_with_carry"],
        tags=["add", "multi-digit"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("sub_single_digit", "arithmetic"),
        display_name="Single Digit Subtraction",
        description="Subtract single digits",
        difficulty=1,
        tags=["foundation", "subtract"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("sub_two_digit", "arithmetic"),
        display_name="Two Digit Subtraction",
        description="Subtract two-digit numbers with potential borrowing",
        difficulty=3,
        prerequisites=["sub_single_digit"],
        tags=["subtract", "multi-digit"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("mul_single_digit", "arithmetic"),
        display_name="Single Digit Multiply",
        description="Multiply two single digits (times table)",
        difficulty=2,
        tags=["foundation", "multiply"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("div_exact", "arithmetic"),
        display_name="Exact Division",
        description="Division with no remainder",
        difficulty=2,
        prerequisites=["mul_single_digit"],
        tags=["divide"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("compare_integers", "arithmetic"),
        display_name="Compare Integers",
        description="Determine which of two integers is larger",
        difficulty=1,
        tags=["foundation", "compare"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("parity", "arithmetic"),
        display_name="Even or Odd",
        description="Determine if a number is even or odd",
        difficulty=1,
        tags=["foundation", "parity"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("digit_sum", "arithmetic"),
        display_name="Digit Sum",
        description="Sum the digits of a number",
        difficulty=2,
        tags=["digits"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("modulo", "arithmetic"),
        display_name="Modulo Operation",
        description="Find remainder after division",
        difficulty=3,
        prerequisites=["div_exact"],
        tags=["modulo", "remainder"],
    ),
]

BINARY_PRIMITIVES = [
    PrimitiveMeta(
        id=PrimitiveId("binary_add_no_carry", "binary"),
        display_name="Binary Add (No Carry)",
        description="Add binary numbers where no column carries",
        difficulty=2,
        tags=["foundation", "add"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("binary_add_with_carry", "binary"),
        display_name="Binary Add (Carry)",
        description="Add binary numbers with carrying",
        difficulty=3,
        prerequisites=["binary_add_no_carry"],
        tags=["add", "carry"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("binary_sub_no_borrow", "binary"),
        display_name="Binary Sub (No Borrow)",
        description="Subtract binary numbers without borrowing",
        difficulty=2,
        tags=["subtract"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("binary_sub_with_borrow", "binary"),
        display_name="Binary Sub (Borrow)",
        description="Subtract binary numbers with borrowing",
        difficulty=3,
        prerequisites=["binary_sub_no_borrow"],
        tags=["subtract", "borrow"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("bitwise_and", "binary"),
        display_name="Bitwise AND",
        description="Compute bitwise AND of two numbers",
        difficulty=2,
        tags=["bitwise"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("bitwise_or", "binary"),
        display_name="Bitwise OR",
        description="Compute bitwise OR of two numbers",
        difficulty=2,
        tags=["bitwise"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("bitwise_xor", "binary"),
        display_name="Bitwise XOR",
        description="Compute bitwise XOR of two numbers",
        difficulty=3,
        tags=["bitwise"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("binary_compare", "binary"),
        display_name="Compare Binary",
        description="Compare magnitude of two binary numbers",
        difficulty=2,
        tags=["compare"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("binary_to_decimal", "binary"),
        display_name="Binary to Decimal",
        description="Convert binary representation to decimal",
        difficulty=2,
        tags=["convert"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("decimal_to_binary", "binary"),
        display_name="Decimal to Binary",
        description="Convert decimal number to binary",
        difficulty=3,
        tags=["convert"],
    ),
]

LOGIC_PRIMITIVES = [
    PrimitiveMeta(
        id=PrimitiveId("truth_table_basic", "logic"),
        display_name="Truth Table (AND/OR/NOT)",
        description="Evaluate basic boolean operations given truth values",
        difficulty=1,
        tags=["foundation", "boolean"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("implication_eval", "logic"),
        display_name="Implication Evaluation",
        description="Evaluate P -> Q given truth values",
        difficulty=2,
        prerequisites=["truth_table_basic"],
        tags=["implication"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("modus_ponens", "logic"),
        display_name="Modus Ponens",
        description="If P then Q. P is true. Conclude Q.",
        difficulty=2,
        prerequisites=["implication_eval"],
        tags=["inference", "deduction"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("modus_tollens", "logic"),
        display_name="Modus Tollens",
        description="If P then Q. Q is false. Conclude not P.",
        difficulty=3,
        prerequisites=["modus_ponens"],
        tags=["inference", "deduction"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("syllogism_all", "logic"),
        display_name="Universal Syllogism",
        description="All A are B. X is A. Conclude X is B.",
        difficulty=2,
        tags=["syllogism", "deduction"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("syllogism_some", "logic"),
        display_name="Existential Syllogism",
        description="Some A are B. What can be concluded?",
        difficulty=3,
        prerequisites=["syllogism_all"],
        tags=["syllogism"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("de_morgan", "logic"),
        display_name="De Morgan's Laws",
        description="Apply NOT(P AND Q) = NOT P OR NOT Q",
        difficulty=4,
        prerequisites=["truth_table_basic"],
        tags=["transformation"],
    ),
]

STRING_PRIMITIVES = [
    PrimitiveMeta(
        id=PrimitiveId("reverse_string", "string"),
        display_name="Reverse String",
        description="Reverse a short string",
        difficulty=1,
        tags=["foundation", "transform"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("count_char", "string"),
        display_name="Count Character",
        description="Count occurrences of a character",
        difficulty=1,
        tags=["foundation", "count"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("substring_check", "string"),
        display_name="Substring Check",
        description="Check if one string contains another",
        difficulty=2,
        tags=["search"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("sort_letters", "string"),
        display_name="Sort Letters",
        description="Sort letters of a word alphabetically",
        difficulty=2,
        tags=["sort"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("palindrome_check", "string"),
        display_name="Palindrome Check",
        description="Check if string reads same forwards and backwards",
        difficulty=2,
        prerequisites=["reverse_string"],
        tags=["pattern"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("string_length", "string"),
        display_name="String Length",
        description="Count characters in a string",
        difficulty=1,
        tags=["foundation", "count"],
    ),
]

CODE_PRIMITIVES = [
    PrimitiveMeta(
        id=PrimitiveId("trace_assignment", "code"),
        display_name="Variable Assignment",
        description="Trace variable assignments (x = 3; y = x + 2)",
        difficulty=1,
        tags=["foundation", "trace"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("trace_if_else", "code"),
        display_name="If-Else Trace",
        description="Trace simple conditional execution",
        difficulty=2,
        prerequisites=["trace_assignment"],
        tags=["control-flow", "trace"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("trace_loop_fixed", "code"),
        display_name="Fixed Loop Trace",
        description="Trace a loop with fixed iterations",
        difficulty=3,
        prerequisites=["trace_if_else"],
        tags=["control-flow", "loop", "trace"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("eval_bool_expr", "code"),
        display_name="Boolean Expression",
        description="Evaluate boolean expression in code",
        difficulty=2,
        tags=["boolean", "expression"],
    ),
    PrimitiveMeta(
        id=PrimitiveId("function_call_trace", "code"),
        display_name="Function Call Trace",
        description="Trace simple function calls and returns",
        difficulty=3,
        prerequisites=["trace_assignment"],
        tags=["function", "trace"],
    ),
]

# All primitives by track
PRIMITIVE_CATALOG: dict[str, list[PrimitiveMeta]] = {
    "arithmetic": ARITHMETIC_PRIMITIVES,
    "binary": BINARY_PRIMITIVES,
    "logic": LOGIC_PRIMITIVES,
    "string": STRING_PRIMITIVES,
    "code": CODE_PRIMITIVES,
}


def get_primitive(name: str, track: Optional[str] = None) -> Optional[PrimitiveMeta]:
    """
    Look up a primitive by name, optionally filtered by track.

    Args:
        name: Primitive name (e.g., "add_single_digit_no_carry")
        track: Optional track filter (e.g., "arithmetic")

    Returns:
        PrimitiveMeta if found, None otherwise
    """
    tracks_to_search = [track] if track else PRIMITIVE_CATALOG.keys()

    for t in tracks_to_search:
        if t not in PRIMITIVE_CATALOG:
            continue
        for prim in PRIMITIVE_CATALOG[t]:
            if prim.id.name == name:
                return prim

    return None


def list_primitives(track: Optional[str] = None) -> list[PrimitiveMeta]:
    """
    List all primitives, optionally filtered by track.

    Args:
        track: Optional track filter

    Returns:
        List of PrimitiveMeta objects
    """
    if track:
        return PRIMITIVE_CATALOG.get(track, [])

    result = []
    for prims in PRIMITIVE_CATALOG.values():
        result.extend(prims)
    return result


def list_tracks() -> list[str]:
    """List all available tracks."""
    return list(PRIMITIVE_CATALOG.keys())
