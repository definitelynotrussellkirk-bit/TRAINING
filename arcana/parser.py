"""
S-Expression Parser for the Arcana DSL.

Grammar:
    program  := form*
    form     := atom | list
    list     := '(' form* ')'
    atom     := string | number | symbol | keyword
    string   := '"' [^"]* '"'
    number   := [-+]?[0-9]+(\.[0-9]+)?
    keyword  := ':' symbol
    symbol   := [a-zA-Z_][-a-zA-Z0-9_?!]*
"""

import re
from typing import Any, Iterator, List, Optional, Union

# Token patterns
TOKEN_RE = re.compile(r'''
    \s*                     # Skip whitespace
    (
        ;[^\n]*           | # Comment (skip to EOL)
        \(                | # Open paren
        \)                | # Close paren
        "[^"\\]*(?:\\.[^"\\]*)*" | # String with escapes
        [^\s()"]+           # Atom (symbol, number, keyword)
    )
''', re.VERBOSE)


class ParseError(Exception):
    """Error during S-expression parsing."""
    pass


def tokenize(source: str) -> Iterator[str]:
    """Tokenize S-expression source into tokens."""
    for match in TOKEN_RE.finditer(source):
        token = match.group(1)
        # Skip comments
        if token.startswith(';'):
            continue
        if token.strip():
            yield token


def _parse_atom(token: str) -> Any:
    """Parse an atomic value (string, number, keyword, or symbol)."""
    # String literal
    if token.startswith('"') and token.endswith('"'):
        # Handle escape sequences
        inner = token[1:-1]
        inner = inner.replace('\\n', '\n')
        inner = inner.replace('\\t', '\t')
        inner = inner.replace('\\"', '"')
        inner = inner.replace('\\\\', '\\')
        return inner

    # Keyword (:foo)
    if token.startswith(':'):
        return token  # Keep as string with colon

    # Boolean literals
    if token == 'true' or token == '#t':
        return True
    if token == 'false' or token == '#f' or token == 'nil':
        return False

    # Try number
    try:
        if '.' in token or 'e' in token.lower():
            return float(token)
        return int(token)
    except ValueError:
        pass

    # Symbol
    return token


def _parse_form(tokens: Iterator[str], depth: int = 0) -> Optional[Any]:
    """Parse a single form (atom or list)."""
    try:
        token = next(tokens)
    except StopIteration:
        return None

    if token == '(':
        # Parse list
        items = []
        while True:
            # Peek at next token
            try:
                peek = next(tokens)
            except StopIteration:
                raise ParseError(f"Unclosed '(' at depth {depth}")

            if peek == ')':
                return items

            # Put token back by parsing it
            if peek == '(':
                items.append(_parse_list_from_open(tokens, depth + 1))
            else:
                items.append(_parse_atom(peek))

    elif token == ')':
        raise ParseError("Unexpected ')'")

    else:
        return _parse_atom(token)


def _parse_list_from_open(tokens: Iterator[str], depth: int) -> List[Any]:
    """Parse a list after seeing '('."""
    items = []
    while True:
        try:
            token = next(tokens)
        except StopIteration:
            raise ParseError(f"Unclosed '(' at depth {depth}")

        if token == ')':
            return items
        elif token == '(':
            items.append(_parse_list_from_open(tokens, depth + 1))
        else:
            items.append(_parse_atom(token))


def parse(source: str) -> List[Any]:
    """Parse S-expression source into Python data structures.

    Returns a list of top-level forms.

    Examples:
        >>> parse('(hero :id dio :level 7)')
        [['hero', ':id', 'dio', ':level', 7]]

        >>> parse('(add 1 2) (mul 3 4)')
        [['add', 1, 2], ['mul', 3, 4]]
    """
    tokens = tokenize(source)
    forms = []

    while True:
        form = _parse_form(tokens)
        if form is None:
            break
        forms.append(form)

    return forms


def parse_file(path: str) -> List[Any]:
    """Parse S-expressions from a file."""
    with open(path, 'r') as f:
        return parse(f.read())


# --- Pretty printing ---

def to_sexpr(obj: Any, indent: int = 0) -> str:
    """Convert Python object back to S-expression string."""
    if obj is None or obj is False:
        return 'nil'
    if obj is True:
        return 'true'
    if isinstance(obj, str):
        if obj.startswith(':'):
            return obj  # Keyword
        if ' ' in obj or '"' in obj or not obj:
            # Quote strings with spaces
            escaped = obj.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return obj  # Symbol or simple string
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, list):
        if not obj:
            return '()'
        inner = ' '.join(to_sexpr(x) for x in obj)
        return f'({inner})'
    if isinstance(obj, dict):
        # Convert dict to property list
        items = []
        for k, v in obj.items():
            key = f':{k}' if not str(k).startswith(':') else k
            items.append(f'{key} {to_sexpr(v)}')
        return '(' + ' '.join(items) + ')'
    return str(obj)


def pprint(obj: Any, indent: int = 0) -> str:
    """Pretty-print with indentation for nested structures."""
    if not isinstance(obj, list) or len(obj) <= 3:
        return to_sexpr(obj)

    # Multi-line for longer lists
    lines = [f"({to_sexpr(obj[0])}"]
    for item in obj[1:]:
        lines.append("  " + pprint(item, indent + 2))
    lines[-1] += ")"
    return "\n".join(lines)


if __name__ == '__main__':
    # Quick test
    test = '''
    ; This is a comment
    (hero :id dio :name "DIO" :level 7)
    (campaign
      :id c-001
      :hero dio
      :step 1500)
    (if (> level 5)
        (grant-skill :skill :fireball)
        (log "Not ready yet"))
    '''

    forms = parse(test)
    for form in forms:
        print(form)
        print("->", to_sexpr(form))
        print()
