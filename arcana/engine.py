"""
Arcana Engine - The LISP evaluator.

Evaluates S-expressions against a World, dispatching to registered verbs.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .parser import parse, parse_file, to_sexpr
from .world import World, get_world


class EvalError(Exception):
    """Error during evaluation."""
    pass


# Type alias for verb functions
# Signature: (engine: Engine, *args) -> Any
VerbFn = Callable[..., Any]


class Engine:
    """
    The Arcana LISP engine.

    Evaluates S-expressions by dispatching to registered verbs.
    Built-in forms: if, when, do, let, quote, eval
    """

    def __init__(self, world: Optional[World] = None, base_dir: Optional[Path] = None):
        self.world = world or get_world(base_dir)
        self.verbs: Dict[str, VerbFn] = {}
        self.env: Dict[str, Any] = {}  # Variable bindings

        # Register built-in forms
        self._register_builtins()

    def register(self, name: str, fn: VerbFn):
        """Register a verb function."""
        self.verbs[name] = fn

    def register_module(self, module):
        """Register all verbs from a module.

        Looks for functions starting with 'verb_'.
        verb_foo becomes the verb 'foo'.
        """
        for name in dir(module):
            if name.startswith('verb_'):
                verb_name = name[5:].replace('_', '-')
                self.register(verb_name, getattr(module, name))

    def _register_builtins(self):
        """Register built-in special forms."""
        self.verbs['if'] = self._builtin_if
        self.verbs['when'] = self._builtin_when
        self.verbs['unless'] = self._builtin_unless
        self.verbs['do'] = self._builtin_do
        self.verbs['let'] = self._builtin_let
        self.verbs['quote'] = self._builtin_quote
        self.verbs['set!'] = self._builtin_set
        self.verbs['def'] = self._builtin_def

        # Comparison operators
        self.verbs['>'] = self._builtin_gt
        self.verbs['<'] = self._builtin_lt
        self.verbs['>='] = self._builtin_gte
        self.verbs['<='] = self._builtin_lte
        self.verbs['='] = self._builtin_eq
        self.verbs['!='] = self._builtin_neq

        # Arithmetic
        self.verbs['+'] = self._builtin_add
        self.verbs['-'] = self._builtin_sub
        self.verbs['*'] = self._builtin_mul
        self.verbs['/'] = self._builtin_div

        # Logic
        self.verbs['and'] = self._builtin_and
        self.verbs['or'] = self._builtin_or
        self.verbs['not'] = self._builtin_not

        # Utility
        self.verbs['print'] = self._builtin_print
        self.verbs['log'] = self._builtin_log
        self.verbs['list'] = self._builtin_list

    # --- Evaluation ---

    def eval(self, form: Any) -> Any:
        """Evaluate a single form."""
        # Atoms
        if form is None:
            return None
        if isinstance(form, bool):
            return form
        if isinstance(form, (int, float)):
            return form
        if isinstance(form, str):
            # Keyword - return as-is
            if form.startswith(':'):
                return form
            # Variable lookup
            if form in self.env:
                return self.env[form]
            # Unknown symbol - return as-is (for entity names, etc.)
            return form

        # List - function call
        if not isinstance(form, list):
            return form

        if not form:
            return []

        head = form[0]
        args = form[1:]

        # Look up verb
        if isinstance(head, str) and head in self.verbs:
            fn = self.verbs[head]
            return fn(self, *args)

        # Unknown form
        raise EvalError(f"Unknown verb: {head}")

    def eval_args(self, args: List[Any]) -> List[Any]:
        """Evaluate a list of arguments."""
        return [self.eval(arg) for arg in args]

    def run(self, source: str) -> List[Any]:
        """Parse and evaluate source code."""
        forms = parse(source)
        results = []
        for form in forms:
            result = self.eval(form)
            results.append(result)
            # Record in world log
            if isinstance(form, list) and form:
                self.world.record_action(
                    action=str(form[0]),
                    args={'form': to_sexpr(form)},
                    result=result
                )
        return results

    def run_file(self, path: str) -> List[Any]:
        """Parse and evaluate a file."""
        forms = parse_file(path)
        results = []
        for form in forms:
            result = self.eval(form)
            results.append(result)
        return results

    # --- Built-in Special Forms ---

    def _builtin_if(self, engine, cond, then_form, *else_forms):
        """(if cond then else?)"""
        cond_val = self.eval(cond)
        if cond_val:
            return self.eval(then_form)
        elif else_forms:
            return self.eval(else_forms[0])
        return None

    def _builtin_when(self, engine, cond, *body):
        """(when cond body...)"""
        if self.eval(cond):
            result = None
            for form in body:
                result = self.eval(form)
            return result
        return None

    def _builtin_unless(self, engine, cond, *body):
        """(unless cond body...)"""
        if not self.eval(cond):
            result = None
            for form in body:
                result = self.eval(form)
            return result
        return None

    def _builtin_do(self, engine, *forms):
        """(do form1 form2 ...)"""
        result = None
        for form in forms:
            result = self.eval(form)
        return result

    def _builtin_let(self, engine, bindings, *body):
        """(let ((x 1) (y 2)) body...)"""
        old_env = self.env.copy()
        try:
            # Process bindings
            for binding in bindings:
                if len(binding) != 2:
                    raise EvalError(f"Invalid binding: {binding}")
                name, value = binding
                self.env[name] = self.eval(value)

            # Evaluate body
            result = None
            for form in body:
                result = self.eval(form)
            return result
        finally:
            self.env = old_env

    def _builtin_quote(self, engine, form):
        """(quote form) - return form without evaluating"""
        return form

    def _builtin_set(self, engine, name, value):
        """(set! name value)"""
        self.env[name] = self.eval(value)
        return self.env[name]

    def _builtin_def(self, engine, name, value):
        """(def name value)"""
        self.env[name] = self.eval(value)
        return self.env[name]

    # --- Comparison ---

    def _builtin_gt(self, engine, a, b):
        return self.eval(a) > self.eval(b)

    def _builtin_lt(self, engine, a, b):
        return self.eval(a) < self.eval(b)

    def _builtin_gte(self, engine, a, b):
        return self.eval(a) >= self.eval(b)

    def _builtin_lte(self, engine, a, b):
        return self.eval(a) <= self.eval(b)

    def _builtin_eq(self, engine, a, b):
        return self.eval(a) == self.eval(b)

    def _builtin_neq(self, engine, a, b):
        return self.eval(a) != self.eval(b)

    # --- Arithmetic ---

    def _builtin_add(self, engine, *args):
        vals = self.eval_args(list(args))
        return sum(vals)

    def _builtin_sub(self, engine, a, *rest):
        result = self.eval(a)
        for x in rest:
            result -= self.eval(x)
        return result

    def _builtin_mul(self, engine, *args):
        result = 1
        for x in args:
            result *= self.eval(x)
        return result

    def _builtin_div(self, engine, a, b):
        return self.eval(a) / self.eval(b)

    # --- Logic ---

    def _builtin_and(self, engine, *args):
        for arg in args:
            if not self.eval(arg):
                return False
        return True

    def _builtin_or(self, engine, *args):
        for arg in args:
            if self.eval(arg):
                return True
        return False

    def _builtin_not(self, engine, arg):
        return not self.eval(arg)

    # --- Utility ---

    def _builtin_print(self, engine, *args):
        vals = self.eval_args(list(args))
        print(*vals)
        return vals[-1] if vals else None

    def _builtin_log(self, engine, *args):
        """Log a message to world log."""
        vals = self.eval_args(list(args))
        msg = ' '.join(str(v) for v in vals)
        self.world.record_action('log', {'message': msg}, msg)
        return msg

    def _builtin_list(self, engine, *args):
        return self.eval_args(list(args))


def create_engine(base_dir: Optional[Path] = None, sync: bool = True) -> Engine:
    """Create an engine with standard verbs loaded."""
    from . import verbs

    world = get_world(base_dir)
    if sync:
        world.sync_from_filesystem()

    engine = Engine(world=world)

    # Register verb modules
    engine.register_module(verbs.core)
    engine.register_module(verbs.training)
    engine.register_module(verbs.query)
    engine.register_module(verbs.curriculum)

    return engine


if __name__ == '__main__':
    # Quick test
    engine = Engine()

    result = engine.run('''
        (def x 10)
        (def y 20)
        (if (> x 5)
            (print "x is greater than 5")
            (print "x is small"))
        (+ x y)
    ''')

    print(f"Results: {result}")
