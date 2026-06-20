"""Calculator built-in tool.

Safe arithmetic evaluation using the ``ast`` module (NOT ``eval``). Only a
whitelist of node types is permitted (numbers, arithmetic operators, unary
minus, parentheses, and a small set of math functions). Attribute access,
calls to arbitrary functions, imports, comprehensions, assignments, and any
form of statement execution are rejected.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from vertai.core.tool import FunctionTool

# Binary operators allowed in expressions.
_BIN_OPS: dict[type[Any], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Unary operators (``-x`` and ``+x``).
_UNARY_OPS: dict[type[Any], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelisted math functions callable as ``sqrt(4)`` etc. We only expose pure
# functions; nothing that touches the filesystem, network, or state.
_SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}

# Constants exposed by name.
_SAFE_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains a disallowed AST node."""


def _eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node against the whitelist.

    Any node type not in the whitelist raises :class:`UnsafeExpressionError`,
    so attribute access (``os.system``), calls to unknown functions, imports,
    comprehensions, assignments, and statements are all rejected.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        # Python 3.8+: numbers and strings are ast.Constant.
        if isinstance(node.value, (int, float)):
            return node.value
        raise UnsafeExpressionError(
            f"Only numeric constants allowed, got {type(node.value).__name__}"
        )
    if isinstance(node, ast.BinOp):
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise UnsafeExpressionError(
                f"Binary operator {type(node.op).__name__} not allowed"
            )
        return op_fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        unary_fn = _UNARY_OPS.get(type(node.op))
        if unary_fn is None:
            raise UnsafeExpressionError(
                f"Unary operator {type(node.op).__name__} not allowed"
            )
        return unary_fn(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        # Only direct calls to whitelisted bare names (e.g. ``sqrt(4)``).
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError(
                "Only direct calls to whitelisted functions are allowed; "
                "attribute/method calls are rejected."
            )
        fn_name = node.func.id
        if fn_name not in _SAFE_FUNCS:
            raise UnsafeExpressionError(
                f"Function '{fn_name}' is not whitelisted"
            )
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are not allowed")
        args = [_eval_node(a) for a in node.args]
        return _SAFE_FUNCS[fn_name](*args)
    if isinstance(node, ast.Name):
        # Bare names resolve to whitelisted constants only.
        if node.id in _SAFE_CONSTS:
            return _SAFE_CONSTS[node.id]
        raise UnsafeExpressionError(
            f"Name '{node.id}' is not allowed; only constants "
            f"({sorted(_SAFE_CONSTS)}) may be referenced by name."
        )
    # Explicit rejection: anything else (Attribute, Subscript, Import,
    # Lambda, comprehension, assignment, statement, ...).
    raise UnsafeExpressionError(
        f"Disallowed expression element: {type(node).__name__}"
    )


def safe_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression safely.

    Parses ``expression`` with :func:`ast.parse` (``mode="eval"``) and
    evaluates only whitelisted nodes. Attribute access, imports, arbitrary
    function calls, and statements are rejected.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpressionError("Expression must be a non-empty string")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid expression syntax: {exc}") from exc
    result = _eval_node(tree)
    if isinstance(result, bool):
        # ``True``/``False`` are ints in Python; surface them as an error so
        # the calculator stays numeric.
        raise UnsafeExpressionError("Boolean results are not supported")
    if not isinstance(result, (int, float)):
        raise UnsafeExpressionError(
            f"Expression did not evaluate to a number (got {type(result).__name__})"
        )
    return result


def _calculator_impl(expression: str) -> str:
    """Evaluate an arithmetic expression safely.

    Supports ``+ - * / // % **``, parentheses, unary minus, and a small set
    of math functions (sqrt, log, log10, log2, exp, sin, cos, tan, floor,
    ceil, abs, round, min, max, pow) plus the constants pi, e, tau.

    Examples:
        ``2 + 3 * 4`` -> ``14``
        ``sqrt(16) + pow(2, 3)`` -> ``12.0``
        ``(10 / 4)`` -> ``2.5``

    Args:
        expression: A Python arithmetic expression (numbers, operators,
            parentheses, and whitelisted functions/constants only).

    Returns:
        The numeric result as a string.
    """
    result: float | int = safe_eval(expression)
    return str(result)


def _failure_message(exc: Exception) -> str:
    return f"Cannot evaluate expression: {exc}"


calculator: FunctionTool = FunctionTool(
    _calculator_impl,
    name_override="calculator",
    failure_error_function=_failure_message,
)


__all__ = ["UnsafeExpressionError", "calculator", "safe_eval"]
