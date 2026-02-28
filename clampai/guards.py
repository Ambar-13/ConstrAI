"""Symbolic precondition guards.

This module provides a tiny, deterministic expression language and evaluator
for *kernel-enforced* action preconditions.

Goal: Turn human-readable guard strings (e.g., "pages >= 3 and tested") into a
pure function: (state, action) -> (ok, message).

Design clampaints:
- Deterministic and side-effect free.
- No Python eval or AST from python stdlib.
- Small surface area: booleans, comparisons, arithmetic (+/-), and/or/not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


class GuardParseError(ValueError):
    pass


# Tokenizer


@dataclass(frozen=True)
class _Tok:
    kind: str
    text: str


_OPS = {
    "<=",
    ">=",
    "==",
    "!=",
    "<",
    ">",
    "+",
    "-",
    "(",
    ")",
}

_KEYWORDS = {"and", "or", "not", "true", "false", "none"}


def _tokenize(src: str) -> List[_Tok]:
    s = src.strip()
    out: List[_Tok] = []
    i = 0

    def peek(n: int = 0) -> str:
        j = i + n
        return s[j] if 0 <= j < len(s) else ""

    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
            continue

        two = s[i : i + 2]
        if two in _OPS:
            out.append(_Tok("op", two))
            i += 2
            continue

        if ch in _OPS:
            out.append(_Tok("op", ch))
            i += 1
            continue

        if ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            if j < len(s) and s[j] == ".":
                j2 = j + 1
                while j2 < len(s) and s[j2].isdigit():
                    j2 += 1
                if j2 == j + 1:
                    raise GuardParseError(f"Invalid float literal near: {s[i:j2]!r}")
                out.append(_Tok("num", s[i:j2]))
                i = j2
            else:
                out.append(_Tok("num", s[i:j]))
                i = j
            continue

        if ch == "_" or ch.isalpha():
            j = i
            while j < len(s) and (s[j] == "_" or s[j].isalnum()):
                j += 1
            ident = s[i:j]
            low = ident.lower()
            if low in _KEYWORDS:
                out.append(_Tok("kw", low))
            else:
                out.append(_Tok("ident", ident))
            i = j
            continue

        if ch == "'" or ch == '"':
            quote = ch
            j = i + 1
            buf = []
            while j < len(s) and s[j] != quote:
                if s[j] == "\\" and j + 1 < len(s):
                    buf.append(s[j + 1])
                    j += 2
                else:
                    buf.append(s[j])
                    j += 1
            if j >= len(s) or s[j] != quote:
                raise GuardParseError("Unterminated string literal")
            out.append(_Tok("str", "".join(buf)))
            i = j + 1
            continue

        raise GuardParseError(f"Unexpected character {ch!r}")

    out.append(_Tok("eof", ""))
    return out


# AST node types


class _Expr:
    def eval(self, env: Dict[str, Any]) -> Any:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class _Literal(_Expr):
    value: Any

    def eval(self, env: Dict[str, Any]) -> Any:
        return self.value


@dataclass(frozen=True)
class _Var(_Expr):
    name: str

    def eval(self, env: Dict[str, Any]) -> Any:
        return env.get(self.name, None)


@dataclass(frozen=True)
class _Unary(_Expr):
    op: str
    inner: _Expr

    def eval(self, env: Dict[str, Any]) -> Any:
        v = self.inner.eval(env)
        if self.op == "not":
            return not bool(v)
        if self.op == "+":
            return +float(v)
        if self.op == "-":
            return -float(v)
        raise GuardParseError(f"Unknown unary operator: {self.op}")


@dataclass(frozen=True)
class _Binary(_Expr):
    op: str
    left: _Expr
    right: _Expr

    def eval(self, env: Dict[str, Any]) -> Any:
        if self.op in ("and", "or"):
            if self.op == "and":
                return bool(self.left.eval(env)) and bool(self.right.eval(env))
            return bool(self.left.eval(env)) or bool(self.right.eval(env))

        lv = self.left.eval(env)
        rv = self.right.eval(env)

        if self.op in ("+", "-"):
            lf = float(lv)
            rf = float(rv)
            return lf + rf if self.op == "+" else lf - rf

        if self.op in ("<", "<=", ">", ">=", "==", "!="):
            # Allow safe numeric comparisons when possible.
            if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                pass
            else:
                # Best-effort numeric coercion for strings like "3" etc.
                try:
                    lf = float(lv)
                    rf = float(rv)
                    lv, rv = lf, rf
                except Exception:
                    # Fall back to raw comparisons for equality/inequality only.
                    if self.op in ("==", "!="):
                        return (lv == rv) if self.op == "==" else (lv != rv)
                    raise GuardParseError(
                        f"Non-numeric comparison: {lv!r} {self.op} {rv!r}"
                    )

            if self.op == "<":
                return lv < rv
            if self.op == "<=":
                return lv <= rv
            if self.op == ">":
                return lv > rv
            if self.op == ">=":
                return lv >= rv
            if self.op == "==":
                return lv == rv
            if self.op == "!=":
                return lv != rv

        raise GuardParseError(f"Unknown binary operator: {self.op}")


# Recursive descent parser


class _Parser:
    def __init__(self, toks: List[_Tok]):
        self._toks = toks
        self._i = 0

    def _cur(self) -> _Tok:
        return self._toks[self._i]

    def _eat(self, kind: str, text: Optional[str] = None) -> _Tok:
        tok = self._cur()
        if tok.kind != kind:
            raise GuardParseError(f"Expected {kind}, got {tok.kind} ({tok.text!r})")
        if text is not None and tok.text != text:
            raise GuardParseError(f"Expected {text!r}, got {tok.text!r}")
        self._i += 1
        return tok

    def parse(self) -> _Expr:
        expr = self._parse_or()
        if self._cur().kind != "eof":
            raise GuardParseError(f"Unexpected token: {self._cur().text!r}")
        return expr

    # or_expr := and_expr ("or" and_expr)*
    def _parse_or(self) -> _Expr:
        left = self._parse_and()
        while self._cur().kind == "kw" and self._cur().text == "or":
            self._eat("kw", "or")
            right = self._parse_and()
            left = _Binary("or", left, right)
        return left

    # and_expr := not_expr ("and" not_expr)*
    def _parse_and(self) -> _Expr:
        left = self._parse_not()
        while self._cur().kind == "kw" and self._cur().text == "and":
            self._eat("kw", "and")
            right = self._parse_not()
            left = _Binary("and", left, right)
        return left

    # not_expr := ("not")* cmp_expr
    def _parse_not(self) -> _Expr:
        if self._cur().kind == "kw" and self._cur().text == "not":
            self._eat("kw", "not")
            return _Unary("not", self._parse_not())
        return self._parse_cmp()

    # cmp_expr := add_expr ((<|<=|>|>=|==|!=) add_expr)?
    def _parse_cmp(self) -> _Expr:
        left = self._parse_add()
        if self._cur().kind == "op" and self._cur().text in ("<", "<=", ">", ">=", "==", "!="):
            op = self._eat("op").text
            right = self._parse_add()
            return _Binary(op, left, right)
        return left

    # add_expr := unary ((+|-) unary)*
    def _parse_add(self) -> _Expr:
        left = self._parse_unary()
        while self._cur().kind == "op" and self._cur().text in ("+", "-"):
            op = self._eat("op").text
            right = self._parse_unary()
            left = _Binary(op, left, right)
        return left

    # unary := (+|-) unary | primary
    def _parse_unary(self) -> _Expr:
        if self._cur().kind == "op" and self._cur().text in ("+", "-"):
            op = self._eat("op").text
            return _Unary(op, self._parse_unary())
        return self._parse_primary()

    # primary := NUMBER | STRING | TRUE|FALSE|NONE | IDENT | "(" expr ")"
    def _parse_primary(self) -> _Expr:
        tok = self._cur()
        if tok.kind == "num":
            self._eat("num")
            if "." in tok.text:
                return _Literal(float(tok.text))
            return _Literal(int(tok.text))
        if tok.kind == "str":
            self._eat("str")
            return _Literal(tok.text)
        if tok.kind == "kw" and tok.text in ("true", "false", "none"):
            self._eat("kw")
            if tok.text == "true":
                return _Literal(True)
            if tok.text == "false":
                return _Literal(False)
            return _Literal(None)
        if tok.kind == "ident":
            self._eat("ident")
            return _Var(tok.text)
        if tok.kind == "op" and tok.text == "(":
            self._eat("op", "(")
            inner = self._parse_or()
            self._eat("op", ")")
            return inner
        raise GuardParseError(f"Unexpected token: {tok.kind}:{tok.text!r}")


def compile_guard(expr: str) -> _Expr:
    """Compile a guard string to an AST expression."""
    toks = _tokenize(expr)
    return _Parser(toks).parse()


def evaluate_guard(expr: str, state: Dict[str, Any]) -> Tuple[bool, str]:
    """Convenience function: compile+evaluate once."""
    ast = compile_guard(expr)
    try:
        ok = bool(ast.eval(state))
    except Exception as e:
        raise GuardParseError(str(e)) from e
    return ok, "ok" if ok else f"Guard failed: {expr}"


def kernel_precondition_from_actions(
    *, all_actions: Iterable[Any]
) -> Callable[[Any, Any], Tuple[bool, str]]:
    """Create a SafetyKernel precondition checker over ActionSpec.preconditions_text.

    Expects ActionSpec-like objects with fields:
      - id: str
      - preconditions_text: Optional[str]

    Returns a function ``(state, action) -> (ok, msg)``. The returned
    function is pure and reads only builtin types.
    """

    compiled: Dict[str, _Expr] = {}

    for a in all_actions:
        pre = getattr(a, "preconditions_text", None)
        if not pre:
            continue

        pre_s = str(pre).strip()
        if not pre_s:
            continue

        # Common shorthand: a single identifier means "that state key is truthy".
        # E.g. preconditions_text="tested" => guard expression "tested == true".
        toks = _tokenize(pre_s)
        if (
            len(toks) >= 2
            and toks[0].kind == "ident"
            and toks[1].kind == "eof"
        ):
            pre_s = f"{toks[0].text} == true"

        compiled[a.id] = compile_guard(pre_s)

    def _check(state: Any, action: Any) -> Tuple[bool, str]:
        ast = compiled.get(action.id)
        if ast is None:
            return True, "no guard"

        # State may be a wrapper object; best-effort extract mapping.
        env: Dict[str, Any]
        if isinstance(state, dict):
            env = state
        elif hasattr(state, "to_dict"):
            env = state.to_dict()
        else:
            _raw: Any = getattr(state, "data", None) or getattr(state, "_data", None)
            env = _raw if _raw is not None else dict(getattr(state, "__dict__", {}))

        try:
            ok = bool(ast.eval(env))
        except Exception as e:
            return False, f"Invalid precondition for {action.id}: {e}"
        if ok:
            return True, "guard ok"
        return False, f"Precondition failed for {action.id}: {getattr(action, 'preconditions_text', '')}"

    return _check
