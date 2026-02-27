"""
Tests for constrai.guards — symbolic precondition expression parser and evaluator.

Covers: tokenizer, parser, evaluator, compile_guard, evaluate_guard,
and kernel_precondition_from_actions.  All code paths are exercised including
error conditions and edge cases.
"""
from __future__ import annotations

from typing import Optional

import pytest

from constrai.formal import State
from constrai.guards import (
    GuardParseError,
    _tokenize,
    compile_guard,
    evaluate_guard,
    kernel_precondition_from_actions,
)


class TestTokenizer:
    def test_tokenizes_integer(self):
        toks = _tokenize("42")
        assert toks[0].kind == "num"
        assert toks[0].text == "42"

    def test_tokenizes_float(self):
        toks = _tokenize("3.14")
        assert toks[0].kind == "num"
        assert toks[0].text == "3.14"

    def test_invalid_float_raises(self):
        with pytest.raises(GuardParseError, match="Invalid float literal"):
            _tokenize("3.")

    def test_tokenizes_string_single_quote(self):
        toks = _tokenize("'hello'")
        assert toks[0].kind == "str"
        assert toks[0].text == "hello"

    def test_tokenizes_string_double_quote(self):
        toks = _tokenize('"world"')
        assert toks[0].kind == "str"
        assert toks[0].text == "world"

    def test_unterminated_string_raises(self):
        with pytest.raises(GuardParseError, match="Unterminated string literal"):
            _tokenize("'unclosed")

    def test_tokenizes_string_with_escape(self):
        toks = _tokenize("'a\\'b'")
        assert toks[0].kind == "str"
        assert toks[0].text == "a'b"

    def test_tokenizes_keywords(self):
        toks = _tokenize("and or not true false none")
        kinds = [t.kind for t in toks[:-1]]
        assert all(k == "kw" for k in kinds)

    def test_tokenizes_operators(self):
        toks = _tokenize("<= >= == != < > + - ( )")
        ops = [t.text for t in toks if t.kind == "op"]
        assert "<=", ">=" in ops
        assert "==" in ops

    def test_tokenizes_identifier(self):
        toks = _tokenize("my_var_123")
        assert toks[0].kind == "ident"
        assert toks[0].text == "my_var_123"

    def test_unexpected_character_raises(self):
        with pytest.raises(GuardParseError, match="Unexpected character"):
            _tokenize("@")

    def test_eof_token_always_appended(self):
        toks = _tokenize("x")
        assert toks[-1].kind == "eof"


class TestCompileAndEvaluate:
    def _eval(self, expr: str, state: dict) -> bool:
        ok, _ = evaluate_guard(expr, state)
        return ok

    def test_integer_literal_true(self):
        assert self._eval("1", {}) is True

    def test_zero_literal_false(self):
        ok, _ = evaluate_guard("0", {})
        assert ok is False

    def test_true_keyword(self):
        assert self._eval("true", {}) is True

    def test_false_keyword(self):
        assert self._eval("false", {}) is False

    def test_none_keyword(self):
        ok, _ = evaluate_guard("none", {})
        assert ok is False

    def test_variable_lookup(self):
        assert self._eval("x", {"x": 1}) is True

    def test_missing_variable_is_none(self):
        ok, _ = evaluate_guard("x", {})
        assert ok is False  # None is falsy

    def test_equality_number(self):
        assert self._eval("x == 5", {"x": 5}) is True
        assert self._eval("x == 5", {"x": 6}) is False

    def test_equality_string(self):
        assert self._eval("x == 'hello'", {"x": "hello"}) is True
        assert self._eval("x == 'hello'", {"x": "world"}) is False

    def test_inequality(self):
        assert self._eval("x != 5", {"x": 3}) is True
        assert self._eval("x != 5", {"x": 5}) is False

    def test_less_than(self):
        assert self._eval("x < 10", {"x": 5}) is True
        assert self._eval("x < 10", {"x": 10}) is False

    def test_less_than_or_equal(self):
        assert self._eval("x <= 10", {"x": 10}) is True
        assert self._eval("x <= 10", {"x": 11}) is False

    def test_greater_than(self):
        assert self._eval("x > 5", {"x": 6}) is True
        assert self._eval("x > 5", {"x": 5}) is False

    def test_greater_than_or_equal(self):
        assert self._eval("x >= 5", {"x": 5}) is True
        assert self._eval("x >= 5", {"x": 4}) is False

    def test_and_both_true(self):
        assert self._eval("x > 0 and y > 0", {"x": 1, "y": 2}) is True

    def test_and_one_false(self):
        assert self._eval("x > 0 and y > 0", {"x": 1, "y": 0}) is False

    def test_or_one_true(self):
        assert self._eval("x > 0 or y > 0", {"x": 0, "y": 1}) is True

    def test_or_both_false(self):
        assert self._eval("x > 0 or y > 0", {"x": 0, "y": 0}) is False

    def test_not_true(self):
        assert self._eval("not false", {}) is True

    def test_not_false(self):
        assert self._eval("not true", {}) is False

    def test_not_variable(self):
        assert self._eval("not x", {"x": 0}) is True
        assert self._eval("not x", {"x": 1}) is False

    def test_parentheses_change_precedence(self):
        # Without parens: not (a and b) vs (not a) and b
        assert self._eval("not (x == 1 and y == 1)", {"x": 1, "y": 1}) is False
        assert self._eval("not (x == 1 and y == 1)", {"x": 1, "y": 0}) is True

    def test_addition(self):
        assert self._eval("x + y == 10", {"x": 3, "y": 7}) is True

    def test_subtraction(self):
        assert self._eval("x - y > 0", {"x": 10, "y": 3}) is True

    def test_unary_minus(self):
        assert self._eval("-x < 0", {"x": 5}) is True

    def test_unary_plus(self):
        assert self._eval("+x > 0", {"x": 1}) is True

    def test_string_literal_in_comparison(self):
        assert self._eval("status == 'active'", {"status": "active"}) is True

    def test_non_numeric_inequality_falls_through(self):
        assert self._eval("name != 'bob'", {"name": "alice"}) is True

    def test_non_numeric_order_raises(self):
        with pytest.raises(GuardParseError, match="Non-numeric comparison"):
            evaluate_guard("name > 'alice'", {"name": "alice"})

    def test_complex_expression(self):
        expr = "pages >= 3 and tested == true and errors <= 0"
        assert self._eval(expr, {"pages": 5, "tested": True, "errors": 0}) is True
        assert self._eval(expr, {"pages": 5, "tested": True, "errors": 1}) is False
        assert self._eval(expr, {"pages": 1, "tested": True, "errors": 0}) is False

    def test_extra_token_raises(self):
        with pytest.raises(GuardParseError, match="Unexpected token"):
            compile_guard("x y")

    def test_unexpected_primary_raises(self):
        with pytest.raises(GuardParseError, match="Unexpected token"):
            compile_guard("==")

    def test_evaluate_guard_returns_message_on_fail(self):
        ok, msg = evaluate_guard("x > 10", {"x": 1})
        assert ok is False
        assert "Guard failed" in msg

    def test_evaluate_guard_returns_ok_on_pass(self):
        ok, msg = evaluate_guard("x > 0", {"x": 5})
        assert ok is True
        assert msg == "ok"

    def test_evaluate_guard_propagates_runtime_error(self):
        with pytest.raises(GuardParseError):
            evaluate_guard("x > 'text'", {"x": 5})

    def test_numeric_coercion_for_string_number(self):
        assert self._eval("x < 10", {"x": "5"}) is True

    def test_double_not(self):
        assert self._eval("not not true", {}) is True


class TestSingleIdentifierShorthand:
    """Single identifier shorthand: 'tested' expands to 'tested == true'."""

    def test_shorthand_truthy(self):
        ok, _ = evaluate_guard("tested", {"tested": True})
        assert ok is True

    def test_shorthand_falsy(self):
        ok, _ = evaluate_guard("tested", {"tested": False})
        assert ok is False


class TestKernelPreconditionFromActions:
    def _make_action(self, action_id: str, precondition: Optional[str] = None):
        class FakeAction:
            def __init__(self, aid: str, pre):
                self.id = aid
                self.preconditions_text = pre
        return FakeAction(action_id, precondition)

    def test_no_precondition_always_passes(self):
        a = self._make_action("deploy", None)
        checker = kernel_precondition_from_actions(all_actions=[a])
        ok, _msg = checker({"tested": True}, a)
        assert ok is True

    def test_empty_precondition_passes(self):
        a = self._make_action("deploy", "")
        checker = kernel_precondition_from_actions(all_actions=[a])
        ok, _ = checker({"tested": True}, a)
        assert ok is True

    def test_satisfied_precondition_passes(self):
        a = self._make_action("deploy", "tested == true")
        checker = kernel_precondition_from_actions(all_actions=[a])
        state = State({"tested": True})
        ok, _ = checker(state, a)
        assert ok is True

    def test_failed_precondition_rejects(self):
        a = self._make_action("deploy", "tested == true")
        checker = kernel_precondition_from_actions(all_actions=[a])
        state = State({"tested": False})
        ok, _ = checker(state, a)
        assert ok is False

    def test_single_identifier_shorthand_truthy(self):
        a = self._make_action("deploy", "tested")
        checker = kernel_precondition_from_actions(all_actions=[a])
        ok, _ = checker(State({"tested": True}), a)
        assert ok is True

    def test_single_identifier_shorthand_falsy(self):
        a = self._make_action("deploy", "tested")
        checker = kernel_precondition_from_actions(all_actions=[a])
        ok, _ = checker(State({"tested": False}), a)
        assert ok is False

    def test_state_dict_input(self):
        a = self._make_action("deploy", "tested == true")
        checker = kernel_precondition_from_actions(all_actions=[a])
        ok, _ = checker({"tested": True}, a)
        assert ok is True

    def test_unknown_action_passes(self):
        a = self._make_action("deploy", "tested == true")
        other = self._make_action("rollback", None)
        checker = kernel_precondition_from_actions(all_actions=[a])
        ok, _ = checker(State({}), other)
        assert ok is True

    def test_invalid_precondition_returns_error(self):
        a = self._make_action("deploy", "x > 'not_a_number'")
        checker = kernel_precondition_from_actions(all_actions=[a])
        ok, msg = checker(State({"x": 5}), a)
        assert ok is False
        assert "Invalid precondition" in msg

    def test_state_with_to_dict_method(self):
        a = self._make_action("deploy", "tested == true")
        checker = kernel_precondition_from_actions(all_actions=[a])
        state = State({"tested": True})  # State has to_dict()
        ok, _ = checker(state, a)
        assert ok is True

    def test_multiple_actions_correct_guard_selected(self):
        a1 = self._make_action("fetch", "ready == true")
        a2 = self._make_action("deploy", "tested == true")
        checker = kernel_precondition_from_actions(all_actions=[a1, a2])
        s = State({"ready": True, "tested": False})
        ok1, _ = checker(s, a1)
        ok2, _ = checker(s, a2)
        assert ok1 is True
        assert ok2 is False
