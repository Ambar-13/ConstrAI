"""
Tests for clampai.adapters.autogen_adapter.

Covers:
  - Exception hierarchy (ClampAIAutoGenError, Budget, Invariant subclasses)
  - ClampAISafeAutoGenAgent initialisation (budget, cost_per_reply, agent_name, fn=None)
  - check() method: budget charging, step counting, state fields, extra_state merge,
    budget exhaustion, invariant violation
  - __call__() method: message extraction from positional list, from kwargs, fn=None path,
    return value wrapping, tuple passthrough, error propagation
  - reset(): kernel rebuilt, step_count cleared, reply_count cleared
  - autogen_reply_fn decorator: returns ClampAISafeAutoGenAgent, all params forwarded,
    agent_name defaults to fn.__name__
"""

from __future__ import annotations

import pytest

from clampai.adapters.autogen_adapter import (
    ClampAIAutoGenBudgetError,
    ClampAIAutoGenError,
    ClampAIAutoGenInvariantError,
    ClampAISafeAutoGenAgent,
    autogen_reply_fn,
)
from clampai.formal import Invariant, State


def _blocking_invariant(name: str, pred) -> Invariant:
    return Invariant(name, pred, f"{name} invariant", enforcement="blocking")


class TestExceptionHierarchy:
    def test_base_is_runtime_error(self):
        assert issubclass(ClampAIAutoGenError, RuntimeError)

    def test_budget_error_is_base(self):
        assert issubclass(ClampAIAutoGenBudgetError, ClampAIAutoGenError)

    def test_invariant_error_is_base(self):
        assert issubclass(ClampAIAutoGenInvariantError, ClampAIAutoGenError)

    def test_can_raise_and_catch_base(self):
        with pytest.raises(ClampAIAutoGenError):
            raise ClampAIAutoGenError("base error")

    def test_can_raise_and_catch_budget_as_base(self):
        with pytest.raises(ClampAIAutoGenError):
            raise ClampAIAutoGenBudgetError("budget gone")

    def test_can_raise_and_catch_invariant_as_base(self):
        with pytest.raises(ClampAIAutoGenError):
            raise ClampAIAutoGenInvariantError("invariant blocked")


class TestClampAISafeAutoGenAgentInit:
    def test_budget_remaining_equals_budget_at_start(self):
        agent = ClampAISafeAutoGenAgent(None, budget=50.0)
        assert agent.budget_remaining == pytest.approx(50.0)

    def test_step_count_zero_at_start(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0)
        assert agent.step_count == 0

    def test_agent_name_stored(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0, agent_name="my_bot")
        assert agent.agent_name == "my_bot"

    def test_default_agent_name(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0)
        assert agent.agent_name == "autogen_agent"

    def test_fn_none_allowed(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0)
        assert agent.fn is None

    def test_fn_callable_stored(self):
        def fn(msgs: list) -> str:
            return "hello"

        agent = ClampAISafeAutoGenAgent(fn, budget=10.0)
        assert agent.fn is fn

    def test_repr_contains_key_fields(self):
        agent = ClampAISafeAutoGenAgent(None, budget=25.0, cost_per_reply=2.5, agent_name="bot")
        r = repr(agent)
        assert "bot" in r
        assert "25.0" in r
        assert "2.5" in r


class TestCheckMethod:
    def test_charges_budget(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0, cost_per_reply=3.0)
        agent.check()
        assert agent.budget_remaining == pytest.approx(7.0)

    def test_increments_step_count(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0)
        agent.check()
        agent.check()
        assert agent.step_count == 2

    def test_message_in_state(self):
        seen_states = []
        inv = _blocking_invariant(
            "capture_msg",
            lambda s: (seen_states.append(dict(s._vars)) or True),
        )
        agent = ClampAISafeAutoGenAgent(None, budget=10.0, invariants=[inv])
        agent.check(message="hello world")
        assert any(d.get("message") == "hello world" for d in seen_states)

    def test_sender_in_state(self):
        seen_states = []
        inv = _blocking_invariant(
            "capture_sender",
            lambda s: (seen_states.append(dict(s._vars)) or True),
        )
        agent = ClampAISafeAutoGenAgent(None, budget=10.0, invariants=[inv])
        agent.check(sender="alice")
        assert any(d.get("sender") == "alice" for d in seen_states)

    def test_reply_count_in_state(self):
        seen_states = []
        inv = _blocking_invariant(
            "capture_count",
            lambda s: (seen_states.append(dict(s._vars)) or True),
        )
        agent = ClampAISafeAutoGenAgent(None, budget=10.0, invariants=[inv])
        agent.check()
        agent.check()
        counts = [d.get("reply_count") for d in seen_states]
        assert 1 in counts
        assert 2 in counts

    def test_extra_state_merged(self):
        seen_states = []
        inv = _blocking_invariant(
            "capture_extra",
            lambda s: (seen_states.append(dict(s._vars)) or True),
        )
        agent = ClampAISafeAutoGenAgent(None, budget=10.0, invariants=[inv])
        agent.check(extra_state={"custom_field": "custom_value"})
        assert any(d.get("custom_field") == "custom_value" for d in seen_states)

    def test_budget_exhaustion_raises_budget_error(self):
        agent = ClampAISafeAutoGenAgent(None, budget=2.0, cost_per_reply=1.0)
        agent.check()
        agent.check()
        with pytest.raises(ClampAIAutoGenBudgetError):
            agent.check()

    def test_invariant_violation_raises_invariant_error(self):
        always_fail = _blocking_invariant("always_fail", lambda s: False)
        agent = ClampAISafeAutoGenAgent(None, budget=100.0, invariants=[always_fail])
        with pytest.raises(ClampAIAutoGenInvariantError):
            agent.check()

    def test_multiple_checks_track_correctly(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0, cost_per_reply=2.0)
        for _ in range(3):
            agent.check()
        assert agent.step_count == 3
        assert agent.budget_remaining == pytest.approx(4.0)

    def test_check_with_fn_none_works(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0)
        agent.check(message="test")
        assert agent.step_count == 1


class TestCallMethod:
    def test_fn_set_returns_true_str_tuple(self):
        agent = ClampAISafeAutoGenAgent(lambda *a, **k: "reply text", budget=10.0)
        result = agent([{"content": "hi"}])
        assert result == (True, "reply text")

    def test_fn_none_returns_false_none(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0)
        result = agent([{"content": "hi"}])
        assert result == (False, None)

    def test_messages_extracted_from_first_list_positional_arg(self):
        captured = []
        inv = _blocking_invariant(
            "capture_call_msg",
            lambda s: (captured.append(s.get("message", "")) or True),
        )
        agent = ClampAISafeAutoGenAgent(
            lambda *a, **k: "ok",
            budget=10.0,
            invariants=[inv],
        )
        agent("ignored_str", [{"content": "extracted content"}], "other")
        assert "extracted content" in captured

    def test_messages_extracted_from_kwargs(self):
        captured = []
        inv = _blocking_invariant(
            "capture_kwarg_msg",
            lambda s: (captured.append(s.get("message", "")) or True),
        )
        agent = ClampAISafeAutoGenAgent(
            lambda *a, **k: "ok",
            budget=10.0,
            invariants=[inv],
        )
        agent(messages=[{"content": "kwarg content"}])
        assert "kwarg content" in captured

    def test_no_messages_passes_empty_string_to_check(self):
        captured = []
        inv = _blocking_invariant(
            "capture_empty",
            lambda s: (captured.append(s.get("message")) or True),
        )
        agent = ClampAISafeAutoGenAgent(
            lambda *a, **k: "ok",
            budget=10.0,
            invariants=[inv],
        )
        agent()
        assert "" in captured

    def test_fn_returning_tuple_passed_through_unchanged(self):
        agent = ClampAISafeAutoGenAgent(lambda *a, **k: (True, "already_tuple"), budget=10.0)
        result = agent([{"content": "msg"}])
        assert result == (True, "already_tuple")

    def test_fn_returning_str_wrapped_as_true_str(self):
        agent = ClampAISafeAutoGenAgent(lambda *a, **k: "string result", budget=10.0)
        result = agent([{"content": "msg"}])
        assert result[0] is True
        assert result[1] == "string result"

    def test_budget_exhaustion_during_call_raises_budget_error(self):
        agent = ClampAISafeAutoGenAgent(
            lambda *a, **k: "reply",
            budget=1.5,
            cost_per_reply=1.0,
        )
        agent([{"content": "msg1"}])
        with pytest.raises(ClampAIAutoGenBudgetError):
            agent([{"content": "msg2"}])

    def test_invariant_violation_during_call_raises_invariant_error(self):
        always_fail = _blocking_invariant("always_fail_call", lambda s: False)
        agent = ClampAISafeAutoGenAgent(
            lambda *a, **k: "reply",
            budget=100.0,
            invariants=[always_fail],
        )
        with pytest.raises(ClampAIAutoGenInvariantError):
            agent([{"content": "blocked"}])

    def test_step_count_increments_after_call(self):
        agent = ClampAISafeAutoGenAgent(lambda *a, **k: "ok", budget=10.0)
        agent([{"content": "msg"}])
        agent([{"content": "msg2"}])
        assert agent.step_count == 2


class TestResetMethod:
    def test_budget_restored_after_reset(self):
        agent = ClampAISafeAutoGenAgent(None, budget=5.0, cost_per_reply=2.0)
        agent.check()
        agent.check()
        assert agent.budget_remaining == pytest.approx(1.0)
        agent.reset()
        assert agent.budget_remaining == pytest.approx(5.0)

    def test_step_count_cleared_after_reset(self):
        agent = ClampAISafeAutoGenAgent(None, budget=10.0)
        agent.check()
        agent.check()
        assert agent.step_count == 2
        agent.reset()
        assert agent.step_count == 0

    def test_new_check_after_exhausted_budget_works_post_reset(self):
        agent = ClampAISafeAutoGenAgent(None, budget=2.0, cost_per_reply=1.0)
        agent.check()
        agent.check()
        with pytest.raises(ClampAIAutoGenBudgetError):
            agent.check()
        agent.reset()
        agent.check()
        assert agent.step_count == 1

    def test_reply_count_resets(self):
        seen_counts = []
        inv = _blocking_invariant(
            "capture_reply_count",
            lambda s: (seen_counts.append(s.get("reply_count")) or True),
        )
        agent = ClampAISafeAutoGenAgent(None, budget=20.0, invariants=[inv])
        agent.check()
        agent.check()
        agent.reset()
        agent.check()
        assert seen_counts[-1] == 1


class TestAutogenReplyFnDecorator:
    def test_result_is_clampai_safe_autogen_agent(self):
        @autogen_reply_fn(budget=10.0)
        def my_fn(messages):
            return "hello"

        assert isinstance(my_fn, ClampAISafeAutoGenAgent)

    def test_budget_passed_through(self):
        @autogen_reply_fn(budget=77.0)
        def my_fn(messages):
            return "x"

        assert my_fn.budget == pytest.approx(77.0)
        assert my_fn.budget_remaining == pytest.approx(77.0)

    def test_cost_per_reply_passed_through(self):
        @autogen_reply_fn(budget=20.0, cost_per_reply=3.5)
        def my_fn(messages):
            return "x"

        assert my_fn.cost_per_reply == pytest.approx(3.5)

    def test_agent_name_defaults_to_fn_name(self):
        @autogen_reply_fn(budget=10.0)
        def specific_function_name(messages):
            return "x"

        assert specific_function_name.agent_name == "specific_function_name"

    def test_agent_name_override(self):
        @autogen_reply_fn(budget=10.0, agent_name="custom_agent")
        def my_fn(messages):
            return "x"

        assert my_fn.agent_name == "custom_agent"

    def test_invariants_passed_through(self):
        inv = _blocking_invariant("pass_inv", lambda s: True)

        @autogen_reply_fn(budget=10.0, invariants=[inv])
        def my_fn(messages):
            return "x"

        assert len(my_fn.invariants) == 1
        assert my_fn.invariants[0] is inv
