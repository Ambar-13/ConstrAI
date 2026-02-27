"""Tests for new distributed and async classes.

Covers ProcessSharedBudgetController (formal.py), AsyncSafetyKernel (formal.py),
AsyncAnthropicAdapter (adapters/anthropic_adapter.py),
AsyncOpenAIAdapter (adapters/openai_adapter.py), and
MockLLMAdapter.acomplete() (reasoning.py).
"""
from __future__ import annotations

import asyncio
import json
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from constrai.adapters.anthropic_adapter import AsyncAnthropicAdapter
from constrai.adapters.openai_adapter import AsyncOpenAIAdapter
from constrai.formal import (
    ActionSpec,
    AsyncSafetyKernel,
    BudgetController,
    Effect,
    ExecutionTrace,
    Invariant,
    ProcessSharedBudgetController,
    State,
)
from constrai.reasoning import MockLLMAdapter

# ─── helpers ─────────────────────────────────────────────────────────────────


def _action(aid: str = "act", cost: float = 1.0) -> ActionSpec:
    return ActionSpec(
        id=aid,
        name=aid,
        description="",
        effects=(Effect("x", "increment", 1),),
        cost=cost,
    )


def _invariants() -> List[Invariant]:
    return [Invariant("always_ok", lambda s: True, "always passes")]


# ─── ProcessSharedBudgetController ───────────────────────────────────────────


class TestProcessSharedBudgetControllerInit:
    def test_valid_budget(self):
        bc = ProcessSharedBudgetController(100.0)
        assert bc.budget == pytest.approx(100.0)

    def test_zero_budget(self):
        bc = ProcessSharedBudgetController(0.0)
        assert bc.budget == pytest.approx(0.0)

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="≥ 0"):
            ProcessSharedBudgetController(-1.0)

    def test_initial_spent_gross_zero(self):
        bc = ProcessSharedBudgetController(100.0)
        assert bc.spent_gross == pytest.approx(0.0)

    def test_initial_spent_refunded_zero(self):
        bc = ProcessSharedBudgetController(100.0)
        assert bc.spent_refunded == pytest.approx(0.0)

    def test_initial_spent_net_zero(self):
        bc = ProcessSharedBudgetController(100.0)
        assert bc.spent_net == pytest.approx(0.0)

    def test_initial_spent_alias_zero(self):
        bc = ProcessSharedBudgetController(100.0)
        assert bc.spent == pytest.approx(0.0)

    def test_initial_remaining_equals_budget(self):
        bc = ProcessSharedBudgetController(50.0)
        assert bc.remaining == pytest.approx(50.0)

    def test_initial_utilization_zero(self):
        bc = ProcessSharedBudgetController(100.0)
        assert bc.utilization() == pytest.approx(0.0)

    def test_initial_ledger_empty(self):
        bc = ProcessSharedBudgetController(100.0)
        assert bc.ledger == []


class TestProcessSharedBudgetControllerCanAfford:
    def test_can_afford_within_budget(self):
        bc = ProcessSharedBudgetController(100.0)
        ok, msg = bc.can_afford(50.0)
        assert ok is True
        assert msg == ""

    def test_can_afford_exact_budget(self):
        bc = ProcessSharedBudgetController(10.0)
        ok, _ = bc.can_afford(10.0)
        assert ok is True

    def test_cannot_afford_over_budget(self):
        bc = ProcessSharedBudgetController(10.0)
        ok, msg = bc.can_afford(10.01)
        assert ok is False
        assert "Cannot afford" in msg

    def test_can_afford_negative_cost_raises(self):
        bc = ProcessSharedBudgetController(100.0)
        with pytest.raises(ValueError, match="≥ 0"):
            bc.can_afford(-1.0)

    def test_can_afford_after_partial_charge(self):
        bc = ProcessSharedBudgetController(10.0)
        bc.charge("a", 6.0)
        ok, _ = bc.can_afford(4.0)
        assert ok is True
        ok2, _ = bc.can_afford(4.01)
        assert ok2 is False


class TestProcessSharedBudgetControllerCharge:
    def test_charge_increases_spent_gross(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 10.0)
        assert bc.spent_gross == pytest.approx(10.0)

    def test_charge_decreases_remaining(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 25.0)
        assert bc.remaining == pytest.approx(75.0)

    def test_charge_updates_spent_net(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 30.0)
        assert bc.spent_net == pytest.approx(30.0)
        assert bc.spent == pytest.approx(30.0)

    def test_charge_records_in_ledger(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("my_action", 5.0)
        entries = bc.ledger
        assert len(entries) == 1
        assert entries[0][0] == "my_action"
        assert entries[0][1] == pytest.approx(5.0)

    def test_charge_negative_cost_raises(self):
        bc = ProcessSharedBudgetController(100.0)
        with pytest.raises(ValueError, match="≥ 0"):
            bc.charge("a", -1.0)

    def test_charge_exceeds_budget_raises(self):
        bc = ProcessSharedBudgetController(10.0)
        with pytest.raises(RuntimeError, match="BUDGET SAFETY VIOLATION"):
            bc.charge("a", 10.01)

    def test_multiple_charges_accumulate(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 10.0)
        bc.charge("b", 20.0)
        assert bc.spent_gross == pytest.approx(30.0)
        assert bc.remaining == pytest.approx(70.0)

    def test_utilization_after_charge(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 50.0)
        assert bc.utilization() == pytest.approx(0.5)


class TestProcessSharedBudgetControllerRefund:
    def test_refund_reduces_spent_net(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 20.0)
        bc.refund("a", 10.0)
        assert bc.spent_net == pytest.approx(10.0)
        assert bc.spent_refunded == pytest.approx(10.0)

    def test_refund_does_not_change_spent_gross(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 20.0)
        bc.refund("a", 5.0)
        assert bc.spent_gross == pytest.approx(20.0)

    def test_refund_increases_remaining(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 20.0)
        bc.refund("a", 10.0)
        assert bc.remaining == pytest.approx(90.0)

    def test_refund_capped_at_gross(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 10.0)
        bc.refund("a", 100.0)  # try to refund more than charged — silently dropped
        assert bc.spent_refunded == pytest.approx(0.0)
        assert bc.spent_gross == pytest.approx(10.0)

    def test_refund_negative_cost_raises(self):
        bc = ProcessSharedBudgetController(100.0)
        with pytest.raises(ValueError):
            bc.refund("a", -5.0)

    def test_refund_records_in_ledger(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 10.0)
        bc.refund("a", 5.0)
        ledger = bc.ledger
        assert any("REFUND:a" in str(e[0]) for e in ledger)


class TestProcessSharedBudgetControllerSummary:
    def test_summary_returns_string(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 10.0)
        s = bc.summary()
        assert isinstance(s, str)
        assert "Budget" in s
        assert "Remaining" in s

    def test_ledger_returns_copy(self):
        bc = ProcessSharedBudgetController(100.0)
        bc.charge("a", 5.0)
        ledger = bc.ledger
        ledger.clear()
        assert len(bc.ledger) == 1  # original unaffected


# ─── AsyncSafetyKernel ────────────────────────────────────────────────────────


class TestAsyncSafetyKernelInit:
    def test_budget_property_returns_budget_controller(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        assert isinstance(k.budget, BudgetController)

    def test_invariants_property(self):
        invs = _invariants()
        k = AsyncSafetyKernel(50.0, invs)
        assert len(k.invariants) == 1

    def test_step_count_starts_zero(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        assert k.step_count == 0

    def test_max_steps_accessible(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        assert k.max_steps > 0

    def test_trace_is_execution_trace(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        assert isinstance(k.trace, ExecutionTrace)

    def test_no_lock_before_first_call(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        assert k._alock is None


class TestAsyncSafetyKernelEvaluate:
    def test_evaluate_approves_valid_action(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        state = State({"x": 0})
        action = _action(cost=1.0)
        verdict = asyncio.run(
            k.evaluate(state, action)
        )
        assert verdict.approved is True

    def test_evaluate_rejects_when_over_budget(self):
        k = AsyncSafetyKernel(0.5, _invariants())
        state = State({"x": 0})
        action = _action(cost=1.0)
        verdict = asyncio.run(
            k.evaluate(state, action)
        )
        assert verdict.approved is False

    def test_evaluate_does_not_increment_step_count(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        state = State({"x": 0})
        action = _action()
        asyncio.run(k.evaluate(state, action))
        assert k.step_count == 0

    def test_evaluate_no_budget_consumed(self):
        k = AsyncSafetyKernel(10.0, _invariants())
        state = State({"x": 0})
        action = _action(cost=5.0)
        asyncio.run(k.evaluate(state, action))
        assert k.budget.remaining == pytest.approx(10.0)


class TestAsyncSafetyKernelExecuteAtomic:
    def test_execute_atomic_increments_step_count(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        state = State({"x": 0})
        action = _action()
        asyncio.run(
            k.execute_atomic(state, action)
        )
        assert k.step_count == 1

    def test_execute_atomic_charges_budget(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        state = State({"x": 0})
        action = _action(cost=5.0)
        asyncio.run(
            k.execute_atomic(state, action)
        )
        assert k.budget.spent == pytest.approx(5.0)

    def test_execute_atomic_returns_new_state_and_entry(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        state = State({"x": 0})
        action = _action()
        new_state, entry = asyncio.run(
            k.execute_atomic(state, action)
        )
        assert isinstance(new_state, State)
        assert entry is not None

    def test_execute_atomic_creates_lock(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        state = State({"x": 0})
        action = _action()
        asyncio.run(k.execute_atomic(state, action))
        assert k._alock is not None

    def test_execute_atomic_raises_on_rejected(self):
        k = AsyncSafetyKernel(0.5, _invariants())
        state = State({"x": 0})
        action = _action(cost=1.0)
        with pytest.raises(RuntimeError):
            asyncio.run(
                k.execute_atomic(state, action)
            )

    def test_concurrent_execute_atomic(self):
        """Multiple coroutines serialize correctly via asyncio.Lock."""
        k = AsyncSafetyKernel(100.0, _invariants())
        state = State({"x": 0})
        action = _action(cost=1.0)

        async def run_n(n: int) -> None:
            for _ in range(n):
                await k.execute_atomic(state, action)

        async def run_all() -> None:
            await asyncio.gather(run_n(3), run_n(3))

        asyncio.run(run_all())
        assert k.step_count == 6
        assert k.budget.spent == pytest.approx(6.0)


class TestAsyncSafetyKernelDelegateMethods:
    def test_add_precondition(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        called = []
        k.add_precondition(lambda s, a: (True, ""))
        # No assertion needed; just checking no error raised

    def test_rollback(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        state_before = State({"x": 0})
        state = State({"x": 0})
        action = _action()
        new_state, _ = asyncio.run(
            k.execute_atomic(state, action)
        )
        rolled = k.rollback(state_before, new_state, action)
        assert isinstance(rolled, State)

    def test_status_returns_string(self):
        k = AsyncSafetyKernel(50.0, _invariants())
        s = k.status()
        assert isinstance(s, str)

    def test_get_lock_reuses_same_instance(self):
        k = AsyncSafetyKernel(50.0, _invariants())

        async def get_two_locks():
            lock1 = k._get_lock()
            lock2 = k._get_lock()
            return lock1 is lock2

        same = asyncio.run(get_two_locks())
        assert same is True


# ─── MockLLMAdapter.acomplete ─────────────────────────────────────────────────


class TestMockLLMAdapterAcomplete:
    def test_acomplete_returns_valid_json(self):
        adapter = MockLLMAdapter()
        prompt = "Action: do_thing [READY] (id=do_thing)\nGoal: 0.0%"
        result = asyncio.run(
            adapter.acomplete(prompt)
        )
        parsed = json.loads(result)
        assert "chosen_action_id" in parsed

    def test_acomplete_matches_complete(self):
        adapter = MockLLMAdapter()
        prompt = "Action: do_thing [READY] (id=do_thing)\nGoal: 0.0%"
        sync_result = adapter.complete(prompt)
        adapter2 = MockLLMAdapter()
        async_result = asyncio.run(
            adapter2.acomplete(prompt)
        )
        assert json.loads(sync_result)["chosen_action_id"] == json.loads(async_result)["chosen_action_id"]

    def test_acomplete_stops_at_goal(self):
        adapter = MockLLMAdapter()
        prompt = "Goal achieved 100.0%"
        result = asyncio.run(
            adapter.acomplete(prompt)
        )
        parsed = json.loads(result)
        assert parsed["should_stop"] is True

    def test_acomplete_no_actions_stops(self):
        adapter = MockLLMAdapter()
        prompt = "No actions here. Goal: 0.0%"
        result = asyncio.run(
            adapter.acomplete(prompt)
        )
        parsed = json.loads(result)
        assert parsed["should_stop"] is True


# ─── AsyncAnthropicAdapter ────────────────────────────────────────────────────


class TestAsyncAnthropicAdapterInterface:
    def test_complete_raises_not_implemented(self):
        client = MagicMock()
        adapter = AsyncAnthropicAdapter(client)
        with pytest.raises(NotImplementedError, match="async-only"):
            adapter.complete("hello")

    def test_repr(self):
        client = MagicMock()
        adapter = AsyncAnthropicAdapter(client, model="claude-opus-4-6")
        assert "claude-opus-4-6" in repr(adapter)

    def test_default_model(self):
        client = MagicMock()
        adapter = AsyncAnthropicAdapter(client)
        assert adapter._model == AsyncAnthropicAdapter.DEFAULT_MODEL


class TestAsyncAnthropicAdapterAcomplete:
    def test_acomplete_non_streaming(self):
        client = MagicMock()
        fake_content = MagicMock()
        fake_content.text = "result text"
        fake_msg = MagicMock()
        fake_msg.content = [fake_content]
        client.messages.create = AsyncMock(return_value=fake_msg)

        adapter = AsyncAnthropicAdapter(client)
        result = asyncio.run(
            adapter.acomplete("Hello")
        )
        assert result == "result text"
        client.messages.create.assert_awaited_once()

    def test_acomplete_uses_default_system_prompt_when_empty(self):
        client = MagicMock()
        fake_content = MagicMock()
        fake_content.text = "ok"
        fake_msg = MagicMock()
        fake_msg.content = [fake_content]
        client.messages.create = AsyncMock(return_value=fake_msg)

        adapter = AsyncAnthropicAdapter(client, default_system_prompt="custom sys")
        asyncio.run(adapter.acomplete("p"))
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "custom sys"

    def test_acomplete_custom_system_prompt(self):
        client = MagicMock()
        fake_content = MagicMock()
        fake_content.text = "ok"
        fake_msg = MagicMock()
        fake_msg.content = [fake_content]
        client.messages.create = AsyncMock(return_value=fake_msg)

        adapter = AsyncAnthropicAdapter(client)
        asyncio.run(
            adapter.acomplete("p", system_prompt="override sys")
        )
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "override sys"

    def test_acomplete_streaming(self):
        client = MagicMock()
        tokens = []

        async def fake_stream():
            for t in ["hello", " world"]:
                yield t

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=None)
        ctx.text_stream = fake_stream()
        client.messages.stream = MagicMock(return_value=ctx)

        adapter = AsyncAnthropicAdapter(client)

        def collect(t: str) -> None:
            tokens.append(t)

        result = asyncio.run(
            adapter.acomplete("p", stream_tokens=collect)
        )
        assert result == "hello world"
        assert tokens == ["hello", " world"]


# ─── AsyncOpenAIAdapter ───────────────────────────────────────────────────────


class TestAsyncOpenAIAdapterInterface:
    def test_complete_raises_not_implemented(self):
        client = MagicMock()
        adapter = AsyncOpenAIAdapter(client)
        with pytest.raises(NotImplementedError, match="async-only"):
            adapter.complete("hello")

    def test_repr(self):
        client = MagicMock()
        adapter = AsyncOpenAIAdapter(client, model="gpt-4o")
        assert "gpt-4o" in repr(adapter)

    def test_default_model(self):
        client = MagicMock()
        adapter = AsyncOpenAIAdapter(client)
        assert adapter._model == AsyncOpenAIAdapter.DEFAULT_MODEL


class TestAsyncOpenAIAdapterAcomplete:
    def test_acomplete_non_streaming(self):
        client = MagicMock()
        fake_choice = MagicMock()
        fake_choice.message.content = "openai result"
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        client.chat.completions.create = AsyncMock(return_value=fake_response)

        adapter = AsyncOpenAIAdapter(client)
        result = asyncio.run(
            adapter.acomplete("Hello")
        )
        assert result == "openai result"
        client.chat.completions.create.assert_awaited_once()

    def test_acomplete_uses_default_system_prompt_when_empty(self):
        client = MagicMock()
        fake_choice = MagicMock()
        fake_choice.message.content = "ok"
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        client.chat.completions.create = AsyncMock(return_value=fake_response)

        adapter = AsyncOpenAIAdapter(client, default_system_prompt="sys default")
        asyncio.run(adapter.acomplete("p"))
        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        sys_msgs = [m for m in messages if m["role"] == "system"]
        assert sys_msgs[0]["content"] == "sys default"

    def test_acomplete_custom_system_prompt(self):
        client = MagicMock()
        fake_choice = MagicMock()
        fake_choice.message.content = "ok"
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        client.chat.completions.create = AsyncMock(return_value=fake_response)

        adapter = AsyncOpenAIAdapter(client)
        asyncio.run(
            adapter.acomplete("p", system_prompt="my sys")
        )
        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        sys_msgs = [m for m in messages if m["role"] == "system"]
        assert sys_msgs[0]["content"] == "my sys"

    def test_acomplete_none_content_returns_empty_string(self):
        client = MagicMock()
        fake_choice = MagicMock()
        fake_choice.message.content = None
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        client.chat.completions.create = AsyncMock(return_value=fake_response)

        adapter = AsyncOpenAIAdapter(client)
        result = asyncio.run(
            adapter.acomplete("Hello")
        )
        assert result == ""

    def test_acomplete_streaming(self):
        client = MagicMock()
        tokens = []

        async def fake_stream():
            for token in ["foo", "bar"]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = token
                yield chunk

        client.chat.completions.create = AsyncMock(return_value=fake_stream())

        adapter = AsyncOpenAIAdapter(client)

        def collect(t: str) -> None:
            tokens.append(t)

        result = asyncio.run(
            adapter.acomplete("Hello", stream_tokens=collect)
        )
        assert result == "foobar"
        assert tokens == ["foo", "bar"]

    def test_acomplete_streaming_skips_none_deltas(self):
        client = MagicMock()
        tokens = []

        async def fake_stream():
            for token in [None, "real", None]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = token
                yield chunk

        client.chat.completions.create = AsyncMock(return_value=fake_stream())

        adapter = AsyncOpenAIAdapter(client)

        def collect(t: str) -> None:
            tokens.append(t)

        result = asyncio.run(
            adapter.acomplete("Hello", stream_tokens=collect)
        )
        assert result == "real"
        assert tokens == ["real"]
