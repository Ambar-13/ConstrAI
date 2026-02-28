"""
Tests for the reconcile_fn parameter of SafetyKernel.

Covers: init storage, call semantics on execute() and
evaluate_and_execute_atomic(), return-value substitution, silent error
swallowing, and integration patterns including OTelTraceExporter.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from clampai.formal import ActionSpec, Effect, Invariant, SafetyKernel, State, TraceEntry
from clampai.testing import make_action, make_state


def _make_kernel(
    budget: float = 100.0,
    *,
    invariants: Optional[List[Invariant]] = None,
    reconcile_fn: Any = None,
    min_action_cost: float = 0.001,
) -> SafetyKernel:
    return SafetyKernel(
        budget,
        invariants or [],
        min_action_cost=min_action_cost,
        reconcile_fn=reconcile_fn,
    )


def _simple_action(cost: float = 1.0, **effects: Any) -> ActionSpec:
    return make_action("step", cost=cost, **effects)


class TestReconcileFnInit:

    def test_default_reconcile_fn_is_none(self) -> None:
        kernel = _make_kernel()
        assert kernel._reconcile_fn is None

    def test_reconcile_fn_stored_on_init(self) -> None:
        fn = lambda state, action, entry: None  # noqa: E731
        kernel = _make_kernel(reconcile_fn=fn)
        assert kernel._reconcile_fn is fn

    def test_explicit_none_same_as_default(self) -> None:
        kernel = _make_kernel(reconcile_fn=None)
        assert kernel._reconcile_fn is None

    def test_various_callable_types_accepted(self) -> None:
        def plain_fn(s: Any, a: Any, e: Any) -> None:
            return None

        lambda_fn = lambda s, a, e: None  # noqa: E731

        class CallableClass:
            def __call__(self, s: Any, a: Any, e: Any) -> None:
                return None

        for fn in (plain_fn, lambda_fn, CallableClass()):
            kernel = _make_kernel(reconcile_fn=fn)
            assert callable(kernel._reconcile_fn)


class TestReconcileFnCalledOnExecute:

    def _run_execute(
        self,
        kernel: SafetyKernel,
        state: State,
        action: ActionSpec,
    ) -> Tuple[State, TraceEntry]:
        verdict = kernel.evaluate(state, action)
        assert verdict.approved
        return kernel.execute(state, action)

    def test_reconcile_fn_called_after_execute(self) -> None:
        calls: List[Any] = []

        def fn(s: Any, a: Any, e: Any) -> None:
            calls.append((s, a, e))
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(x=0)
        action = _simple_action(cost=1.0, x=1)
        self._run_execute(kernel, state, action)
        assert len(calls) == 1

    def test_reconcile_fn_receives_correct_new_state(self) -> None:
        received: List[State] = []

        def fn(s: State, a: Any, e: Any) -> None:
            received.append(s)
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(count=0)
        action = _simple_action(cost=1.0, count=42)
        self._run_execute(kernel, state, action)
        assert len(received) == 1
        assert received[0].get("count") == 42

    def test_reconcile_fn_receives_correct_action(self) -> None:
        received_actions: List[ActionSpec] = []

        def fn(s: Any, a: ActionSpec, e: Any) -> None:
            received_actions.append(a)
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(v=0)
        action = _simple_action(cost=1.0, v=1)
        self._run_execute(kernel, state, action)
        assert received_actions[0].id == action.id

    def test_reconcile_fn_receives_trace_entry(self) -> None:
        received_entries: List[TraceEntry] = []

        def fn(s: Any, a: Any, e: TraceEntry) -> None:
            received_entries.append(e)
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(v=0)
        action = _simple_action(cost=1.0, v=1)
        self._run_execute(kernel, state, action)
        entry = received_entries[0]
        assert isinstance(entry.step, int)
        assert entry.step == 1

    def test_reconcile_fn_not_called_when_budget_exhausted(self) -> None:
        calls: List[Any] = []

        def fn(s: Any, a: Any, e: Any) -> None:
            calls.append(True)
            return None

        kernel = _make_kernel(budget=2.0, reconcile_fn=fn, min_action_cost=1.0)
        state = make_state(v=0)
        ok_action = _simple_action(cost=2.0, v=1)
        big_action = _simple_action(cost=5.0, v=99)

        verdict = kernel.evaluate(state, ok_action)
        assert verdict.approved
        kernel.execute(state, ok_action)

        assert len(calls) == 1

        verdict2 = kernel.evaluate(state, big_action)
        assert not verdict2.approved

        assert len(calls) == 1

    def test_reconcile_fn_call_count_is_one_per_execute(self) -> None:
        call_count = [0]

        def fn(s: Any, a: Any, e: Any) -> None:
            call_count[0] += 1
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(n=0)
        action = _simple_action(cost=1.0, n=1)
        self._run_execute(kernel, state, action)
        assert call_count[0] == 1


class TestReconcileFnCalledOnAtomic:

    def test_reconcile_fn_called_after_atomic(self) -> None:
        calls: List[Any] = []

        def fn(s: Any, a: Any, e: Any) -> None:
            calls.append(True)
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(x=0)
        action = _simple_action(cost=1.0, x=1)
        kernel.evaluate_and_execute_atomic(state, action)
        assert len(calls) == 1

    def test_atomic_returns_reconcile_value_when_non_none(self) -> None:
        real_world = make_state(x=999)

        def fn(s: Any, a: Any, e: Any) -> State:
            return real_world

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(x=0)
        action = _simple_action(cost=1.0, x=1)
        returned, _ = kernel.evaluate_and_execute_atomic(state, action)
        assert returned.get("x") == 999

    def test_atomic_returns_model_state_when_reconcile_returns_none(self) -> None:
        def fn(s: Any, a: Any, e: Any) -> None:
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(x=0)
        action = _simple_action(cost=1.0, x=7)
        returned, _ = kernel.evaluate_and_execute_atomic(state, action)
        assert returned.get("x") == 7

    def test_reconcile_fn_not_called_when_atomic_fails(self) -> None:
        calls: List[Any] = []

        def fn(s: Any, a: Any, e: Any) -> None:
            calls.append(True)
            return None

        kernel = _make_kernel(budget=1.0, reconcile_fn=fn, min_action_cost=1.0)
        state = make_state(v=0)
        big_action = _simple_action(cost=10.0, v=99)

        with pytest.raises(RuntimeError):
            kernel.evaluate_and_execute_atomic(state, big_action)

        assert len(calls) == 0

    def test_atomic_call_count_is_one(self) -> None:
        call_count = [0]

        def fn(s: Any, a: Any, e: Any) -> None:
            call_count[0] += 1
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(n=0)
        action = _simple_action(cost=1.0, n=1)
        kernel.evaluate_and_execute_atomic(state, action)
        assert call_count[0] == 1

    def test_reconcile_fn_receives_trace_entry_with_step_value(self) -> None:
        received_entries: List[TraceEntry] = []

        def fn(s: Any, a: Any, e: TraceEntry) -> None:
            received_entries.append(e)
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(n=0)
        action1 = _simple_action(cost=1.0, n=1)
        action2 = _simple_action(cost=1.0, n=2)
        kernel.evaluate_and_execute_atomic(state, action1)
        state2 = make_state(n=1)
        kernel.evaluate_and_execute_atomic(state2, action2)
        assert received_entries[0].step == 1
        assert received_entries[1].step == 2


class TestReconcileFnReturnValue:

    def _atomic(
        self,
        kernel: SafetyKernel,
        state: State,
        action: ActionSpec,
    ) -> Tuple[State, TraceEntry]:
        return kernel.evaluate_and_execute_atomic(state, action)

    def test_none_return_uses_model_state(self) -> None:
        def fn(s: Any, a: Any, e: Any) -> None:
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(val=0)
        action = _simple_action(cost=1.0, val=55)
        returned, _ = self._atomic(kernel, state, action)
        assert returned.get("val") == 55

    def test_non_none_return_replaces_model_state(self) -> None:
        corrected = make_state(val=100)

        def fn(s: Any, a: Any, e: Any) -> State:
            return corrected

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(val=0)
        action = _simple_action(cost=1.0, val=1)
        returned, _ = self._atomic(kernel, state, action)
        assert returned is corrected

    def test_returned_state_has_reconcile_fn_values(self) -> None:
        def fn(s: Any, a: Any, e: Any) -> State:
            return make_state(source="real_world", value=42)

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(source="model", value=0)
        action = _simple_action(cost=1.0, value=1)
        returned, _ = self._atomic(kernel, state, action)
        assert returned.get("source") == "real_world"
        assert returned.get("value") == 42

    def test_model_state_values_preserved_when_none_returned(self) -> None:
        def fn(s: Any, a: Any, e: Any) -> None:
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(a=1, b=2)
        action = _simple_action(cost=1.0, a=10)
        returned, _ = self._atomic(kernel, state, action)
        assert returned.get("a") == 10
        assert returned.get("b") == 2

    def test_reconcile_fn_can_return_state_with_different_keys(self) -> None:
        def fn(s: Any, a: Any, e: Any) -> State:
            return make_state(completely_new_key="surprise")

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(original_key=0)
        action = _simple_action(cost=1.0, original_key=1)
        returned, _ = self._atomic(kernel, state, action)
        assert returned.get("completely_new_key") == "surprise"
        assert not returned.has("original_key")


class TestReconcileFnErrorSilenced:

    def _run_execute(
        self,
        kernel: SafetyKernel,
        state: State,
        action: ActionSpec,
    ) -> Tuple[State, TraceEntry]:
        verdict = kernel.evaluate(state, action)
        assert verdict.approved
        return kernel.execute(state, action)

    def test_runtime_error_in_reconcile_fn_does_not_propagate(self) -> None:
        def fn(s: Any, a: Any, e: Any) -> None:
            raise RuntimeError("simulated reconciliation failure")

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(x=0)
        action = _simple_action(cost=1.0, x=1)
        returned, entry = self._run_execute(kernel, state, action)
        assert returned.get("x") == 1
        assert isinstance(entry, TraceEntry)

    def test_value_error_causes_model_state_to_be_used(self) -> None:
        def fn(s: Any, a: Any, e: Any) -> None:
            raise ValueError("reconcile rejected")

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(n=0)
        action = _simple_action(cost=1.0, n=99)
        returned, _ = self._run_execute(kernel, state, action)
        assert returned.get("n") == 99

    def test_raising_reconcile_fn_does_not_prevent_trace_append(self) -> None:
        def fn(s: Any, a: Any, e: Any) -> None:
            raise RuntimeError("boom")

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(v=0)
        action = _simple_action(cost=1.0, v=1)
        self._run_execute(kernel, state, action)
        assert kernel.trace.length == 1
        assert kernel.trace.entries[0].approved is True

    def test_raising_reconcile_fn_budget_still_charged(self) -> None:
        def fn(s: Any, a: Any, e: Any) -> None:
            raise RuntimeError("crash in reconcile")

        kernel = _make_kernel(budget=10.0, reconcile_fn=fn)
        state = make_state(v=0)
        action = _simple_action(cost=3.0, v=1)
        self._run_execute(kernel, state, action)
        assert abs(kernel.budget.remaining - 7.0) < 0.01


class TestReconcileFnIntegration:

    def test_reconcile_fn_tracks_real_world_divergence(self) -> None:
        real_db: dict = {"rows": 0}

        def sync_from_db(model_state: State, action: ActionSpec, entry: TraceEntry) -> State:
            actual_rows = real_db["rows"]
            return model_state.with_updates({"rows": actual_rows, "synced": True})

        kernel = _make_kernel(reconcile_fn=sync_from_db)
        state = make_state(rows=0, synced=False)
        action = _simple_action(cost=1.0, rows=1)

        real_db["rows"] = 5

        returned, _ = kernel.evaluate_and_execute_atomic(state, action)
        assert returned.get("rows") == 5
        assert returned.get("synced") is True

    def test_otel_exporter_make_reconcile_fn_returns_callable(self) -> None:
        from clampai.adapters.metrics import OTelTraceExporter

        class _FakeTracer:
            def start_as_current_span(self, name: str) -> Any:
                from contextlib import contextmanager

                @contextmanager
                def _ctx() -> Any:
                    yield _FakeSpan()

                return _ctx()

        class _FakeSpan:
            def set_attribute(self, k: str, v: Any) -> None:
                pass

            def set_status(self, status: Any) -> None:
                pass

        exporter = OTelTraceExporter.__new__(OTelTraceExporter)
        object.__setattr__(exporter, "_tracer", _FakeTracer()) if False else None
        exporter._tracer = _FakeTracer()  # type: ignore[attr-defined]
        fn = exporter.make_reconcile_fn()
        assert callable(fn)

    def test_reconcile_fn_called_n_times_for_n_executes(self) -> None:
        call_count = [0]

        def fn(s: Any, a: Any, e: Any) -> None:
            call_count[0] += 1
            return None

        kernel = _make_kernel(reconcile_fn=fn)
        state = make_state(n=0)
        n_steps = 7
        for i in range(n_steps):
            action = _simple_action(cost=1.0, n=i + 1)
            state, _ = kernel.evaluate_and_execute_atomic(state, action)

        assert call_count[0] == n_steps

    def test_reconcile_fn_returned_state_not_rechecked_by_invariants(self) -> None:
        inv = Invariant(
            name="counter_below_10",
            predicate=lambda s: s.get("counter", 0) < 10,
            description="counter must stay below 10",
            severity="error",
            enforcement="blocking",
        )

        def fn(s: State, a: Any, e: Any) -> State:
            return s.with_updates({"counter": 999})

        kernel = _make_kernel(invariants=[inv], reconcile_fn=fn)
        state = make_state(counter=0)
        action = _simple_action(cost=1.0, counter=1)

        returned, _ = kernel.evaluate_and_execute_atomic(state, action)
        assert returned.get("counter") == 999

    def test_none_reconcile_fn_is_safe_default(self) -> None:
        kernel = _make_kernel(reconcile_fn=None)
        state = make_state(x=0)
        action = _simple_action(cost=1.0, x=1)
        returned, entry = kernel.evaluate_and_execute_atomic(state, action)
        assert returned.get("x") == 1
        assert entry.step == 1
        assert kernel.budget.remaining == pytest.approx(99.0, abs=0.01)
