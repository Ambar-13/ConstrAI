"""
Tests for the @safe function (clampai/api.py).

Covers:
  - Budget enforcement: blocked after budget exhausted (T1)
  - Invariant enforcement: blocking mode prevents execution (T3)
  - Monitoring mode invariants: do NOT block execution
  - Audit log accumulation
  - reset() restores full budget and clears history
  - state_fn merges external state for invariant evaluation
  - action_name override in trace
  - Thread safety: concurrent calls are serialized correctly
  - SafetyViolation carries the verdict
  - Wrapped function's return value is preserved
  - functools.update_wrapper: __name__, __doc__ preserved
"""

import threading
import time

import pytest

from clampai import (
    Invariant,
    SafetyViolation,
    safe,
)
from clampai.api import _SafeWrapper


@pytest.fixture()
def simple_fn():
    """A wrapped function that returns its argument doubled."""
    @safe(budget=20.0, cost_per_call=5.0)
    def double(x: int) -> int:
        return x * 2
    return double


class TestReturnValue:
    def test_returns_underlying_result(self, simple_fn):
        assert simple_fn(3) == 6
        assert simple_fn(0) == 0
        assert simple_fn(-7) == -14

    def test_functools_wraps_preserved(self, simple_fn):
        """__name__ and __doc__ should reflect the wrapped function."""
        assert simple_fn.__name__ == "double"

    def test_wrapper_is_safe_wrapper(self, simple_fn):
        assert isinstance(simple_fn, _SafeWrapper)


class TestBudgetEnforcement:
    def test_blocks_when_budget_exceeded(self):
        @safe(budget=10.0, cost_per_call=4.0)
        def action() -> str:
            return "ok"

        assert action() == "ok"  # cost=4, remaining=6
        assert action() == "ok"  # cost=4, remaining=2
        with pytest.raises(SafetyViolation):
            action()  # would cost 4, remaining=2 → blocked

    def test_exact_budget_boundary(self):
        """Budget exactly matches total cost: last allowed call succeeds."""
        @safe(budget=15.0, cost_per_call=5.0)
        def work() -> int:
            return 1

        results = [work() for _ in range(3)]  # 3 * $5 = $15 exactly
        assert results == [1, 1, 1]
        with pytest.raises(SafetyViolation):
            work()  # 4th call, remaining=0

    def test_violation_carries_verdict(self):
        @safe(budget=5.0, cost_per_call=5.0)
        def single() -> None:
            pass

        single()  # uses up the budget

        with pytest.raises(SafetyViolation) as exc_info:
            single()
        assert exc_info.value.verdict is not None
        assert not exc_info.value.verdict.approved
        assert len(exc_info.value.verdict.rejection_reasons) > 0

    def test_violation_message_contains_action_name(self):
        @safe(budget=5.0, cost_per_call=10.0, action_name="my_op")
        def op() -> None:
            pass

        with pytest.raises(SafetyViolation, match="my_op"):
            op()


class TestInvariantEnforcement:
    def test_blocking_invariant_prevents_execution(self):
        """T3 checks the projected next state (after effects), not current state.

        Invariant: call_count < 2
        Call 1: projected call_count=0+1=1 → 1<2=True → APPROVED
        Call 2: projected call_count=1+1=2 → 2<2=False → BLOCKED
        """
        call_count = [0]

        @safe(
            budget=100.0,
            cost_per_call=1.0,
            invariants=[
                Invariant(
                    "max_two",
                    lambda s: s.get("call_count", 0) < 2,
                    "Maximum 2 calls",
                ),
            ],
        )
        def fn() -> None:
            call_count[0] += 1

        fn()  # projected call_count=1, 1<2 → APPROVED
        assert call_count[0] == 1

        with pytest.raises(SafetyViolation):
            fn()  # projected call_count=2, 2<2 → BLOCKED

        assert call_count[0] == 1  # underlying fn not called on blocked attempt

    def test_invariant_checked_on_projected_state(self):
        """Invariant predicate receives next state (after effects), not current."""
        @safe(
            budget=100.0,
            cost_per_call=1.0,
            invariants=[
                Invariant(
                    "never_exceed_3",
                    lambda s: s.get("call_count", 0) <= 3,
                    "Never more than 3 calls",
                ),
            ],
        )
        def fn() -> int:
            return 1

        # Calls 1-3: next state has call_count=1,2,3 — all ≤ 3, approved
        for _ in range(3):
            fn()
        # 4th call: next state would have call_count=4 → 4≤3=False → BLOCKED
        with pytest.raises(SafetyViolation):
            fn()

    def test_invariant_violation_does_not_call_function(self):
        """Underlying function must NOT be called when invariant blocks."""
        called = [False]

        @safe(
            budget=100.0,
            cost_per_call=50.0,
            invariants=[
                Invariant("always_block", lambda s: False, "Always blocked"),
            ],
        )
        def side_effect_fn() -> None:
            called[0] = True

        with pytest.raises(SafetyViolation):
            side_effect_fn()

        assert not called[0], "Function was called despite invariant violation"

    def test_monitoring_invariant_does_not_block(self):
        """Monitoring-mode invariants log but do NOT prevent execution."""
        @safe(
            budget=100.0,
            cost_per_call=1.0,
            invariants=[
                Invariant(
                    "warn_always",
                    lambda s: False,  # always "violated"
                    "Monitoring warning (never blocks)",
                    enforcement="monitoring",
                ),
            ],
        )
        def fn() -> str:
            return "executed"

        result = fn()
        assert result == "executed"
        assert len(fn.audit_log) == 1


class TestAuditLog:
    def test_audit_log_grows_on_approved_calls(self, simple_fn):
        simple_fn(1)
        simple_fn(2)
        simple_fn(3)
        assert len(simple_fn.audit_log) == 3

    def test_audit_log_empty_at_start(self):
        @safe(budget=10.0)
        def fn() -> None:
            pass
        assert fn.audit_log == []

    def test_audit_log_structure(self, simple_fn):
        simple_fn(42)
        entry = simple_fn.audit_log[0]
        assert "step" in entry
        assert "action" in entry
        assert "cost" in entry
        assert "timestamp" in entry
        assert "approved" in entry
        assert entry["approved"] is True

    def test_rejected_calls_not_in_audit_log(self):
        @safe(budget=4.0, cost_per_call=5.0)
        def fn() -> None:
            pass

        with pytest.raises(SafetyViolation):
            fn()
        assert len(fn.audit_log) == 0

    def test_audit_log_step_numbers_sequential(self, simple_fn):
        simple_fn(1)
        simple_fn(2)
        steps = [e["step"] for e in simple_fn.audit_log]
        assert steps == [1, 2]


class TestReset:
    def test_reset_restores_full_budget(self):
        @safe(budget=10.0, cost_per_call=5.0)
        def fn() -> None:
            pass

        fn()
        fn()  # budget exhausted
        with pytest.raises(SafetyViolation):
            fn()

        fn.reset()
        assert fn.kernel.budget.remaining == pytest.approx(10.0)

    def test_reset_clears_audit_log(self):
        @safe(budget=20.0, cost_per_call=5.0)
        def fn() -> None:
            pass

        fn()
        fn()
        assert len(fn.audit_log) == 2

        fn.reset()
        assert len(fn.audit_log) == 0

    def test_reset_allows_reuse(self):
        @safe(budget=5.0, cost_per_call=5.0)
        def fn() -> str:
            return "result"

        assert fn() == "result"
        with pytest.raises(SafetyViolation):
            fn()

        fn.reset()
        assert fn() == "result"


class TestStateFn:
    def test_state_fn_provides_extra_context(self):
        """state_fn fields are merged into the evaluation state.

        Invariant: not free OR call_count < 2
        Call 1: projected call_count=1, 1<2=True → APPROVED
        Call 2: projected call_count=2, 2<2=False → BLOCKED
        """
        external = {"user_tier": "free"}

        @safe(
            budget=100.0,
            cost_per_call=1.0,
            invariants=[
                Invariant(
                    "free_limit",
                    lambda s: not (s.get("user_tier") == "free")
                              or s.get("call_count", 0) < 2,
                    "Free tier: max 2 calls",
                ),
            ],
            state_fn=lambda: dict(external),
        )
        def api_call() -> str:
            return "ok"

        result = api_call()  # projected call_count=1, 1<2 → APPROVED
        assert result == "ok"

        with pytest.raises(SafetyViolation):
            api_call()  # projected call_count=2, 2<2 → BLOCKED

    def test_state_fn_exception_does_not_crash(self):
        """If state_fn raises, the wrapper still works (graceful degradation)."""
        call_count = [0]

        def bad_state_fn():
            call_count[0] += 1
            raise RuntimeError("database unavailable")

        @safe(
            budget=10.0,
            cost_per_call=1.0,
            state_fn=bad_state_fn,
        )
        def fn() -> str:
            return "ok"

        result = fn()
        assert result == "ok"
        assert call_count[0] >= 1  # state_fn was called


class TestActionName:
    def test_action_name_appears_in_trace(self):
        @safe(budget=20.0, cost_per_call=5.0, action_name="custom_name")
        def my_fn() -> None:
            pass

        my_fn()
        entry = my_fn.audit_log[0]
        assert "custom_name" in entry["action"]

    def test_default_action_name_is_function_name(self):
        @safe(budget=20.0, cost_per_call=5.0)
        def my_special_function() -> None:
            pass

        my_special_function()
        entry = my_special_function.audit_log[0]
        assert "my_special_function" in entry["action"]


class TestThreadSafety:
    def test_concurrent_calls_do_not_exceed_budget(self):
        """Budget must not be over-drawn even under concurrent access."""
        budget = 10.0
        cost_per_call = 1.0
        expected_max_calls = int(budget / cost_per_call)  # 10

        @safe(budget=budget, cost_per_call=cost_per_call)
        def fn() -> None:
            time.sleep(0.001)

        approved_count = [0]
        rejected_count = [0]
        lock = threading.Lock()

        def worker():
            try:
                fn()
                with lock:
                    approved_count[0] += 1
            except SafetyViolation:
                with lock:
                    rejected_count[0] += 1

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert approved_count[0] == expected_max_calls
        assert rejected_count[0] == 20 - expected_max_calls
        assert fn.kernel.budget.spent_net <= budget + 1e-9

    def test_audit_log_count_matches_approved_calls(self):
        """Audit log must have exactly as many entries as approved calls."""
        @safe(budget=5.0, cost_per_call=1.0)
        def fn() -> None:
            pass

        for _ in range(10):
            try:
                fn()
            except SafetyViolation:
                pass

        assert len(fn.audit_log) <= 5
        for entry in fn.audit_log:
            assert entry["approved"] is True


class TestKernelProperty:
    def test_kernel_is_safety_kernel(self, simple_fn):
        from clampai import SafetyKernel
        assert isinstance(simple_fn.kernel, SafetyKernel)

    def test_kernel_budget_starts_at_configured_value(self):
        @safe(budget=42.0, cost_per_call=1.0)
        def fn() -> None:
            pass
        assert fn.kernel.budget.remaining == pytest.approx(42.0)

    def test_kernel_budget_decreases_after_call(self, simple_fn):
        initial = simple_fn.kernel.budget.remaining
        simple_fn(1)
        assert simple_fn.kernel.budget.remaining == pytest.approx(initial - 5.0)


class TestEdgeCases:
    def test_zero_budget_blocks_all_calls(self):
        @safe(budget=0.5, cost_per_call=1.0)
        def fn() -> None:
            pass

        with pytest.raises(SafetyViolation):
            fn()

    def test_function_with_keyword_args(self):
        @safe(budget=20.0, cost_per_call=1.0)
        def fn(a: int, b: int = 0, *, c: int = 0) -> int:
            return a + b + c

        assert fn(1, b=2, c=3) == 6

    def test_function_with_exception_charges_budget(self):
        """Budget IS charged when the action is approved, even if the function raises.

        ClampAI charges for the action, not its result. The function's exception
        is propagated unmodified.
        """
        @safe(budget=20.0, cost_per_call=5.0)
        def boom() -> None:
            raise ValueError("function error")

        budget_before = 20.0
        with pytest.raises(ValueError, match="function error"):
            boom()

        assert boom.kernel.budget.remaining < budget_before

    def test_multiple_independent_wrappers_have_separate_budgets(self):
        """Each @safe decoration creates an independent kernel."""
        @safe(budget=5.0, cost_per_call=5.0)
        def fn_a() -> str:
            return "a"

        @safe(budget=5.0, cost_per_call=5.0)
        def fn_b() -> str:
            return "b"

        fn_a()  # exhausts fn_a's budget
        assert fn_b() == "b"  # fn_b still has budget

        with pytest.raises(SafetyViolation):
            fn_a()
        assert fn_b.kernel.budget.remaining == pytest.approx(0.0)

    def test_clampai_safe_alias_works(self):
        """clampai_safe is a backwards-compat alias for safe."""
        from clampai import clampai_safe
        assert clampai_safe is safe


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
