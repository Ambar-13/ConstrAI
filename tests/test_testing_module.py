"""
Tests for clampai.testing — SafetyHarness, make_state, make_action.

Covers: all public methods of SafetyHarness, both helper constructors,
assertion success and failure paths, context manager protocol, reset,
and integration scenarios.
"""
from __future__ import annotations

import pytest

from clampai.formal import ActionSpec, Effect, State
from clampai.invariants import (
    no_action_after_flag_invariant,
    no_delete_invariant,
    rate_limit_invariant,
    resource_ceiling_invariant,
)
from clampai.testing import SafetyHarness, make_action, make_state

# ─── make_state ────────────────────────────────────────────────────────────────


class TestMakeState:
    def test_empty(self):
        s = make_state()
        assert isinstance(s, State)
        assert s.to_dict() == {}

    def test_single_key(self):
        s = make_state(x=1)
        assert s.get("x") == 1

    def test_multiple_keys(self):
        s = make_state(a=1, b="hello", c=True)
        assert s.get("a") == 1
        assert s.get("b") == "hello"
        assert s.get("c") is True

    def test_none_value(self):
        s = make_state(k=None)
        assert s.get("k") is None

    def test_nested_dict(self):
        s = make_state(config={"timeout": 30})
        assert s.get("config") == {"timeout": 30}

    def test_list_value(self):
        s = make_state(items=[1, 2, 3])
        assert s.get("items") == [1, 2, 3]

    def test_returns_immutable_state(self):
        s = make_state(x=1)
        with pytest.raises((AttributeError, TypeError)):
            s._vars["x"] = 99  # type: ignore[index]

    def test_float_value(self):
        s = make_state(cost=3.14)
        assert abs(s.get("cost") - 3.14) < 1e-9


# ─── make_action ───────────────────────────────────────────────────────────────


class TestMakeAction:
    def test_returns_action_spec(self):
        a = make_action("test")
        assert isinstance(a, ActionSpec)

    def test_name_is_set(self):
        a = make_action("do_thing")
        assert a.name == "do_thing"

    def test_default_cost_is_one(self):
        a = make_action("x")
        assert a.cost == pytest.approx(1.0)

    def test_custom_cost(self):
        a = make_action("y", cost=5.0)
        assert a.cost == pytest.approx(5.0)

    def test_no_effects_by_default(self):
        a = make_action("noop")
        assert a.effects == ()

    def test_single_effect(self):
        a = make_action("inc", counter=1)
        assert len(a.effects) == 1
        e = a.effects[0]
        assert e.variable == "counter"
        assert e.mode == "set"
        assert e.value == 1

    def test_multiple_effects(self):
        a = make_action("multi", x=10, y="hello", z=True)
        assert len(a.effects) == 3
        vars_ = {e.variable for e in a.effects}
        assert vars_ == {"x", "y", "z"}

    def test_id_is_unique_per_call(self):
        a1 = make_action("same")
        a2 = make_action("same")
        assert a1.id != a2.id

    def test_id_contains_name(self):
        a = make_action("my_action")
        assert "my_action" in a.id

    def test_reversible_default_true(self):
        a = make_action("r")
        assert a.reversible is True

    def test_reversible_false(self):
        a = make_action("r", reversible=False)
        assert a.reversible is False

    def test_tags_empty_by_default(self):
        a = make_action("t")
        assert a.tags == ()

    def test_tags_set(self):
        a = make_action("t", tags=["read", "safe"])
        assert "read" in a.tags
        assert "safe" in a.tags


# ─── SafetyHarness.assert_allowed ──────────────────────────────────────────────


class TestSafetyHarnessAssertAllowed:
    def test_passes_for_affordable_action(self):
        with SafetyHarness(budget=10.0) as h:
            h.assert_allowed(make_state(), make_action("a", cost=1.0))

    def test_raises_for_blocked_action(self):
        with SafetyHarness(budget=1.0) as h:
            with pytest.raises(AssertionError, match="BLOCKED"):
                h.assert_allowed(make_state(), make_action("big", cost=100.0))

    def test_custom_msg_in_error(self):
        with SafetyHarness(budget=0.5) as h:
            with pytest.raises(AssertionError, match="my test"):
                h.assert_allowed(
                    make_state(), make_action("x", cost=5.0), msg="my test"
                )

    def test_passes_when_invariant_holds(self):
        inv = rate_limit_invariant("count", 5)
        with SafetyHarness(budget=10.0, invariants=[inv]) as h:
            h.assert_allowed(make_state(count=3), make_action("a", cost=1.0, count=4))

    def test_raises_when_invariant_violated(self):
        inv = rate_limit_invariant("count", 2)
        with SafetyHarness(budget=10.0, invariants=[inv]) as h:
            with pytest.raises(AssertionError):
                h.assert_allowed(make_state(count=2), make_action("a", cost=1.0, count=3))


# ─── SafetyHarness.assert_blocked ──────────────────────────────────────────────


class TestSafetyHarnessAssertBlocked:
    def test_passes_for_overspend_action(self):
        with SafetyHarness(budget=1.0) as h:
            h.assert_blocked(make_state(), make_action("big", cost=100.0))

    def test_raises_for_approved_action(self):
        with SafetyHarness(budget=10.0) as h:
            with pytest.raises(AssertionError, match="APPROVED"):
                h.assert_blocked(make_state(), make_action("cheap", cost=1.0))

    def test_reason_contains_match(self):
        with SafetyHarness(budget=0.5) as h:
            h.assert_blocked(
                make_state(), make_action("x", cost=5.0),
                reason_contains="afford"
            )

    def test_reason_contains_mismatch_raises(self):
        with SafetyHarness(budget=0.5) as h:
            with pytest.raises(AssertionError, match="reasons do not contain"):
                h.assert_blocked(
                    make_state(), make_action("x", cost=5.0),
                    reason_contains="invariant"  # budget is the real reason
                )

    def test_custom_msg_in_error(self):
        with SafetyHarness(budget=10.0) as h:
            with pytest.raises(AssertionError, match="context info"):
                h.assert_blocked(
                    make_state(), make_action("ok", cost=1.0), msg="context info"
                )

    def test_invariant_block(self):
        inv = no_delete_invariant("required_key")
        with SafetyHarness(budget=10.0, invariants=[inv]) as h:
            h.assert_blocked(
                make_state(required_key=None),
                make_action("bad", cost=1.0, required_key=None),
            )


# ─── SafetyHarness.assert_budget_remaining ─────────────────────────────────────


class TestSafetyHarnessAssertBudget:
    def test_initial_budget_equals_full(self):
        with SafetyHarness(budget=10.0) as h:
            h.assert_budget_remaining(10.0)

    def test_passes_with_tolerance(self):
        with SafetyHarness(budget=10.0) as h:
            h.assert_budget_remaining(10.005, tol=0.01)

    def test_fails_outside_tolerance(self):
        with SafetyHarness(budget=10.0) as h:
            with pytest.raises(AssertionError, match=r"10\.0"):
                h.assert_budget_remaining(5.0, tol=0.01)

    def test_decreases_after_execute(self):
        with SafetyHarness(budget=10.0) as h:
            s = h.execute(make_state(), make_action("a", cost=3.0))
            _ = s
            h.assert_budget_remaining(7.0)


# ─── SafetyHarness.assert_step_count ───────────────────────────────────────────


class TestSafetyHarnessAssertStepCount:
    def test_initial_step_count_zero(self):
        with SafetyHarness(budget=10.0) as h:
            h.assert_step_count(0)

    def test_increments_after_execute(self):
        with SafetyHarness(budget=10.0) as h:
            h.execute(make_state(), make_action("a", cost=1.0))
            h.assert_step_count(1)

    def test_increments_twice(self):
        with SafetyHarness(budget=10.0) as h:
            s = h.execute(make_state(), make_action("a", cost=1.0))
            h.execute(s, make_action("b", cost=1.0))
            h.assert_step_count(2)

    def test_fails_for_wrong_count(self):
        with SafetyHarness(budget=10.0) as h:
            with pytest.raises(AssertionError, match="step_count"):
                h.assert_step_count(5)


# ─── SafetyHarness.execute ─────────────────────────────────────────────────────


class TestSafetyHarnessExecute:
    def test_returns_new_state(self):
        with SafetyHarness(budget=10.0) as h:
            s0 = make_state(x=0)
            s1 = h.execute(s0, make_action("set_x", cost=1.0, x=42))
            assert s1.get("x") == 42

    def test_original_state_unchanged(self):
        with SafetyHarness(budget=10.0) as h:
            s0 = make_state(x=0)
            h.execute(s0, make_action("set_x", cost=1.0, x=99))
            assert s0.get("x") == 0

    def test_raises_for_blocked_action(self):
        with SafetyHarness(budget=0.5) as h:
            with pytest.raises(RuntimeError):
                h.execute(make_state(), make_action("big", cost=10.0))

    def test_reasoning_accepted(self):
        with SafetyHarness(budget=10.0) as h:
            h.execute(make_state(), make_action("a", cost=1.0), reasoning="test reason")

    def test_chain_of_executions(self):
        with SafetyHarness(budget=10.0) as h:
            s = make_state(count=0)
            s = h.execute(s, make_action("step1", cost=1.0, count=1))
            s = h.execute(s, make_action("step2", cost=1.0, count=2))
            assert s.get("count") == 2


# ─── SafetyHarness context manager and reset ───────────────────────────────────


class TestSafetyHarnessContextManager:
    def test_returns_self_from_enter(self):
        h = SafetyHarness(budget=5.0)
        entered = h.__enter__()
        assert entered is h
        h.__exit__(None, None, None)

    def test_with_statement(self):
        with SafetyHarness(budget=5.0) as h:
            assert isinstance(h, SafetyHarness)

    def test_reset_restores_budget(self):
        with SafetyHarness(budget=5.0) as h:
            h.execute(make_state(), make_action("a", cost=3.0))
            h.assert_budget_remaining(2.0)
            h.reset()
            h.assert_budget_remaining(5.0)

    def test_reset_restores_step_count(self):
        with SafetyHarness(budget=10.0) as h:
            h.execute(make_state(), make_action("a", cost=1.0))
            h.assert_step_count(1)
            h.reset()
            h.assert_step_count(0)

    def test_kernel_property(self):
        with SafetyHarness(budget=5.0) as h:
            from clampai.formal import SafetyKernel
            assert isinstance(h.kernel, SafetyKernel)

    def test_emergency_actions_passed(self):
        with SafetyHarness(budget=0.0, emergency_actions={"escape"}) as h:
            emergency = ActionSpec(
                id="escape", name="escape", description="emergency",
                effects=(), cost=0.0,
            )
            h.assert_allowed(make_state(), emergency)


# ─── Integration scenarios ─────────────────────────────────────────────────────


class TestSafetyHarnessIntegration:
    def test_budget_exhaustion_scenario(self):
        with SafetyHarness(budget=3.0) as h:
            s = make_state()
            s = h.execute(s, make_action("a", cost=1.0))
            s = h.execute(s, make_action("b", cost=1.0))
            s = h.execute(s, make_action("c", cost=1.0))
            h.assert_budget_remaining(0.0)
            h.assert_blocked(s, make_action("d", cost=1.0), reason_contains="afford")

    def test_invariant_scenario(self):
        inv = resource_ceiling_invariant("temperature", 100.0)
        with SafetyHarness(budget=100.0, invariants=[inv]) as h:
            s = make_state(temperature=50.0)
            h.assert_allowed(s, make_action("warm", cost=1.0, temperature=80.0))
            h.assert_blocked(s, make_action("overheat", cost=1.0, temperature=120.0))

    def test_flag_stops_all_actions(self):
        inv = no_action_after_flag_invariant("terminated")
        with SafetyHarness(budget=100.0, invariants=[inv]) as h:
            # Actions on non-terminated state succeed (next state won't have flag set)
            s = make_state(terminated=False)
            s = h.execute(s, make_action("step", cost=1.0))
            # Once state has terminated=True, any action whose next state keeps it True
            # is blocked by the invariant
            flagged = make_state(terminated=True)
            h.assert_blocked(flagged, make_action("more", cost=1.0))

    def test_assert_blocked_case_insensitive_reason(self):
        with SafetyHarness(budget=0.1) as h:
            h.assert_blocked(
                make_state(), make_action("big", cost=1.0),
                reason_contains="AFFORD"  # uppercase: tests case-insensitive matching
            )
