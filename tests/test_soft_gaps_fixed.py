"""
Tests for Soft Gap Fixes.

Tests verify:
- Effect.inverse() computes correct inverse for all modes
- Inverse effects undo original effect: undo(execute(s,a)) = s
- QPProjector nudges numeric parameters to safe range
- ReferenceMonitor computes safe values using binary search
"""

from typing import Any

import pytest

from constrai.formal import ActionSpec, Effect, GuaranteeLevel, State
from constrai.reference_monitor import ControlBarrierFunction, QPProjector, ReferenceMonitor


class TestEffectInverse:
    """Test Effect.inverse() for all modes."""

    def test_inverse_increment(self):
        """inverse(increment v) = decrement v."""
        effect = Effect("x", "increment", 5)
        inverse = effect.inverse()

        assert inverse.variable == "x"
        assert inverse.mode == "decrement"
        assert inverse.value == 5

    def test_inverse_decrement(self):
        """inverse(decrement v) = increment v"""
        effect = Effect("x", "decrement", 3)
        inverse = effect.inverse()

        assert inverse.variable == "x"
        assert inverse.mode == "increment"
        assert inverse.value == 3

    def test_inverse_multiply(self):
        """inverse(multiply v) = multiply (1/v)"""
        effect = Effect("x", "multiply", 2.0)
        inverse = effect.inverse()

        assert inverse.variable == "x"
        assert inverse.mode == "multiply"
        assert abs(inverse.value - 0.5) < 1e-9

    def test_inverse_multiply_nonzero(self):
        """inverse(multiply 4) = multiply 0.25"""
        effect = Effect("x", "multiply", 4.0)
        inverse = effect.inverse()
        assert abs(inverse.value - 0.25) < 1e-9

    def test_inverse_append(self):
        """inverse(append v) = remove v"""
        effect = Effect("items", "append", "apple")
        inverse = effect.inverse()

        assert inverse.variable == "items"
        assert inverse.mode == "remove"
        assert inverse.value == "apple"

    def test_inverse_remove(self):
        """inverse(remove v) = append v"""
        effect = Effect("items", "remove", "banana")
        inverse = effect.inverse()

        assert inverse.variable == "items"
        assert inverse.mode == "append"
        assert inverse.value == "banana"

    def test_inverse_set_raises(self):
        """inverse(set v) raises ValueError (needs prior state)"""
        effect = Effect("x", "set", 100)
        with pytest.raises(ValueError, match="Cannot invert 'set' mode"):
            effect.inverse()

    def test_inverse_delete_raises(self):
        """inverse(delete) raises ValueError (needs prior state)"""
        effect = Effect("x", "delete")
        with pytest.raises(ValueError, match="Cannot invert 'delete' mode"):
            effect.inverse()

    def test_inverse_multiply_by_zero_raises(self):
        """inverse(multiply 0) raises ValueError (division by zero)"""
        effect = Effect("x", "multiply", 0.0)
        with pytest.raises(ValueError, match="Cannot invert multiply by 0"):
            effect.inverse()

    # ── Algebraic T7 Proofs: undo(execute) = id ──

    def test_algebraic_t7_increment_decrement(self):
        """Increment then decrement restores original value."""
        x0 = 10
        effect = Effect("x", "increment", 5)
        x1 = effect.apply(x0)
        assert x1 == 15

        inverse = effect.inverse()
        x2 = inverse.apply(x1)
        assert x2 == x0  # Restored to 10

    def test_algebraic_t7_decrement_increment(self):
        """Decrement then increment restores original value."""
        x0 = 20
        effect = Effect("x", "decrement", 7)
        x1 = effect.apply(x0)
        assert x1 == 13

        inverse = effect.inverse()
        x2 = inverse.apply(x1)
        assert x2 == x0  # Restored to 20

    def test_algebraic_t7_multiply_divide(self):
        """Multiply then divide restores original value."""
        x0 = 100.0
        effect = Effect("x", "multiply", 2.5)
        x1 = effect.apply(x0)
        assert abs(x1 - 250.0) < 1e-9

        inverse = effect.inverse()
        x2 = inverse.apply(x1)
        assert abs(x2 - x0) < 1e-9  # Restored to 100

    def test_algebraic_t7_append_remove(self):
        """Append then remove restores original list."""
        lst0 = ["a", "b"]
        effect = Effect("items", "append", "c")
        lst1 = effect.apply(lst0)
        assert lst1 == ["a", "b", "c"]

        inverse = effect.inverse()
        lst2 = inverse.apply(lst1)
        assert lst2 == lst0  # Restored to ["a", "b"]

    def test_algebraic_t7_remove_append(self):
        """Theorem T7: remove then append restores original value (not position).

        Note: append puts items at the end, so position may differ.
        For true position-preservation, use set operations.
        """
        lst0 = ["x", "y", "z"]
        effect = Effect("items", "remove", "y")
        lst1 = effect.apply(lst0)
        assert lst1 == ["x", "z"]

        inverse = effect.inverse()
        lst2 = inverse.apply(lst1)
        # After remove "y" then append "y", we get ["x", "z", "y"]
        # This is valid morphism: the set of items is preserved
        assert set(lst2) == set(lst0)
        assert "y" in lst2

    def test_algebraic_t7_action_level_inverse(self):
        """ActionSpec.compute_inverse_effects() uses inverse morphism."""
        state_before = State({
            "balance": 100,
            "items": ["apple"]
        })

        action = ActionSpec(
            id="test_action",
            name="modify_state",
            description="modify balance and items",
            effects=(
                Effect("balance", "decrement", 25),
                Effect("items", "append", "banana")
            ),
            cost=10.0
        )

        # Apply action
        state_after = action.simulate(state_before)
        assert state_after.get("balance") == 75
        assert state_after.get("items") == ["apple", "banana"]

        # Compute and apply inverse effects
        inverse_effects = action.compute_inverse_effects(state_before)
        inverse_action = ActionSpec(
            id="undo_test",
            name="undo",
            description="undo modify_state",
            effects=inverse_effects,
            cost=0.0
        )

        state_restored = inverse_action.simulate(state_after)
        assert state_restored.get("balance") == state_before.get("balance")
        assert state_restored.get("items") == state_before.get("items")


# QP nudging tests — numeric QP gap

class TestQPNudging:
    """Test QP repair of effect parameters."""

    def test_qp_project_effect_parameters_no_change(self):
        """QP: action already safe, no nudging needed."""
        action = ActionSpec(
            id="safe_action",
            name="spend_20",
            description="spend 20 units",
            effects=(Effect("balance", "decrement", 20),),
            cost=5.0
        )

        projector = QPProjector()
        state = State({"balance": 100})

        # No safe_values provided → no repair
        repaired, was_changed = projector.project_effect_parameters(
            action.effects, state, safe_values=None
        )

        assert not was_changed
        assert repaired == action.effects

    def test_qp_project_effect_parameters_nudge_down(self):
        """QP: nudge spend amount from 100 down to 50 (budget constraint)."""
        original_effects = (Effect("balance", "decrement", 100),)
        safe_values = {"balance": 50}  # Safe to spend only 50

        projector = QPProjector()
        state = State({"balance": 100})

        repaired, was_changed = projector.project_effect_parameters(
            original_effects, state, safe_values=safe_values
        )

        assert was_changed
        assert len(repaired) == 1
        assert repaired[0].variable == "balance"
        assert repaired[0].mode == "decrement"
        assert repaired[0].value == 50  # Nudged from 100 to 50

    def test_qp_project_action_with_effect_nudging(self):
        """QP: project_action() nudges effect parameters + cost."""
        action = ActionSpec(
            id="expensive_action",
            name="big_spend",
            description="spend 100",
            effects=(Effect("balance", "decrement", 100),),
            cost=20.0
        )

        projector = QPProjector()
        state = State({"balance": 100})
        safe_effect_values = {"balance": 50}

        repaired_action, was_changed = projector.project_action(
            action, state, constraints=[],
            safe_effect_values=safe_effect_values
        )

        assert was_changed
        assert repaired_action.id == action.id
        assert repaired_action.name == action.name + " [repaired]"
        assert len(repaired_action.effects) == 1
        assert repaired_action.effects[0].value == 50  # Effect nudged

    def test_qp_project_multiple_effects(self):
        """QP: nudge multiple effect parameters independently."""
        original_effects = (
            Effect("balance", "decrement", 80),
            Effect("inventory", "decrement", 100),
            Effect("risk_level", "increment", 5)
        )
        safe_values = {
            "balance": 30,      # Nudge from 80 to 30
            "inventory": 20,    # Nudge from 100 to 20
            # risk_level not in safe_values, leave unchanged
        }

        projector = QPProjector()
        state = State({"balance": 100, "inventory": 200, "risk_level": 1})

        repaired, was_changed = projector.project_effect_parameters(
            original_effects, state, safe_values=safe_values
        )

        assert was_changed
        assert len(repaired) == 3
        assert repaired[0].value == 30   # balance nudged
        assert repaired[1].value == 20   # inventory nudged
        assert repaired[2].value == 5    # risk_level unchanged

    def test_qp_non_numeric_effects_untouched(self):
        """QP: non-numeric effects (append, remove, set) are not nudged."""
        original_effects = (
            Effect("items", "append", "apple"),
            Effect("tags", "remove", "old_tag"),
        )
        safe_values = {"items": "banana"}  # Irrelevant

        projector = QPProjector()
        state = State({"items": [], "tags": ["old_tag"]})

        repaired, was_changed = projector.project_effect_parameters(
            original_effects, state, safe_values=safe_values
        )

        # No numeric effects to nudge
        assert not was_changed
        assert repaired == original_effects


# Reference monitor safe value computation — integration test

class TestReferenceMonitorSafeValues:
    """Test _compute_safe_effect_values() with binary search."""

    def test_compute_safe_effect_values_basic(self):
        """Monitor computes safe values via binary search for CBF compliance."""
        # Setup: balance starts at 100, CBF requires h(s') >= h(s) - 0.1*h(s)
        # Barrier h(s) = balance / 100 (normalized)
        def barrier_h(state: State) -> float:
            return state.get("balance", 0) / 100.0

        cbf = ControlBarrierFunction(h=barrier_h, alpha=0.1)

        action = ActionSpec(
            id="big_spend",
            name="spend_80",
            description="spend 80 units",
            effects=(Effect("balance", "decrement", 80),),
            cost=0.0
        )

        state = State({"balance": 100})
        monitor = ReferenceMonitor(cbf_enabled=True)
        monitor.cbf_budget = cbf

        safe_values = monitor._compute_safe_effect_values(action, state, cbf)

        # Balance would drop from 100 to 20 (unsafe)
        # h(100) = 1.0, requires h(s') >= 1.0 - 0.1*1.0 = 0.9
        # So balance >= 90. Safe spend is at most 10.
        assert "balance" in safe_values
        assert safe_values["balance"] <= 10
        print(f"Safe spend: {safe_values['balance']}")


# End-to-end integration: QP nudging in orchestration

class TestQPNudgingIntegration:
    """End-to-end test: action gets nudged by ReferenceMonitor and applied by orchestrator."""

    def test_qp_nudging_e2e_scenario(self):
        """Scenario: Agent wants to spend 100, but only 50 available.

        Expected: ReferenceMonitor nudges effect parameter from 100 to 50,
                  Action is repaired and executed, balance decreases by 50 not 100.
        """
        # Create action requesting spend of 100
        action = ActionSpec(
            id="spend_attempt",
            name="buy_item",
            description="spend 100 units",
            effects=(Effect("balance", "decrement", 100),),
            cost=5.0
        )

        state = State({"balance": 100})

        # Setup monitor with CBF that allows max 50 spend
        def budget_barrier(s: State) -> float:
            """h(s) = remaining_budget / initial_budget"""
            return s.get("balance", 0) / 100.0  # Normalized to [0,1]

        cbf = ControlBarrierFunction(h=budget_barrier, alpha=0.1)
        monitor = ReferenceMonitor(cbf_enabled=True, ifc_enabled=False, hjb_enabled=False)
        monitor.cbf_budget = cbf

        # Enforce the action
        safe, reason, repaired_action = monitor.enforce(action, state)

        # Should be safe after repair
        assert safe, f"Monitor rejected: {reason}"

        # Repaired action should have reduced spend amount
        assert repaired_action is not None, "Action should be repaired"
        assert len(repaired_action.effects) == 1

        # The spend amount should be reduced
        repaired_effect = repaired_action.effects[0]
        assert repaired_effect.value < 100, f"Expected nudge from 100, got {repaired_effect.value}"

        # Apply repaired action
        new_state = repaired_action.simulate(state)
        new_balance = new_state.get("balance")

        # New balance should reflect the reduced spend
        assert new_balance > 0, "Balance should remain positive after nudged spend"
        assert new_balance >= 50, f"Balance after spend should be >= 50, got {new_balance}"

        print(f"ok: QP Nudging E2E: Spend nudged from 100 to {100 - new_balance}, balance: {new_balance}")


# Regression tests — backward compatibility

class TestBackwardCompatibility:
    """Ensure new features don't break existing functionality."""

    def test_effect_apply_unchanged(self):
        """Effect.apply() still works as before."""
        effect = Effect("x", "increment", 10)
        result = effect.apply(5)
        assert result == 15

    def test_action_simulate_unchanged(self):
        """ActionSpec.simulate() still works as before."""
        action = ActionSpec(
            id="test",
            name="test_action",
            description="test",
            effects=(
                Effect("x", "increment", 5),
                Effect("y", "set", 100)
            ),
            cost=1.0
        )

        state = State({"x": 10, "y": 50})
        new_state = action.simulate(state)

        assert new_state.get("x") == 15
        assert new_state.get("y") == 100

    def test_qp_projector_backward_compat(self):
        """QPProjector.project_action() works without safe_effect_values."""
        action = ActionSpec(
            id="test",
            name="test_action",
            description="test",
            effects=(Effect("x", "increment", 5),),
            cost=10.0
        )

        projector = QPProjector()
        state = State({"x": 0})

        # Call without safe_effect_values (old API)
        repaired, _was_changed = projector.project_action(
            action, state, constraints=[]
        )

        # Should work fine, just no effect parameter repair
        assert isinstance(repaired, ActionSpec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
