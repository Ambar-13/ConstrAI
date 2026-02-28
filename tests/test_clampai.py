"""
ClampAI Test Suite - All Theorems + Integration + Adversarial
Converted to proper pytest format.
"""
import json
import math
import random
import threading
import time

from clampai import (
    ActionSpec,
    ActionValueComputer,
    Belief,
    BeliefState,
    BudgetController,
    CausalGraph,
    CheckResult,
    Effect,
    ExecutionResult,
    ExecutionTrace,
    Invariant,
    InverseAlgebra,
    MockLLMAdapter,
    Orchestrator,
    ProgressMonitor,
    ReasoningRequest,
    ReasoningResponse,
    RejectionFormatter,
    RollbackRecord,
    SafetyKernel,
    SafetyVerdict,
    State,
    TaskDefinition,
    TerminationReason,
    TraceEntry,
    parse_llm_response,
)
from clampai.formal import _BudgetLogic

# T1 — Budget safety: spent_net(t) ≤ B₀ for all t

class TestT1BudgetSafety:

    def test_t1_1_can_afford_within_budget(self):
        bc = BudgetController(100.0)
        ok, _ = bc.can_afford(50.0)
        assert ok, "T1.1 can afford within budget"

    def test_t1_2_cannot_afford_over_budget(self):
        bc = BudgetController(100.0)
        bc.charge("a1", 50.0)
        ok, _ = bc.can_afford(60.0)
        assert not ok, "T1.2 cannot afford over budget"

    def test_t1_3_cannot_afford_at_limit(self):
        bc = BudgetController(100.0)
        bc.charge("a1", 50.0)
        bc.charge("a2", 50.0)
        ok, _ = bc.can_afford(0.01)
        assert not ok, "T1.3 cannot afford at limit"

    def test_t1_4_spent_equals_budget(self):
        bc = BudgetController(100.0)
        bc.charge("a1", 50.0)
        bc.charge("a2", 50.0)
        assert bc.spent == 100.0, "T1.4 spent == budget"

    def test_t1_5_kernel_rejects_over_budget(self):
        kernel = SafetyKernel(budget=10.0, invariants=[], min_action_cost=1.0)
        state = State({"x": 0})
        expensive = ActionSpec(id="big", name="Big", description="Costly",
                               effects=(Effect("x", "increment", 1),), cost=15.0)
        v = kernel.evaluate(state, expensive)
        assert not v.approved, "T1.5 kernel rejects over-budget"

    def test_t1_6_negative_cost_rejected(self):
        try:
            BudgetController(100.0).charge("bad", -5.0)
            assert False, "T1.6 negative cost rejected"
        except (ValueError, AssertionError):
            pass

    def test_t1_7_budget_logic_to_from_millicents(self):
        """_BudgetLogic round-trips: to_millicents → from_millicents == original."""
        for val in (0.0, 1.0, 0.12345, 99999.99999, 0.00001):
            mc = _BudgetLogic.to_millicents(val)
            back = _BudgetLogic.from_millicents(mc)
            assert abs(back - val) < 1e-9, f"T1.7 round-trip failed for {val}"

    def test_t1_8_budget_logic_can_afford_i(self):
        """_BudgetLogic.can_afford_i enforces T1 guard at the millicent level."""
        budget_i = _BudgetLogic.to_millicents(10.0)
        gross_i  = _BudgetLogic.to_millicents(6.0)
        ref_i    = _BudgetLogic.to_millicents(1.0)   # net = 5.0
        assert     _BudgetLogic.can_afford_i(budget_i, gross_i, ref_i,
                       _BudgetLogic.to_millicents(5.0)), "T1.8a exactly affordable"
        assert not _BudgetLogic.can_afford_i(budget_i, gross_i, ref_i,
                       _BudgetLogic.to_millicents(5.01)), "T1.8b 5.01 over limit"

    def test_t1_9_budget_logic_remaining_i(self):
        """_BudgetLogic.remaining_i matches BudgetController.remaining property."""
        bc = BudgetController(10.0)
        bc.charge("x", 3.0)
        bc.refund("x", 1.0)
        expected_mc = _BudgetLogic.remaining_i(
            bc._budget_i, bc._spent_gross_i, bc._refunded_i)
        assert _BudgetLogic.from_millicents(expected_mc) == bc.remaining, \
            "T1.9 remaining_i matches property"

    def test_t1_10_budget_logic_utilization(self):
        """_BudgetLogic.utilization matches BudgetController.utilization()."""
        bc = BudgetController(20.0)
        bc.charge("x", 8.0)
        bc.refund("x", 2.0)    # net = 6.0 → util = 6/20 = 0.3
        expected = _BudgetLogic.utilization(
            bc._budget_i, bc._spent_gross_i, bc._refunded_i)
        assert abs(expected - 0.3) < 1e-9, "T1.10 utilization = 0.3"
        assert abs(bc.utilization() - 0.3) < 1e-9, "T1.10 BC.utilization() = 0.3"

    def test_t1_11_can_refund_i_normal(self):
        """can_refund_i allows refunds up to total gross charged."""
        gross_i   = _BudgetLogic.to_millicents(10.0)
        refunded_i = _BudgetLogic.to_millicents(3.0)
        # Refunding 7 more brings total refunded to 10 == gross → OK
        assert _BudgetLogic.can_refund_i(gross_i, refunded_i,
                   _BudgetLogic.to_millicents(7.0)), "T1.11a refund to gross OK"
        # Refunding 7.01 would over-refund → rejected
        assert not _BudgetLogic.can_refund_i(gross_i, refunded_i,
                       _BudgetLogic.to_millicents(7.01)), "T1.11b over-refund blocked"

    def test_t1_12_budget_controller_refund_delegation(self):
        """BudgetController.refund() correctly delegates over-refund guard to can_refund_i."""
        bc = BudgetController(20.0)
        bc.charge("a", 5.0)
        # Legitimate refund: restores net spend
        bc.refund("a", 5.0)
        assert bc.spent_net == 0.0, "T1.12a refund restores net spend"
        assert bc.spent_gross == 5.0, "T1.12b gross spend unchanged (T4)"
        # Over-refund attempt: silently ignored (guard in can_refund_i)
        bc.refund("extra", 1.0)   # would over-refund beyond gross
        assert bc.spent_net == 0.0, "T1.12c over-refund ignored"
        # After refund, can_afford reflects restored capacity
        ok, _ = bc.can_afford(20.0)
        assert ok, "T1.12d refund restores affordability"

    def test_t1_13_budget_controller_scale_shared(self):
        """BudgetController._SCALE equals _BudgetLogic._SCALE — single source of truth."""
        assert BudgetController._SCALE == _BudgetLogic._SCALE, \
            "T1.13 _SCALE shared between BudgetController and _BudgetLogic"


# T2 — Bounded termination: halts in ≤ ⌊B₀/ε⌋ steps

class TestT2Termination:

    def test_t2_1_max_steps_formula(self):
        kernel2 = SafetyKernel(budget=5.0, invariants=[], min_action_cost=1.0)
        assert kernel2.max_steps == 5, "T2.1 max_steps = floor(B/epsilon)"

    def test_t2_2_terminates_at_budget(self):
        kernel2 = SafetyKernel(budget=5.0, invariants=[], min_action_cost=1.0)
        state = State({"n": 0})
        action = ActionSpec(id="inc", name="Inc", description="+1",
                            effects=(Effect("n", "increment", 1),), cost=1.0)
        count = 0
        while count < 100:
            v = kernel2.evaluate(state, action)
            if not v.approved:
                break
            state, _ = kernel2.execute(state, action)
            count += 1
        assert count == 5, "T2.2 terminated at 5"

    def test_t2_3_budget_exhausted(self):
        kernel2 = SafetyKernel(budget=5.0, invariants=[], min_action_cost=1.0)
        state = State({"n": 0})
        action = ActionSpec(id="inc", name="Inc", description="+1",
                            effects=(Effect("n", "increment", 1),), cost=1.0)
        while True:
            v = kernel2.evaluate(state, action)
            if not v.approved:
                break
            state, _ = kernel2.execute(state, action)
        assert kernel2.budget.remaining == 0.0, "T2.3 budget exhausted"


# T3 — Invariant preservation: I(s₀) = True ⟹ I(sₜ) = True for all t

class TestT3InvariantPreservation:

    def _make_kernel(self):
        inv_max5 = Invariant("max_5", lambda s: (s.get("n", 0)) <= 5,
                             description="n <= 5")
        return SafetyKernel(budget=100.0, invariants=[inv_max5], min_action_cost=0.1)

    def _make_inc(self):
        return ActionSpec(id="inc", name="Inc", description="+1",
                          effects=(Effect("n", "increment", 1),), cost=1.0)

    def test_t3_1_increment_1_allowed(self):
        kernel3 = self._make_kernel()
        state3 = State({"n": 0})
        v = kernel3.evaluate(state3, self._make_inc())
        assert v.approved, "T3.1 increment 1 allowed"

    def test_t3_2_increment_2_allowed(self):
        kernel3 = self._make_kernel()
        state3 = State({"n": 0})
        inc = self._make_inc()
        state3, _ = kernel3.execute(state3, inc)
        v = kernel3.evaluate(state3, inc)
        assert v.approved, "T3.2 increment 2 allowed"

    def test_t3_3_increment_3_allowed(self):
        kernel3 = self._make_kernel()
        state3 = State({"n": 0})
        inc = self._make_inc()
        for _ in range(2):
            state3, _ = kernel3.execute(state3, inc)
        assert kernel3.evaluate(state3, inc).approved, "T3.3 increment 3 allowed"

    def test_t3_4_increment_4_allowed(self):
        kernel3 = self._make_kernel()
        state3 = State({"n": 0})
        inc = self._make_inc()
        for _ in range(3):
            state3, _ = kernel3.execute(state3, inc)
        assert kernel3.evaluate(state3, inc).approved, "T3.4 increment 4 allowed"

    def test_t3_5_increment_5_allowed(self):
        kernel3 = self._make_kernel()
        state3 = State({"n": 0})
        inc = self._make_inc()
        for _ in range(4):
            state3, _ = kernel3.execute(state3, inc)
        assert kernel3.evaluate(state3, inc).approved, "T3.5 increment 5 allowed"

    def test_t3_6_sixth_increment_blocked(self):
        kernel3 = self._make_kernel()
        state3 = State({"n": 0})
        inc = self._make_inc()
        for _ in range(5):
            state3, _ = kernel3.execute(state3, inc)
        assert not kernel3.evaluate(state3, inc).approved, "T3.6 6th increment blocked"

    def test_t3_7_n_still_5(self):
        kernel3 = self._make_kernel()
        state3 = State({"n": 0})
        inc = self._make_inc()
        for _ in range(5):
            state3, _ = kernel3.execute(state3, inc)
        kernel3.evaluate(state3, inc)  # attempt 6th (blocked)
        assert state3.get("n") == 5, "T3.7 n still 5"

    def test_t3_8_exception_is_violation(self):
        inc = self._make_inc()
        bad_inv = Invariant("bad", lambda s: 1/0, description="Raises")
        kernel_bad = SafetyKernel(budget=100.0, invariants=[bad_inv])
        v = kernel_bad.evaluate(State({}), inc)
        assert not v.approved, "T3.8 exception = violation"

    def test_t3_9_monitoring_invariant_does_not_block(self):
        inc = self._make_inc()
        monitor_only = Invariant(
            "monitor_only",
            lambda s: False,
            description="Should be logged but not block",
            severity="warning",
        )
        kernel_monitor = SafetyKernel(budget=100.0, invariants=[monitor_only])
        v = kernel_monitor.evaluate(State({"n": 0}), inc)
        assert v.approved, "T3.9 monitoring invariant does not block"

    def test_t3_10_monitoring_invariant_recorded_as_fail(self):
        inc = self._make_inc()
        monitor_only = Invariant(
            "monitor_only",
            lambda s: False,
            description="Should be logged but not block",
            severity="warning",
        )
        kernel_monitor = SafetyKernel(budget=100.0, invariants=[monitor_only])
        v = kernel_monitor.evaluate(State({"n": 0}), inc)
        assert any(result == CheckResult.FAIL_INVARIANT for _, result, _ in v.checks), \
            "T3.10 monitoring invariant recorded as FAIL_INVARIANT"


# T4 — Monotone spend: spent_gross is non-decreasing

class TestT4MonotoneSpend:

    def test_t4_monotone_spend(self):
        bc4 = BudgetController(100.0)
        prev = 0.0
        rng = random.Random(42)
        for i in range(10):
            c = rng.uniform(0.5, 5.0)
            if bc4.remaining >= c:
                bc4.charge(f"a{i}", c)
                assert bc4.spent >= prev, \
                    f"T4.{i+1} monotone: spent={bc4.spent} < prev={prev}"
                prev = bc4.spent


# T5 — Atomicity: transitions are all-or-nothing

class TestT5Atomicity:

    def _make_kernel_and_action(self):
        kernel5 = SafetyKernel(budget=100.0, invariants=[
            Invariant("positive", lambda s: s.get("x", 0) >= 0, description="x >= 0")
        ])
        state5 = State({"x": 5})
        bad_action = ActionSpec(id="neg", name="Neg", description="x to -10",
                                effects=(Effect("x", "set", -10),), cost=1.0)
        return kernel5, state5, bad_action

    def test_t5_1_bad_action_rejected(self):
        kernel5, state5, bad_action = self._make_kernel_and_action()
        v = kernel5.evaluate(state5, bad_action)
        assert not v.approved, "T5.1 bad action rejected"

    def test_t5_2_state_unchanged(self):
        kernel5, state5, bad_action = self._make_kernel_and_action()
        kernel5.evaluate(state5, bad_action)
        assert state5.get("x") == 5, "T5.2 state unchanged"

    def test_t5_3_budget_unchanged(self):
        kernel5, state5, bad_action = self._make_kernel_and_action()
        kernel5.evaluate(state5, bad_action)
        assert kernel5.budget.spent == 0.0, "T5.3 budget unchanged"


# T6 — Trace integrity: append-only, SHA-256 hash-chained

class TestT6TraceIntegrity:

    def _make_entry(self, step, state_before, state_after, action_id="a1", cost=1.0):
        return TraceEntry(
            step=step,
            action_id=action_id,
            action_name=f"Action {action_id}",
            state_before_fp=state_before.fingerprint,
            state_after_fp=state_after.fingerprint,
            cost=cost,
            timestamp=time.time(),
            approved=True,
        )

    def test_t6_1_empty_trace_valid(self):
        trace = ExecutionTrace()
        ok, msg = trace.verify_integrity()
        assert ok, f"T6.1 empty trace valid: {msg}"

    def test_t6_2_single_entry_valid(self):
        trace = ExecutionTrace()
        s0 = State({"x": 0})
        s1 = State({"x": 1})
        trace.append(self._make_entry(0, s0, s1))
        ok, msg = trace.verify_integrity()
        assert ok, f"T6.2 single entry valid: {msg}"

    def test_t6_3_multi_entry_valid(self):
        trace = ExecutionTrace()
        state = State({"x": 0})
        for i in range(5):
            next_state = State({"x": i + 1})
            trace.append(self._make_entry(i, state, next_state))
            state = next_state
        ok, msg = trace.verify_integrity()
        assert ok, f"T6.3 multi-entry trace valid: {msg}"

    def test_t6_4_length_increases(self):
        trace = ExecutionTrace()
        state = State({"x": 0})
        for i in range(3):
            next_state = State({"x": i + 1})
            trace.append(self._make_entry(i, state, next_state))
            state = next_state
            assert trace.length == i + 1, f"T6.4 length == {i+1} after {i+1} appends"

    def test_t6_5_entries_immutable(self):
        # TraceEntry is frozen (dataclass(frozen=True))
        entry = TraceEntry(
            step=0, action_id="test", action_name="Test",
            state_before_fp="abc", state_after_fp="def",
            cost=1.0, timestamp=0.0, approved=True,
        )
        try:
            entry.step = 99  # type: ignore
            assert False, "T6.5 entry should be immutable"
        except (AttributeError, TypeError):
            pass

    def test_t6_6_kernel_trace_valid_after_execution(self):
        kernel = SafetyKernel(budget=10.0, invariants=[], min_action_cost=1.0)
        state = State({"n": 0})
        action = ActionSpec(id="inc", name="Inc", description="+1",
                            effects=(Effect("n", "increment", 1),), cost=1.0)
        for _ in range(3):
            v = kernel.evaluate(state, action)
            if not v.approved:
                break
            state, _ = kernel.execute(state, action)
        ok, msg = kernel.trace.verify_integrity()
        assert ok, f"T6.6 kernel trace valid after execution: {msg}"


# T7 — Rollback exactness: undo(execute(s, a)) == s

class TestT7Rollback:

    def _make_action_and_states(self):
        action = ActionSpec(
            id="set_x", name="SetX", description="Set x to 42",
            effects=(Effect("x", "set", 42),), cost=1.0,
        )
        state_before = State({"x": 10, "y": 5})
        state_after = action.simulate(state_before)
        return action, state_before, state_after

    def test_t7_1_inverse_from_states_computed(self):
        action, before, after = self._make_action_and_states()
        inverse_effects = InverseAlgebra.compute_inverse_from_states(
            before, after, action
        )
        assert len(inverse_effects) > 0, "T7.1 inverse effects computed"

    def test_t7_2_rollback_restores_state(self):
        action, before, after = self._make_action_and_states()
        record = InverseAlgebra.make_rollback_record(
            action, before, after, time.time()
        )
        restored = record.apply_rollback(after)
        assert restored == before, f"T7.2 rollback restores state: {restored} != {before}"

    def test_t7_3_verify_inverse_correct(self):
        action, before, after = self._make_action_and_states()
        inverse_effects = InverseAlgebra.compute_inverse_from_states(
            before, after, action
        )
        ok, msg = InverseAlgebra.verify_inverse_correctness(before, action, inverse_effects)
        assert ok, f"T7.3 inverse is correct: {msg}"

    def test_t7_4_increment_rollback(self):
        action = ActionSpec(
            id="inc", name="Inc", description="+1",
            effects=(Effect("n", "increment", 1),), cost=1.0,
        )
        before = State({"n": 7})
        after = action.simulate(before)
        assert after.get("n") == 8, "T7.4a simulate increments"
        record = InverseAlgebra.make_rollback_record(action, before, after, time.time())
        restored = record.apply_rollback(after)
        assert restored.get("n") == 7, "T7.4 increment rollback restores n=7"

    def test_t7_5_delete_rollback(self):
        action = ActionSpec(
            id="del_key", name="DeleteKey", description="Delete 'tmp'",
            effects=(Effect("tmp", "delete"),), cost=1.0,
        )
        before = State({"x": 1, "tmp": "hello"})
        after = action.simulate(before)
        assert not after.has("tmp"), "T7.5a delete removed key"
        record = InverseAlgebra.make_rollback_record(action, before, after, time.time())
        restored = record.apply_rollback(after)
        assert restored.get("tmp") == "hello", "T7.5 delete rollback restores tmp"


# T8 — Emergency escape: escape action always executable

class TestT8EmergencyEscape:

    def test_t8_1_emergency_bypasses_min_cost(self):
        kernel = SafetyKernel(budget=10.0, invariants=[], min_action_cost=1.0)
        kernel.register_emergency_action("safe_halt")
        state = State({"x": 0})
        halt = ActionSpec(
            id="safe_halt", name="SafeHalt", description="Halt immediately",
            effects=(), cost=0.0,
        )
        v = kernel.evaluate(state, halt)
        assert v.approved, "T8.1 emergency action passes despite zero cost"

    def test_t8_2_emergency_bypasses_step_limit(self):
        # Exhaust all steps, then try emergency action
        kernel = SafetyKernel(budget=5.0, invariants=[], min_action_cost=1.0)
        kernel.register_emergency_action("safe_halt")
        state = State({"n": 0})
        action = ActionSpec(id="inc", name="Inc", description="+1",
                            effects=(Effect("n", "increment", 1),), cost=1.0)
        halt = ActionSpec(
            id="safe_halt", name="SafeHalt", description="Halt immediately",
            effects=(), cost=0.0,
        )
        # Run until budget exhausted
        while True:
            v = kernel.evaluate(state, action)
            if not v.approved:
                break
            state, _ = kernel.execute(state, action)
        # Now emergency action should still be approved
        v_halt = kernel.evaluate(state, halt)
        assert v_halt.approved, "T8.2 emergency action bypasses step limit"

    def test_t8_3_normal_zero_cost_rejected(self):
        kernel = SafetyKernel(budget=10.0, invariants=[], min_action_cost=1.0)
        state = State({"x": 0})
        zero_cost = ActionSpec(
            id="free", name="Free", description="Zero cost, not emergency",
            effects=(Effect("x", "increment", 1),), cost=0.0,
        )
        v = kernel.evaluate(state, zero_cost)
        assert not v.approved, "T8.3 non-emergency zero-cost rejected"


# R9 — Reasoning layer: Bayesian beliefs, causal graph, action values

class TestR9ReasoningLayer:

    def test_r9_1_belief_uniform_prior(self):
        b = Belief()
        assert abs(b.mean - 0.5) < 1e-9, "R9.1 default belief mean = 0.5"

    def test_r9_2_belief_update_success(self):
        b = Belief()
        for _ in range(10):
            b = b.observe(True)
        assert b.mean > 0.5, "R9.2 success updates mean upward"

    def test_r9_3_belief_update_failure(self):
        b = Belief()
        for _ in range(10):
            b = b.observe(False)
        assert b.mean < 0.5, "R9.3 failure updates mean downward"

    def test_r9_4_belief_state_tracks_multiple(self):
        bs = BeliefState()
        bs.observe("action:a:succeeds", True)
        bs.observe("action:a:succeeds", True)
        bs.observe("action:b:succeeds", False)
        assert bs.get("action:a:succeeds").mean > bs.get("action:b:succeeds").mean, \
            "R9.4 belief state tracks multiple propositions"

    def test_r9_5_causal_graph_blocks_unmet_dep(self):
        cg = CausalGraph()
        cg.add_action("step_a")
        cg.add_action("step_b", depends_on=[("step_a", "Need step_a first")])
        can, unmet = cg.can_execute("step_b")
        assert not can, "R9.5 step_b blocked (dep unmet)"
        assert "step_a" in unmet, "R9.5 step_a in unmet list"

    def test_r9_6_causal_graph_unblocks_on_completion(self):
        cg = CausalGraph()
        cg.add_action("step_a")
        cg.add_action("step_b", depends_on=[("step_a", "Need step_a first")])
        cg.mark_completed("step_a")
        can, _ = cg.can_execute("step_b")
        assert can, "R9.6 step_b allowed after step_a completed"

    def test_r9_7_causal_graph_no_cycle(self):
        cg = CausalGraph()
        cg.add_action("a")
        cg.add_action("b", depends_on=[("a", "b needs a")])
        cg.add_action("c", depends_on=[("b", "c needs b")])
        assert not cg.has_cycle(), "R9.7 linear chain has no cycle"

    def test_r9_8_parse_llm_response_valid(self):
        raw = json.dumps({
            "chosen_action_id": "action1",
            "reasoning": "It makes sense",
            "expected_outcome": "progress",
            "risk_assessment": "low",
            "alternative_considered": "action2",
            "should_stop": False,
            "stop_reason": "",
        })
        resp = parse_llm_response(raw, {"action1", "action2"})
        assert resp.chosen_action_id == "action1", "R9.8 parse chosen_action_id"
        assert resp.is_valid, "R9.8 response is_valid"

    def test_r9_9_parse_llm_response_invalid_action(self):
        raw = json.dumps({
            "chosen_action_id": "nonexistent_action",
            "reasoning": "picked something weird",
            "expected_outcome": "unknown",
            "risk_assessment": "unknown",
            "alternative_considered": "none",
            "should_stop": False,
            "stop_reason": "",
        })
        resp = parse_llm_response(raw, {"action1", "action2"})
        assert not resp.is_valid, "R9.9 invalid action id rejected"

    def test_r9_10_action_value_score_finite(self):
        avc = ActionValueComputer()
        action = ActionSpec(id="act", name="Act", description="test",
                            effects=(Effect("x", "increment", 1),), cost=2.0)
        bs = BeliefState()
        av = avc.compute(action, State({"x": 0}), bs,
                         budget_remaining=10.0, goal_progress=0.3, steps_remaining=5)
        assert math.isfinite(av.value_score), "R9.10 value_score is finite"
        assert av.action_id == "act", "R9.10 action_id matches"


# O9 — Orchestrator integration: full task execution loop

class TestO9OrchestratorIntegration:

    def _make_simple_task(self, n_steps=3, budget=50.0):
        actions = [
            ActionSpec(
                id=f"step_{i}", name=f"Step {i}",
                description=f"Perform step {i}",
                effects=(Effect("progress", "increment", 1),),
                cost=1.0,
            )
            for i in range(n_steps)
        ]
        return TaskDefinition(
            goal=f"Complete {n_steps} steps",
            initial_state=State({"progress": 0}),
            available_actions=actions,
            invariants=[
                Invariant("bound",
                          lambda s, n=n_steps+5: s.get("progress", 0) <= n),
            ],
            budget=budget,
            goal_predicate=lambda s, n=n_steps: s.get("progress", 0) >= n,
        )

    def test_o9_1_goal_achieved(self):
        task = self._make_simple_task(n_steps=3, budget=50.0)
        engine = Orchestrator(task, llm=MockLLMAdapter())
        result = engine.run()
        assert result.goal_achieved, "O9.1 goal achieved"

    def test_o9_2_cost_within_budget(self):
        task = self._make_simple_task(n_steps=3, budget=50.0)
        engine = Orchestrator(task, llm=MockLLMAdapter())
        result = engine.run()
        assert result.total_cost <= 50.0, "O9.2 cost within budget"

    def test_o9_3_trace_valid_after_run(self):
        task = self._make_simple_task(n_steps=3, budget=50.0)
        engine = Orchestrator(task, llm=MockLLMAdapter())
        engine.run()
        ok, msg = engine.kernel.trace.verify_integrity()
        assert ok, f"O9.3 trace valid after orchestrator run: {msg}"

    def test_o9_4_budget_exhaustion_terminates(self):
        # Very small budget — goal unreachable, should terminate gracefully
        actions = [
            ActionSpec(id="inc", name="Inc", description="+1",
                       effects=(Effect("n", "increment", 1),), cost=5.0)
        ]
        task = TaskDefinition(
            goal="Reach n=100",
            initial_state=State({"n": 0}),
            available_actions=actions,
            invariants=[],
            budget=10.0,
            goal_predicate=lambda s: s.get("n", 0) >= 100,
        )
        engine = Orchestrator(task, llm=MockLLMAdapter())
        result = engine.run()
        assert not result.goal_achieved, "O9.4 goal not achieved (budget too small)"
        assert result.total_cost <= 10.0, "O9.4 total cost <= budget"


# D10 — Domain tests: specific scenario coverage

class TestD10DomainTests:

    def test_d10_1_rejection_formatter_llm_message(self):
        kernel = SafetyKernel(budget=10.0, invariants=[
            Invariant("no_delete", lambda s: not s.get("deleted", False))
        ])
        state = State({"deleted": False})
        bad = ActionSpec(id="del", name="Delete", description="Deletes",
                         effects=(Effect("deleted", "set", True),), cost=1.0)
        verdict = kernel.evaluate(state, bad)
        assert not verdict.approved
        msg = RejectionFormatter.llm_message(verdict, bad)
        assert "REJECTED" in msg, "D10.1 llm_message contains REJECTED"
        audit = RejectionFormatter.audit_record(verdict, bad)
        assert audit["action_id"] == "del", "D10.1 audit_record action_id"
        assert not audit["approved"], "D10.1 audit_record not approved"

    def test_d10_2_invariant_suggestion_in_rejection(self):
        inv_with_hint = Invariant(
            "cap_files",
            lambda s: s.get("files", 0) <= 5,
            description="Max 5 files",
            suggestion="Delete old files before creating new ones",
        )
        kernel = SafetyKernel(budget=100.0, invariants=[inv_with_hint])
        state = State({"files": 5})
        create = ActionSpec(id="create", name="Create", description="Add file",
                            effects=(Effect("files", "increment", 1),), cost=1.0)
        verdict = kernel.evaluate(state, create)
        assert not verdict.approved
        user_msg = RejectionFormatter.user_message(verdict, create)
        assert "blocked" in user_msg.lower(), "D10.2 user_message contains 'blocked'"

    def test_d10_3_multi_invariant_all_checked(self):
        invs = [
            Invariant(f"inv_{i}", lambda s, i=i: s.get("x", 0) != i)
            for i in range(3)
        ]
        kernel = SafetyKernel(budget=100.0, invariants=invs)
        state = State({"x": 0})
        # This action sets x=1, which violates inv_1
        action = ActionSpec(id="set1", name="Set1", description="x=1",
                            effects=(Effect("x", "set", 1),), cost=1.0)
        verdict = kernel.evaluate(state, action)
        assert not verdict.approved, "D10.3 multi-invariant: violation caught"

    def test_d10_4_state_applies_effects_correctly(self):
        state = State({"n": 10, "s": "hello"})
        action = ActionSpec(
            id="compound", name="Compound", description="Multiple effects",
            effects=(
                Effect("n", "increment", 5),
                Effect("s", "set", "world"),
                Effect("new_key", "set", 42),
            ),
            cost=1.0,
        )
        next_state = action.simulate(state)
        assert next_state.get("n") == 15, "D10.4 increment effect"
        assert next_state.get("s") == "world", "D10.4 set effect on string"
        assert next_state.get("new_key") == 42, "D10.4 new key created"
        assert state.get("n") == 10, "D10.4 original state unchanged"

    def test_d10_5_invariant_timeout_treated_as_violation(self):
        import time as _time

        def slow_predicate(s):
            _time.sleep(5.0)  # Way too slow
            return True

        inv_slow = Invariant(
            "slow_check",
            slow_predicate,
            description="Should timeout",
            max_eval_ms=50.0,  # 50ms timeout
        )
        # Direct check should timeout and return (False, message)
        t0 = _time.time()
        holds, msg = inv_slow.check(State({}))
        elapsed = _time.time() - t0
        assert not holds, "D10.5 timeout treated as violation"
        assert "timed out" in msg.lower(), "D10.5 timeout message"
        assert elapsed < 2.0, f"D10.5 timeout respected (elapsed={elapsed:.2f}s)"


# A11 — Adversarial: attempts to bypass the safety kernel

class TestA11Adversarial:

    def test_a11_1_concurrent_kernel_thread_safe(self):
        """Multiple threads hammering the kernel — budget must never be exceeded."""
        kernel = SafetyKernel(budget=20.0, invariants=[], min_action_cost=1.0)
        state = State({"n": 0})
        action = ActionSpec(id="inc", name="Inc", description="+1",
                            effects=(Effect("n", "increment", 1),), cost=1.0)
        errors = []

        def worker():
            try:
                for _ in range(5):
                    v = kernel.evaluate(state, action)
                    if v.approved:
                        kernel.execute(state, action)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"A11.1 no exceptions in concurrent use: {errors}"
        assert kernel.budget.spent_net <= 20.0, "A11.1 budget never exceeded"

    def test_a11_2_pathological_near_zero_budget(self):
        """Budget so small that only 1 action is possible."""
        kernel = SafetyKernel(budget=1.001, invariants=[], min_action_cost=1.0)
        state = State({"n": 0})
        action = ActionSpec(id="inc", name="Inc", description="+1",
                            effects=(Effect("n", "increment", 1),), cost=1.0)
        v1 = kernel.evaluate(state, action)
        if v1.approved:
            state, _ = kernel.execute(state, action)
        v2 = kernel.evaluate(state, action)
        assert not v2.approved, "A11.2 second action blocked near budget boundary"

    def test_a11_3_invariant_with_expensive_predicate(self):
        """Invariant timeout test at kernel level."""
        import time as _time

        def slow_predicate(s):
            _time.sleep(10.0)  # Will timeout
            return True

        inv = Invariant("slow", slow_predicate, max_eval_ms=100.0)
        kernel = SafetyKernel(budget=100.0, invariants=[inv])
        state = State({})
        action = ActionSpec(id="a", name="A", description="Test",
                            effects=(), cost=1.0)
        t0 = _time.time()
        v = kernel.evaluate(state, action)
        elapsed = _time.time() - t0
        assert not v.approved, "A11.3 slow invariant treated as violation"
        assert elapsed < 5.0, f"A11.3 kernel evaluation respected timeout (elapsed={elapsed:.2f}s)"

    def test_a11_4_state_immutable_cannot_be_modified(self):
        """State object resists direct mutation."""
        s = State({"x": 10, "data": [1, 2, 3]})
        # Try dict-style mutation (should fail)
        try:
            s._vars["x"] = 99
            # If we got here, check that the state value is unchanged
            assert s.get("x") == 10, "A11.4 value unchanged despite attempted mutation"
        except (TypeError, AttributeError):
            pass  # Expected: MappingProxyType or frozen

    def test_a11_5_invalid_effect_mode_raises(self):
        """ActionSpec with invalid effect mode should fail at simulation."""
        action = ActionSpec(
            id="bad", name="Bad", description="Invalid mode",
            effects=(Effect("x", "invalid_mode_xyz", 1),), cost=1.0,
        )
        state = State({"x": 0})
        try:
            result = action.simulate(state)
            # Some implementations silently ignore unknown modes — check state unchanged
            assert result.get("x") == state.get("x"), \
                "A11.5 unknown effect mode leaves state unchanged"
        except (ValueError, KeyError, AttributeError):
            pass  # Also acceptable

    def test_a11_6_massive_cost_blocked(self):
        """Action costing 10x budget is always rejected."""
        kernel = SafetyKernel(budget=10.0, invariants=[], min_action_cost=0.1)
        state = State({})
        action = ActionSpec(id="huge", name="Huge", description="Costs 1000",
                            effects=(), cost=1000.0)
        v = kernel.evaluate(state, action)
        assert not v.approved, "A11.6 massive cost blocked"
