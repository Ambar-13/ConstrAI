"""
Tests for constrai.orchestrator coverage:
  Outcome, ProgressMonitor, ExecutionResult, Orchestrator termination scenarios.
"""
from __future__ import annotations

import pytest

from constrai import (
    ActionSpec,
    Effect,
    ExecutionResult,
    Invariant,
    MockLLMAdapter,
    Orchestrator,
    Outcome,
    OutcomeType,
    ProgressMonitor,
    State,
    TaskDefinition,
    TerminationReason,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _inc_action(cost: float = 1.0) -> ActionSpec:
    return ActionSpec(
        id="inc",
        name="Increment",
        description="Increment counter by 1",
        effects=(Effect("counter", "increment", 1),),
        cost=cost,
    )


def _make_task(
    *,
    budget: float = 10.0,
    min_action_cost: float = 0.001,
    invariants=None,
    goal_fn=None,
    max_consecutive_failures: int = 5,
    actions=None,
    priors=None,
) -> TaskDefinition:
    acts = actions if actions is not None else [_inc_action()]
    goal = goal_fn or (lambda s: s.get("counter", 0) >= 100)
    return TaskDefinition(
        goal="Reach counter 100",
        initial_state=State({"counter": 0}),
        available_actions=acts,
        invariants=invariants or [],
        budget=budget,
        goal_predicate=goal,
        min_action_cost=min_action_cost,
        max_consecutive_failures=max_consecutive_failures,
        priors=priors,
    )


def _make_exec_result(
    *,
    goal_achieved: bool = True,
    reason: TerminationReason = TerminationReason.GOAL_ACHIEVED,
) -> ExecutionResult:
    return ExecutionResult(
        goal_achieved=goal_achieved,
        termination_reason=reason,
        final_state=State({}),
        total_cost=1.0,
        total_steps=1,
        goal_progress=1.0,
        execution_time_s=0.01,
        trace_length=1,
        beliefs_summary="",
        budget_summary="",
    )


# ── Outcome ────────────────────────────────────────────────────────────────────


class TestOutcome:
    def _make(self, otype: OutcomeType, *, same_state: bool = True) -> Outcome:
        s = State({"x": 1})
        t = State({"x": 1}) if same_state else State({"x": 2})
        return Outcome(action_id="a1", outcome_type=otype,
                       actual_state=s, expected_state=t)

    def test_succeeded_true_for_success(self):
        assert self._make(OutcomeType.SUCCESS).succeeded is True

    def test_succeeded_false_for_partial(self):
        assert self._make(OutcomeType.PARTIAL).succeeded is False

    def test_succeeded_false_for_failure(self):
        assert self._make(OutcomeType.FAILURE).succeeded is False

    def test_state_matches_expected_true(self):
        assert self._make(OutcomeType.SUCCESS, same_state=True).state_matches_expected is True

    def test_state_matches_expected_false(self):
        assert self._make(OutcomeType.SUCCESS, same_state=False).state_matches_expected is False


# ── ProgressMonitor ────────────────────────────────────────────────────────────


class TestProgressMonitor:
    def test_empty_current_progress_is_zero(self):
        pm = ProgressMonitor()
        assert pm.current_progress == 0.0

    def test_record_updates_current_progress(self):
        pm = ProgressMonitor()
        pm.record(1, 0.5)
        assert pm.current_progress == 0.5

    def test_multiple_records_last_value(self):
        pm = ProgressMonitor()
        pm.record(1, 0.2)
        pm.record(2, 0.7)
        assert pm.current_progress == 0.7

    def test_progress_rate_empty_is_zero(self):
        pm = ProgressMonitor()
        assert pm.progress_rate == 0.0

    def test_progress_rate_single_entry_is_zero(self):
        pm = ProgressMonitor()
        pm.record(1, 0.5)
        assert pm.progress_rate == 0.0

    def test_progress_rate_computed(self):
        pm = ProgressMonitor()
        pm.record(0, 0.0)
        pm.record(2, 0.4)
        # dp = 0.4, ds = 2  → rate = 0.2
        assert abs(pm.progress_rate - 0.2) < 1e-9

    def test_estimated_steps_none_when_no_progress(self):
        pm = ProgressMonitor()
        pm.record(0, 0.5)
        pm.record(1, 0.5)
        assert pm.estimated_steps_to_goal() is None

    def test_estimated_steps_with_positive_rate(self):
        pm = ProgressMonitor()
        pm.record(0, 0.0)
        pm.record(1, 0.5)
        # rate = 0.5, remaining = 0.5  → steps = 1
        assert pm.estimated_steps_to_goal() == 1

    def test_is_stuck_false_insufficient_history(self):
        pm = ProgressMonitor(patience=5)
        for i in range(4):
            pm.record(i, 0.1)
        assert pm.is_stuck is False

    def test_is_stuck_true_no_improvement(self):
        pm = ProgressMonitor(patience=3)
        for i in range(3):
            pm.record(i, 0.5)
        assert pm.is_stuck is True

    def test_is_stuck_false_improving(self):
        pm = ProgressMonitor(patience=3)
        pm.record(0, 0.1)
        pm.record(1, 0.5)
        pm.record(2, 0.9)
        assert pm.is_stuck is False

    def test_to_llm_text_contains_progress(self):
        pm = ProgressMonitor()
        pm.record(1, 0.42)
        text = pm.to_llm_text()
        assert "Progress:" in text

    def test_to_llm_text_contains_rate_when_positive(self):
        pm = ProgressMonitor()
        pm.record(0, 0.0)
        pm.record(1, 0.3)
        text = pm.to_llm_text()
        assert "Rate:" in text

    def test_to_llm_text_no_rate_when_zero(self):
        pm = ProgressMonitor()
        pm.record(0, 0.5)
        text = pm.to_llm_text()
        assert "Rate:" not in text


# ── ExecutionResult ────────────────────────────────────────────────────────────


class TestExecutionResult:
    def test_to_dict_has_required_keys(self):
        result = _make_exec_result()
        d = result.to_dict()
        for key in ("goal_achieved", "termination_reason", "total_cost",
                    "total_steps", "goal_progress", "execution_time_s",
                    "actions_attempted", "actions_succeeded",
                    "actions_rejected_safety", "rollbacks", "llm_calls", "errors"):
            assert key in d

    def test_to_dict_goal_achieved_true(self):
        d = _make_exec_result(goal_achieved=True).to_dict()
        assert d["goal_achieved"] is True

    def test_to_dict_termination_reason_is_string(self):
        d = _make_exec_result().to_dict()
        assert isinstance(d["termination_reason"], str)

    def test_to_dict_errors_is_list(self):
        d = _make_exec_result().to_dict()
        assert isinstance(d["errors"], list)

    def test_summary_non_empty(self):
        assert len(_make_exec_result().summary()) > 0

    def test_summary_achieved_shows_checkmark(self):
        assert "GOAL ACHIEVED" in _make_exec_result(goal_achieved=True).summary()

    def test_summary_not_achieved_shows_x(self):
        assert "GOAL NOT ACHIEVED" in _make_exec_result(
            goal_achieved=False, reason=TerminationReason.STEP_LIMIT).summary()


# ── Orchestrator: init & priors ───────────────────────────────────────────────


class TestOrchestratorInit:
    def test_default_llm_is_mock(self):
        engine = Orchestrator(_make_task())
        assert isinstance(engine.llm, MockLLMAdapter)

    def test_priors_initialize_beliefs(self):
        task = _make_task(priors={"action:pay:succeeds": (9.0, 1.0)})
        engine = Orchestrator(task)
        belief = engine.beliefs.get("action:pay:succeeds")
        assert abs(belief.mean - 9.0 / 10.0) < 1e-9

    def test_no_priors_gives_default_belief_mean(self):
        engine = Orchestrator(_make_task())
        # Default Beta(1,1) → mean = 0.5
        belief = engine.beliefs.get("some_key")
        assert abs(belief.mean - 0.5) < 1e-9


# ── Orchestrator: initial state violation ─────────────────────────────────────


class TestOrchestratorInitialStateViolation:
    def _blocking_inv(self) -> Invariant:
        # Fails on counter=0 (the initial state)
        return Invariant(
            "counter_must_be_positive",
            lambda s: s.get("counter", 0) > 0,
            "Counter must be > 0",
        )

    def test_blocking_invariant_returns_error(self):
        task = _make_task(invariants=[self._blocking_inv()])
        result = Orchestrator(task).run()
        assert result.termination_reason == TerminationReason.ERROR

    def test_error_result_not_goal_achieved(self):
        task = _make_task(invariants=[self._blocking_inv()])
        result = Orchestrator(task).run()
        assert result.goal_achieved is False

    def test_monitoring_invariant_does_not_block(self):
        inv = Invariant(
            "monitor_only",
            lambda s: s.get("counter", 0) > 0,
            "Counter > 0",
            enforcement="monitoring",
        )
        task = _make_task(invariants=[inv], goal_fn=lambda s: True)
        result = Orchestrator(task).run()
        assert result.termination_reason == TerminationReason.GOAL_ACHIEVED


# ── Orchestrator: goal achieved immediately ───────────────────────────────────


class TestOrchestratorGoalAchievedInitially:
    def test_goal_satisfied_at_start_terminates(self):
        task = _make_task(goal_fn=lambda s: True)
        result = Orchestrator(task).run()
        assert result.termination_reason == TerminationReason.GOAL_ACHIEVED

    def test_goal_achieved_flag_true(self):
        task = _make_task(goal_fn=lambda s: True)
        result = Orchestrator(task).run()
        assert result.goal_achieved is True

    def test_total_steps_zero_when_immediate(self):
        task = _make_task(goal_fn=lambda s: True)
        result = Orchestrator(task).run()
        assert result.total_steps == 0


# ── Orchestrator: goal achieved via action ────────────────────────────────────


class TestOrchestratorGoalAchievedViaAction:
    def test_goal_achieved_after_one_inc(self):
        task = _make_task(
            goal_fn=lambda s: s.get("counter", 0) >= 1,
            budget=5.0,
        )
        result = Orchestrator(task).run()
        assert result.goal_achieved is True

    def test_goal_reason_is_goal_achieved(self):
        task = _make_task(
            goal_fn=lambda s: s.get("counter", 0) >= 1,
            budget=5.0,
        )
        result = Orchestrator(task).run()
        assert result.termination_reason == TerminationReason.GOAL_ACHIEVED

    def test_final_state_reflects_action(self):
        task = _make_task(
            goal_fn=lambda s: s.get("counter", 0) >= 1,
            budget=5.0,
        )
        result = Orchestrator(task).run()
        assert result.final_state.get("counter", 0) >= 1


# ── Orchestrator: step limit ──────────────────────────────────────────────────


class TestOrchestratorStepLimit:
    def _step_limit_task(self) -> TaskDefinition:
        # budget=3.0, min_action_cost=1.0  → max_steps = 3
        return _make_task(budget=3.0, min_action_cost=1.0)

    def test_step_limit_terminates(self):
        result = Orchestrator(self._step_limit_task()).run()
        assert result is not None

    def test_step_limit_reason(self):
        result = Orchestrator(self._step_limit_task()).run()
        assert result.termination_reason == TerminationReason.STEP_LIMIT

    def test_step_limit_not_goal_achieved(self):
        result = Orchestrator(self._step_limit_task()).run()
        assert result.goal_achieved is False

    def test_step_limit_total_steps_equals_max(self):
        task = self._step_limit_task()
        result = Orchestrator(task).run()
        # max_steps = int(3.0 / 1.0) = 3
        assert result.total_steps == 3


# ── Orchestrator: max consecutive failures ────────────────────────────────────


class TestOrchestratorMaxFailures:
    def _blocking_task(self, max_failures: int = 3) -> TaskDefinition:
        # invariant that blocks the result of "inc" (counter=0 must stay 0)
        inv = Invariant(
            "freeze_counter",
            lambda s: s.get("counter", 0) == 0,
            "Counter must stay at 0",
        )
        return _make_task(
            budget=100.0,
            invariants=[inv],
            max_consecutive_failures=max_failures,
        )

    def test_max_failures_terminates(self):
        result = Orchestrator(self._blocking_task()).run()
        assert result is not None

    def test_max_failures_reason(self):
        result = Orchestrator(self._blocking_task()).run()
        assert result.termination_reason == TerminationReason.MAX_FAILURES

    def test_max_failures_not_goal_achieved(self):
        result = Orchestrator(self._blocking_task()).run()
        assert result.goal_achieved is False

    def test_max_failures_with_fewer_allowed_failures(self):
        result = Orchestrator(self._blocking_task(max_failures=2)).run()
        assert result.termination_reason == TerminationReason.MAX_FAILURES


# ── Orchestrator: budget exhausted ────────────────────────────────────────────


class TestOrchestratorBudgetExhausted:
    def test_budget_too_small_exhausted(self):
        # action costs 5.0, budget is only 2.0
        task = _make_task(
            budget=2.0,
            actions=[_inc_action(cost=5.0)],
            min_action_cost=0.001,
        )
        result = Orchestrator(task).run()
        assert result.termination_reason == TerminationReason.BUDGET_EXHAUSTED

    def test_budget_exhausted_not_goal_achieved(self):
        task = _make_task(
            budget=2.0,
            actions=[_inc_action(cost=5.0)],
            min_action_cost=0.001,
        )
        result = Orchestrator(task).run()
        assert result.goal_achieved is False

    def test_budget_exhausted_total_cost_zero(self):
        task = _make_task(
            budget=2.0,
            actions=[_inc_action(cost=5.0)],
            min_action_cost=0.001,
        )
        result = Orchestrator(task).run()
        assert result.total_cost == 0.0


# ── Orchestrator: fallback selection ─────────────────────────────────────────


class TestOrchestratorFallback:
    def test_fallback_picks_ready_action(self):
        task = _make_task()
        engine = Orchestrator(task)
        avail = engine._get_available_actions()
        vals = engine._compute_action_values(avail)
        response = engine._fallback_selection(avail, vals)
        assert response.chosen_action_id == "inc"

    def test_fallback_not_should_stop_when_actions_exist(self):
        task = _make_task()
        engine = Orchestrator(task)
        avail = engine._get_available_actions()
        vals = engine._compute_action_values(avail)
        response = engine._fallback_selection(avail, vals)
        assert response.should_stop is False

    def test_fallback_should_stop_when_no_ready(self):
        task = _make_task()
        engine = Orchestrator(task)
        # Pass empty lists to force should_stop
        response = engine._fallback_selection([], [])
        assert response.should_stop is True

    def test_fallback_reasoning_is_non_empty(self):
        task = _make_task()
        engine = Orchestrator(task)
        avail = engine._get_available_actions()
        vals = engine._compute_action_values(avail)
        response = engine._fallback_selection(avail, vals)
        assert len(response.reasoning) > 0
