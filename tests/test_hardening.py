"""Tests for clampai.hardening — defence-in-depth layer.

Covers SubprocessAttestor, PredicateAttestor, AttestationGate, ReadinessProbe,
TemporalDependency, TemporalCausalGraph, CostAwarePriorFactory,
EnvironmentReconciler, MultiDimensionalAttestor, DependencyDiscovery,
ResourceTracker, and the HARDENING_CLAIMS registry.
"""
from __future__ import annotations

import subprocess
from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest

from clampai.formal import ActionSpec, Effect, State
from clampai.hardening import (
    HARDENING_CLAIMS,
    Attestation,
    AttestationGate,
    AttestationResult,
    CostAwarePriorFactory,
    DependencyDiscovery,
    EnvironmentDriftError,
    EnvironmentProbe,
    EnvironmentReconciler,
    FailurePattern,
    MultiDimensionalAttestor,
    Permission,
    PredicateAttestor,
    QualityDimension,
    QualityScore,
    ReadinessProbe,
    ReconciliationResult,
    ResourceDescriptor,
    ResourceState,
    ResourceTracker,
    SubprocessAttestor,
    TemporalCausalGraph,
    TemporalDependency,
)
from clampai.reasoning import BeliefState, CausalGraph

# ─── helpers ─────────────────────────────────────────────────────────────────

def _action(aid: str, cost: float = 1.0, reversible: bool = True,
            risk_level: str = "low") -> ActionSpec:
    return ActionSpec(
        id=aid,
        name=aid,
        description="",
        effects=(Effect("x", "increment", 1),),
        cost=cost,
        reversible=reversible,
        risk_level=risk_level,
    )


def _state(**kw) -> State:
    return State(kw)


# ─── Attestation.is_positive ─────────────────────────────────────────────────

class TestAttestationIsPositive:
    def test_verified_is_positive(self):
        a = Attestation("x", AttestationResult.VERIFIED, "ok", 0.0, "fp")
        assert a.is_positive() is True

    def test_failed_is_not_positive(self):
        a = Attestation("x", AttestationResult.FAILED, "fail", 0.0, "fp")
        assert a.is_positive() is False

    def test_error_is_not_positive(self):
        a = Attestation("x", AttestationResult.ERROR, "err", 0.0, "fp")
        assert a.is_positive() is False

    def test_timeout_is_not_positive(self):
        a = Attestation("x", AttestationResult.TIMEOUT, "timeout", 0.0, "fp")
        assert a.is_positive() is False

    def test_skipped_is_not_positive(self):
        a = Attestation("x", AttestationResult.SKIPPED, "skip", 0.0, "fp")
        assert a.is_positive() is False


# ─── SubprocessAttestor ───────────────────────────────────────────────────────

class TestSubprocessAttestorInit:
    def test_empty_command_raises(self):
        with pytest.raises(ValueError, match="empty"):
            SubprocessAttestor("test", [])

    def test_binary_not_in_allowlist_raises(self):
        with pytest.raises(ValueError, match="not in allowlist"):
            SubprocessAttestor("test", ["rm", "-rf", "/"])

    def test_invalid_working_dir_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            SubprocessAttestor("test", ["ls"], working_dir="/nonexistent_abc_xyz_123")

    def test_valid_command(self):
        a = SubprocessAttestor("mytest", ["ls"])
        assert a.name == "mytest"

    def test_custom_allowlist(self):
        a = SubprocessAttestor("t", ["mybin"], command_allowlist=frozenset({"mybin"}))
        assert a.name == "t"

    def test_binary_extracted_from_path(self):
        # Full path to binary — basename must be in allowlist.
        a = SubprocessAttestor("t", ["/usr/bin/ls"])
        assert a.name == "t"


class TestSubprocessAttestorVerify:
    def test_successful_run_returns_verified(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            a = SubprocessAttestor("t", ["ls"])
            att = a.verify(_state(), "goal", timeout_s=5.0)
        assert att.result == AttestationResult.VERIFIED
        assert att.attestor_name == "t"

    def test_nonzero_returncode_returns_failed(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"
        with patch("subprocess.run", return_value=mock_result):
            a = SubprocessAttestor("t", ["ls"])
            att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.FAILED

    def test_success_pattern_matches(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hello world"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            a = SubprocessAttestor("t", ["ls"], success_pattern="hello")
            att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.VERIFIED

    def test_success_pattern_no_match(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hello world"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            a = SubprocessAttestor("t", ["ls"], success_pattern="NOTHERE")
            att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.FAILED

    def test_timeout_returns_timeout_attestation(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["ls"], 5)):
            a = SubprocessAttestor("t", ["ls"])
            att = a.verify(_state(), "goal", timeout_s=5.0)
        assert att.result == AttestationResult.TIMEOUT
        assert "5" in att.evidence

    def test_exception_returns_error_attestation(self):
        with patch("subprocess.run", side_effect=OSError("permission denied")):
            a = SubprocessAttestor("t", ["ls"])
            att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.ERROR
        assert "permission denied" in att.evidence


# ─── PredicateAttestor ────────────────────────────────────────────────────────

class TestPredicateAttestor:
    def test_verified_when_check_passes(self):
        a = PredicateAttestor("p", lambda: (True, "all good"))
        att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.VERIFIED
        assert att.attestor_name == "p"

    def test_failed_when_check_fails(self):
        a = PredicateAttestor("p", lambda: (False, "not ready"))
        att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.FAILED
        assert "not ready" in att.evidence

    def test_error_when_check_raises(self):
        def bad():
            raise RuntimeError("probe crashed")
        a = PredicateAttestor("p", bad)
        att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.ERROR
        assert "probe crashed" in att.evidence

    def test_name_property(self):
        a = PredicateAttestor("myname", lambda: (True, "ok"))
        assert a.name == "myname"


# ─── AttestationGate ─────────────────────────────────────────────────────────

class TestAttestationGate:
    def test_goal_predicate_false_returns_false(self):
        gate = AttestationGate()
        ok, atts = gate.verify_goal(
            _state(), "goal", goal_predicate=lambda s: False
        )
        assert ok is False
        assert atts == []

    def test_no_attestors_returns_skipped(self):
        gate = AttestationGate()
        ok, atts = gate.verify_goal(
            _state(), "goal", goal_predicate=lambda s: True
        )
        assert ok is True
        assert len(atts) == 1
        assert atts[0].result == AttestationResult.SKIPPED

    def test_quorum_met(self):
        gate = AttestationGate(quorum=1)
        gate.add_attestor(PredicateAttestor("p1", lambda: (True, "ok")))
        gate.add_attestor(PredicateAttestor("p2", lambda: (False, "fail")))
        ok, atts = gate.verify_goal(
            _state(), "goal", goal_predicate=lambda s: True
        )
        assert ok is True
        assert len(atts) == 2

    def test_quorum_not_met(self):
        gate = AttestationGate(quorum=2)
        gate.add_attestor(PredicateAttestor("p1", lambda: (True, "ok")))
        gate.add_attestor(PredicateAttestor("p2", lambda: (False, "fail")))
        ok, _atts = gate.verify_goal(
            _state(), "goal", goal_predicate=lambda s: True
        )
        assert ok is False

    def test_attestation_history_recorded(self):
        gate = AttestationGate()
        gate.add_attestor(PredicateAttestor("p", lambda: (True, "ok")))
        gate.verify_goal(_state(), "goal", goal_predicate=lambda s: True)
        gate.verify_goal(_state(), "goal", goal_predicate=lambda s: True)
        hist = gate.attestation_history
        assert len(hist) == 2

    def test_attestation_history_is_copy(self):
        gate = AttestationGate()
        gate.add_attestor(PredicateAttestor("p", lambda: (True, "ok")))
        gate.verify_goal(_state(), "goal", goal_predicate=lambda s: True)
        hist = gate.attestation_history
        hist.clear()
        assert len(gate.attestation_history) == 1


# ─── ReadinessProbe ───────────────────────────────────────────────────────────

class TestReadinessProbe:
    def test_ready_immediately(self):
        probe = ReadinessProbe(
            name="p",
            check_fn=lambda: (True, "ready"),
            interval_s=0.0,
        )
        ok, detail, wait = probe.wait_until_ready()
        assert ok is True
        assert "ready" in detail
        assert wait == 0.0

    def test_ready_after_retries(self):
        calls = [0]

        def check():
            calls[0] += 1
            return (calls[0] >= 3, f"attempt {calls[0]}")

        probe = ReadinessProbe(
            name="p",
            check_fn=check,
            interval_s=0.0,
            max_retries=10,
        )
        ok, _, _ = probe.wait_until_ready()
        assert ok is True
        assert calls[0] == 3

    def test_not_ready_exhausts_retries(self):
        probe = ReadinessProbe(
            name="p",
            check_fn=lambda: (False, "not ready"),
            interval_s=0.0,
            max_retries=3,
        )
        ok, detail, _ = probe.wait_until_ready()
        assert ok is False
        assert "3 attempts" in detail

    def test_probe_exception_treated_as_not_ready(self):
        calls = [0]

        def flaky():
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("transient error")
            return (True, "recovered")

        probe = ReadinessProbe(
            name="p",
            check_fn=flaky,
            interval_s=0.0,
            max_retries=5,
        )
        ok, _, _ = probe.wait_until_ready()
        assert ok is True

    def test_initial_delay_counted(self):
        probe = ReadinessProbe(
            name="p",
            check_fn=lambda: (True, "ok"),
            initial_delay_s=2.0,
            interval_s=0.0,
        )
        ok, _, wait = probe.wait_until_ready()
        assert ok is True
        assert wait == 2.0


# ─── TemporalDependency ───────────────────────────────────────────────────────

class TestTemporalDependency:
    def test_not_satisfied_before_completion(self):
        dep = TemporalDependency("action_a", "needs a")
        ok, reason = dep.is_satisfied(current_time=10.0)
        assert ok is False
        assert "action_a" in reason

    def test_satisfied_after_completion_no_delay(self):
        dep = TemporalDependency("action_a", "needs a", min_delay_s=0.0)
        dep.mark_completed(at_time=5.0)
        ok, reason = dep.is_satisfied(current_time=5.0)
        assert ok is True
        assert reason == ""

    def test_delay_not_elapsed(self):
        dep = TemporalDependency("action_a", "needs a", min_delay_s=10.0)
        dep.mark_completed(at_time=0.0)
        ok, reason = dep.is_satisfied(current_time=5.0)
        assert ok is False
        assert "need 10.0s" in reason

    def test_delay_elapsed(self):
        dep = TemporalDependency("action_a", "needs a", min_delay_s=5.0)
        dep.mark_completed(at_time=0.0)
        ok, _ = dep.is_satisfied(current_time=10.0)
        assert ok is True

    def test_with_readiness_probe_fails(self):
        probe = ReadinessProbe(
            name="probe",
            check_fn=lambda: (False, "not ready"),
            interval_s=0.0,
            max_retries=1,
        )
        dep = TemporalDependency("a", "reason", readiness_probe=probe)
        dep.mark_completed(at_time=0.0)
        ok, reason = dep.is_satisfied(current_time=0.0)
        assert ok is False
        assert "not ready" in reason

    def test_with_readiness_probe_passes(self):
        probe = ReadinessProbe(
            name="probe",
            check_fn=lambda: (True, "ready"),
            interval_s=0.0,
        )
        dep = TemporalDependency("a", "reason", readiness_probe=probe)
        dep.mark_completed(at_time=0.0)
        ok, _ = dep.is_satisfied(current_time=0.0)
        assert ok is True


# ─── TemporalCausalGraph ──────────────────────────────────────────────────────

class TestTemporalCausalGraph:
    def _make_graph(self):
        base = CausalGraph()
        base.add_action("a", [])
        base.add_action("b", [("a", "needs a")])
        return TemporalCausalGraph(base)

    def test_can_execute_base_unmet(self):
        g = self._make_graph()
        g.set_time(0.0)
        ok, reasons = g.can_execute("b")
        assert ok is False
        assert len(reasons) > 0

    def test_can_execute_after_mark_completed(self):
        g = self._make_graph()
        g.set_time(0.0)
        g.mark_completed("a")
        ok, _ = g.can_execute("b")
        assert ok is True

    def test_temporal_dep_blocks_execution(self):
        g = self._make_graph()
        g.mark_completed("a", at_time=0.0)
        tdep = TemporalDependency("a", "must wait", min_delay_s=100.0)
        g.add_temporal_dep("b", tdep)
        g.set_time(0.0)
        ok, reasons = g.can_execute("b")
        assert ok is False
        assert len(reasons) > 0

    def test_temporal_dep_satisfied_after_time(self):
        g = self._make_graph()
        # Add temporal dep BEFORE marking completed so propagation fires.
        tdep = TemporalDependency("a", "must wait", min_delay_s=5.0)
        g.add_temporal_dep("b", tdep)
        g.mark_completed("a", at_time=0.0)
        g.set_time(10.0)
        ok, _ = g.can_execute("b")
        assert ok is True

    def test_mark_completed_propagates_to_temporal_deps(self):
        g = self._make_graph()
        tdep = TemporalDependency("a", "must wait", min_delay_s=5.0)
        g.add_temporal_dep("b", tdep)
        g.mark_completed("a", at_time=0.0)
        g.set_time(10.0)
        ok, _ = g.can_execute("b")
        assert ok is True

    def test_ready_actions_filters(self):
        g = self._make_graph()
        g.set_time(0.0)
        ready = g.ready_actions(["a", "b"])
        assert "a" in ready
        assert "b" not in ready


# ─── CostAwarePriorFactory ────────────────────────────────────────────────────

class TestCostAwarePriorFactory:
    def _factory(self, budget: float = 100.0) -> CostAwarePriorFactory:
        return CostAwarePriorFactory(total_budget=budget, pessimism_factor=5.0,
                                     expensive_threshold=0.1, critical_threshold=0.3)

    def test_explore_tier(self):
        f = self._factory()
        a = _action("a", cost=1.0)  # cost_ratio=0.01 → EXPLORE
        alpha, _beta, tier = f.compute_prior(a)
        assert tier == "EXPLORE"
        assert alpha == 3.0

    def test_cautious_tier(self):
        f = self._factory()
        a = _action("a", cost=5.0)  # cost_ratio=0.05 → CAUTIOUS
        _, _, tier = f.compute_prior(a)
        assert tier == "CAUTIOUS"

    def test_guarded_tier_by_cost(self):
        f = self._factory()
        a = _action("a", cost=15.0)  # cost_ratio=0.15 → GUARDED
        _, _, tier = f.compute_prior(a)
        assert tier == "GUARDED"

    def test_guarded_tier_by_risk(self):
        f = self._factory()
        a = _action("a", cost=1.0, risk_level="critical")  # critical → GUARDED
        _, _, tier = f.compute_prior(a)
        assert tier == "GUARDED"

    def test_gated_tier_unauthorized(self):
        f = self._factory()
        a = _action("a", cost=40.0, reversible=False, risk_level="critical")
        alpha, _, tier = f.compute_prior(a)
        assert tier == "GATED"
        assert alpha == pytest.approx(0.01)

    def test_gated_tier_authorized(self):
        f = self._factory()
        a = _action("a", cost=40.0, reversible=False, risk_level="critical")
        f.authorize("a")
        _, _, tier = f.compute_prior(a)
        assert tier == "GATED_AUTHORIZED"

    def test_gated_by_cost_ratio(self):
        f = self._factory(budget=10.0)
        a = _action("a", cost=5.0)  # cost_ratio=0.5 >= critical_threshold=0.3 → GATED
        _, _, tier = f.compute_prior(a)
        assert tier == "GATED"

    def test_initialize_beliefs(self):
        f = self._factory()
        beliefs = BeliefState()
        actions = [_action("a", 1.0), _action("b", 15.0)]
        tiers = f.initialize_beliefs(beliefs, actions)
        assert "a" in tiers
        assert "b" in tiers

    def test_summary_returns_string(self):
        f = self._factory()
        actions = [_action("a", 1.0), _action("b", 15.0)]
        s = f.summary(actions)
        assert "Cost-Aware Priors" in s
        assert "a" in s
        assert "b" in s


# ─── ReconciliationResult ─────────────────────────────────────────────────────

class TestReconciliationResult:
    def test_is_consistent_true(self):
        r = ReconciliationResult(
            matches=[("x", 1, 1)], mismatches=[], unprobed=[]
        )
        assert r.is_consistent is True

    def test_is_consistent_false(self):
        r = ReconciliationResult(
            matches=[], mismatches=[("x", 1, 2)], unprobed=[]
        )
        assert r.is_consistent is False

    def test_drift_severity_zero_when_consistent(self):
        r = ReconciliationResult(
            matches=[("x", 1, 1)], mismatches=[], unprobed=[]
        )
        assert r.drift_severity == pytest.approx(0.0)

    def test_drift_severity_calculated(self):
        r = ReconciliationResult(
            matches=[("x", 1, 1)], mismatches=[("y", 2, 3)], unprobed=[]
        )
        assert r.drift_severity == pytest.approx(0.5)

    def test_drift_severity_zero_total(self):
        r = ReconciliationResult(matches=[], mismatches=[], unprobed=[])
        assert r.drift_severity == pytest.approx(0.0)


# ─── EnvironmentDriftError ───────────────────────────────────────────────────

class TestEnvironmentDriftError:
    def test_attributes(self):
        result = ReconciliationResult([], [("x", 1, 2)], [])
        err = EnvironmentDriftError("drift detected", result)
        assert str(err) == "drift detected"
        assert err.result is result


# ─── EnvironmentReconciler ───────────────────────────────────────────────────

class TestEnvironmentReconciler:
    def test_add_probe_and_snapshot(self):
        rec = EnvironmentReconciler()
        rec.add_probe(EnvironmentProbe("x", lambda: 42))
        snap = rec.snapshot()
        assert snap["x"] == 42

    def test_snapshot_handles_probe_error(self):
        rec = EnvironmentReconciler()
        rec.add_probe(EnvironmentProbe("bad", lambda: 1 / 0))
        snap = rec.snapshot()
        assert "PROBE_ERROR" in snap["bad"]

    def test_reconcile_all_matching(self):
        rec = EnvironmentReconciler(drift_threshold=0.0, halt_on_drift=True)
        rec.add_probe(EnvironmentProbe("count", lambda: 5))
        s = _state(count=5)
        result = rec.reconcile(s, label="test")
        assert result.is_consistent
        assert len(result.matches) == 1

    def test_reconcile_mismatch_raises_drift_error(self):
        rec = EnvironmentReconciler(drift_threshold=0.0, halt_on_drift=True)
        rec.add_probe(EnvironmentProbe("count", lambda: 99))
        s = _state(count=5)
        with pytest.raises(EnvironmentDriftError):
            rec.reconcile(s)

    def test_reconcile_mismatch_no_halt(self):
        rec = EnvironmentReconciler(drift_threshold=0.0, halt_on_drift=False)
        rec.add_probe(EnvironmentProbe("count", lambda: 99))
        s = _state(count=5)
        result = rec.reconcile(s)
        assert not result.is_consistent

    def test_reconcile_probe_error_is_mismatch(self):
        rec = EnvironmentReconciler(halt_on_drift=False)
        rec.add_probe(EnvironmentProbe("x", lambda: 1 / 0))
        s = _state(x=5)
        result = rec.reconcile(s)
        assert len(result.mismatches) == 1

    def test_reconcile_records_unprobed_vars(self):
        rec = EnvironmentReconciler(halt_on_drift=False)
        s = _state(count=5, extra=10)
        result = rec.reconcile(s)
        assert "extra" in result.unprobed or "count" in result.unprobed

    def test_reconciliation_history_is_copy(self):
        rec = EnvironmentReconciler(halt_on_drift=False)
        s = _state()
        rec.reconcile(s)
        hist = rec.reconciliation_history
        hist.clear()
        assert len(rec.reconciliation_history) == 1


# ─── MultiDimensionalAttestor ─────────────────────────────────────────────────

class TestMultiDimensionalAttestor:
    def test_all_checks_pass(self):
        a = MultiDimensionalAttestor("multi", threshold=0.7)
        a.add_check(QualityDimension.EXISTENCE, 1.0, lambda: (1.0, "exists"))
        a.add_check(QualityDimension.CORRECTNESS, 1.0, lambda: (1.0, "correct"))
        att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.VERIFIED

    def test_one_zero_score_fails(self):
        a = MultiDimensionalAttestor("multi", threshold=0.5)
        a.add_check(QualityDimension.EXISTENCE, 1.0, lambda: (1.0, "exists"))
        a.add_check(QualityDimension.CORRECTNESS, 1.0, lambda: (0.0, "wrong"))
        att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.FAILED

    def test_overall_below_threshold_fails(self):
        a = MultiDimensionalAttestor("multi", threshold=0.9)
        a.add_check(QualityDimension.QUALITY, 1.0, lambda: (0.5, "partial"))
        att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.FAILED

    def test_check_fn_exception_scores_zero(self):
        def boom():
            raise RuntimeError("kaboom")
        a = MultiDimensionalAttestor("multi", threshold=0.5)
        a.add_check(QualityDimension.SAFETY, 1.0, boom)
        att = a.verify(_state(), "goal")
        # Exception is caught; score becomes 0 → overall fails threshold.
        assert att.result == AttestationResult.FAILED
        assert "safety=0.00" in att.evidence

    def test_no_checks_returns_failed(self):
        a = MultiDimensionalAttestor("multi", threshold=0.5)
        att = a.verify(_state(), "goal")
        assert att.result == AttestationResult.FAILED

    def test_name_property(self):
        a = MultiDimensionalAttestor("myname", threshold=0.7)
        assert a.name == "myname"

    def test_evidence_contains_scores(self):
        a = MultiDimensionalAttestor("multi", threshold=0.5)
        a.add_check(QualityDimension.EXISTENCE, 1.0, lambda: (0.8, "ok"))
        att = a.verify(_state(), "goal")
        assert "existence" in att.evidence


# ─── DependencyDiscovery ─────────────────────────────────────────────────────

class TestDependencyDiscovery:
    def _disc(self, min_obs: int = 5) -> DependencyDiscovery:
        return DependencyDiscovery(significance_threshold=0.05, min_observations=min_obs)

    def test_observe_records_patterns(self):
        d = self._disc(min_obs=1)
        d.observe("b", succeeded=False,
                   completed_actions=set(),
                   all_known_actions={"a", "b"})
        assert ("b", "a") in d._patterns

    def test_discover_skips_low_observations(self):
        d = self._disc(min_obs=10)
        for _ in range(3):
            d.observe("b", succeeded=False,
                       completed_actions=set(),
                       all_known_actions={"a", "b"})
        result = d.discover()
        assert "b" not in result

    def test_discover_finds_strong_dependency(self):
        d = self._disc(min_obs=5)
        # Simulate: b always fails when a is NOT done, succeeds when a IS done.
        for _ in range(15):
            d.observe("b", succeeded=True,
                       completed_actions={"a"},
                       all_known_actions={"a", "b"})
        for _ in range(15):
            d.observe("b", succeeded=False,
                       completed_actions=set(),
                       all_known_actions={"a", "b"})
        result = d.discover()
        # Should discover b depends on a.
        assert "b" in result

    def test_inject_into_graph(self):
        d = self._disc(min_obs=5)
        for _ in range(15):
            d.observe("b", succeeded=True,
                       completed_actions={"a"},
                       all_known_actions={"a", "b"})
        for _ in range(15):
            d.observe("b", succeeded=False,
                       completed_actions=set(),
                       all_known_actions={"a", "b"})
        graph = CausalGraph()
        graph.add_action("a", [])
        graph.add_action("b", [])
        injected = d.inject_into_graph(graph)
        assert len(injected) > 0

    def test_inject_skips_existing_deps(self):
        d = self._disc(min_obs=5)
        for _ in range(15):
            d.observe("b", succeeded=True,
                       completed_actions={"a"},
                       all_known_actions={"a", "b"})
        for _ in range(15):
            d.observe("b", succeeded=False,
                       completed_actions=set(),
                       all_known_actions={"a", "b"})
        graph = CausalGraph()
        graph.add_action("a", [])
        graph.add_action("b", [("a", "already exists")])
        injected = d.inject_into_graph(graph)
        # Dep already exists, should not be injected again.
        assert len(injected) == 0

    def test_chi2p_zero_n(self):
        p = FailurePattern("b", "a")
        val = DependencyDiscovery._chi2p(p)
        assert val == pytest.approx(1.0)

    def test_chi2p_zero_row_or_col(self):
        p = FailurePattern("b", "a")
        p.successes_without_missing = 5
        p.successes_with_missing = 0
        p.failures_without_missing = 0
        p.failures_with_missing = 0
        p.observations = 5
        val = DependencyDiscovery._chi2p(p)
        assert val == pytest.approx(1.0)

    def test_chi2p_strong_association(self):
        p = FailurePattern("b", "a")
        p.successes_without_missing = 20
        p.successes_with_missing = 1
        p.failures_without_missing = 1
        p.failures_with_missing = 20
        p.observations = 42
        val = DependencyDiscovery._chi2p(p)
        assert val < 0.01

    def test_summary_no_discoveries(self):
        d = self._disc()
        s = d.summary()
        assert "none" in s

    def test_summary_with_discoveries(self):
        d = self._disc(min_obs=5)
        for _ in range(15):
            d.observe("b", succeeded=True,
                       completed_actions={"a"},
                       all_known_actions={"a", "b"})
        for _ in range(15):
            d.observe("b", succeeded=False,
                       completed_actions=set(),
                       all_known_actions={"a", "b"})
        d.discover()
        s = d.summary()
        assert "Discovered" in s


# ─── ResourceTracker ─────────────────────────────────────────────────────────

class TestResourceTracker:
    def test_register_and_get(self):
        t = ResourceTracker()
        r = ResourceDescriptor(kind="file", identifier="f1")
        t.register(r)
        got = t.get("f1")
        assert got is not None
        assert got.kind == "file"

    def test_get_missing_returns_none(self):
        t = ResourceTracker()
        assert t.get("nonexistent") is None

    def test_valid_transition(self):
        t = ResourceTracker()
        r = ResourceDescriptor(kind="file", identifier="f1",
                                state=ResourceState.ABSENT)
        t.register(r)
        ok, msg = t.transition("f1", ResourceState.CREATING)
        assert ok is True
        assert "ABSENT" in msg.upper() or "absent" in msg.lower()

    def test_invalid_transition(self):
        t = ResourceTracker()
        r = ResourceDescriptor(kind="file", identifier="f1",
                                state=ResourceState.ABSENT)
        t.register(r)
        ok, msg = t.transition("f1", ResourceState.READY)  # ABSENT→READY invalid
        assert ok is False
        assert "Invalid" in msg

    def test_unregistered_transition(self):
        t = ResourceTracker()
        ok, msg = t.transition("nope", ResourceState.READY)
        assert ok is False
        assert "not registered" in msg

    def test_all_ready_true(self):
        t = ResourceTracker()
        t.register(ResourceDescriptor("f", "f1", state=ResourceState.READY))
        t.register(ResourceDescriptor("f", "f2", state=ResourceState.READY))
        assert t.all_ready() is True

    def test_all_ready_false(self):
        t = ResourceTracker()
        t.register(ResourceDescriptor("f", "f1", state=ResourceState.READY))
        t.register(ResourceDescriptor("f", "f2", state=ResourceState.CREATING))
        assert t.all_ready() is False

    def test_all_ready_with_ids(self):
        t = ResourceTracker()
        t.register(ResourceDescriptor("f", "f1", state=ResourceState.READY))
        t.register(ResourceDescriptor("f", "f2", state=ResourceState.CREATING))
        assert t.all_ready(ids=["f1"]) is True
        assert t.all_ready(ids=["f2"]) is False

    def test_summary_returns_string(self):
        t = ResourceTracker()
        t.register(ResourceDescriptor("file", "f1", state=ResourceState.READY))
        s = t.summary()
        assert "f1" in s
        assert "ready" in s

    def test_resource_descriptor_to_dict(self):
        r = ResourceDescriptor(
            kind="file",
            identifier="f1",
            state=ResourceState.READY,
            permissions=frozenset({Permission.READ}),
        )
        d = r.to_dict()
        assert d["kind"] == "file"
        assert d["state"] == "ready"
        assert "read" in d["permissions"]

    def test_transition_sets_created_at_on_creating(self):
        t = ResourceTracker()
        r = ResourceDescriptor("f", "f1", state=ResourceState.ABSENT)
        t.register(r)
        t.transition("f1", ResourceState.CREATING)
        updated = t.get("f1")
        assert updated.state == ResourceState.CREATING
        assert updated.created_at > 0.0


# ─── HARDENING_CLAIMS registry ───────────────────────────────────────────────

class TestHardeningClaims:
    def test_claims_exist(self):
        assert len(HARDENING_CLAIMS) == 7

    def test_claim_names(self):
        names = {c.name for c in HARDENING_CLAIMS}
        assert "H1" in names
        assert "H7" in names
