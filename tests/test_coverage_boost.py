"""Coverage boost tests targeting specific uncovered lines.

Modules covered:
  - clampai/saliency.py         (69% → 100%)
  - clampai/verification_log.py (79% → 100%)
  - clampai/safe_hover.py       (91% → ~100%)
  - clampai/inverse_algebra.py  (81% → ~90%)

Each test class is focused on one module.
"""

from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import MagicMock

import pytest

from clampai.formal import _SENTINEL_DELETE, ActionSpec, Effect, State
from clampai.inverse_algebra import (
    InverseAlgebra,
    InverseEffect,
    RollbackRecord,
    action_with_inverse_guarantee,
)
from clampai.reasoning import ActionValue
from clampai.reference_monitor import CaptureBasin
from clampai.safe_hover import AuthoritativeHJBBarrier, SafeHoverSignal
from clampai.saliency import SaliencyEngine, SaliencyResult
from clampai.verification_log import ProofRecord, ProofStep, _sha256, write_proof

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(action_id: str, var: str, value: object = 99, cost: float = 1.0) -> ActionSpec:
    return ActionSpec(
        id=action_id,
        name=action_id.capitalize(),
        description="test action",
        effects=(Effect(var, "set", value),),
        cost=cost,
    )


def _av(action_id: str, score: float = 1.0) -> ActionValue:
    return ActionValue(
        action_id=action_id,
        expected_progress=0.5,
        information_gain=0.0,
        cost=1.0,
        risk=0.0,
        opportunity_cost=0.0,
        value_score=score,
        reasoning_hint="test",
    )


# ---------------------------------------------------------------------------
# SaliencyEngine (saliency.py lines 87-88, 97-102, 112-113, 118, 123, 130-133)
# ---------------------------------------------------------------------------

class TestSaliencyEngineDisabled:
    """Lines 87-88: early-return when threshold <= 0 or max_keys <= 0."""

    def test_disabled_by_zero_threshold(self):
        engine = SaliencyEngine(threshold=0.0)
        state = State({"a": 1, "b": 2})
        result = engine.analyze(state=state, available_actions=[], action_values=[])
        assert set(result.kept_keys) == {"a", "b"}
        assert result.dropped_keys == ()
        assert all(v == float("inf") for v in result.scores.values())

    def test_disabled_by_negative_threshold(self):
        engine = SaliencyEngine(threshold=-1.0)
        state = State({"x": 10})
        result = engine.analyze(state=state, available_actions=[], action_values=[])
        assert "x" in result.kept_keys
        assert result.scores["x"] == float("inf")

    def test_disabled_by_zero_max_keys(self):
        engine = SaliencyEngine(max_keys=0)
        state = State({"p": True, "q": False})
        result = engine.analyze(state=state, available_actions=[], action_values=[])
        assert set(result.kept_keys) == {"p", "q"}
        assert result.dropped_keys == ()


class TestSaliencyEngineActionScoring:
    """Lines 97-102: action-based scoring loop."""

    def test_action_with_positive_value_boosts_variable_score(self):
        engine = SaliencyEngine(threshold=0.5)
        state = State({"x": 10, "y": 20, "z": 30})
        action = _action("a1", "x")
        av = _av("a1", score=5.0)
        result = engine.analyze(state=state, available_actions=[action], action_values=[av])
        # x is boosted; y and z have score 0 < threshold → dropped
        assert "x" in result.kept_keys
        assert "y" not in result.kept_keys
        assert "z" not in result.kept_keys

    def test_action_with_zero_value_skipped(self):
        engine = SaliencyEngine(threshold=0.5)
        state = State({"x": 1})
        action = _action("a1", "x")
        av = _av("a1", score=0.0)   # zero value → skipped by `if w <= 0`
        result = engine.analyze(state=state, available_actions=[action], action_values=[av])
        # x score stays 0, below threshold → dropped
        assert "x" not in result.kept_keys

    def test_action_affecting_unknown_variable_is_harmless(self):
        engine = SaliencyEngine(threshold=0.5)
        state = State({"x": 1})
        # action affects "nonexistent" which is not in state
        action = ActionSpec(
            id="a1", name="A1", description="d",
            effects=(Effect("nonexistent", "set", 99),),
            cost=0.0,
        )
        av = _av("a1", score=3.0)
        result = engine.analyze(state=state, available_actions=[action], action_values=[av])
        # "nonexistent" not in scores dict, so `if k in scores` guards it
        assert "x" not in result.kept_keys


class TestSaliencyEngineJacobianOverride:
    """Lines 112-113: jacobian fusion forces critical variables to inf."""

    def test_jacobian_critical_var_kept_despite_low_action_score(self):
        mock_jf = MagicMock()
        mock_report = MagicMock()
        mock_report.critical_variables = ["budget_remaining"]
        mock_jf.compute_jacobian.return_value = mock_report

        engine = SaliencyEngine(threshold=10.0, jacobian_fusion=mock_jf)
        state = State({"budget_remaining": 50.0, "unrelated": 1})
        result = engine.analyze(state=state, available_actions=[], action_values=[])

        # budget_remaining should be forced to inf → kept
        assert "budget_remaining" in result.kept_keys
        assert result.scores["budget_remaining"] == float("inf")

    def test_jacobian_unknown_critical_var_does_not_crash(self):
        mock_jf = MagicMock()
        mock_report = MagicMock()
        mock_report.critical_variables = ["var_not_in_state"]
        mock_jf.compute_jacobian.return_value = mock_report

        engine = SaliencyEngine(threshold=10.0, jacobian_fusion=mock_jf)
        state = State({"x": 1})
        # Should not raise; "var_not_in_state" not in scores → guarded by `if var in scores`
        result = engine.analyze(state=state, available_actions=[], action_values=[])
        assert isinstance(result, SaliencyResult)


class TestSaliencyEngineMetaKeys:
    """Line 118: keys starting with '_' are always kept (score set to inf)."""

    def test_underscore_key_always_kept(self):
        engine = SaliencyEngine(threshold=100.0)   # very high threshold
        state = State({"_internal": "metadata", "regular": 0})
        result = engine.analyze(state=state, available_actions=[], action_values=[])
        # "_internal" starts with '_' → score = inf → kept
        assert "_internal" in result.kept_keys
        # "regular" has score 0 < threshold → dropped
        assert "regular" not in result.kept_keys

    def test_multiple_meta_keys(self):
        engine = SaliencyEngine(threshold=50.0)
        state = State({"_ts": 123, "_ver": 2, "data": "stuff"})
        result = engine.analyze(state=state, available_actions=[], action_values=[])
        assert "_ts" in result.kept_keys
        assert "_ver" in result.kept_keys
        assert "data" not in result.kept_keys


class TestSaliencyEngineMaxKeysTruncation:
    """Line 123: truncation when len(kept) > max_keys."""

    def test_max_keys_truncates_to_top_n(self):
        # Three meta keys all get inf score; max_keys=2 → only 2 kept
        engine = SaliencyEngine(threshold=0.01, max_keys=2)
        state = State({"_a": 1, "_b": 2, "_c": 3})
        result = engine.analyze(state=state, available_actions=[], action_values=[])
        assert len(result.kept_keys) == 2

    def test_max_keys_does_not_truncate_when_under_limit(self):
        engine = SaliencyEngine(threshold=0.01, max_keys=10)
        state = State({"_a": 1, "_b": 2})
        result = engine.analyze(state=state, available_actions=[], action_values=[])
        assert len(result.kept_keys) == 2


class TestSaliencyEnginePruneState:
    """Lines 130-133: prune_state() method."""

    def test_prune_state_empty_kept_keys_returns_original(self):
        engine = SaliencyEngine()
        state = State({"a": 1, "b": 2})
        saliency = SaliencyResult(
            kept_keys=(),
            scores={},
            dropped_keys=("a", "b"),
        )
        pruned = engine.prune_state(state=state, saliency=saliency)
        assert pruned is state  # original returned unchanged

    def test_prune_state_removes_dropped_keys(self):
        engine = SaliencyEngine()
        state = State({"a": 1, "b": 2, "c": 3})
        saliency = SaliencyResult(
            kept_keys=("a", "c"),
            scores={"a": 1.0, "c": 0.5},
            dropped_keys=("b",),
        )
        pruned = engine.prune_state(state=state, saliency=saliency)
        assert pruned.get("a") == 1
        assert pruned.get("c") == 3
        assert pruned.get("b") is None

    def test_prune_state_handles_key_not_in_state(self):
        engine = SaliencyEngine()
        state = State({"a": 1})
        # kept_keys includes "ghost" which is not in state; should not crash
        saliency = SaliencyResult(
            kept_keys=("a", "ghost"),
            scores={"a": 1.0, "ghost": 1.0},
            dropped_keys=(),
        )
        pruned = engine.prune_state(state=state, saliency=saliency)
        assert pruned.get("a") == 1
        assert pruned.get("ghost") is None


# ---------------------------------------------------------------------------
# VerificationLog (verification_log.py lines 22, 52-54, 58-61)
# ---------------------------------------------------------------------------

class TestVerificationLog:
    """Tests for _sha256(), ProofRecord.to_json(), and write_proof()."""

    def test_sha256_returns_hex_string(self):
        # Line 22: _sha256()
        result = _sha256("hello world")
        assert isinstance(result, str)
        assert len(result) == 64
        assert result == _sha256("hello world")  # deterministic

    def test_sha256_different_inputs_differ(self):
        assert _sha256("abc") != _sha256("xyz")

    def _make_record(self) -> ProofRecord:
        step = ProofStep(
            step_index=0,
            action_id="act_1",
            action_name="SendEmail",
            approved=True,
            reason="All invariants passed",
            kernel_reasons=["budget ok", "rate limit ok"],
            monitor_reason="",
            cbf_margin=0.15,
            ifc_ok=True,
            hjb_ok=True,
            prompt_tokens_est=100,
            output_tokens_est=50,
        )
        return ProofRecord(
            version="0.4.0",
            created_at=1700000000.0,
            goal="send_email",
            budget=50.0,
            trace_hash_head="abc123",
            steps=[step],
        )

    def test_proof_record_to_json_valid_json(self):
        # Lines 52-54
        import json
        record = self._make_record()
        text = record.to_json()
        parsed = json.loads(text)
        assert parsed["version"] == "0.4.0"
        assert parsed["goal"] == "send_email"
        assert "record_hash" in parsed
        assert isinstance(parsed["record_hash"], str)

    def test_proof_record_to_json_hash_is_deterministic(self):
        record = self._make_record()
        assert record.to_json() == record.to_json()

    def test_write_proof_creates_file(self):
        # Lines 58-61
        record = self._make_record()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "proof.json")
            write_proof(path, record)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            import json
            parsed = json.loads(content)
            assert parsed["goal"] == "send_email"

    def test_write_proof_creates_parent_directories(self):
        record = self._make_record()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deep", "nested", "dir", "log.json")
            write_proof(path, record)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Safe Hover (safe_hover.py lines 114, 128-129)
# ---------------------------------------------------------------------------

class TestSafeHover:
    """Tests for AuthoritativeHJBBarrier."""

    def test_check_action_leads_to_danger_safe_path(self):
        # Line 114: PROCEED return when no basin is violated
        barrier = AuthoritativeHJBBarrier(capture_basins=[])
        state = State({"x": 1})
        action = _action("move", "x", value=2)
        result = barrier.check_action_leads_to_danger(state, action, [])
        assert result.signal == SafeHoverSignal.PROCEED
        assert result.violated_basin is None
        assert not result.requires_immediate_rollback

    def test_check_action_leads_to_danger_with_basin_violation(self):
        bad_basin = CaptureBasin(
            name="danger_zone",
            is_bad=lambda s: (s.get("x") or 0) > 50,
        )
        barrier = AuthoritativeHJBBarrier(capture_basins=[bad_basin])
        state = State({"x": 1})
        # action sets x=100, which violates the basin
        action = _action("big_jump", "x", value=100)
        result = barrier.check_action_leads_to_danger(state, action, [])
        assert result.signal == SafeHoverSignal.TERMINATE_AND_ROLLBACK

    def test_enforce_safe_hover_safe_state(self):
        # Lines 128-129: enforce_safe_hover delegates to check_state_safety
        barrier = AuthoritativeHJBBarrier(capture_basins=[])
        state = State({"mode": "normal"})
        signal = barrier.enforce_safe_hover(state)
        assert signal == SafeHoverSignal.PROCEED

    def test_enforce_safe_hover_dangerous_state(self):
        bad_basin = CaptureBasin(
            name="unsafe",
            is_bad=lambda s: s.get("alarm") is True,
        )
        barrier = AuthoritativeHJBBarrier(capture_basins=[bad_basin])
        state = State({"alarm": True})
        signal = barrier.enforce_safe_hover(state)
        assert signal == SafeHoverSignal.TERMINATE_AND_ROLLBACK


# ---------------------------------------------------------------------------
# InverseAlgebra (inverse_algebra.py lines 34-37, 200-212)
# ---------------------------------------------------------------------------

class TestInverseEffect:
    """Lines 34-37: InverseEffect.to_effect()."""

    def test_to_effect_with_sentinel_returns_delete(self):
        # Line 34-35: prior_value is _SENTINEL_DELETE → delete effect
        ie = InverseEffect(variable="x", mode="delete", prior_value=_SENTINEL_DELETE)
        eff = ie.to_effect()
        assert eff.variable == "x"
        assert eff.mode == "delete"
        assert eff.value is None

    def test_to_effect_with_value_returns_set(self):
        # Lines 36-37: prior_value is a real value → set effect
        ie = InverseEffect(variable="counter", mode="set", prior_value=42)
        eff = ie.to_effect()
        assert eff.variable == "counter"
        assert eff.mode == "set"
        assert eff.value == 42

    def test_to_effect_with_string_value(self):
        ie = InverseEffect(variable="name", mode="set", prior_value="alice")
        eff = ie.to_effect()
        assert eff.value == "alice"

    def test_to_effect_with_none_value_returns_set_none(self):
        # None is falsy but NOT _SENTINEL_DELETE → takes the else branch
        ie = InverseEffect(variable="opt", mode="set", prior_value=None)
        eff = ie.to_effect()
        assert eff.mode == "set"
        assert eff.value is None


class TestActionWithInverseGuarantee:
    """Lines 200-212: action_with_inverse_guarantee()."""

    def test_returns_original_spec_and_inverse_effects(self):
        state = State({"x": 10, "y": 5})
        action = ActionSpec(
            id="set_x",
            name="SetX",
            description="Sets x to 20",
            effects=(Effect("x", "set", 20),),
            cost=1.0,
        )
        returned_spec, inv_effects = action_with_inverse_guarantee(action, state)
        assert returned_spec is action
        assert len(inv_effects) == 1
        # The inverse should restore x to 10
        assert inv_effects[0].variable == "x"
        assert inv_effects[0].mode == "set"
        assert inv_effects[0].value == 10

    def test_inverse_guarantee_multi_effect_action(self):
        state = State({"a": 1, "b": 2})
        action = ActionSpec(
            id="multi",
            name="Multi",
            description="Changes a and b",
            effects=(
                Effect("a", "set", 100),
                Effect("b", "set", 200),
            ),
            cost=0.0,
        )
        spec, inv_effects = action_with_inverse_guarantee(action, state)
        assert spec is action
        assert len(inv_effects) == 2
        restored_vars = {e.variable for e in inv_effects}
        assert restored_vars == {"a", "b"}

    def test_inverse_guarantee_increment_action(self):
        state = State({"count": 5})
        action = ActionSpec(
            id="inc",
            name="Inc",
            description="Increments count",
            effects=(Effect("count", "increment", 1),),
            cost=0.0,
        )
        spec, inv_effects = action_with_inverse_guarantee(action, state)
        assert spec is action
        # inverse of increment by 1 on count=5 → set count back to 5
        assert len(inv_effects) == 1

    def test_inverse_guarantee_add_new_key_action(self):
        state = State({"existing": True})
        action = ActionSpec(
            id="add",
            name="Add",
            description="Adds new_key",
            effects=(Effect("new_key", "set", "hello"),),
            cost=0.0,
        )
        spec, inv_effects = action_with_inverse_guarantee(action, state)
        assert spec is action
        # new_key was added → inverse is delete
        assert len(inv_effects) == 1
        assert inv_effects[0].variable == "new_key"
        assert inv_effects[0].mode == "delete"


# ---------------------------------------------------------------------------
# InverseAlgebra — additional failure paths
# (inverse_algebra.py lines 63, 185-186)
# ---------------------------------------------------------------------------

class TestInverseAlgebraFailurePaths:
    """Cover error branches not reached by the happy-path tests above."""

    def test_verify_inverse_correctness_returns_false_for_wrong_inverse(self):
        # Lines 185-186: the else branch when restored != state_before
        state = State({"x": 10})
        action = ActionSpec(
            id="setx",
            name="SetX",
            description="sets x",
            effects=(Effect("x", "set", 20),),
            cost=0.0,
        )
        # Deliberately broken inverse: sets x to 999 instead of restoring 10
        wrong_inverse = (Effect("x", "set", 999),)
        ok, msg = InverseAlgebra.verify_inverse_correctness(state, action, wrong_inverse)
        assert ok is False
        assert "Inverse failed" in msg

    def test_rollback_record_raises_on_fingerprint_mismatch(self):
        # Line 63: ValueError when current state fingerprint ≠ recorded after-state
        state_before = State({"x": 1})
        state_after = State({"x": 2})
        record = RollbackRecord(
            action_id="a1",
            action_name="A1",
            state_before_fingerprint=state_before.fingerprint,
            state_before_snapshot=state_before,
            state_after_fingerprint=state_after.fingerprint,
            inverse_effects=(Effect("x", "set", 1),),
            timestamp=0.0,
        )
        # Pass a state whose fingerprint does NOT match state_after
        wrong_state = State({"x": 999})
        with pytest.raises(ValueError, match="Cannot rollback"):
            record.apply_rollback(wrong_state)
