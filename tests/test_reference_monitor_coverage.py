"""
Tests for clampai.reference_monitor coverage:
  DataLabel algebra, ControlBarrierFunction, CaptureBasin,
  ContractSpecification, OperadicComposition, SafeHoverState,
  ReferenceMonitor registration.
"""
from __future__ import annotations

import pytest

from clampai import (
    ActionSpec,
    CaptureBasin,
    ContractSpecification,
    ControlBarrierFunction,
    DataLabel,
    Effect,
    OperadicComposition,
    QPProjector,
    ReferenceMonitor,
    SafeHoverState,
    SecurityLevel,
    State,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _label(level: SecurityLevel, tags=None) -> DataLabel:
    return DataLabel(level, tags)


def _pub() -> DataLabel:
    return _label(SecurityLevel.PUBLIC)


def _int() -> DataLabel:
    return _label(SecurityLevel.INTERNAL)


def _pii() -> DataLabel:
    return _label(SecurityLevel.PII)


def _sec() -> DataLabel:
    return _label(SecurityLevel.SECRET)


def _state(**kw) -> State:
    return State(dict(kw))


def _noop_action(action_id: str = "noop") -> ActionSpec:
    return ActionSpec(
        id=action_id,
        name="No-op",
        description="Does nothing",
        effects=(),
        cost=0.001,
    )


def _set_action(key: str, value, action_id: str = "set_x") -> ActionSpec:
    return ActionSpec(
        id=action_id,
        name=f"Set {key}",
        description=f"Sets {key}",
        effects=(Effect(key, "set", value),),
        cost=0.001,
    )


# ── DataLabel algebra ──────────────────────────────────────────────────────────


class TestDataLabelLattice:
    def test_public_le_internal(self):
        assert _pub() <= _int()

    def test_internal_le_pii(self):
        assert _int() <= _pii()

    def test_pii_le_secret(self):
        assert _pii() <= _sec()

    def test_public_le_secret(self):
        assert _pub() <= _sec()

    def test_secret_not_le_public(self):
        assert not (_sec() <= _pub())

    def test_internal_not_le_public(self):
        assert not (_int() <= _pub())

    def test_ge_is_inverse_of_le(self):
        assert _sec() >= _pub()
        assert not (_pub() >= _sec())

    def test_equal_labels(self):
        assert _label(SecurityLevel.PII) == _label(SecurityLevel.PII)

    def test_different_levels_not_equal(self):
        assert _pub() != _sec()

    def test_hash_same_for_equal_labels(self):
        a = _label(SecurityLevel.PII, {"tag1"})
        b = _label(SecurityLevel.PII, {"tag1"})
        assert hash(a) == hash(b)

    def test_hash_usable_in_set(self):
        labels = {_pub(), _int(), _pii(), _sec()}
        assert len(labels) == 4

    def test_hash_usable_as_dict_key(self):
        d = {_pub(): "low", _sec(): "high"}
        assert d[_pub()] == "low"


class TestDataLabelJoin:
    def test_join_two_takes_max_level(self):
        joined = DataLabel.join(_pub(), _sec())
        assert joined.level == SecurityLevel.SECRET

    def test_join_three_labels(self):
        joined = DataLabel.join(_pub(), _int(), _pii())
        assert joined.level == SecurityLevel.PII

    def test_join_merges_tags(self):
        a = _label(SecurityLevel.PUBLIC, {"a"})
        b = _label(SecurityLevel.SECRET, {"b"})
        joined = DataLabel.join(a, b)
        assert "a" in joined.tags and "b" in joined.tags

    def test_join_single_label_preserved(self):
        joined = DataLabel.join(_pii())
        assert joined.level == SecurityLevel.PII


# ── SafeHoverState ─────────────────────────────────────────────────────────────


class TestSafeHoverState:
    def test_to_action_id(self):
        hover = SafeHoverState()
        assert hover.to_action().id == "SAFE_HOVER"

    def test_to_action_cost_zero(self):
        hover = SafeHoverState()
        assert hover.to_action().cost == 0.0

    def test_to_action_effects_empty(self):
        hover = SafeHoverState()
        assert hover.to_action().effects == ()

    def test_to_action_description_propagated(self):
        desc = "Waiting for clearance"
        hover = SafeHoverState(desc)
        assert hover.to_action().description == desc

    def test_to_action_returns_action_spec(self):
        assert isinstance(SafeHoverState().to_action(), ActionSpec)


# ── ControlBarrierFunction ─────────────────────────────────────────────────────


class TestControlBarrierFunction:
    def test_passes_when_delta_equals_threshold(self):
        # h(s) = 1.0, alpha=0.1 → threshold = -0.1
        # h(next) = 0.9 → delta = -0.1 == threshold → passes
        cbf = ControlBarrierFunction(h=lambda s: s.get("h", 1.0), alpha=0.1)
        ok, _ = cbf.evaluate(_state(h=1.0), _state(h=0.9))
        assert ok

    def test_passes_when_delta_above_threshold(self):
        cbf = ControlBarrierFunction(h=lambda s: s.get("h", 1.0), alpha=0.1)
        ok, _ = cbf.evaluate(_state(h=1.0), _state(h=1.0))  # delta=0 > -0.1
        assert ok

    def test_fails_when_barrier_violated(self):
        # h(s)=1.0, alpha=0.1, threshold=-0.1  — next h < 0.9 violates
        cbf = ControlBarrierFunction(h=lambda s: s.get("h", 1.0), alpha=0.1)
        ok, _ = cbf.evaluate(_state(h=1.0), _state(h=0.0))  # delta=-1.0 < -0.1
        assert not ok

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            ControlBarrierFunction(h=lambda s: 1.0, alpha=0.0)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError):
            ControlBarrierFunction(h=lambda s: 1.0, alpha=1.0)

    def test_diagnostic_contains_cbf(self):
        cbf = ControlBarrierFunction(h=lambda s: 1.0, alpha=0.1)
        _, msg = cbf.evaluate(_state(), _state())
        assert "CBF" in msg


# ── CaptureBasin ───────────────────────────────────────────────────────────────


class TestCaptureBasin:
    def _bad_basin(self) -> CaptureBasin:
        return CaptureBasin(
            name="overflow",
            is_bad=lambda s: s.get("x", 0) > 10,
            max_steps=3,
        )

    def test_safe_when_no_bad_states_reachable(self):
        basin = self._bad_basin()
        ok, _ = basin.evaluate_reachability(
            _state(x=0),
            [_noop_action()],  # no-op: x stays 0, not > 10
        )
        assert ok

    def test_unsafe_when_already_in_bad_region(self):
        basin = self._bad_basin()
        ok, _ = basin.evaluate_reachability(_state(x=11), [_noop_action()])
        assert not ok

    def test_unsafe_when_action_leads_to_bad(self):
        basin = self._bad_basin()
        big_set = _set_action("x", 20)
        ok, _ = basin.evaluate_reachability(_state(x=0), [big_set])
        assert not ok

    def test_safe_diagnostic_contains_name(self):
        basin = self._bad_basin()
        _, msg = basin.evaluate_reachability(_state(x=0), [_noop_action()])
        assert "overflow" in msg

    def test_unsafe_diagnostic_contains_action_name(self):
        basin = self._bad_basin()
        action = _set_action("x", 20)
        _, msg = basin.evaluate_reachability(_state(x=0), [action])
        assert "Set x" in msg


# ── ContractSpecification ──────────────────────────────────────────────────────


class TestContractSpecification:
    def _always_true(self, s: State) -> bool:
        return True

    def _x_positive(self, s: State) -> bool:
        return s.get("x", 0) > 0

    def test_satisfied_when_both_hold(self):
        spec = ContractSpecification(
            name="pos_contract",
            assume=self._always_true,
            guarantee=self._x_positive,
        )
        ok, msg = spec.is_satisfied_by(_state(x=1), _state(x=5))
        assert ok
        assert "satisfied" in msg

    def test_assume_violated_returns_false(self):
        spec = ContractSpecification(
            name="strict",
            assume=lambda s: s.get("ready", False),
            guarantee=self._always_true,
        )
        ok, msg = spec.is_satisfied_by(_state(ready=False), _state())
        assert not ok
        assert "Assume violated" in msg

    def test_guarantee_violated_returns_false(self):
        spec = ContractSpecification(
            name="g_fail",
            assume=self._always_true,
            guarantee=lambda s: s.get("done", False),
        )
        ok, msg = spec.is_satisfied_by(_state(), _state(done=False))
        assert not ok
        assert "Guarantee violated" in msg

    def test_diagnostic_contains_contract_name(self):
        spec = ContractSpecification(
            name="my_contract",
            assume=self._always_true,
            guarantee=self._always_true,
        )
        _, msg = spec.is_satisfied_by(_state(), _state())
        assert "my_contract" in msg

    def test_contract_is_frozen(self):
        spec = ContractSpecification(
            name="frozen",
            assume=self._always_true,
            guarantee=self._always_true,
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "changed"  # type: ignore[misc]


# ── OperadicComposition ────────────────────────────────────────────────────────


class TestOperadicComposition:
    def _spec(self, name: str, assume_val=True, guarantee_val=True,
               side_effects=()) -> ContractSpecification:
        return ContractSpecification(
            name=name,
            assume=lambda s: assume_val,
            guarantee=lambda s: guarantee_val,
            side_effects=tuple(side_effects),
        )

    def test_compose_returns_contract(self):
        composed = OperadicComposition.compose(self._spec("A"), self._spec("B"))
        assert isinstance(composed, ContractSpecification)

    def test_composed_name_format(self):
        composed = OperadicComposition.compose(self._spec("A"), self._spec("B"))
        assert "(A;B)" == composed.name

    def test_composed_assume_from_first(self):
        spec_a = ContractSpecification(
            name="A",
            assume=lambda s: s.get("ready", False),
            guarantee=lambda s: True,
        )
        spec_b = self._spec("B")
        composed = OperadicComposition.compose(spec_a, spec_b)
        # If A's assume fires, composed assume also fires
        assert not composed.assume(_state(ready=False))
        assert composed.assume(_state(ready=True))

    def test_composed_guarantee_from_second(self):
        spec_a = self._spec("A")
        spec_b = ContractSpecification(
            name="B",
            assume=lambda s: True,
            guarantee=lambda s: s.get("done", False),
        )
        composed = OperadicComposition.compose(spec_a, spec_b)
        assert not composed.guarantee(_state(done=False))
        assert composed.guarantee(_state(done=True))

    def test_composed_side_effects_union(self):
        spec_a = self._spec("A", side_effects=("x",))
        spec_b = self._spec("B", side_effects=("y",))
        composed = OperadicComposition.compose(spec_a, spec_b)
        assert set(composed.side_effects) == {"x", "y"}


# ── ReferenceMonitor registration ─────────────────────────────────────────────


class TestReferenceMonitor:
    def test_set_ifc_label_stored(self):
        rm = ReferenceMonitor()
        rm.set_ifc_label("email", _sec())
        assert rm.ifc_labels["email"].level == SecurityLevel.SECRET

    def test_add_cbf_sets_cbf_budget(self):
        rm = ReferenceMonitor()
        rm.add_cbf(h=lambda s: 1.0, alpha=0.1)
        assert rm.cbf_budget is not None

    def test_add_capture_basin_appended(self):
        rm = ReferenceMonitor()
        basin = CaptureBasin("test_basin", lambda s: False)
        rm.add_capture_basin(basin)
        assert len(rm.capture_basins) == 1
        assert rm.capture_basins[0].name == "test_basin"

    def test_register_contract_stored(self):
        rm = ReferenceMonitor()
        spec = ContractSpecification(
            name="test_spec",
            assume=lambda s: True,
            guarantee=lambda s: True,
        )
        rm.register_contract(spec)
        assert "test_spec" in rm.contracts

    def test_init_validates_cbf_binary_search_tol(self):
        with pytest.raises(ValueError):
            ReferenceMonitor(cbf_binary_search_tol=0.0)

    def test_init_validates_cbf_max_iterations(self):
        with pytest.raises(ValueError):
            ReferenceMonitor(cbf_max_iterations=0)
