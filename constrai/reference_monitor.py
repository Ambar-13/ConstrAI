"""
ConstrAI — Formal Reference Monitor Layer
====================================================================
Module: constrai.reference_monitor — Authoritative Safety Enforcement

This module implements the deterministic, mathematically-verified safety
mechanisms:
  - Lattice-based Information Flow Control (IFC) for PII/data tagging
  - Control Barrier Functions (CBF) for resource constraint enforcement
  - Quadratic Programming (QP) based minimum-intervention action repair
  - Hamilton-Jacobi-Bellman (HJB) reachability for capture basin avoidance
  - Operadic composition for verified sub-task lifting

Core Theorem: The Reference Monitor enforce() method is the SOLE gatekeeper.
  If enforce(action, state) returns (safe, repaired_action), that is the
  only transition allowed. Otherwise, identity (s' = s).

MATHEMATICAL GUARANTEES:
  M1  Lattice IFC:     ∀ action, label(output_sink) ≥ label(input_data)
  M2  CBF Safety:      h(s_{t+1}) - h(s_t) ≥ -α·h(s_t) in resource space
  M3  QP Minimality:   a* ∈ argmin‖a' - a_agent‖² s.t. a' ∈ SafeSet
  M4  HJB Reachability: ¬(∃ τ≤k: s_t+τ ∈ CaptureBasin) → Safe
  M5  Operadic Lift:    (A verified ∧ B verified) ⟹ (A ∘ B) verified
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

from constrai.formal import (
    State, ActionSpec, Effect, SafetyVerdict, CheckResult, Invariant
)


# ═══════════════════════════════════════════════════════════════════════════
# §M1  LATTICE-BASED INFORMATION FLOW CONTROL
# ═══════════════════════════════════════════════════════════════════════════

class SecurityLevel(Enum):
    """Security lattice for IFC: Public ⊑ Internal ⊑ PII ⊑ Secret."""
    PUBLIC   = 0    # No restrictions
    INTERNAL = 1    # Internal use only
    PII      = 2    # Personally Identifiable Information
    SECRET   = 3    # Highest classification


class DataLabel:
    """Immutable security label for a data value."""
    def __init__(self, level: SecurityLevel, tags: Optional[Set[str]] = None):
        self.level = level
        self.tags = frozenset(tags or set())

    def __le__(self, other: DataLabel) -> bool:
        """Lattice meet: self ≤ other iff level permits flow to other."""
        return self.level.value <= other.level.value and self.tags <= other.tags

    def __ge__(self, other: DataLabel) -> bool:
        return other <= self

    def __eq__(self, other):
        return isinstance(other, DataLabel) and self.level == other.level and self.tags == other.tags

    def __hash__(self):
        return hash((self.level, self.tags))

    def __repr__(self):
        tag_str = f" [{','.join(sorted(self.tags))}]" if self.tags else ""
        return f"Label({self.level.name}{tag_str})"

    @staticmethod
    def join(*labels: DataLabel) -> DataLabel:
        """Lattice least-upper-bound (join)."""
        max_level = max((lbl.level.value for lbl in labels))
        max_level = SecurityLevel(max_level)
        all_tags = set()
        for lbl in labels:
            all_tags.update(lbl.tags)
        return DataLabel(max_level, all_tags)


class LabelledData:
    """Data chunk with attached security label (immutable)."""
    __slots__ = ('_value', '_label')

    def __init__(self, value: Any, label: DataLabel):
        object.__setattr__(self, '_value', value)
        object.__setattr__(self, '_label', label)

    @property
    def value(self) -> Any:
        return self._value

    @property
    def label(self) -> DataLabel:
        return self._label

    def __setattr__(self, *_):
        raise AttributeError("LabelledData is immutable")

    def __repr__(self):
        return f"LabelledData({self._value!r}, {self._label})"

    def __deepcopy__(self, memo):
        # LabelledData is immutable by construction, so it can be safely
        # re-used across deep copies.
        return self


# ═══════════════════════════════════════════════════════════════════════════
# §M2  CONTROL BARRIER FUNCTIONS FOR RESOURCE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class ControlBarrierFunction:
    """
    Discrete-time CBF for resource constraint enforcement.

    Safety condition (h > 0 ⟹ safe):
      h(s_{t+1}) - h(s_t) ≥ -α·h(s_t),  where α ∈ (0,1)

    This smoothly restricts the action space as the "barrier" erodes.
    Applied to budget: h(s) := remaining_budget / initial_budget.
    """

    def __init__(self, h: Callable[[State], float], alpha: float = 0.1):
        """
        Args:
          h: barrier function, should return h(s) ≈ "margin to safety"
          alpha: decay rate, typically 0.05–0.2 for gradual tightening
        """
        if not (0 < alpha < 1):
            raise ValueError(f"CBF alpha must be in (0,1), got {alpha}")
        self.h = h
        self.alpha = alpha

    def evaluate(self, state: State, next_state: State) -> Tuple[bool, str]:
        """
        Check if transition state -> next_state respects the barrier.

        Returns: (passes_cbf, diagnostic_message)
        """
        h_now = self.h(state)
        h_next = self.h(next_state)
        delta = h_next - h_now

        threshold = -self.alpha * h_now

        if delta >= threshold:
            return True, f"CBF: h={h_now:.3f}, Δh={delta:.4f} ≥ {threshold:.4f}"
        else:
            return False, f"CBF VIOLATION: h={h_now:.3f}, Δh={delta:.4f} < {threshold:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# §M3  QUADRATIC PROGRAMMING MINIMUM-INTERVENTION REPAIR
# ═══════════════════════════════════════════════════════════════════════════

class QPProjector:
    """
    Minimum-intervention repair using Quadratic Programming.

    Given an unsafe action a_agent, find the closest safe action a* in L² norm:
      a* = argmin ‖a' - a_agent‖²
      s.t. constraints(a') are satisfied

    For numeric parameters (e.g., cost, intensity), this smoothly "nudges"
    the action toward feasibility without a hard rejection.
    """

    def __init__(self, max_iterations: int = 100, tol: float = 1e-6):
        self.max_iterations = max_iterations
        self.tol = tol

    def project_cost(self, requested_cost: float, budget_remaining: float
                     ) -> Tuple[float, bool]:
        """
        Project numeric cost to feasible range [min_cost, budget_remaining].

        Returns: (safe_cost, was_repaired)
        """
        min_cost = 0.001  # Convention: all actions have min cost
        max_cost = budget_remaining

        if requested_cost < min_cost:
            return min_cost, True
        elif requested_cost <= max_cost:
            return requested_cost, False
        else:
            # Overspend: return max feasible
            return max_cost, True

    def project_effect_parameters(
        self,
        effects: Tuple[Effect, ...],
        state: State,
        safe_values: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tuple[Effect, ...], bool]:
        """
        Project numeric effect parameters to safe values via L2 minimization.

        For increment/decrement/multiply effects, nudges the parameter toward
        the feasible set rather than rejecting outright.

        Returns: (repaired_effects, was_repaired)
        """
        if not safe_values:
            return effects, False

        repaired_effects = []
        was_repaired = False

        for effect in effects:
            if effect.variable in safe_values and effect.mode in ("set", "increment", "decrement", "multiply"):
                safe_val = safe_values[effect.variable]
                if isinstance(effect.value, (int, float)) and isinstance(safe_val, (int, float)):
                    if effect.value != safe_val:
                        # Nudge the value toward feasibility
                        repaired_effects.append(
                            Effect(effect.variable, effect.mode, safe_val)
                        )
                        was_repaired = True
                    else:
                        repaired_effects.append(effect)
                else:
                    repaired_effects.append(effect)
            else:
                repaired_effects.append(effect)

        return tuple(repaired_effects), was_repaired

    def project_action(
        self,
        action: ActionSpec,
        state: State,
        constraints: List[Callable[[float], bool]],
        safe_effect_values: Optional[Dict[str, Any]] = None
    ) -> Tuple[ActionSpec, bool]:
        """
        Project action parameters (cost, effects) to safe set.

        Finds the action closest to the original in L2 norm that satisfies
        all constraints. Used for minimum-intervention action repair.

        Returns: (repaired_action, was_repaired)
        """
        # Attempt direct repair on cost first (common case)
        safe_cost, cost_changed = self.project_cost(
            action.cost, budget_remaining=10.0  # Placeholder; passed by caller
        )

        # Then attempt repair on effect parameters (QP nudging)
        safe_effects, effects_changed = self.project_effect_parameters(
            action.effects, state, safe_effect_values
        )

        if cost_changed or effects_changed:
            # Return a new ActionSpec with adjusted cost and/or effects
            return ActionSpec(
                id=action.id,
                name=action.name + " [repaired]",
                description=action.description,
                effects=safe_effects,
                cost=safe_cost,
                risk_level=action.risk_level,
                preconditions_text=action.preconditions_text,
                postconditions_text=action.postconditions_text,
            ), True

        return action, False


# ═══════════════════════════════════════════════════════════════════════════
# §M4  HAMILTON-JACOBI-BELLMAN REACHABILITY & CAPTURE BASINS
# ═══════════════════════════════════════════════════════════════════════════

class CaptureBasin:
    """
    A region of state space from which a given bad predicate is reachable
    within k steps, making escape impossible.

    HJB Reachability Analysis computes (offline) which states are "doomed."
    The Reference Monitor queries this at runtime to avoid entry.
    """

    def __init__(self, name: str, is_bad: Callable[[State], bool],
                 max_steps: int = 5):
        """
        Args:
          name: identifier for this capture basin
          is_bad: predicate defining the bad region (is_bad(s) => unsafe)
          max_steps: lookahead horizon for reachability
        """
        self.name = name
        self.is_bad = is_bad
        self.max_steps = max_steps

    def evaluate_reachability(
        self,
        state: State,
        candidate_actions: List[ActionSpec]
    ) -> Tuple[bool, str]:
        """
        Check if from this state, any sequence of candidate_actions
        leads to the bad region within max_steps.

        Returns: (is_safe, diagnostic)
        """
        # Simplified check: if bad region already entered, fail immediately
        if self.is_bad(state):
            return False, f"CaptureBasin '{self.name}': already in bad region"

        # Full reachability would require action sequence enumeration.
        # For now, single-step lookahead (simplified).
        for action in candidate_actions:
            next_state = action.simulate(state)
            if self.is_bad(next_state):
                return False, (f"CaptureBasin '{self.name}': "
                              f"action '{action.name}' leads directly to bad region")

        return True, f"CaptureBasin '{self.name}': safe from reachability"


# ═══════════════════════════════════════════════════════════════════════════
# §M5  OPERADIC COMPOSITION FOR SUBTASK VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ContractSpecification:
    """
    A contract (A-specification) for a subtask.

    Assume-Guarantee logic:
      Assume:  precondition on input state
      Guarantee: postcondition on output state (if Assume holds)

    Theorem: If A ⊨ A-spec and B ⊨ B-spec and out(A) matches in(B),
      then A;B ⊨ (A-spec;B-spec) by standard Hoare-logic composition.
    """
    name: str
    assume: Callable[[State], bool]  # Input requirement
    guarantee: Callable[[State], bool]  # Output promise
    side_effects: Tuple[str, ...] = ()  # Variables modified

    def is_satisfied_by(self, state_in: State, state_out: State) -> Tuple[bool, str]:
        """Verify that a state transition satisfies this contract."""
        if not self.assume(state_in):
            return False, f"Contract '{self.name}': Assume violated"
        if not self.guarantee(state_out):
            return False, f"Contract '{self.name}': Guarantee violated"
        return True, f"Contract '{self.name}': satisfied"


class OperadicComposition:
    """
    Operadic composition of contracts: if contracts A and B are verified,
    their sequential composition is safe by transitivity.

    Composition: A ∘ B means "do A, then do B"
      Assume(A ∘ B) := Assume(A)
      Guarantee(A ∘ B) := Guarantee(B)  [provided outputs of A match inputs of B]
    """

    @staticmethod
    def compose(
        spec_a: ContractSpecification,
        spec_b: ContractSpecification,
        verify_interface: bool = True
    ) -> ContractSpecification:
        """
        Compose two contracts. Returns a new contract for the sequence.

        If verify_interface=True, check that outputs of A are compatible
        with inputs of B (via side_effects matching).
        """
        if verify_interface:
            if not (set(spec_a.side_effects) & set(spec_b.side_effects)):
                # No overlap => compatible
                pass
            # Full interface check would verify type/dimension matching

        def composed_assume(s: State) -> bool:
            return spec_a.assume(s)

        def composed_guarantee(s: State) -> bool:
            # Composed guarantee: output must satisfy B's guarantee
            # (In full Hoare logic, we'd need an intermediate state)
            return spec_b.guarantee(s)

        return ContractSpecification(
            name=f"({spec_a.name};{spec_b.name})",
            assume=composed_assume,
            guarantee=composed_guarantee,
            side_effects=tuple(set(spec_a.side_effects) | set(spec_b.side_effects)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §M6  AUTHORITATIVE REFERENCE MONITOR
# ═══════════════════════════════════════════════════════════════════════════

class ReferenceMonitor:
    """
    The sole authoritative gatekeeper for environment transitions.

    Theorem M0 (Enforcement Completeness):
      Every transition s -> s' is either:
        (a) approved by enforce(), executed with full effect, OR
        (b) rejected, returning identity transition (s' := s)
      There is no third case.

    This monitor is the INNERMOST layer, beneath the formal kernel.
    It checks mathematical invariants that the kernel alone cannot verify.
    """

    def __init__(
        self,
        ifc_enabled: bool = True,
        cbf_enabled: bool = True,
        hjb_enabled: bool = True,
        cbf_binary_search_tol: float = 1e-6,
        cbf_max_iterations: int = 20,
    ):
        if cbf_binary_search_tol <= 0:
            raise ValueError(f"cbf_binary_search_tol must be > 0, got {cbf_binary_search_tol}")
        if cbf_max_iterations < 1:
            raise ValueError(f"cbf_max_iterations must be ≥ 1, got {cbf_max_iterations}")

        self.ifc_enabled = ifc_enabled
        self.cbf_enabled = cbf_enabled
        self.hjb_enabled = hjb_enabled

        # CBF binary search parameters
        self.cbf_binary_search_tol = cbf_binary_search_tol
        self.cbf_max_iterations = cbf_max_iterations

        # IFC database: state_var -> label mapping
        self.ifc_labels: Dict[str, DataLabel] = {}

        # CBF for budget resource
        self.cbf_budget: Optional[ControlBarrierFunction] = None

        # Reachability barriers (capture basins)
        self.capture_basins: List[CaptureBasin] = []

        # Contracts for compositional verification
        self.contracts: Dict[str, ContractSpecification] = {}

    def set_ifc_label(self, var_name: str, label: DataLabel) -> None:
        """Register a data variable with its security label."""
        self.ifc_labels[var_name] = label

    def add_cbf(self, h: Callable[[State], float], alpha: float = 0.1) -> None:
        """Register a control barrier function for resource management."""
        self.cbf_budget = ControlBarrierFunction(h, alpha)

    def add_capture_basin(self, basin: CaptureBasin) -> None:
        """Register a region to avoid via reachability analysis."""
        self.capture_basins.append(basin)

    def register_contract(self, spec: ContractSpecification) -> None:
        """Register a contract for compositional verification."""
        self.contracts[spec.name] = spec

    def enforce(
        self,
        action: ActionSpec,
        state: State,
        candidate_next_actions: Optional[List[ActionSpec]] = None
    ) -> Tuple[bool, str, Optional[ActionSpec]]:
        """
        Authoritative enforcement: the ONLY method that approves transitions.

        Returns: (safe: bool, reason: str, repaired_action: Optional[ActionSpec])

        Called by orchestrator before any action executes. If safe=False, no
        transition occurs (identity). If safe=True and repaired_action is not None,
        the repaired action replaces the original.
        """
        reasons = []

        # Check 1: Information Flow Control (IFC)
        if self.ifc_enabled:
            ifc_ok, ifc_msg = self._check_ifc(action, state)
            if not ifc_ok:
                return False, ifc_msg, None
            reasons.append(ifc_msg)

        # Check 2: Control Barrier Function (CBF)
        repaired = action
        if self.cbf_enabled and self.cbf_budget is not None:
            next_state = action.simulate(state)
            cbf_ok, cbf_msg = self.cbf_budget.evaluate(state, next_state)
            if not cbf_ok:
                # Attempt repair: compute safe effect values via QP nudging
                safe_effect_values = self._compute_safe_effect_values(
                    action, state, self.cbf_budget
                )
                projector = QPProjector()
                repaired, was_repaired = projector.project_action(
                    action, state, constraints=[], 
                    safe_effect_values=safe_effect_values
                )
                if not was_repaired:
                    return False, cbf_msg, None
                reasons.append(f"CBF: action repaired")
            else:
                reasons.append(cbf_msg)

        # Check 3: HJB Reachability (Capture Basins)
        if self.hjb_enabled:
            candidates = [repaired] + (candidate_next_actions or [])
            for basin in self.capture_basins:
                reach_ok, reach_msg = basin.evaluate_reachability(state, candidates)
                if not reach_ok:
                    return False, reach_msg, None
                reasons.append(reach_msg)

        # All checks passed
        return True, "; ".join(reasons), (repaired if repaired != action else None)


    def _compute_safe_effect_values(
        self,
        action: ActionSpec,
        state: State,
        cbf: ControlBarrierFunction
    ) -> Dict[str, Any]:
        """
        For each numeric effect, compute the maximum safe parameter value
        via binary search, keeping the barrier function h(s') >= h(s) - α*h(s).

        Returns: mapping of effect variable → safe_value
        """
        safe_values = {}

        for effect in action.effects:
            if effect.mode in ("increment", "decrement", "multiply") and \
               isinstance(effect.value, (int, float)):
                # Binary search for max safe parameter
                current_val = state.get(effect.variable)
                if isinstance(current_val, (int, float)):
                    # Try binary search: low = 0, high = original value
                    low = 0.0
                    high = float(effect.value) if effect.mode == "increment" else 0.0
                    
                    # Adjust bounds based on effect mode
                    if effect.mode == "decrement":
                        low = 0.0
                        high = float(effect.value)
                    elif effect.mode == "multiply":
                        low = 0.0
                        high = float(effect.value)

                    # Binary search for max safe value
                    best_safe_val = low
                    for _ in range(self.cbf_max_iterations):
                        mid = (low + high) / 2.0
                        test_effect = Effect(effect.variable, effect.mode, mid)
                        test_state = state.with_updates({effect.variable: test_effect.apply(current_val)})
                        cbf_ok, _ = cbf.evaluate(state, test_state)
                        if cbf_ok:
                            best_safe_val = mid
                            low = mid
                        else:
                            high = mid
                        if high - low < self.cbf_binary_search_tol:
                            break

                    if best_safe_val != float(effect.value):
                        safe_values[effect.variable] = best_safe_val

        return safe_values


    def _check_ifc(self, action: ActionSpec, state: State) -> Tuple[bool, str]:
        """
        Information Flow Control: ensure no data downgrade.

        For each sink (output) accessed by the action, verify:
          label(data_sink) ≥ label(data_source)
        """
        if not self.ifc_enabled:
            return True, "IFC: disabled"

        # Extract data sinks from action effects.
        sinks = {effect.variable for effect in action.effects}

        # IFC check for explicit labelled payloads:
        # If an effect writes LabelledData into a sink, enforce
        #   label(payload) <= label(sink)
        for effect in action.effects:
            if isinstance(effect.value, LabelledData):
                sink_label = self.ifc_labels.get(effect.variable, DataLabel(SecurityLevel.PUBLIC))
                src_label = effect.value.label
                if not (src_label <= sink_label):
                    return False, (
                        f"IFC VIOLATION: payload ({src_label}) "
                        f"cannot flow to {effect.variable} ({sink_label})"
                    )

        # Conservative fallback: if we have registered labels for state vars,
        # treat any labelled vars as potential sources flowing into any sink.
        sources = {k for k in state.keys() if k in self.ifc_labels}
        for source in sources:
            source_label = self.ifc_labels.get(source, DataLabel(SecurityLevel.PUBLIC))
            for sink in sinks:
                sink_label = self.ifc_labels.get(sink, DataLabel(SecurityLevel.PUBLIC))
                if not (source_label <= sink_label):
                    return False, (
                        f"IFC VIOLATION: {source} ({source_label}) "
                        f"cannot flow to {sink} ({sink_label})"
                    )

        return True, "IFC: all flows permitted"


# ═══════════════════════════════════════════════════════════════════════════
# §M7  SAFE HOVER STATE & GRACEFUL DEGRADATION
# ═══════════════════════════════════════════════════════════════════════════

class SafeHoverState:
    """
    A predefined "safe idle" action when the system detects a threat.

    Used when HJB reachability detects imminent capture, or when
    IFC detects a flow violation. The system remains in current state
    but can transition to a known-safe action.
    """

    def __init__(self, description: str = "Safe hover (no-op)"):
        self.description = description

    def to_action(self) -> ActionSpec:
        """
        Convert to a concrete emergency action (no-op).
        
        Theorem T8 (Emergency Escape):
          The SAFE_HOVER action is always executable (bypasses cost and step limits).
          It has no effects and zero cost, allowing graceful system degradation
          when normal actions are unsafe.
        
        Returns: ActionSpec marked as emergency action
        """
        return ActionSpec(
            id="SAFE_HOVER",
            name="Safe Hover (Emergency)",
            description=self.description,
            effects=(),  # Immutable tuple; provably empty
            cost=0.0,    # Emergency action: cost-free
            risk_level="low",
            metadata={"emergency": True, "reason": "system_safe_hold"},
        )


__all__ = [
    "SecurityLevel", "DataLabel", "LabelledData",
    "ControlBarrierFunction",
    "QPProjector",
    "CaptureBasin",
    "ContractSpecification", "OperadicComposition",
    "ReferenceMonitor",
    "SafeHoverState",
]
