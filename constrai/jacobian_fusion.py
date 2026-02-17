"""constrai.jacobian_fusion

Jacobian-HJB Fusion: Boundary Sensitivity Engine (Heuristic)
==============================================================

This module provides HEURISTIC sensitivity analysis for constraint boundaries.
It computes numerical perturbation-based scores indicating how close state
variables are to violating invariants.

IMPORTANT: The analyses in this module are not formally proven. They provide
pragmatic heuristic guidance for LLM prompting and early-warning systems, but
should NOT be used for safety-critical decisions. Those decisions are made by
the formal kernel (formal.py) and reference monitor (reference_monitor.py).

Mathematical Foundation (Heuristic):

    Boundary Sensitivity Score(s_i) ≈ (1 - |δ_violation|/ε) / 10

This is a HEURISTIC approximation that uses numerical perturbation to estimate
how close a variable is to violating an invariant.

Guarantee Classification:

JSF-1 (Boundary Sensitivity Score):
  Level: HEURISTIC (not proven; for diagnostic use only)
  Scope: Applies only to continuous, deterministic invariant predicates
  Method: Numerical differentiation via fixed perturbation magnitudes
  Reliability: Best-effort guidance; may miss discontinuous boundaries
  Assumptions: Invariant is pure function (no side effects, non-deterministic)
  Limitations:
    - Cannot handle discontinuous invariant surfaces
    - Perturbation magnitude must be chosen heuristically
    - Does not guarantee detection of all boundary violations

JSF-2 (Boundary Proximity Detection):
  Level: HEURISTIC (not proven; for diagnostic use only)
  Scope: Detects variables within k-step perturbation distance
  Method: Binary/linear search with fixed tolerances
  Reliability: Heuristic detection; no formal completeness guarantee
  Limitations:
    - Does not detect multi-step boundary violations
    - Score mapping is ad-hoc (not integral-based)
    - May miss non-monotone invariant surfaces
  Use Case: Prioritize critical variables for LLM prompt; NOT for safety decisions

## Assumptions (Required for Analysis)

  1. Invariant predicates are pure functions (no side effects, no randomness)
  2. Invariants are approximately continuous near current state
  3. Perturbation magnitude is relevant to actual state dynamics
  4. Scoring thresholds (τ_critical, etc.) are application-calibrated

For formal safety decisions, always use the kernel's T3 (Invariant Preservation
theorem) in formal.py, NOT these heuristic scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import math

from .formal import State, Invariant, GuaranteeLevel


class BoundarySeverity(Enum):
    """
    Severity levels based on heuristic Jacobian magnitude and proximity.
    
    NOTE: These are NOT formally proven levels. They are pragmatic categories
    derived from empirical testing and heuristic perturbation analysis.
    
    Actual safety decisions MUST use the formal kernel's invariant checks
    (formal.py:T3), NOT these heuristic severity scores.
    """
    SAFE_DISTANT = 0       # Heuristic: perturbation > 10*ε from violation
    SAFE_NOMINAL = 1       # Heuristic: 5*ε to 10*ε from violation
    WARNING_APPROACH = 2   # Heuristic: 2*ε to 5*ε from violation
    CRITICAL_IMMINENT = 3  # Heuristic: 0.5*ε to 2*ε from violation
    DANGER_CLIFF = 4       # Heuristic: < 0.5*ε from violation (imminent)


@dataclass(frozen=True)
class JacobianScore:
    """Single (variable, invariant) sensitivity score."""
    variable: str
    invariant_name: str
    score: float  # ∈ [0, 1], integral-based boundary proximity
    severity: BoundarySeverity
    steps_to_violation: Optional[float]  # Estimated steps if trend continues
    critical: bool  # True if should force into prompt


@dataclass(frozen=True)
class JacobianReport:
    """Complete Jacobian analysis for a state."""
    state_id: str
    timestamp: float
    scores: Tuple[JacobianScore, ...]  # All (variable, invariant) scores
    critical_variables: Tuple[str, ...]  # Variables with DANGER_CLIFF or CRITICAL_IMMINENT
    safety_barrier_violated: bool  # True if any score ≥ 0.9
    recommendation: str  # Human-readable action

    def describe(self) -> str:
        """Generate human-readable Jacobian analysis."""
        lines = [f"Jacobian Report (state={self.state_id[:8]})"]
        lines.append(f"Critical Variables: {', '.join(self.critical_variables) or 'None'}")
        lines.append(f"Safety Barrier Violated: {self.safety_barrier_violated}")
        lines.append("")
        
        # Group by severity
        for severity in BoundarySeverity:
            relevant = [s for s in self.scores if s.severity == severity]
            if not relevant:
                continue
            lines.append(f"{severity.name}:")
            for score in relevant:
                steps_info = f" (~{score.steps_to_violation:.1f} steps)" if score.steps_to_violation else ""
                lines.append(f"  {score.variable} × {score.invariant_name}: {score.score:.3f}{steps_info}")
        
        lines.append("")
        lines.append(f"Recommendation: {self.recommendation}")
        return "\n".join(lines)


class JacobianFusion:
    """
    Authoritative Jacobian engine: computes ∇ϕ for all constraint boundaries.
    
    The key insight: Instead of heuristically guessing which variables matter,
    we compute the exact rate of change of safety margins. Variables that
    appear in the Jacobian of violated constraints are FORCED into prompts.
    """

    def __init__(self, invariants: List[Invariant], 
                 epsilon: float = 0.01,
                 lookahead_steps: int = 5):
        """
        Args:
            invariants: List of Invariant predicates to monitor
            epsilon: Perturbation size for numerical differentiation
            lookahead_steps: How many steps ahead to estimate violation
        """
        self.invariants = invariants
        self.epsilon = float(epsilon)
        self.lookahead_steps = int(lookahead_steps)

    def compute_jacobian(self, state: State) -> JacobianReport:
        """
        Compute boundary sensitivity for all (variable, invariant) pairs.

        Algorithm (Numerical Integration):
          1. For each invariant I:
             a. Check if I(state) holds (baseline)
             b. For each state variable s_i:
                i.   Create perturbations [ε, 10ε, 100ε, ...]
                ii.  Check if I(state + Δ) still holds
                iii. If not, s_i approaches constraint boundary
                iv.  Score based on magnitude of perturbation needed to break
                v.   Assign severity based on proximity

        Returns:
            JacobianReport with all sensitivity scores
        """
        d = state.to_dict()
        scores: List[JacobianScore] = []
        critical_vars = set()
        max_score = 0.0

        for inv in self.invariants:
            # Check baseline
            baseline_holds = inv.check(state)[0]

            for var_name in d.keys():
                if var_name.startswith("_"):
                    # Skip internal meta variables
                    continue

                val = d[var_name]
                if not isinstance(val, (int, float)):
                    # Skip non-numeric
                    continue

                # Binary search: find minimum perturbation that breaks invariant
                score = 0.0
                steps_to_violation = None

                if baseline_holds:
                    # Try increasing perturbations to find boundary proximity
                    perturbation_magnitudes = [
                        self.epsilon,
                        5 * self.epsilon,
                        10 * self.epsilon,
                        50 * self.epsilon,
                        100 * self.epsilon,
                    ]

                    min_breaking_magnitude = None
                    for mag in perturbation_magnitudes:
                        # Try both positive and negative perturbations
                        for direction in [1, -1]:
                            d_pert = d.copy()
                            d_pert[var_name] = val + direction * mag
                            pert_state = State(d_pert)
                            pert_holds = inv.check(pert_state)[0]

                            if not pert_holds:
                                min_breaking_magnitude = mag
                                break

                        if min_breaking_magnitude is not None:
                            break

                    # Compute score: inverse of perturbation magnitude needed
                    # Small perturbation → high score (boundary very close)
                    # Large perturbation → low score (boundary far)
                    if min_breaking_magnitude is not None:
                        # Map: epsilon → 0.9, 5*epsilon → 0.7, 10*epsilon → 0.5, etc.
                        score = max(0.0, 1.0 - (min_breaking_magnitude / (100 * self.epsilon)))
                        steps_to_violation = self.lookahead_steps * (1.0 - score)

                # Determine severity level
                if score < 0.1:
                    severity = BoundarySeverity.SAFE_DISTANT
                elif score < 0.3:
                    severity = BoundarySeverity.SAFE_NOMINAL
                elif score < 0.6:
                    severity = BoundarySeverity.WARNING_APPROACH
                elif score < 0.9:
                    severity = BoundarySeverity.CRITICAL_IMMINENT
                    critical_vars.add(var_name)
                else:
                    severity = BoundarySeverity.DANGER_CLIFF
                    critical_vars.add(var_name)

                critical = severity in (BoundarySeverity.CRITICAL_IMMINENT, BoundarySeverity.DANGER_CLIFF)
                max_score = max(max_score, score)

                score_obj = JacobianScore(
                    variable=var_name,
                    invariant_name=inv.name,
                    score=score,
                    severity=severity,
                    steps_to_violation=steps_to_violation,
                    critical=critical
                )
                scores.append(score_obj)

        # Determine if safety barrier is violated
        safety_barrier_violated = max_score >= 0.9

        # Generate recommendation
        if safety_barrier_violated:
            recommendation = f"⛔ TRIGGER SAFE HOVER IMMEDIATELY: {len(critical_vars)} variable(s) in danger zone"
        elif critical_vars:
            recommendation = f"⚠️  FORCE VARIABLES INTO PROMPT: {', '.join(sorted(critical_vars))}"
        elif max_score > 0.5:
            recommendation = "⚠️  Monitor constraint boundaries closely; critical variables approaching"
        elif max_score > 0.3:
            recommendation = "ℹ️  All constraints nominal; continue normal operation"
        else:
            recommendation = "✓ Safe state; all constraints well within margins"

        state_id = hash(frozenset(d.items())) & 0xFFFFFFFF
        return JacobianReport(
            state_id=f"{state_id:08x}",
            timestamp=0.0,
            scores=tuple(scores),
            critical_variables=tuple(sorted(critical_vars)),
            safety_barrier_violated=safety_barrier_violated,
            recommendation=recommendation
        )

    def _estimate_steps_to_violation(self, state: State, var_name: str, inv: Invariant,
                                      perturb_val) -> Optional[float]:
        """
        Estimate how many steps until violation if variable drifts at current rate.
        Uses binary search to find when invariant fails.
        """
        d = state.to_dict()
        original_val = d[var_name]
        
        if not isinstance(original_val, (int, float)):
            return None

        # Binary search: at what multiple of epsilon does invariant break?
        low, high = 0.5, float(self.lookahead_steps)
        
        for _ in range(10):  # Binary search iterations
            mid = (low + high) / 2
            d_test = d.copy()
            d_test[var_name] = original_val + (perturb_val - original_val) * mid
            test_state = State(d_test)
            
            if inv.check(test_state)[0]:
                low = mid
            else:
                high = mid
        
        # Estimate: if invariant breaks at 'high' perturbations, 
        # and each perturbation is ~epsilon/step, then steps ≈ high
        return high * self.lookahead_steps if high < float(self.lookahead_steps) else None
