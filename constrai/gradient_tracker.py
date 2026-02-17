"""
Gradient Tracker for Formal Safety Margins (Heuristic)
=======================================================

This module implements HEURISTIC "Jacobian of Safety" — estimation of
how close each state variable is to violating an invariant.

IMPORTANT: This analysis is NOT formally proven. It provides pragmatic
heuristic guidance for LLM prompting and early warnings, but should NOT
be used for safety-critical decisions.

Key idea: For each state variable k and each invariant I, estimate:
    ∇ₖ I(s) := heuristic sensitivity score indicating variable criticality

This enables:
  1. **Prioritization**: which variables matter most? (heuristic)
  2. **Warning signals**: how close are we to invariant boundary? (heuristic)
  3. **State pruning**: drop irrelevant variables from prompts (heuristic)
  4. **Adaptive constraints**: tighten limits when approaching boundary (heuristic)

Assumptions (required for reliability):
  - Invariant predicates are pure functions (no side effects, non-deterministic)
  - Invariants are approximately continuous near current state
  - Perturbation magnitude is relevant to actual dynamics

For formal invariant guarantees, use kernel's T3 (formal.py), NOT these heuristics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
from .formal import State, Invariant, GuaranteeLevel


class SensitivityLevel(Enum):
    """Categorical safety distance."""
    CRITICAL = 0     # Violating now
    SEVERE = 1       # Very close to violation
    WARNING = 2      # Approaching boundary
    ACCEPTABLE = 3   # Safe distance
    FAR = 4          # Well-safe


@dataclass
class GradientScore:
    """Sensitivity of one variable to one invariant."""
    variable: str
    invariant_name: str
    numeric_score: float      # Usually in [0, 1], 0=critical, 1=safe
    level: SensitivityLevel
    diagnostic: str = ""      # Human explanation
    
    @property
    def is_critical(self) -> bool:
        return self.level in (SensitivityLevel.CRITICAL, SensitivityLevel.SEVERE)


@dataclass
class GradientReport:
    """Summary of safety gradients across all variables and invariants."""
    state_fingerprint: str
    gradients: List[GradientScore] = field(default_factory=list)
    critical_variables: List[str] = field(default_factory=list)
    safe_variables: List[str] = field(default_factory=list)
    overall_safety_margin: float = 1.0  # Minimum across all invariants
    
    def rank_by_criticality(self) -> List[GradientScore]:
        """Return gradients sorted by how close to critical."""
        return sorted(self.gradients, key=lambda g: g.numeric_score)
    
    def describe(self) -> str:
        """Human-readable safety report."""
        lines = [
            f"Safety Gradient Report (state={self.state_fingerprint[:8]})",
            f"Overall margin: {self.overall_safety_margin:.1%}",
        ]
        if self.critical_variables:
            lines.append(f"🔴 CRITICAL: {', '.join(self.critical_variables)}")
        if self.safe_variables:
            lines.append(f"✅ SAFE: {', '.join(self.safe_variables)}")
        return "\n".join(lines)


class GradientTracker:
    """
    Compute formal safety margins (gradients) for each state variable
    with respect to each invariant.
    """
    
    def __init__(self, invariants: List[Invariant]):
        self.invariants = invariants
    
    def compute_gradients(self, state: State) -> GradientReport:
        """
        Analyze how close state variables are to violating invariants.
        
        Algorithm:
          For each variable k in state:
            For each invariant I:
              1. Check if I holds on state → baseline
              2. Perturb k (increment/decrement by small ε)
              3. Check if I still holds on perturbed state
              4. Assign sensitivity score
        
        Theorem: This is a LOCAL linear approximation of the constraint
        boundary. It's exact only for linear constraints, but works as a
        heuristic for nonlinear ones.
        """
        report = GradientReport(state_fingerprint=state.fingerprint)
        epsilon = 0.01  # Small perturbation for finite difference
        
        all_scores = []
        
        for var_key in state.keys():
            var_value = state.get(var_key)
            
            for inv in self.invariants:
                # ─ Baseline: does invariant hold now? ─
                baseline_ok, baseline_msg = inv.check(state)
                
                if not baseline_ok:
                    # Already violating → critical
                    score = GradientScore(
                        variable=var_key,
                        invariant_name=inv.name,
                        numeric_score=0.0,
                        level=SensitivityLevel.CRITICAL,
                        diagnostic=f"Invariant already violated: {baseline_msg}"
                    )
                    all_scores.append(score)
                    continue
                
                # ─ Perturbation test: perturb var_key ─
                # Try to find a perturbation that breaks the invariant
                perturb_delta = epsilon
                if isinstance(var_value, (int, float)):
                    perturb_delta = epsilon * (abs(var_value) if var_value else 1.0)
                
                sensitivity_score = 1.0  # Default: "safe distance"
                level = SensitivityLevel.FAR
                diagnostic = "Variable not involved in invariant"
                
                # Try small perturbation
                if isinstance(var_value, (int, float)):
                    for test_val in [var_value + perturb_delta, var_value - perturb_delta]:
                        perturbed_state = state.with_updates({var_key: test_val})
                        perturbed_ok, perturbed_msg = inv.check(perturbed_state)
                        
                        if not perturbed_ok:
                            # Small change broke invariant → variable is sensitive!
                            sensitivity_score = 0.1
                            level = SensitivityLevel.WARNING
                            diagnostic = f"Small change to {var_key} breaks invariant"
                            break
                    else:
                        # Try larger perturbation
                        for test_val in [var_value + 10*perturb_delta, var_value - 10*perturb_delta]:
                            perturbed_state = state.with_updates({var_key: test_val})
                            perturbed_ok, perturbed_msg = inv.check(perturbed_state)
                            
                            if not perturbed_ok:
                                sensitivity_score = 0.3
                                level = SensitivityLevel.ACCEPTABLE
                                diagnostic = f"Larger change to {var_key} breaks invariant"
                                break
                
                score = GradientScore(
                    variable=var_key,
                    invariant_name=inv.name,
                    numeric_score=sensitivity_score,
                    level=level,
                    diagnostic=diagnostic
                )
                all_scores.append(score)
        
        # ─ Aggregate ─
        report.gradients = all_scores
        report.overall_safety_margin = min(
            (s.numeric_score for s in all_scores),
            default=1.0
        )
        
        report.critical_variables = list(set(
            s.variable for s in all_scores
            if s.level in (SensitivityLevel.CRITICAL, SensitivityLevel.SEVERE)
        ))
        
        report.safe_variables = list(set(
            s.variable for s in all_scores
            if s.level == SensitivityLevel.FAR
        ))
        
        return report
    
    def should_trigger_safe_hover(self, report: GradientReport) -> Tuple[bool, str]:
        """
        Decide if we should enter Safe Hover mode based on gradient report.
        
        Heuristic: If ANY critical variable is detected, or overall margin < 10%,
        enter safe hover.
        """
        if report.critical_variables:
            return True, f"Critical variables: {report.critical_variables}"
        
        if report.overall_safety_margin < 0.1:
            return True, f"Overall safety margin dangerously low: {report.overall_safety_margin:.1%}"
        
        return False, "Safety margins acceptable"


# ═════════════════════════════════════════════════════════════════════════════
# Integration: Per-Invariant Safety Budgets
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PerInvariantBudget:
    """
    Adaptive "margin" for each invariant, tightening as we approach boundary.
    
    Like Control Barrier Functions, but applied per-invariant.
    """
    invariant_name: str
    margin_percent: float = 0.2  # 20% safety buffer
    margin_threshold: float = 0.1  # If margin drops below 10%, trigger warning
    
    def compute_buffer(self, gradient: GradientScore) -> float:
        """
        Given a gradient score, compute how much "margin" we have left.
        
        Returns: fraction in [0, 1], where 1.0 = fully safe.
        """
        # Gradient.numeric_score is already in [0,1] where 1=safe
        # Subtract margin_percent for buffer
        buffered = max(0.0, gradient.numeric_score - self.margin_percent)
        return buffered
    
    def is_safe(self, gradient: GradientScore) -> Tuple[bool, str]:
        """Check if we're still safely above the threshold."""
        buffered = self.compute_buffer(gradient)
        if buffered < self.margin_threshold:
            return False, f"Margin for {self.invariant_name} exhausted: {buffered:.1%} < {self.margin_threshold:.1%}"
        return True, f"Safe for {self.invariant_name}: {buffered:.1%} margin"
