"""
Active HJB Safety Barrier (Heuristic)
======================================

This module implements heuristic k-step lookahead reachability analysis for
capture basin avoidance. It is NOT complete reachability analysis.

IMPORTANT: This analysis is HEURISTIC. It provides pragmatic early-warning
detection of immediate (1–3 step) dangers when action set is small. It does
NOT provide formal completeness guarantees or cover the full state space.

For formal invariant preservation, use the kernel's T3 (formal.py).

Wire the Hamilton-Jacobi-Bellman (HJB) reachability analysis directly into
the orchestrator's decision loop, so it forces Safe Hover when capture
basins are detected within lookahead window.

This module acts as a "Physical Barrier" — it's not passive; it actively
constrains what actions can be taken based on reachability heuristics.

Limitations:

1. **Exponential Complexity**: k-step reachability with n actions has
   complexity O(n^k). Scales poorly. Recommended: k ≤ 3, n ≤ 10.

2. **Incomplete**: Only explores actions provided. Does not explore all
   possible state space (which is infinite in general).

3. **No Memoization**: Revisits states across branches. Can be optimized
   with reachability caching, but not implemented.

4. **Approximation**: Assumes actions are deterministic and have no
   stochastic outcomes.

Use Case: Early-warning detection of immediate (1–3 step) dangers when
action set is small and bounded. NOT suitable for long-horizon safety
guarantees or large action spaces.

For formal reachability proofs, use external model checker (e.g., SPIN, TLA+).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from .formal import State, ActionSpec, GuaranteeLevel
from .reference_monitor import CaptureBasin


class SafetyBarrierViolation(Enum):
    """Type of HJB barrier violation."""
    ALREADY_IN_BASIN = "already_in_basin"        # s is already in bad region
    ACTION_DIRECTLY_ENTERS = "direct_entry"      # Action leads directly to basin
    REACHABLE_IN_STEPS = "reachable_in_steps"    # Basin reachable within k steps
    NO_VIOLATION = "safe"                         # No violation


@dataclass
class HJBBarrierCheck:
    """Result of an HJB safety barrier evaluation."""
    violation_type: SafetyBarrierViolation
    is_safe: bool  # = (violation_type == SafetyBarrierViolation.NO_VIOLATION)
    basin_name: Optional[str]
    distance_to_basin: Optional[int]  # Steps until capture basin (if computable)
    recommendation: str  # Human explanation + recovery suggestion


class ActiveHJBBarrier:
    """
    Active HJB Barrier with k-step lookahead reachability (Heuristic, Incomplete)
    
    Guarantee Level: HEURISTIC
    
    This module provides heuristic detection of dangerous states within k lookahead
    steps. It is NOT complete reachability analysis.
    
    Limitations:
        1. **Exponential Complexity**: k-step reachability with n actions has
           complexity O(n^k). This scales poorly. Recommended: k ≤ 3, n ≤ 10 actions.
        
        2. **Incomplete**: Only explores actions provided. Does not explore all
           possible state space (which is infinite in general).
        
        3. **No Memoization**: Revisits states across branches. Can be optimized
           with reachability caching, but not implemented.
        
        4. **Approximation**: Assumes actions are deterministic and have no
           stochastic outcomes.
    
    Use Case:
        Use for early-warning detection of immediate (1–3 step) dangers when
        action set is small and bounded. Not suitable for long-horizon safety
        guarantees or large action spaces.
        
        For formal reachability proofs, use external model checker (e.g., SPIN, TLA+).
    """
    
    def __init__(self, basins: List[CaptureBasin], max_lookahead: int = 3):
        """
        Initialize HJB barrier.
        
        Args:
            basins: Capture basins to avoid
            max_lookahead: Lookahead depth k. Complexity is O(|actions|^k).
                          Recommended: k ≤ 3. Will warn if k > 5.
        """
        if max_lookahead > 5:
            print(f"⚠️  HJB lookahead depth {max_lookahead} may be exponentially slow. "
                  f"Consider k ≤ 3.")
        self.basins = basins
        self.max_lookahead = max_lookahead
    
    def check_and_enforce(
        self,
        state: State,
        proposed_action: ActionSpec,
        available_actions: List[ActionSpec],
        current_step: int,
        max_steps: int
    ) -> Tuple[bool, HJBBarrierCheck]:
        """
        Check if proposed_action violates any capture basin.
        
        Returns: (is_safe, check_result)
        
        If is_safe=False, the orchestrator MUST:
          1. Reject the action
          2. Enter Safe Hover
          3. Attempt recovery or halt gracefully
        """
        
        for basin in self.basins:
            # ─ Check 1: Already in basin? ─
            if basin.is_bad(state):
                return False, HJBBarrierCheck(
                    violation_type=SafetyBarrierViolation.ALREADY_IN_BASIN,
                    is_safe=False,
                    basin_name=basin.name,
                    distance_to_basin=0,
                    recommendation=(
                        f"🔴 CRITICAL: Already in capture basin '{basin.name}'. "
                        f"System must enter Safe Hover and attempt recovery."
                    )
                )
            
            # ─ Check 2: Proposed action directly enters basin? ─
            next_state = proposed_action.simulate(state)
            if basin.is_bad(next_state):
                return False, HJBBarrierCheck(
                    violation_type=SafetyBarrierViolation.ACTION_DIRECTLY_ENTERS,
                    is_safe=False,
                    basin_name=basin.name,
                    distance_to_basin=1,
                    recommendation=(
                        f"🛑 BARRIER: Action '{proposed_action.name}' would immediately "
                        f"enter capture basin '{basin.name}'. Action rejected. "
                        f"Proposing Safe Hover."
                    )
                )
            
            # ─ Check 3: Reachable within lookahead? ─
            # (Simplified: only check next few actions from current state)
            if self._is_basin_reachable(state, available_actions, basin, steps=self.max_lookahead):
                steps_left = max_steps - current_step
                if steps_left < self.max_lookahead:
                    # Very close to the edge
                    return False, HJBBarrierCheck(
                        violation_type=SafetyBarrierViolation.REACHABLE_IN_STEPS,
                        is_safe=False,
                        basin_name=basin.name,
                        distance_to_basin=steps_left,
                        recommendation=(
                            f"⚠️ WARNING: Capture basin '{basin.name}' reachable "
                            f"within {steps_left} steps (all remaining budget). "
                            f"Entering Safe Hover to preserve margin."
                        )
                    )
        
        # ─ All checks passed ─
        return True, HJBBarrierCheck(
            violation_type=SafetyBarrierViolation.NO_VIOLATION,
            is_safe=True,
            basin_name=None,
            distance_to_basin=None,
            recommendation="✅ HJB barrier clear"
        )
    
    def _is_basin_reachable(
        self,
        state: State,
        actions: List[ActionSpec],
        basin: CaptureBasin,
        steps: int = 3
    ) -> bool:
        """
        BFS-style reachability: can we reach basin in 'steps' actions?
        (Simplified version — real implementation would enumerate paths)
        """
        if steps <= 0:
            return False
        
        for action in actions:
            next_state = action.simulate(state)
            if basin.is_bad(next_state):
                return True
            
            # Recurse (limited depth)
            if self._is_basin_reachable(next_state, actions, basin, steps - 1):
                return True
        
        return False


# ═════════════════════════════════════════════════════════════════════════════
# Recovery Strategy Enum
# ═════════════════════════════════════════════════════════════════════════════

class RecoveryStrategy(Enum):
    """What to do when HJB barrier is triggered."""
    SAFE_HOVER = "safe_hover"          # Stay in current state, do nothing
    ROLLBACK_ONE = "rollback_one"      # Undo last action
    ROLLBACK_TO_SAFE = "rollback_to_safe"  # Undo until we're far from basin
    HUMAN_INTERVENTION = "human_ask"   # Ask human for decision
    GRACEFUL_HALT = "halt"             # Stop gracefully


def choose_recovery_strategy(
    barrier_check: HJBBarrierCheck,
    current_step: int,
    max_steps: int,
    is_reversible_available: bool
) -> RecoveryStrategy:
    """
    Decide what recovery action to take given the barrier violation.
    """
    
    # If we can rollback and still have budget, try it
    if is_reversible_available and current_step > 0:
        return RecoveryStrategy.ROLLBACK_ONE
    
    # If we're very close to the end, just hover
    if current_step > max_steps * 0.8:
        return RecoveryStrategy.SAFE_HOVER
    
    # Default: safe hover
    return RecoveryStrategy.SAFE_HOVER
