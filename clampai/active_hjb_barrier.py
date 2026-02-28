"""
clampai.active_hjb_barrier — k-step lookahead for capture-basin avoidance.

Implements heuristic reachability analysis. Given the current action set,
it evaluates all paths up to k steps ahead and raises a Safe Hover signal
if any reachable state falls inside a declared capture basin. This is not
complete reachability analysis and has known limits:

  - Complexity is O(n^k): keep k ≤ 3 and the action set small (n ≤ 10).
  - Only explores the provided action set; the full state space is not covered.
  - No memoisation; states may be revisited across branches.
  - Assumes deterministic action effects.

For formal invariant preservation on each committed action, use T3 in
formal.py. This module provides an early-warning complement to that check,
not a replacement. For full reachability proofs, use a model checker such
as TLA+ or SPIN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .formal import ActionSpec, State
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
            print(f"WARNING: HJB lookahead depth {max_lookahead} may be exponentially slow. "
                  f"Consider k <= 3.")
        self.basins = basins
        self.max_lookahead = max_lookahead

    def check_and_enforce(
        self,
        state: State,
        proposed_action: ActionSpec,
        available_actions: List[ActionSpec],
        current_step: int,
        max_steps: int,
    ) -> Tuple[bool, HJBBarrierCheck]:
        """
        Evaluate whether proposed_action is safe under the HJB barrier.

        Returns:
            (is_safe, check_result)

        If is_safe=False, the caller must reject the action and trigger
        recovery (Safe Hover or rollback). The LLM cannot override this.
        """
        # Fresh per-call memoization cache (key = fingerprint + basin + depth).
        cache: Dict[Tuple[str, str, int], bool] = {}

        for basin in self.basins:
            # Check 1: Already trapped.
            if basin.is_bad(state):
                return False, HJBBarrierCheck(
                    violation_type=SafetyBarrierViolation.ALREADY_IN_BASIN,
                    is_safe=False,
                    basin_name=basin.name,
                    distance_to_basin=0,
                    recommendation=(
                        f"CRITICAL: Current state is already inside capture basin "
                        f"'{basin.name}'. Must enter Safe Hover and attempt rollback."
                    ),
                )

            # Check 2: Action enters basin immediately.
            next_state = proposed_action.simulate(state)
            if basin.is_bad(next_state):
                return False, HJBBarrierCheck(
                    violation_type=SafetyBarrierViolation.ACTION_DIRECTLY_ENTERS,
                    is_safe=False,
                    basin_name=basin.name,
                    distance_to_basin=1,
                    recommendation=(
                        f"BARRIER: Action '{proposed_action.name}' would immediately "
                        f"enter capture basin '{basin.name}'. Action rejected."
                    ),
                )

            # Check 3: Basin reachable within lookahead from post-action state.
            if self._is_basin_reachable(
                next_state, available_actions, basin,
                steps_remaining=self.max_lookahead, cache=cache,
            ):
                steps_left = max_steps - current_step
                if steps_left <= self.max_lookahead:
                    return False, HJBBarrierCheck(
                        violation_type=SafetyBarrierViolation.REACHABLE_IN_STEPS,
                        is_safe=False,
                        basin_name=basin.name,
                        distance_to_basin=steps_left,
                        recommendation=(
                            f"WARNING: Capture basin '{basin.name}' reachable within "
                            f"{steps_left} remaining step(s). Entering Safe Hover."
                        ),
                    )

        return True, HJBBarrierCheck(
            violation_type=SafetyBarrierViolation.NO_VIOLATION,
            is_safe=True,
            basin_name=None,
            distance_to_basin=None,
            recommendation="HJB barrier clear",
        )

    def _is_basin_reachable(
        self,
        state: State,
        actions: List[ActionSpec],
        basin: CaptureBasin,
        steps_remaining: int,
        cache: Dict[Tuple[str, str, int], bool],
    ) -> bool:
        """
        Recursive DFS reachability check with per-call memoization.

        Cache key: (state_fingerprint, basin.name, steps_remaining).
        Avoids re-exploring sub-trees when multiple branches converge on
        the same (state, depth). Cache is provided by the caller and scoped
        to one top-level check_and_enforce() invocation.
        """
        if steps_remaining <= 0:
            return False

        key = (state.fingerprint, basin.name, steps_remaining)
        if key in cache:
            return cache[key]

        result = False
        for action in actions:
            next_state = action.simulate(state)
            if basin.is_bad(next_state):
                result = True
                break
            if self._is_basin_reachable(next_state, actions, basin,
                                         steps_remaining - 1, cache):
                result = True
                break

        cache[key] = result
        return result


# Recovery strategy

class RecoveryStrategy(Enum):
    """What the orchestrator should do when the HJB barrier fires."""
    SAFE_HOVER          = "safe_hover"          # Stay in place; do nothing
    ROLLBACK_ONE        = "rollback_one"        # Undo the last committed action
    ROLLBACK_TO_SAFE    = "rollback_to_safe"    # Undo until well clear of basin
    HUMAN_INTERVENTION  = "human_ask"           # Escalate to a human operator
    GRACEFUL_HALT       = "halt"                # Stop execution cleanly


def choose_recovery_strategy(
    barrier_check: HJBBarrierCheck,
    current_step: int,
    max_steps: int,
    is_reversible_available: bool,
) -> RecoveryStrategy:
    """
    Select the least-disruptive recovery for a barrier violation.

    Priority:
      1. Rollback one step if rollback history is available (cheapest recovery).
      2. Safe Hover if we are near the step/budget limit.
      3. Default: Safe Hover.
    """
    if is_reversible_available and current_step > 0:
        return RecoveryStrategy.ROLLBACK_ONE

    if current_step > max_steps * 0.8:
        return RecoveryStrategy.SAFE_HOVER

    return RecoveryStrategy.SAFE_HOVER
